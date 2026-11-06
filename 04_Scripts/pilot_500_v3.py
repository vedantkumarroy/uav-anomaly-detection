import json
import os
import random
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
DBINFO_CACHE = "pilot/dbinfo.json"
OUTDIR = "pilot_500_v3"

N_LOGS = 500
SEED = 7

MIN_DUR = 60.0
MAX_DUR = 3600.0
VALID_AUTOSTART = "4001"

MOTOR_SAT_THRESHOLD = 0.95
NAV_FORBIDDEN = {10, 17, 18}


def decode_ver_sw(v):
    if v is None or v == "":
        return None
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def ver_band(ver_sw):
    decoded = decode_ver_sw(ver_sw)
    if decoded is None:
        return None
    major, minor, patch, rtype = decoded
    if major != 1:
        return None
    if minor <= 13:
        return "v1.12-1.13"
    if minor in (14, 15):
        return "v1.14-1.15"
    if minor >= 16:
        return "v1.16+"
    return None


def stream_to_temp(log_id, session):
    url = BASE + DOWNLOAD_PATH.format(log_id=log_id)
    try:
        r = session.get(url, timeout=180, stream=True)
        if r.status_code != 200:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg", dir=OUTDIR, prefix="tmp_")
        n = 0
        with open(tmp.name, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
                n += len(chunk)
        if n < 1024:
            os.remove(tmp.name)
            return None
        return tmp.name
    except Exception:
        return None


def longest_run_seconds(values_bool, timestamps_s):
    """Return longest consecutive run of True, in seconds."""
    if len(values_bool) == 0:
        return 0.0
    best = 0.0
    start = None
    for i, v in enumerate(values_bool):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dur = timestamps_s[i] - timestamps_s[start]
            if dur > best:
                best = dur
            start = None
    if start is not None:
        dur = timestamps_s[-1] - timestamps_s[start]
        if dur > best:
            best = dur
    return best


def get_airborne_mask(u, target_len, target_ts):
    """Return a boolean mask (True = airborne) aligned to target timestamps."""
    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is None or "landed" not in ld.data or "timestamp" not in ld.data:
        return np.ones(target_len, dtype=bool)
    ld_ts = np.asarray(ld.data["timestamp"], dtype=np.float64)
    ld_landed = np.asarray(ld.data["landed"], dtype=bool)
    if len(ld_ts) == 0:
        return np.ones(target_len, dtype=bool)
    # interpolate landed status onto target timestamps using nearest neighbor
    idx = np.searchsorted(ld_ts, target_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ld_landed) - 1)
    landed_at_target = ld_landed[idx]
    return ~landed_at_target


def compute_actuator_stats(u):
    """Return dict with the five actuator scalars."""
    out = {
        "sat_frac": np.nan,
        "sat_max_run_s": np.nan,
        "motor_spread_p95": np.nan,
        "n_motors": np.nan,
        "actuator_source": "none",
        "pwm_source": "none",
    }

    motors = next((d for d in u.data_list if d.name == "actuator_motors"), None)
    outputs = next((d for d in u.data_list if d.name == "actuator_outputs"), None)

    if motors is not None:
        out["actuator_source"] = "actuator_motors"
        ts = np.asarray(motors.data.get("timestamp", []), dtype=np.float64)
        cols = sorted([k for k in motors.data.keys() if k.startswith("control[")])
        # extract only columns with any non-NaN value
        active = []
        for c in cols:
            arr = np.asarray(motors.data[c], dtype=np.float64)
            if np.isnan(arr).all():
                continue
            if np.nanmax(np.abs(arr)) == 0:
                continue
            active.append(c)
        if not active:
            return out
        arr = np.column_stack([np.asarray(motors.data[c], dtype=np.float64) for c in active])
        out["n_motors"] = len(active)

        # airborne mask
        mask = get_airborne_mask(u, len(ts), ts)
        if mask.sum() < 2:
            return out
        arr_air = arr[mask]
        ts_air = ts[mask]

        # per-sample: max across motors
        max_motor = np.nanmax(arr_air, axis=1)
        min_motor = np.nanmin(arr_air, axis=1)

        # sat_frac
        saturated = max_motor >= MOTOR_SAT_THRESHOLD
        out["sat_frac"] = float(np.mean(saturated))

        # sat_max_run_s
        out["sat_max_run_s"] = longest_run_seconds(saturated, (ts_air - ts_air[0]) / 1e6)

        # motor_spread_p95
        spread = max_motor - min_motor
        spread = spread[np.isfinite(spread)]
        if len(spread) > 0:
            out["motor_spread_p95"] = float(np.percentile(spread, 95))

        return out

    if outputs is not None:
        out["actuator_source"] = "actuator_outputs"
        ts = np.asarray(outputs.data.get("timestamp", []), dtype=np.float64)
        noutputs = outputs.data.get("noutputs")
        cols = sorted([k for k in outputs.data.keys() if k.startswith("output[")])

        # only consider first noutputs channels
        if noutputs is not None:
            try:
                n = int(np.asarray(noutputs).flatten()[0])
                cols = [c for c in cols if int(c.split("[")[1].rstrip("]")) < n]
            except Exception:
                pass

        # drop all-zero columns
        active = []
        for c in cols:
            arr = np.asarray(outputs.data[c], dtype=np.float64)
            if np.nanmax(np.abs(arr)) == 0:
                continue
            active.append(c)
        if not active:
            return out
        arr = np.column_stack([np.asarray(outputs.data[c], dtype=np.float64) for c in active])
        out["n_motors"] = len(active)

        # PWM parameters
        params = u.initial_parameters or {}
        pwm_min, pwm_max, src = get_pwm_range(params, active)

        if pwm_min is None or pwm_max is None or pwm_max <= pwm_min:
            out["pwm_source"] = "none"
            return out

        out["pwm_source"] = src

        # normalize
        norm = (arr - pwm_min) / (pwm_max - pwm_min)
        norm = np.clip(norm, 0.0, 1.0)

        # airborne mask
        mask = get_airborne_mask(u, len(ts), ts)
        if mask.sum() < 2:
            return out
        norm_air = norm[mask]
        ts_air = ts[mask]

        max_motor = np.nanmax(norm_air, axis=1)
        min_motor = np.nanmin(norm_air, axis=1)

        saturated = max_motor >= MOTOR_SAT_THRESHOLD
        out["sat_frac"] = float(np.mean(saturated))
        out["sat_max_run_s"] = longest_run_seconds(saturated, (ts_air - ts_air[0]) / 1e6)

        spread = max_motor - min_motor
        spread = spread[np.isfinite(spread)]
        if len(spread) > 0:
            out["motor_spread_p95"] = float(np.percentile(spread, 95))

        return out

    return out


def get_pwm_range(params, active_cols):
    """Return (pwm_min, pwm_max, source). Source: per_channel, bus, global, none."""
    # per-channel PWM_MAIN_MIN1, PWM_MAIN_MAX1
    try:
        channel_indices = [int(c.split("[")[1].rstrip("]")) for c in active_cols]
    except Exception:
        channel_indices = []

    per_min, per_max = [], []
    ok_per = True
    for idx in channel_indices:
        k_min = f"PWM_MAIN_MIN{idx + 1}"
        k_max = f"PWM_MAIN_MAX{idx + 1}"
        v_min = params.get(k_min)
        v_max = params.get(k_max)
        if v_min is None or v_max is None or float(v_min) <= 0 or float(v_max) <= 0:
            ok_per = False
            break
        per_min.append(float(v_min))
        per_max.append(float(v_max))

    if ok_per and per_min:
        # use the min of mins and max of maxes so we do not clip
        return min(per_min), max(per_max), "per_channel"

    # bus-level
    v_min = params.get("PWM_MAIN_MIN")
    v_max = params.get("PWM_MAIN_MAX")
    if v_min is not None and v_max is not None:
        try:
            if float(v_min) > 0 and float(v_max) > float(v_min):
                return float(v_min), float(v_max), "bus"
        except Exception:
            pass

    # global
    v_min = params.get("PWM_MIN")
    v_max = params.get("PWM_MAX")
    if v_min is not None and v_max is not None:
        try:
            if float(v_min) > 0 and float(v_max) > float(v_min):
                return float(v_min), float(v_max), "global"
        except Exception:
            pass

    return None, None, "none"


def compute_indicators(u):
    """Compute the five indicators as floats."""
    out = {
        "failsafe_frac": np.nan,
        "nav_state_forbidden_frac": np.nan,
        "filter_fault_flags_frac": np.nan,
        "landed_at_end": None,
    }

    vs = next((d for d in u.data_list if d.name == "vehicle_status"), None)
    if vs is not None and "failsafe" in vs.data:
        fs = np.asarray(vs.data["failsafe"]).astype(bool)
        if len(fs) > 0:
            out["failsafe_frac"] = float(np.mean(fs))

    if vs is not None and "nav_state" in vs.data:
        ns = np.asarray(vs.data["nav_state"])
        if len(ns) > 0:
            out["nav_state_forbidden_frac"] = float(np.mean(np.isin(ns, list(NAV_FORBIDDEN))))

    es = next((d for d in u.data_list if d.name == "estimator_status"), None)
    if es is not None and "filter_fault_flags" in es.data:
        ff = np.asarray(es.data["filter_fault_flags"])
        if len(ff) > 0:
            out["filter_fault_flags_frac"] = float(np.mean(ff != 0))

    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is not None and "landed" in ld.data:
        landed = np.asarray(ld.data["landed"]).astype(bool)
        if len(landed) > 0:
            out["landed_at_end"] = bool(landed[-1])

    return out


def write_parquet_and_count(u, log_id):
    logdir = os.path.join(OUTDIR, "logs", log_id)
    os.makedirs(logdir, exist_ok=True)
    per_topic = {}
    total = 0
    for d in u.data_list:
        try:
            arrays = {k: pa.array(v) for k, v in d.data.items()}
            table = pa.table(arrays)
            path = os.path.join(logdir, f"{d.name}.parquet")
            pq.write_table(table, path, compression="zstd")
            size = os.path.getsize(path)
            per_topic[d.name] = size
            total += size
        except Exception:
            continue
    return total, per_topic


def process(rec, session):
    log_id = rec["log_id"]
    result = {"log_id": log_id, "status": "unknown"}

    t0 = time.time()
    path = stream_to_temp(log_id, session)
    if path is None:
        result["status"] = "download_failed"
        return result

    try:
        from pyulog import ULog
        try:
            u = ULog(path)
        except Exception:
            result["status"] = "parse_failed"
            return result

        info = u.msg_info_dict or {}
        ver_sw = info.get("ver_sw_release")
        result["ver_sw"] = ver_sw
        result["band"] = ver_band(ver_sw)

        indicators = compute_indicators(u)
        result.update(indicators)

        act = compute_actuator_stats(u)
        result.update(act)

        result["ulg_bytes"] = os.path.getsize(path)
        size, per_topic = write_parquet_and_count(u, log_id)
        result["parquet_bytes"] = size
        result["status"] = "ok"

        # store per-topic sizes in a shared dict for later
        result["_per_topic_sizes"] = per_topic

        u = None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
        result["elapsed_s"] = round(time.time() - t0, 1)

    return result


def assign_groups(df, seed):
    rng = random.Random(seed)
    df["group"] = None
    for band in df["band"].dropna().unique():
        idx = df[df["band"] == band].index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_a = int(round(0.20 * n))
        n_b = int(round(0.40 * n))
        df.loc[idx[:n_a], "group"] = "A"
        df.loc[idx[n_a:n_a + n_b], "group"] = "B"
        df.loc[idx[n_a + n_b:], "group"] = "C"
    return df


def main():
    os.makedirs(os.path.join(OUTDIR, "logs"), exist_ok=True)

    with open(DBINFO_CACHE) as f:
        records = json.load(f)

    cand = []
    for r in records:
        if not r.get("log_id"):
            continue
        if str(r.get("mav_type", "")).lower() != "quadrotor":
            continue
        if str(r.get("sys_autostart_id", "")) != VALID_AUTOSTART:
            continue
        try:
            d = float(r.get("duration_s") or r.get("duration") or 0)
        except (TypeError, ValueError):
            continue
        if not (MIN_DUR <= d <= MAX_DUR):
            continue
        cand.append(r)

    print(f"candidates (4001 pre-filtered): {len(cand)}")
    random.seed(SEED)
    sample = random.sample(cand, min(N_LOGS, len(cand)))
    print(f"seed = {SEED}, streaming {len(sample)} logs\n")

    session = requests.Session()
    session.headers["User-Agent"] = "px4-pilot/3.0"

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process, r, session): r["log_id"] for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0:
                print(f"  processed {i}/{len(sample)}")

    elapsed = round(time.time() - t_start, 1)

    # extract per-topic sizes
    per_topic_agg = defaultdict(list)
    for r in results:
        sizes = r.pop("_per_topic_sizes", {})
        for topic, s in sizes.items():
            per_topic_agg[topic].append(s)

    df = pd.DataFrame(results)

    # drop non-PX4 logs
    df["excluded_reason"] = ""
    for i, row in df.iterrows():
        if pd.isna(row["band"]) and row["status"] == "ok":
            ver = row["ver_sw"]
            if pd.isna(ver):
                df.at[i, "excluded_reason"] = "missing_ver_sw"
            else:
                try:
                    v = int(float(ver))
                    major = (v >> 24) & 0xFF
                    if major != 1:
                        df.at[i, "excluded_reason"] = f"not_px4_v{major}"
                except Exception:
                    df.at[i, "excluded_reason"] = "unparseable_ver_sw"

    df_clean = df[(df["status"] == "ok") & (df["band"].notna())].copy()
    df_clean = assign_groups(df_clean, SEED)

    # merge back into full df
    df = df.merge(df_clean[["log_id", "group"]], on="log_id", how="left")

    # summary
    print()
    print(f"=== Summary ===")
    print(f"elapsed: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"attempted: {len(df)}")
    print(f"ok: {int((df['status']=='ok').sum())}")
    print(f"clean (PX4, has band): {len(df_clean)}")
    print()

    kept = df[df["status"] == "ok"]
    print(f"total ulg bytes:      {kept['ulg_bytes'].sum()/1e6:.1f} MB")
    print(f"total parquet bytes:  {kept['parquet_bytes'].sum()/1e6:.1f} MB")
    if kept['ulg_bytes'].sum() > 0:
        print(f"compression ratio:    {100*kept['parquet_bytes'].sum()/kept['ulg_bytes'].sum():.1f}%")

    print()
    print("band x group (clean only):")
    print(pd.crosstab(df_clean["band"], df_clean["group"]))

    print()
    print("=== Actuator source distribution ===")
    print(kept["actuator_source"].value_counts())
    print()
    print("=== PWM source distribution (actuator_outputs only) ===")
    outputs_only = kept[kept["actuator_source"] == "actuator_outputs"]
    if len(outputs_only) > 0:
        print(outputs_only["pwm_source"].value_counts())

    print()
    print("=== Five indicator summary ===")
    for col in ["failsafe_frac", "nav_state_forbidden_frac",
                "filter_fault_flags_frac", "actuator_saturation_frac"]:
        if col not in kept.columns:
            continue
        vals = kept[col].dropna()
        print(f"{col}:")
        print(f"  count > 0: {int((vals > 0).sum())} of {len(vals)}")
        print(f"  median:    {vals.median():.4f}")
        print(f"  max:       {vals.max():.4f}")

    print()
    print("=== Actuator scalars ===")
    for col in ["sat_frac", "sat_max_run_s", "motor_spread_p95"]:
        vals = kept[col].dropna()
        print(f"{col}:")
        print(f"  count non-NaN: {len(vals)}")
        print(f"  median:        {vals.median():.4f}")
        print(f"  max:           {vals.max():.4f}")

    landed = kept["landed_at_end"].dropna().astype(bool)
    print()
    print(f"landed_at_end:")
    print(f"  True:  {int(landed.sum())} of {len(landed)}")
    print(f"  False: {int((~landed).sum())} of {len(landed)}")

    # present_frac by band for five indicators
    print()
    print("=== Indicator present_frac by band (fraction of logs with value > 0) ===")
    bands = ["v1.12-1.13", "v1.14-1.15", "v1.16+"]
    indicators = ["failsafe_frac", "nav_state_forbidden_frac", "filter_fault_flags_frac",
                  "actuator_saturation_frac", "landed_at_end"]
    print(f"{'indicator':<30}" + "".join(f"{b:>14}" for b in bands))
    for ind in indicators:
        if ind not in df_clean.columns:
            continue
        line = ind.ljust(30)
        for b in bands:
            sub = df_clean[df_clean["band"] == b]
            if ind == "landed_at_end":
                # landed present means true, absent means false; count False as "tripped"
                vals = sub[ind].dropna().astype(bool)
                frac = float((~vals).mean()) if len(vals) > 0 else float("nan")
            else:
                vals = sub[ind].dropna()
                frac = float((vals > 0).mean()) if len(vals) > 0 else float("nan")
            line += f"{frac:.3f}".rjust(14)
        print(line)

    # save CSVs
    df.to_csv("pilot_500_v3_index.csv", index=False)
    print(f"\nsaved pilot_500_v3_index.csv")

    # per-topic bytes
    topic_rows = []
    for topic, sizes in per_topic_agg.items():
        topic_rows.append({
            "topic": topic,
            "n_logs": len(sizes),
            "total_bytes": int(sum(sizes)),
            "median_bytes": int(np.median(sizes)) if sizes else 0,
        })
    pt_df = pd.DataFrame(topic_rows).sort_values("total_bytes", ascending=False)
    pt_df.to_csv("pilot_500_v3_topic_bytes.csv", index=False)
    print("saved pilot_500_v3_topic_bytes.csv")
    print()
    print("=== Top 20 topics by total bytes ===")
    print(pt_df.head(20).to_string(index=False))

    with open("pilot_500_v3_seed.txt", "w") as f:
        f.write(f"seed={SEED}\nn_logs={len(sample)}\n")
    print("saved pilot_500_v3_seed.txt")


if __name__ == "__main__":
    main()