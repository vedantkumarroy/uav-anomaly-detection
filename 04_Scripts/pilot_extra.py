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
OUTDIR = "pilot_extra"

N_LOGS = 500
SEED = 11

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
    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is None or "landed" not in ld.data or "timestamp" not in ld.data:
        return np.ones(target_len, dtype=bool)
    ld_ts = np.asarray(ld.data["timestamp"], dtype=np.float64)
    ld_landed = np.asarray(ld.data["landed"], dtype=bool)
    if len(ld_ts) == 0:
        return np.ones(target_len, dtype=bool)
    idx = np.searchsorted(ld_ts, target_ts, side="right") - 1
    idx = np.clip(idx, 0, len(ld_landed) - 1)
    return ~ld_landed[idx]


def get_pwm_range(params, active_cols):
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
        return min(per_min), max(per_max), "per_channel"
    v_min = params.get("PWM_MAIN_MIN")
    v_max = params.get("PWM_MAIN_MAX")
    if v_min is not None and v_max is not None:
        try:
            if float(v_min) > 0 and float(v_max) > float(v_min):
                return float(v_min), float(v_max), "bus"
        except Exception:
            pass
    v_min = params.get("PWM_MIN")
    v_max = params.get("PWM_MAX")
    if v_min is not None and v_max is not None:
        try:
            if float(v_min) > 0 and float(v_max) > float(v_min):
                return float(v_min), float(v_max), "global"
        except Exception:
            pass
    return None, None, "none"


def compute_actuator_stats(u):
    out = {
        "sat_frac": np.nan, "sat_max_run_s": np.nan, "motor_spread_p95": np.nan,
        "n_motors": np.nan, "actuator_source": "none", "pwm_source": "none",
    }
    motors = next((d for d in u.data_list if d.name == "actuator_motors"), None)
    outputs = next((d for d in u.data_list if d.name == "actuator_outputs"), None)

    if motors is not None:
        out["actuator_source"] = "actuator_motors"
        ts = np.asarray(motors.data.get("timestamp", []), dtype=np.float64)
        cols = sorted([k for k in motors.data.keys() if k.startswith("control[")])
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
        mask = get_airborne_mask(u, len(ts), ts)
        if mask.sum() < 2:
            return out
        arr_air = arr[mask]
        ts_air = ts[mask]
        all_nan = np.isnan(arr_air).all(axis=1)
        if all_nan.all():
            return out
        arr_v = arr_air[~all_nan]
        ts_v = ts_air[~all_nan]
        if len(arr_v) == 0:
            return out
        mx = np.nanmax(arr_v, axis=1)
        mn = np.nanmin(arr_v, axis=1)
        sat = mx >= MOTOR_SAT_THRESHOLD
        out["sat_frac"] = float(np.mean(sat))
        out["sat_max_run_s"] = longest_run_seconds(sat, (ts_v - ts_v[0]) / 1e6)
        sp = mx - mn
        sp = sp[np.isfinite(sp)]
        if len(sp) > 0:
            out["motor_spread_p95"] = float(np.percentile(sp, 95))
        return out

    if outputs is not None:
        out["actuator_source"] = "actuator_outputs"
        ts = np.asarray(outputs.data.get("timestamp", []), dtype=np.float64)
        noutputs = outputs.data.get("noutputs")
        cols = sorted([k for k in outputs.data.keys() if k.startswith("output[")])
        if noutputs is not None:
            try:
                n = int(np.asarray(noutputs).flatten()[0])
                cols = [c for c in cols if int(c.split("[")[1].rstrip("]")) < n]
            except Exception:
                pass
        active = []
        for c in cols:
            arr = np.asarray(outputs.data[c], dtype=np.float64)
            if np.nanmax(np.abs(arr)) == 0:
                continue
            active.append(c)
        if not active:
            return out
        out["n_motors"] = len(active)
        params = u.initial_parameters or {}
        pwm_min, pwm_max, src = get_pwm_range(params, active)
        out["pwm_source"] = src
        if pwm_min is None or pwm_max is None or pwm_max <= pwm_min:
            return out
        arr = np.column_stack([np.asarray(outputs.data[c], dtype=np.float64) for c in active])
        norm = np.clip((arr - pwm_min) / (pwm_max - pwm_min), 0.0, 1.0)
        mask = get_airborne_mask(u, len(ts), ts)
        if mask.sum() < 2:
            return out
        norm_air = norm[mask]
        ts_air = ts[mask]
        all_nan = np.isnan(norm_air).all(axis=1)
        if all_nan.all():
            return out
        arr_v = norm_air[~all_nan]
        ts_v = ts_air[~all_nan]
        if len(arr_v) == 0:
            return out
        mx = np.nanmax(arr_v, axis=1)
        mn = np.nanmin(arr_v, axis=1)
        sat = mx >= MOTOR_SAT_THRESHOLD
        out["sat_frac"] = float(np.mean(sat))
        out["sat_max_run_s"] = longest_run_seconds(sat, (ts_v - ts_v[0]) / 1e6)
        sp = mx - mn
        sp = sp[np.isfinite(sp)]
        if len(sp) > 0:
            out["motor_spread_p95"] = float(np.percentile(sp, 95))
        return out

    return out


def compute_indicators(u):
    out = {"failsafe_frac": np.nan, "nav_state_forbidden_frac": np.nan,
           "filter_fault_flags_frac": np.nan, "landed_at_end": None}
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
        result.update(compute_indicators(u))
        result.update(compute_actuator_stats(u))
        result["ulg_bytes"] = os.path.getsize(path)
        size, per_topic = write_parquet_and_count(u, log_id)
        result["parquet_bytes"] = size
        result["status"] = "ok"
        result["_per_topic_sizes"] = per_topic
        u = None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
        result["elapsed_s"] = round(time.time() - t0, 1)
    return result


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
    print(f"candidates: {len(cand)}")
    random.seed(SEED)
    sample = random.sample(cand, min(N_LOGS, len(cand)))
    print(f"seed = {SEED}, streaming {len(sample)} logs\n")
    session = requests.Session()
    session.headers["User-Agent"] = "px4-extra/1.0"
    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process, r, session): r["log_id"] for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"  processed {i}/{len(sample)}")
    elapsed = round(time.time() - t_start, 1)
    per_topic_agg = defaultdict(list)
    for r in results:
        sizes = r.pop("_per_topic_sizes", {})
        for topic, s in sizes.items():
            per_topic_agg[topic].append(s)
    df = pd.DataFrame(results)
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
    print()
    print(f"=== Summary ===")
    print(f"elapsed: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"attempted: {len(df)}")
    print(f"ok: {int((df['status']=='ok').sum())}")
    print(f"clean: {int(df['band'].notna().sum())}")
    kept = df[df["status"] == "ok"]
    print(f"total parquet bytes: {kept['parquet_bytes'].sum()/1e6:.1f} MB")
    df.to_csv("pilot_extra_index.csv", index=False)
    print(f"\nsaved pilot_extra_index.csv")
    with open("pilot_extra_seed.txt", "w") as f:
        f.write(f"seed={SEED}\nn_logs={len(sample)}\n")
    print("saved pilot_extra_seed.txt")


if __name__ == "__main__":
    main()