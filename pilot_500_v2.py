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
OUTDIR = "pilot_500_v2"

N_LOGS = 500
SEED = 6

MIN_DUR = 60.0
MAX_DUR = 3600.0
VALID_AUTOSTART = "4001"

# actuator saturation thresholds
MOTOR_HIGH = 0.99
MOTOR_LOW = 0.01
OUTPUT_HIGH = 1950
OUTPUT_LOW = 1050

# nav_state forbidden values
FORBIDDEN_NAV = {10, 17, 18}


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


def compute_indicators(u):
    """Compute the five indicators. Returns dict of floats."""
    out = {
        "failsafe_frac": np.nan,
        "nav_state_forbidden_frac": np.nan,
        "filter_fault_flags_frac": np.nan,
        "landed_at_end": None,
        "actuator_saturation_frac": np.nan,
        "rate_ratios": 0.0,
        "dropout_frac": np.nan,
    }

    # 1. failsafe
    vs = next((d for d in u.data_list if d.name == "vehicle_status"), None)
    if vs is not None and "failsafe" in vs.data:
        fs = np.asarray(vs.data["failsafe"])
        out["failsafe_frac"] = float(np.mean(fs.astype(bool))) if len(fs) > 0 else np.nan

    # 2. nav_state forbidden
    if vs is not None and "nav_state" in vs.data:
        ns = np.asarray(vs.data["nav_state"])
        if len(ns) > 0:
            forbidden_mask = np.isin(ns, list(FORBIDDEN_NAV))
            out["nav_state_forbidden_frac"] = float(np.mean(forbidden_mask))

    # 3. filter fault flags
    es = next((d for d in u.data_list if d.name == "estimator_status"), None)
    if es is not None and "filter_fault_flags" in es.data:
        ff = np.asarray(es.data["filter_fault_flags"])
        if len(ff) > 0:
            out["filter_fault_flags_frac"] = float(np.mean(ff != 0))

    # 4. landed at end
    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is not None and "landed" in ld.data:
        landed = np.asarray(ld.data["landed"])
        if len(landed) > 0:
            out["landed_at_end"] = bool(landed[-1])

    # 5. actuator saturation
    act = None
    act_topic = None
    for name in ("actuator_motors", "actuator_outputs"):
        d = next((x for x in u.data_list if x.name == name), None)
        if d is not None:
            act = d
            act_topic = name
            break
    if act is not None:
        cols = [k for k in act.data.keys()
                if k.startswith("output") or "motor" in k.lower()]
        # exclude non-motor columns
        cols = [c for c in cols if "timestamp" not in c]
        if cols:
            arr = np.column_stack([np.asarray(act.data[c], dtype=float) for c in cols])
            if act_topic == "actuator_motors":
                hi, lo = MOTOR_HIGH, MOTOR_LOW
            else:
                hi, lo = OUTPUT_HIGH, OUTPUT_LOW
            sat_any = np.any((arr >= hi) | (arr <= lo), axis=1)
            out["actuator_saturation_frac"] = float(np.mean(sat_any))

    # rate_ratios
    for d in u.data_list:
        if d.name != "estimator_innovation_test_ratios":
            continue
        ts = d.data.get("timestamp")
        if ts is None or len(ts) < 2:
            continue
        span = (float(ts[-1]) - float(ts[0])) / 1e6
        if span > 0:
            out["rate_ratios"] = (len(ts) - 1) / span
        break

    # dropout_frac
    try:
        dur = (u.last_timestamp - u.start_timestamp) / 1e6
    except Exception:
        dur = 0.0
    if dur > 0:
        drop_ms = sum(getattr(d, "duration", 0) for d in (u.dropouts or []))
        out["dropout_frac"] = (drop_ms / 1000.0) / dur

    return out


def write_parquet(u, log_id):
    logdir = os.path.join(OUTDIR, "logs", log_id)
    os.makedirs(logdir, exist_ok=True)
    total = 0
    for d in u.data_list:
        try:
            arrays = {k: pa.array(v) for k, v in d.data.items()}
            table = pa.table(arrays)
            path = os.path.join(logdir, f"{d.name}.parquet")
            pq.write_table(table, path, compression="zstd")
            total += os.path.getsize(path)
        except Exception:
            continue
    return total


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

        result["ulg_bytes"] = os.path.getsize(path)
        size = write_parquet(u, log_id)
        result["parquet_bytes"] = size
        result["status"] = "ok"

        u = None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
        result["elapsed_s"] = round(time.time() - t0, 1)

    return result


def assign_groups(df, seed):
    """Assign A/B/C per band using 20/40/40 split. One shuffle per band."""
    rng = random.Random(seed)
    df["group"] = None
    for band in df["band"].dropna().unique():
        idx = df[df["band"] == band].index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_a = int(round(0.20 * n))
        n_b = int(round(0.40 * n))
        # A = first n_a, B = next n_b, C = rest
        a_idx = idx[:n_a]
        b_idx = idx[n_a:n_a + n_b]
        c_idx = idx[n_a + n_b:]
        df.loc[a_idx, "group"] = "A"
        df.loc[b_idx, "group"] = "B"
        df.loc[c_idx, "group"] = "C"
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
    session.headers["User-Agent"] = "px4-pilot/2.0"

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process, r, session): r["log_id"] for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0:
                print(f"  processed {i}/{len(sample)}")

    elapsed = round(time.time() - t_start, 1)

    df = pd.DataFrame(results)

    # assign A/B/C groups
    df = assign_groups(df, SEED)

    # summary
    ok = df[df["status"] == "ok"]
    print()
    print(f"=== Summary ===")
    print(f"elapsed: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"attempted: {len(df)}")
    print(f"ok: {len(ok)}")
    print(f"failed: {len(df) - len(ok)}")
    print()
    print(f"total ulg bytes:       {ok['ulg_bytes'].sum()/1e6:.1f} MB")
    print(f"total parquet bytes:   {ok['parquet_bytes'].sum()/1e6:.1f} MB")
    if ok['ulg_bytes'].sum() > 0:
        print(f"compression ratio:     {100*ok['parquet_bytes'].sum()/ok['ulg_bytes'].sum():.1f}%")

    print()
    print("=== Band x Group counts ===")
    print(pd.crosstab(df["band"], df["group"]))

    print()
    print("=== Five indicators (summary stats) ===")
    for col in ["failsafe_frac", "nav_state_forbidden_frac",
                "filter_fault_flags_frac", "actuator_saturation_frac"]:
        vals = ok[col].dropna()
        print(f"{col}:")
        print(f"  count > 0: {int((vals > 0).sum())} of {len(vals)}")
        print(f"  median:    {vals.median():.4f}")
        print(f"  max:       {vals.max():.4f}")

    landed = ok["landed_at_end"].dropna()
    print(f"landed_at_end:")
    print(f"  True:  {int(landed.sum())} of {len(landed)}")
    print(f"  False: {int((~landed).sum())} of {len(landed)}")

    df.to_csv("pilot_500_v2_index.csv", index=False)
    print(f"\nsaved pilot_500_v2_index.csv")

    # write seed record
    with open("pilot_500_v2_seed.txt", "w") as f:
        f.write(f"seed={SEED}\nn_logs={len(sample)}\n")
    print(f"saved pilot_500_v2_seed.txt")


if __name__ == "__main__":
    main()