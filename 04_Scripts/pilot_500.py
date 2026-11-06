import json
import os
import random
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
DBINFO_CACHE = "pilot/dbinfo.json"
OUTDIR = "pilot_500"

N_LOGS = 500
SEED = 5

MIN_DUR = 60.0
MAX_DUR = 3600.0

MIN_RATE_POSITION = 5.0
MIN_RATE_ATTITUDE = 10.0
MIN_RATE_RATIOS = 1.5
MIN_AIRBORNE_S = 45.0
MAX_DROPOUT_FRAC = 0.02

VALID_AUTOSTART = "4001"


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


def apply_runtime_filters(u):
    vs = next((d for d in u.data_list if d.name == "vehicle_status"), None)
    if vs is not None and "failsafe" in vs.data:
        if bool(np.any(vs.data["failsafe"])):
            return False, "failsafe_true"

    if vs is not None and "nav_state" in vs.data:
        ns = np.asarray(vs.data["nav_state"])
        forbidden = {10, 17, 18}
        if any(int(v) in forbidden for v in np.unique(ns)):
            return False, "nav_state_forbidden"

    es = next((d for d in u.data_list if d.name == "estimator_status"), None)
    if es is not None and "filter_fault_flags" in es.data:
        ff = np.asarray(es.data["filter_fault_flags"])
        if np.any(ff != 0):
            return False, "filter_fault_flags"

    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is None or "landed" not in ld.data:
        return False, "land_det_missing"
    landed = np.asarray(ld.data["landed"])
    if len(landed) == 0 or not bool(landed[-1]):
        return False, "never_landed"

    return True, "ok"


def compute_gates(u):
    present = set()
    rates = {}
    ratio_topic = None
    for d in u.data_list:
        present.add(d.name)
        ts = d.data.get("timestamp")
        if ts is None or len(ts) < 2:
            continue
        span = (float(ts[-1]) - float(ts[0])) / 1e6
        if span <= 0:
            continue
        hz = (len(ts) - 1) / span
        rates[d.name] = max(rates.get(d.name, 0.0), hz)
        if d.name == "estimator_innovation_test_ratios":
            ratio_topic = d

    if ratio_topic is None:
        return False, "no_ratios_topic"

    rate_pos = rates.get("vehicle_local_position", 0.0)
    rate_att = rates.get("vehicle_attitude", 0.0)
    rate_rat = rates.get("estimator_innovation_test_ratios", 0.0)
    if rate_pos < MIN_RATE_POSITION:
        return False, "rate_position"
    if rate_att < MIN_RATE_ATTITUDE:
        return False, "rate_attitude"
    if rate_rat < MIN_RATE_RATIOS:
        return False, "rate_ratios"

    try:
        dur = (u.last_timestamp - u.start_timestamp) / 1e6
    except Exception:
        return False, "duration_unknown"
    if dur < MIN_DUR:
        return False, "duration"

    drop_ms = sum(getattr(d, "duration", 0) for d in (u.dropouts or []))
    dropout_frac = (drop_ms / 1000.0) / dur if dur > 0 else 1.0
    if dropout_frac > MAX_DROPOUT_FRAC:
        return False, "dropout"

    airborne = 0.0
    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is not None and "landed" in ld.data and len(ld.data["timestamp"]) > 1:
        ts, landed = ld.data["timestamp"], ld.data["landed"]
        for i in range(len(ts) - 1):
            if not landed[i]:
                airborne += (ts[i + 1] - ts[i]) / 1e6
    else:
        lp = next((d for d in u.data_list if d.name == "vehicle_local_position"), None)
        if lp is not None and "z" in lp.data and len(lp.data["z"]) > 1:
            z = lp.data["z"]
            if (max(z) - min(z)) > 2.0:
                airborne = dur
    if airborne < MIN_AIRBORNE_S:
        return False, "airborne"

    return True, "ok"


def write_parquet(u, log_id):
    logdir = os.path.join(OUTDIR, log_id)
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

        ok, reason = apply_runtime_filters(u)
        if not ok:
            result["status"] = f"filtered_{reason}"
            return result

        ok, reason = compute_gates(u)
        if not ok:
            result["status"] = f"gate_{reason}"
            return result

        size = write_parquet(u, log_id)
        result["status"] = "kept"
        result["parquet_bytes"] = size
        result["ulg_bytes"] = os.path.getsize(path)

        u = None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
        result["elapsed_s"] = round(time.time() - t0, 1)

    return result


def main():
    os.makedirs(OUTDIR, exist_ok=True)

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
    session.headers["User-Agent"] = "px4-pilot/1.0"

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process, r, session): r["log_id"] for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0:
                print(f"  processed {i}/{len(sample)}")

    elapsed = round(time.time() - t_start, 1)

    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1

    kept = [r for r in results if r["status"] == "kept"]
    total_parquet = sum(r.get("parquet_bytes", 0) for r in kept)
    total_ulg = sum(r.get("ulg_bytes", 0) for r in kept)

    print()
    print(f"=== Summary ===")
    print(f"elapsed: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"attempted: {len(results)}")
    print(f"kept: {len(kept)}")
    print(f"yield: {100*len(kept)/len(results):.1f}%")
    print()
    print("status breakdown:")
    for k in sorted(status_counts, key=lambda x: -status_counts[x]):
        print(f"  {k:30s} {status_counts[k]:4d}")
    print()
    if kept:
        print(f"original ulg total:      {total_ulg/1e6:.1f} MB")
        print(f"parquet zstd total:      {total_parquet/1e6:.1f} MB")
        print(f"ratio:                   {100*total_parquet/total_ulg:.1f}%")

    band_counts = defaultdict(int)
    for r in kept:
        b = r.get("band")
        if b is None:
            b = "unknown"
        band_counts[b] += 1
    print()
    print("band counts among kept:")
    for b in sorted(band_counts):
        print(f"  {b}: {band_counts[b]}")

    import pandas as pd
    pd.DataFrame(results).to_csv("pilot_500_results.csv", index=False)
    print(f"\nsaved pilot_500_results.csv")


if __name__ == "__main__":
    main()