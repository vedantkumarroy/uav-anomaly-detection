import json
import os
import random
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
DBINFO_CACHE = "pilot/dbinfo.json"

N_LOGS = 2000
SEED = 4

MIN_DUR = 60.0
MAX_DUR = 3600.0

MIN_RATE_POSITION = 5.0
MIN_RATE_ATTITUDE = 10.0
MIN_RATE_RATIOS = 1.5
MIN_AIRBORNE_S = 45.0
MAX_DROPOUT_FRAC = 0.02

VALID_AUTOSTART = {"4001"}


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


def stream_download(log_id, session):
    url = BASE + DOWNLOAD_PATH.format(log_id=log_id)
    try:
        r = session.get(url, timeout=180, stream=True)
        if r.status_code != 200:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg", dir="pilot")
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


def analyze(path):
    from pyulog import ULog
    try:
        u = ULog(path)
    except Exception:
        return None

    info = u.msg_info_dict or {}
    ver_sw = info.get("ver_sw_release")
    band = ver_band(ver_sw)
    if band is None:
        return None

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
        return None

    rate_pos = rates.get("vehicle_local_position", 0.0)
    rate_att = rates.get("vehicle_attitude", 0.0)
    rate_rat = rates.get("estimator_innovation_test_ratios", 0.0)
    if rate_pos < MIN_RATE_POSITION:
        return None
    if rate_att < MIN_RATE_ATTITUDE:
        return None
    if rate_rat < MIN_RATE_RATIOS:
        return None

    try:
        dur = (u.last_timestamp - u.start_timestamp) / 1e6
    except Exception:
        return None
    if dur < MIN_DUR:
        return None

    drop_ms = sum(getattr(d, "duration", 0) for d in (u.dropouts or []))
    dropout_frac = (drop_ms / 1000.0) / dur if dur > 0 else 1.0
    if dropout_frac > MAX_DROPOUT_FRAC:
        return None

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
        return None

    params = u.initial_parameters or {}
    autostart = str(params.get("SYS_AUTOSTART", ""))
    if autostart not in VALID_AUTOSTART:
        return None

    field_info = {}
    for field, arr in ratio_topic.data.items():
        if field in ("timestamp", "timestamp_sample"):
            continue
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 2:
            field_info[field] = {"present": True, "varying": False}
            continue
        varying = bool(np.nanstd(arr) > 0)
        field_info[field] = {"present": True, "varying": varying}

    return {
        "ver_sw": ver_sw,
        "band": band,
        "dropout_frac": dropout_frac,
        "rate_ratios": rate_rat,
        "field_info": field_info,
    }


def process(log_id, session):
    path = stream_download(log_id, session)
    if path is None:
        return None
    try:
        r = analyze(path)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    if r is None:
        return None
    r["log_id"] = log_id
    return r


def main():
    with open(DBINFO_CACHE) as f:
        records = json.load(f)

    cand = []
    for r in records:
        if not r.get("log_id"):
            continue
        if str(r.get("mav_type", "")).lower() != "quadrotor":
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
    session.headers["User-Agent"] = "px4-matrix/1.0"

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process, r["log_id"], session): r["log_id"]
                for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 25 == 0:
                print(f"  processed {i}/{len(sample)}")

    print(f"\npassed all gates: {len(rows)} logs\n")

    bands = ["v1.12-1.13", "v1.14-1.15", "v1.16+"]
    band_counts = {b: 0 for b in bands}
    band_present = defaultdict(lambda: defaultdict(int))
    band_varying = defaultdict(lambda: defaultdict(int))

    all_fields = set()
    has_baro_fluct = []
    for r in rows:
        b = r["band"]
        if b not in bands:
            continue
        band_counts[b] += 1
        if "is_baro_fluctuation" in r["field_info"]:
            has_baro_fluct.append(r["log_id"])
        for field, info in r["field_info"].items():
            all_fields.add(field)
            if info["present"]:
                band_present[b][field] += 1
            if info["varying"]:
                band_varying[b][field] += 1

    all_fields = sorted(all_fields)

    print("=== Band counts (post-gate) ===")
    for b in bands:
        print(f"  {b}: {band_counts[b]}")
    print(f"  TOTAL: {sum(band_counts.values())}")

    print("\n=== Field matrix ===")
    print(f"{'field':<26}" + "".join(f"{b:>16}" for b in bands))
    print(f"{'':<26}" + "".join(f"{'present/vary':>16}" for b in bands))
    for field in all_fields:
        line = field.ljust(26)
        for b in bands:
            n_band = band_counts[b]
            n_present = band_present[b][field]
            n_varying = band_varying[b][field]
            p_frac = n_present / n_band if n_band > 0 else 0.0
            v_frac = n_varying / n_present if n_present > 0 else 0.0
            line += f"{p_frac:.2f}/{v_frac:.2f}".rjust(16)
        print(line)

    print(f"\n=== is_baro_fluctuation ===")
    print(f"logs with this field: {len(has_baro_fluct)}")
    if has_baro_fluct:
        for lid in has_baro_fluct[:20]:
            print(f"  {lid}")

    per_log = []
    for r in rows:
        row = {
            "log_id": r["log_id"],
            "ver_sw": r["ver_sw"],
            "band": r["band"],
            "dropout_frac": r["dropout_frac"],
            "rate_ratios": r["rate_ratios"],
        }
        for field, info in r["field_info"].items():
            row[f"present_{field}"] = int(info["present"])
            row[f"varying_{field}"] = int(info["varying"])
        per_log.append(row)

    pd.DataFrame(per_log).to_csv("band_matrix_v2_2000.csv", index=False)
    print("\nsaved: band_matrix_v2_2000.csv")


if __name__ == "__main__":
    main()