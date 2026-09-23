import json
import os
import random
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np

BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
DBINFO_CACHE = "pilot/dbinfo.json"

N_LOGS = 300
MIN_DUR = 60.0
MAX_DUR = 3600.0

def ver_band(ver_sw):
    """Map ver_sw to a firmware band. ver_sw is a numeric string.
    Real PX4 versions map approximately:
       17000000-17399999  -> v1.12-1.13
       17400000-17599999  -> v1.14-1.15
       17600000-17759999  -> v1.16
       17760000+          -> v1.17+
    """
    if not ver_sw:
        return None
    try:
        v = int(ver_sw)
    except (TypeError, ValueError):
        return None
    if 17000000 <= v <= 17399999:
        return "v1.12-1.13"
    if 17400000 <= v <= 17599999:
        return "v1.14-1.15"
    if 17600000 <= v <= 17759999:
        return "v1.16"
    if v >= 17760000:
        return "v1.17+"
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
    ver_sw = str(info.get("ver_sw_release") or info.get("ver_sw") or "")
    band = ver_band(ver_sw)

    ratio_topic = None
    for d in u.data_list:
        if d.name == "estimator_innovation_test_ratios":
            ratio_topic = d
            break
    if ratio_topic is None:
        return None

    varying = []
    for field, arr in ratio_topic.data.items():
        if field in ("timestamp", "timestamp_sample"):
            continue
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 2:
            continue
        if np.nanstd(arr) > 0:
            varying.append(field)

    return {"ver_sw": ver_sw, "band": band, "varying": sorted(varying)}

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
    random.seed(1)
    sample = random.sample(cand, min(N_LOGS, len(cand)))
    print(f"streaming {len(sample)} logs\n")

    session = requests.Session()
    session.headers["User-Agent"] = "px4-bands/1.0"

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

    # group by band
    print(f"\nparsed: {len(rows)} logs\n")
    band_varying = defaultdict(set)
    band_counts = defaultdict(int)
    for r in rows:
        if r["band"] is None:
            continue
        band_counts[r["band"]] += 1
        band_varying[r["band"]].update(r["varying"])

    print("=== Bands ===")
    for band in ["v1.12-1.13", "v1.14-1.15", "v1.16", "v1.17+"]:
        n = band_counts.get(band, 0)
        nv = len(band_varying.get(band, set()))
        print(f"{band:12s}  logs={n:4d}  unique varying ratios={nv}")

    # matrix
    all_fields = sorted(set().union(*band_varying.values()))
    bands = ["v1.12-1.13", "v1.14-1.15", "v1.16", "v1.17+"]

    print("\n=== Vary matrix ===")
    header = "field".ljust(22) + "".join(b.rjust(14) for b in bands)
    print(header)
    for f in all_fields:
        line = f.ljust(22)
        for b in bands:
            line += ("X" if f in band_varying[b] else ".").rjust(14)
        print(line)

    # save
    import pandas as pd
    df = pd.DataFrame([{"log_id": r["log_id"], "ver_sw": r["ver_sw"],
                        "band": r["band"], "n_varying": len(r["varying"])}
                       for r in rows])
    df.to_csv("version_bands_table.csv", index=False)
    print("\nsaved: version_bands_table.csv")

if __name__ == "__main__":
    main()