import json
import os
import random
import time
import io
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
DBINFO_CACHE = "pilot/dbinfo.json"

MIN_DUR = 60.0
MAX_DUR = 3600.0
N_LOGS = 200

# sensor topics to measure rate for
TOPICS = ["vehicle_local_position", "vehicle_attitude", "sensor_combined",
          "estimator_innovation_test_ratios"]

# write streaming data to a temp file in %TEMP% (pyulog needs a path)
import tempfile

def stream_download(log_id, session, timeout=180):
    url = BASE + DOWNLOAD_PATH.format(log_id=log_id)
    try:
        r = session.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg", dir="pilot")
        n = 0
        with open(tmp.name, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
                n += len(chunk)
        if n < 1024:
            os.remove(tmp.name)
            return None, "too_small"
        return tmp.name, "ok"
    except Exception as e:
        return None, f"err_{type(e).__name__}"

def measure_rates(path):
    from pyulog import ULog
    try:
        u = ULog(path)
    except Exception:
        return None

    info = u.msg_info_dict or {}
    ver_sw = str(info.get("ver_sw_release") or info.get("ver_sw") or "")

    durations = {}
    for d in u.data_list:
        if d.name not in TOPICS:
            continue
        ts = d.data.get("timestamp")
        if ts is None or len(ts) < 2:
            continue
        span = (float(ts[-1]) - float(ts[0])) / 1e6
        if span <= 0:
            continue
        hz = (len(ts) - 1) / span
        # take the max across instances (dual-IMU case)
        durations[d.name] = max(durations.get(d.name, 0.0), hz)

    return {
        "ver_sw": ver_sw,
        "rate_position": durations.get("vehicle_local_position", 0.0),
        "rate_attitude": durations.get("vehicle_attitude", 0.0),
        "rate_imu": durations.get("sensor_combined", 0.0),
        "rate_ratios": durations.get("estimator_innovation_test_ratios", 0.0),
    }

def process_one(log_id, session):
    path, status = stream_download(log_id, session)
    if path is None:
        return {"log_id": log_id, "status": status}
    try:
        rates = measure_rates(path)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    if rates is None:
        return {"log_id": log_id, "status": "parse_failed"}
    rates["log_id"] = log_id
    rates["status"] = "ok"
    return rates

def main():
    with open(DBINFO_CACHE) as f:
        records = json.load(f)

    # same quadrotor filter as smoke test
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

    print(f"candidates after filter: {len(cand)}")
    random.seed(0)
    sample = random.sample(cand, min(N_LOGS, len(cand)))
    print(f"sampling {len(sample)} logs (streaming, no disk)\n")

    session = requests.Session()
    session.headers["User-Agent"] = "px4-rate/1.0 (research)"

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(process_one, r["log_id"], session): r["log_id"]
                for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 20 == 0:
                print(f"  processed {i}/{len(sample)}")

    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"]
    df.to_csv("sensor_rates.csv", index=False)
    print(f"\nsaved {len(df)} rows -> sensor_rates.csv")

    # summary stats
    for col in ["rate_position", "rate_attitude", "rate_imu", "rate_ratios"]:
        print(f"\n{col}:")
        print(f"  min={df[col].min():.1f}  p05={df[col].quantile(0.05):.1f}"
              f"  median={df[col].median():.1f}  p95={df[col].quantile(0.95):.1f}"
              f"  max={df[col].max():.1f}")

if __name__ == "__main__":
    main()