import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import matplotlib.pyplot as plt

BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
DBINFO_CACHE = "pilot/dbinfo.json"

N_LOGS = 100
MIN_DUR = 60.0
MAX_DUR = 3600.0


def stream_and_measure(log_id, duration_s, session):
    url = BASE + DOWNLOAD_PATH.format(log_id=log_id)
    try:
        r = session.get(url, timeout=180, stream=True)
        if r.status_code != 200:
            return None
        size_header = r.headers.get("Content-Length")
        size_bytes = int(size_header) if size_header else None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg", dir="pilot")
        n = 0
        with open(tmp.name, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
                n += len(chunk)
        if n < 1024:
            os.remove(tmp.name)
            return None

        # measure ratios rate
        try:
            from pyulog import ULog
            u = ULog(tmp.name)
            ratio_topic = None
            for d in u.data_list:
                if d.name == "estimator_innovation_test_ratios":
                    ratio_topic = d
                    break
            if ratio_topic is None:
                ratio_rate = 0.0
            else:
                ts = ratio_topic.data.get("timestamp")
                if ts is None or len(ts) < 2:
                    ratio_rate = 0.0
                else:
                    span = (float(ts[-1]) - float(ts[0])) / 1e6
                    ratio_rate = (len(ts) - 1) / span if span > 0 else 0.0
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass

        return {
            "log_id": log_id,
            "size_bytes": size_bytes if size_bytes else n,
            "duration_s": duration_s,
            "size_per_s": (size_bytes if size_bytes else n) / duration_s,
            "ratio_rate_hz": ratio_rate,
        }
    except Exception:
        return None


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
        cand.append((r["log_id"], d))

    print(f"candidates: {len(cand)}")
    random.seed(2)
    sample = random.sample(cand, min(N_LOGS, len(cand)))
    print(f"streaming {len(sample)} logs\n")

    session = requests.Session()
    session.headers["User-Agent"] = "px4-size/1.0"

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(stream_and_measure, log_id, d, session): log_id
                for log_id, d in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 20 == 0:
                print(f"  processed {i}/{len(sample)}")

    print(f"\nparsed: {len(rows)} logs")

    sizes_per_s = np.array([r["size_per_s"] for r in rows])
    rates = np.array([r["ratio_rate_hz"] for r in rows])

    # filter out zero-rate logs for the correlation
    mask = rates > 0
    sps = sizes_per_s[mask]
    rt = rates[mask]

    corr = np.corrcoef(sps, rt)[0, 1] if len(sps) > 1 else float("nan")
    print(f"\n=== Correlation ===")
    print(f"n = {len(sps)}")
    print(f"corr(size/duration, ratio_rate) = {corr:.3f}")

    # plot
    plt.figure(figsize=(8, 5))
    plt.scatter(sps / 1e6, rt, alpha=0.6, s=20)
    plt.xlabel("file size / duration (MB/s)")
    plt.ylabel("ratio logging rate (Hz)")
    plt.title(f"Size-per-second vs logging rate (n={len(sps)}, r={corr:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("size_rate_correlation.png", dpi=120)
    plt.show()
    print("saved: size_rate_correlation.png")


if __name__ == "__main__":
    main()