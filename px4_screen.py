#!/usr/bin/env python3
"""
px4_screen.py -- survival screening for the PX4 Flight Review public log corpus.

Answers one question: of N candidate logs, what fraction actually carry the
topics, rates and flight content needed for a conformal anomaly-detection study?

    pip install requests pyulog pandas
    python px4_screen.py --n 200 --outdir ./screen_run

Outputs
    <outdir>/dbinfo.json     cached metadata index (fetched once)
    <outdir>/logs/*.ulg      downloaded logs (resumable; delete to re-pull)
    <outdir>/per_log.csv     one row per log with every screening flag
    <outdir>/funnel.txt      the survival funnel

Notes
    - Logs are CC-BY PX4. Keep concurrency low; this is a community server.
    - The /dbinfo and /download endpoints are what download_logs.py uses. If PX4
      changes them, patch BASE/DBINFO_PATH/DOWNLOAD_PATH below rather than the
      rest of the script.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import requests

BASE = "https://logs.px4.io"
DBINFO_PATH = "/dbinfo"
DOWNLOAD_PATH = "/download?log={log_id}"

# ---------------------------------------------------------------------------
# Topic requirements.
#
# Each entry is a list of ALTERNATIVES -- satisfied if ANY member is present.
# The alternatives exist because PX4 renamed/split several of these topics
# across firmware generations. This fragmentation is the main thing the screen
# is measuring, so do not collapse these into single names.
# ---------------------------------------------------------------------------
REQUIRED = {
    "position":   ["vehicle_local_position"],
    "attitude":   ["vehicle_attitude"],
    "status":     ["vehicle_status"],
    "imu":        ["sensor_combined"],
    "actuator":   ["actuator_motors", "actuator_outputs"],
    "att_sp":     ["vehicle_attitude_setpoint"],
    "rates_sp":   ["vehicle_rates_setpoint"],
    "battery":    ["battery_status"],
    # The EKF consistency-check substrate. Newer firmware splits test ratios
    # into their own topic; older firmware carries *_test_ratio fields inside
    # estimator_status.
    "ekf_ratios": ["estimator_innovation_test_ratios", "estimator_status"],
    "ekf_innov":  ["estimator_innovations", "estimator_status"],
    "land_det":   ["vehicle_land_detected"],
}

# Minimum acceptable median publication rate, Hz, for the topics we will
# actually build residuals from.
MIN_RATE_HZ = {
    "vehicle_local_position": 10.0,
    "vehicle_attitude": 20.0,
    "sensor_combined": 50.0,
}

MIN_DURATION_S = 60.0
MIN_AIRBORNE_S = 45.0
MAX_DROPOUT_FRAC = 0.01


# ---------------------------------------------------------------------------
# Metadata index
# ---------------------------------------------------------------------------
def fetch_dbinfo(outdir, force=False):
    path = os.path.join(outdir, "dbinfo.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    print(f"fetching {BASE}{DBINFO_PATH} (this can take a minute; it is cached server-side)")
    r = requests.get(BASE + DBINFO_PATH, timeout=300)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):                      # tolerate a wrapper key
        for k in ("logs", "data", "results"):
            if k in data:
                data = data[k]
                break
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  {len(data)} log records cached -> {path}")
    return data


def filter_dbinfo(records, mav_type=None, autostart=None,
                  min_dur=MIN_DURATION_S, max_dur=3600.0):
    """Cheap metadata-level prefilter. Runs before any bytes are downloaded."""
    out = []
    for r in records:
        if not r.get("log_id"):
            continue
        d = r.get("duration_s") or r.get("duration") or 0
        try:
            d = float(d)
        except (TypeError, ValueError):
            continue
        if not (min_dur <= d <= max_dur):
            continue
        if mav_type and str(r.get("mav_type", "")).lower() != mav_type.lower():
            continue
        if autostart and str(r.get("sys_autostart_id")) != str(autostart):
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_one(log_id, logdir, session, delay=0.4):
    dest = os.path.join(logdir, f"{log_id}.ulg")
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest, "cached", os.path.getsize(dest)
    url = BASE + DOWNLOAD_PATH.format(log_id=log_id)
    try:
        time.sleep(delay)
        with session.get(url, timeout=180, stream=True) as r:
            if r.status_code != 200:
                return None, f"http_{r.status_code}", 0
            tmp = dest + ".part"
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n += len(chunk)
            if n < 1024:
                os.remove(tmp)
                return None, "too_small", n
            os.replace(tmp, dest)
            return dest, "ok", n
    except Exception as e:
        return None, f"err_{type(e).__name__}", 0


# ---------------------------------------------------------------------------
# Parse + screen (one process per log; pyulog is single-threaded and slow)
# ---------------------------------------------------------------------------
def screen_one(path):
    from pyulog import ULog

    row = {"file": os.path.basename(path), "parse_ok": False}
    try:
        ulog = ULog(path)
    except Exception as e:
        row["parse_error"] = f"{type(e).__name__}: {e}"[:200]
        return row
    row["parse_ok"] = True

    info = ulog.msg_info_dict or {}
    row["ver_sw"] = info.get("ver_sw_release") or info.get("ver_sw") or ""
    row["sys_name"] = info.get("sys_name", "")
    row["hw"] = info.get("ver_hw", "")
    params = ulog.initial_parameters or {}
    row["sys_autostart"] = params.get("SYS_AUTOSTART", "")
    row["mav_type"] = params.get("MAV_TYPE", "")
    row["ekf2_en"] = params.get("EKF2_EN", "")

    # topic inventory + effective rates
    present, rates, counts = set(), {}, {}
    for d in ulog.data_list:
        present.add(d.name)
        ts = d.data.get("timestamp")
        if ts is None or len(ts) < 2:
            continue
        span = (float(ts[-1]) - float(ts[0])) / 1e6
        if span <= 0:
            continue
        counts[d.name] = counts.get(d.name, 0) + len(ts)
        rates[d.name] = max(rates.get(d.name, 0.0), (len(ts) - 1) / span)

    row["n_topics"] = len(present)
    for label, alts in REQUIRED.items():
        hit = next((a for a in alts if a in present), None)
        row[f"has_{label}"] = hit is not None
        row[f"src_{label}"] = hit or ""

    # legacy estimator_status must actually carry the test-ratio fields
    if "estimator_innovation_test_ratios" not in present and "estimator_status" in present:
        es = next((d for d in ulog.data_list if d.name == "estimator_status"), None)
        legacy = es is not None and any(k.endswith("test_ratio") for k in es.data)
        row["has_ekf_ratios"] = legacy
        row["src_ekf_ratios"] = "estimator_status(legacy)" if legacy else ""

    for t, minhz in MIN_RATE_HZ.items():
        r = rates.get(t, 0.0)
        row[f"rate_{t}"] = round(r, 1)
        row[f"rate_ok_{t}"] = r >= minhz

    # duration
    try:
        dur = (ulog.last_timestamp - ulog.start_timestamp) / 1e6
    except Exception:
        dur = 0.0
    row["duration_s"] = round(dur, 1)
    row["duration_ok"] = dur >= MIN_DURATION_S

    # dropouts
    drop_ms = sum(getattr(d, "duration", 0) for d in (ulog.dropouts or []))
    row["dropout_frac"] = round((drop_ms / 1000.0) / dur, 4) if dur > 0 else 1.0
    row["dropout_ok"] = row["dropout_frac"] <= MAX_DROPOUT_FRAC

    # airborne time: prefer the land detector, fall back to altitude excursion
    airborne = 0.0
    ld = next((d for d in ulog.data_list if d.name == "vehicle_land_detected"), None)
    if ld is not None and "landed" in ld.data and len(ld.data["timestamp"]) > 1:
        ts, landed = ld.data["timestamp"], ld.data["landed"]
        for i in range(len(ts) - 1):
            if not landed[i]:
                airborne += (ts[i + 1] - ts[i]) / 1e6
        row["airborne_src"] = "land_detected"
    else:
        lp = next((d for d in ulog.data_list if d.name == "vehicle_local_position"), None)
        if lp is not None and "z" in lp.data and len(lp.data["z"]) > 1:
            z = lp.data["z"]
            if (max(z) - min(z)) > 2.0:
                airborne = row["duration_s"]      # crude: assume mostly flying
            row["airborne_src"] = "altitude_excursion"
        else:
            row["airborne_src"] = "none"
    row["airborne_s"] = round(airborne, 1)
    row["airborne_ok"] = airborne >= MIN_AIRBORNE_S

    row["n_errors"] = len([m for m in (ulog.logged_messages or [])
                           if getattr(m, "log_level", 0) <= ord("3")])

    row["topics_ok"] = all(row[f"has_{k}"] for k in REQUIRED)
    row["rates_ok"] = all(row[f"rate_ok_{t}"] for t in MIN_RATE_HZ)
    row["SURVIVES"] = bool(row["topics_ok"] and row["rates_ok"]
                           and row["duration_ok"] and row["airborne_ok"]
                           and row["dropout_ok"])
    return row


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="logs to sample and screen")
    ap.add_argument("--outdir", default="./screen_run")
    ap.add_argument("--mav-type", default=None,
                    help="dbinfo mav_type filter, e.g. 'Quadrotor'")
    ap.add_argument("--autostart", default=None,
                    help="SYS_AUTOSTART id filter, e.g. '4001'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="download threads")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--refresh-dbinfo", action="store_true")
    args = ap.parse_args()

    logdir = os.path.join(args.outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    records = fetch_dbinfo(args.outdir, force=args.refresh_dbinfo)
    cand = filter_dbinfo(records, mav_type=args.mav_type, autostart=args.autostart)
    print(f"metadata prefilter: {len(records)} -> {len(cand)} candidates")
    if not cand:
        sys.exit("no candidates; loosen the filters or check dbinfo field names")

    random.seed(args.seed)
    sample = random.sample(cand, min(args.n, len(cand)))
    meta = {r["log_id"]: r for r in sample}

    # download
    paths, dl_status = [], Counter()
    session = requests.Session()
    session.headers["User-Agent"] = "px4-screen/1.0 (research; contact: you@bits-dubai.ac.ae)"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, r["log_id"], logdir, session): r["log_id"]
                for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            p, status, nbytes = fut.result()
            dl_status[status] += 1
            if p:
                paths.append((p, futs[fut], nbytes))
            if i % 25 == 0:
                print(f"  downloaded {i}/{len(sample)}")
    print("download status:", dict(dl_status))
    total_mb = sum(n for _, _, n in paths) / 1e6
    print(f"bytes on disk: {total_mb:.1f} MB for {len(paths)} logs "
          f"({total_mb / max(len(paths), 1):.2f} MB/log)")

    # parse + screen
    rows = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = {ex.submit(screen_one, p): lid for p, lid, _ in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            row["log_id"] = futs[fut]
            m = meta.get(row["log_id"], {})
            row["db_duration_s"] = m.get("duration_s")
            row["db_mav_type"] = m.get("mav_type")
            row["db_date"] = m.get("log_date")
            rows.append(row)
            if i % 25 == 0:
                print(f"  screened {i}/{len(paths)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "per_log.csv")
    df.to_csv(csv_path, index=False)

    # funnel
    L = []
    L.append(f"sampled                 {len(sample)}")
    L.append(f"downloaded              {len(paths)}")
    L.append(f"parsed by pyulog        {int(df['parse_ok'].sum())}")
    ok = df[df["parse_ok"]]
    for label in REQUIRED:
        L.append(f"  has {label:<20} {int(ok[f'has_{label}'].sum()):>5}"
                 f"   ({100 * ok[f'has_{label}'].mean():.0f}%)")
    L.append(f"all required topics     {int(ok['topics_ok'].sum())}")
    L.append(f"  + rate gates          {int((ok['topics_ok'] & ok['rates_ok']).sum())}")
    L.append(f"  + duration >= {MIN_DURATION_S:.0f}s     "
             f"{int((ok['topics_ok'] & ok['rates_ok'] & ok['duration_ok']).sum())}")
    L.append(f"  + airborne >= {MIN_AIRBORNE_S:.0f}s     "
             f"{int((ok['topics_ok'] & ok['rates_ok'] & ok['duration_ok'] & ok['airborne_ok']).sum())}")
    L.append(f"SURVIVES ALL GATES      {int(ok['SURVIVES'].sum())}"
             f"   ({100 * ok['SURVIVES'].mean():.1f}% of parsed)")
    L.append("")
    L.append("EKF test-ratio source (this drives firmware fragmentation):")
    for k, v in ok["src_ekf_ratios"].value_counts().items():
        L.append(f"  {str(k) or '(none)':<32} {v}")
    L.append("")
    L.append("firmware versions among survivors:")
    for k, v in ok[ok["SURVIVES"]]["ver_sw"].value_counts().head(15).items():
        L.append(f"  {str(k) or '(unknown)':<32} {v}")
    L.append("")
    L.append("SYS_AUTOSTART among survivors:")
    for k, v in ok[ok["SURVIVES"]]["sys_autostart"].value_counts().head(15).items():
        L.append(f"  {str(k):<32} {v}")

    text = "\n".join(L)
    print("\n" + text)
    with open(os.path.join(args.outdir, "funnel.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nper-log detail -> {csv_path}")


if __name__ == "__main__":
    main()