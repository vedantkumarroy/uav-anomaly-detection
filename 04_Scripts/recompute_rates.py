import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
INDEX_CSV = "pilot_2270_index.csv"


def compute_rate(logdir, topic):
    p = os.path.join(logdir, topic + ".parquet")
    if not os.path.exists(p):
        return 0.0, 0
    t = pq.read_table(p)
    if "timestamp" not in t.column_names:
        return 0.0, 0
    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if len(ts) < 2:
        return 0.0, len(ts)
    span = (ts[-1] - ts[0]) / 1e6
    if span <= 0:
        return 0.0, len(ts)
    return (len(ts) - 1) / span, len(ts)


def compute_airborne_s(logdir):
    p = os.path.join(logdir, "vehicle_land_detected.parquet")
    if not os.path.exists(p):
        return 0.0
    t = pq.read_table(p)
    if "landed" not in t.column_names or "timestamp" not in t.column_names:
        return 0.0
    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    landed = t["landed"].to_numpy(zero_copy_only=False).astype(bool)
    if len(ts) < 2:
        return 0.0
    air = 0.0
    for i in range(len(ts) - 1):
        if not landed[i]:
            air += (ts[i + 1] - ts[i]) / 1e6
    return air


def compute_duration_s(logdir):
    p = os.path.join(logdir, "vehicle_status.parquet")
    if os.path.exists(p):
        t = pq.read_table(p)
        if "timestamp" in t.column_names:
            ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
            if len(ts) >= 2:
                return (ts[-1] - ts[0]) / 1e6
    # fallback to any topic with timestamp
    for topic in ["vehicle_attitude", "vehicle_local_position", "sensor_combined"]:
        p = os.path.join(logdir, topic + ".parquet")
        if os.path.exists(p):
            t = pq.read_table(p)
            if "timestamp" in t.column_names:
                ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
                if len(ts) >= 2:
                    return (ts[-1] - ts[0]) / 1e6
    return 0.0


df = pd.read_csv(INDEX_CSV)
print(f"rows: {len(df)}")

rate_pos = []
rate_att = []
rate_rat = []
dur = []
air = []

for i, row in df.iterrows():
    logdir = os.path.join(LOGDIR, row["log_id"])
    if not os.path.isdir(logdir):
        rate_pos.append(np.nan)
        rate_att.append(np.nan)
        rate_rat.append(np.nan)
        dur.append(np.nan)
        air.append(np.nan)
        continue

    rp, _ = compute_rate(logdir, "vehicle_local_position")
    ra, _ = compute_rate(logdir, "vehicle_attitude")
    rr, _ = compute_rate(logdir, "estimator_innovation_test_ratios")
    d = compute_duration_s(logdir)
    a = compute_airborne_s(logdir)

    rate_pos.append(rp)
    rate_att.append(ra)
    rate_rat.append(rr)
    dur.append(d)
    air.append(a)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(df)}")

df["rate_position_hz"] = rate_pos
df["rate_attitude_hz"] = rate_att
df["rate_ratios_hz"] = rate_rat
df["duration_s"] = dur
df["airborne_s"] = air

df.to_csv(INDEX_CSV, index=False)
print(f"\nupdated {INDEX_CSV}")

print()
print("=== rate summary ===")
for c in ["rate_position_hz", "rate_attitude_hz", "rate_ratios_hz", "duration_s", "airborne_s"]:
    vals = df[c].dropna()
    print(f"{c}: median={vals.median():.2f} p05={vals.quantile(0.05):.2f} min={vals.min():.2f}")