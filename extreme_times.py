import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ANALYSIS_CSV = "combined_analysis_set.csv"

RATIOS_EXTREME = ["gps_hpos[0]", "gps_hpos[1]", "gps_vpos"]
EXTREME_THRESHOLD = 1e9  # 10^9

def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


df = pd.read_csv(ANALYSIS_CSV)
print(f"loaded {len(df)} flights")

records = []

for i, row in df.iterrows():
    log_id = row["log_id"]
    logdir = logdir_for(log_id)
    if logdir is None:
        continue

    rat_p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
    if not os.path.exists(rat_p):
        continue
    rat = pq.read_table(rat_p)
    if "timestamp" not in rat.column_names:
        continue

    rat_ts = rat["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    t0 = rat_ts[0]

    # first xy_valid time from vehicle_local_position
    first_xy_valid_ts = None
    lp_p = os.path.join(logdir, "vehicle_local_position.parquet")
    if os.path.exists(lp_p):
        try:
            lp = pq.read_table(lp_p)
            if "xy_valid" in lp.column_names and "timestamp" in lp.column_names:
                lp_ts = lp["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
                xy_valid = lp["xy_valid"].to_numpy(zero_copy_only=False).astype(bool)
                valid_idx = np.where(xy_valid)[0]
                if len(valid_idx) > 0:
                    first_xy_valid_ts = lp_ts[valid_idx[0]]
        except Exception:
            pass

    for ratio in RATIOS_EXTREME:
        if ratio not in rat.column_names:
            continue
        arr = rat[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
        extreme_mask = arr > EXTREME_THRESHOLD
        if not extreme_mask.any():
            continue
        extreme_ts = rat_ts[extreme_mask]
        for ts in extreme_ts:
            time_from_start_s = (ts - t0) / 1e6
            if first_xy_valid_ts is not None:
                time_from_xy_valid_s = (ts - first_xy_valid_ts) / 1e6
            else:
                time_from_xy_valid_s = np.nan
            records.append({
                "log_id": log_id,
                "band": row["band"],
                "ratio": ratio,
                "timestamp": ts,
                "time_from_start_s": time_from_start_s,
                "time_from_xy_valid_s": time_from_xy_valid_s,
            })

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)}")

rdf = pd.DataFrame(records)
print()
print(f"total extreme samples: {len(rdf)}")
print(f"unique flights with extreme samples: {rdf['log_id'].nunique()}")
print()

# histogram by time from start
print("=" * 78)
print("Distribution of extreme-sample times (relative to log start)")
print("=" * 78)
print()

bins = [0, 1, 2, 5, 10, 30, 60, 120, 600, 3600]
labels = ["0-1s", "1-2s", "2-5s", "5-10s", "10-30s", "30-60s", "1-2min", "2-10min", ">10min"]
counts = pd.cut(rdf["time_from_start_s"], bins=bins, labels=labels, right=False).value_counts().reindex(labels)

print("Time from log start:")
for label, count in counts.items():
    pct = 100 * count / len(rdf)
    print(f"  {label:>10s}: {int(count):>6d}  ({pct:>5.1f}%)")

print()
print("Time from first xy_valid:")
counts2 = pd.cut(rdf["time_from_xy_valid_s"], bins=bins, labels=labels, right=False).value_counts().reindex(labels)
for label, count in counts2.items():
    pct = 100 * count / len(rdf)
    print(f"  {label:>10s}: {int(count):>6d}  ({pct:>5.1f}%)")

# figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(rdf["time_from_start_s"], bins=np.logspace(-2, 3.5, 40), edgecolor="black", alpha=0.75)
ax.set_xscale("log")
ax.set_xlabel("Time from log start (s, log scale)")
ax.set_ylabel("Number of extreme samples")
ax.set_title(f"Extreme samples (>10^9) by time from log start\n(n={len(rdf)}, {rdf['log_id'].nunique()} flights)")
ax.grid(alpha=0.3)
ax.axvline(1, color="red", linestyle="--", linewidth=0.8, label="1s")
ax.axvline(10, color="orange", linestyle="--", linewidth=0.8, label="10s")
ax.legend()

ax = axes[1]
valid = rdf["time_from_xy_valid_s"].dropna()
ax.hist(valid, bins=np.logspace(-2, 3.5, 40), edgecolor="black", alpha=0.75, color="seagreen")
ax.set_xscale("log")
ax.set_xlabel("Time from first xy_valid (s, log scale)")
ax.set_ylabel("Number of extreme samples")
ax.set_title(f"Extreme samples by time from EKF xy_valid start\n(n={len(valid)})")
ax.grid(alpha=0.3)
ax.axvline(1, color="red", linestyle="--", linewidth=0.8, label="1s")
ax.axvline(10, color="orange", linestyle="--", linewidth=0.8, label="10s")
ax.legend()

plt.tight_layout()
plt.savefig("extreme_times_histogram.png", dpi=150)
plt.show()
print()
print("saved extreme_times_histogram.png")

rdf.to_csv("extreme_times.csv", index=False)
print("saved extreme_times.csv")