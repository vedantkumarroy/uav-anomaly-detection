import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ANALYSIS_CSV = "combined_analysis_set.csv"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

PX4_THRESHOLD = 1.0

a = pd.read_csv(ANALYSIS_CSV)
print(f"analysis flights: {len(a)}")
print(f"columns: {a.columns.tolist()}")
print()

# determine source for each log
# check whether log exists in pilot_2270 or pilot_extra
def log_path(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id, "estimator_innovation_test_ratios.parquet")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id, "estimator_innovation_test_ratios.parquet")
    if os.path.exists(p2):
        return p2
    return None

per_flight = []
for i, row in a.iterrows():
    log_id = row["log_id"]
    band = row.get("band", None)
    group = row.get("group", None)

    p = log_path(log_id)
    if p is None:
        continue
    t = pq.read_table(p)
    if "timestamp" not in t.column_names:
        continue

    out = {"log_id": log_id, "band": band, "group": group}
    for ratio in RATIOS:
        if ratio not in t.column_names:
            continue
        arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        flags = finite > PX4_THRESHOLD
        out[f"rate_{ratio}"] = float(flags.mean())
    per_flight.append(out)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(a)}")

df = pd.DataFrame(per_flight)
df.to_csv("c1_per_flight.csv", index=False)
print(f"\nsaved c1_per_flight.csv ({len(df)} flights)")

# aggregate
print()
print("=" * 78)
print("C1: PX4 EKF false alarm rate on mostly-healthy flights (threshold = 1.0)")
print("=" * 78)
print()
print(f"{'ratio':>18s}  {'n':>6s}  {'mean':>8s}  {'median':>8s}  {'p95':>8s}  "
      f"{'>1%':>8s}  {'>5%':>8s}  {'>10%':>8s}")

summary = []
for ratio in RATIOS:
    col = f"rate_{ratio}"
    if col not in df.columns:
        continue
    vals = df[col].dropna()
    n = len(vals)
    mean = vals.mean()
    median = vals.median()
    p95 = vals.quantile(0.95)
    gt1 = float((vals > 0.01).mean())
    gt5 = float((vals > 0.05).mean())
    gt10 = float((vals > 0.10).mean())
    print(f"{ratio:>18s}  {n:>6d}  {mean:>8.4f}  {median:>8.4f}  {p95:>8.4f}  "
          f"{gt1:>8.3f}  {gt5:>8.3f}  {gt10:>8.3f}")
    summary.append({
        "ratio": ratio, "n_flights": n, "mean_rate": mean,
        "median_rate": median, "p95_rate": p95,
        "frac_gt_1pct": gt1, "frac_gt_5pct": gt5, "frac_gt_10pct": gt10,
    })

sdf = pd.DataFrame(summary)
sdf.to_csv("c1_summary.csv", index=False)
print()
print("saved c1_summary.csv")

# flight-level aggregation
print()
print("=" * 78)
print("Flight-level: fraction of flights with any ratio above threshold")
print("=" * 78)
print()
cols = [f"rate_{r}" for r in RATIOS if f"rate_{r}" in df.columns]
for thr_label, thr in [("1%", 0.01), ("5%", 0.05), ("10%", 0.10)]:
    any_above = (df[cols] > thr).any(axis=1)
    print(f"flights with any ratio flagged in > {thr_label} of samples: "
          f"{int(any_above.sum())} of {len(df)} ({100*any_above.mean():.1f}%)")

# histograms
fig, axes = plt.subplots(5, 2, figsize=(12, 14))
axes = axes.flatten()
for i, ratio in enumerate(RATIOS):
    col = f"rate_{ratio}"
    if col not in df.columns:
        axes[i].set_title(f"{ratio} (no data)")
        continue
    vals = df[col].dropna() * 100
    axes[i].hist(vals, bins=50, edgecolor="black", alpha=0.7)
    axes[i].set_title(f"{ratio} (n={len(vals)})")
    axes[i].set_xlabel("Percent of samples flagged by PX4 (threshold = 1.0)")
    axes[i].set_ylabel("Number of flights")
    axes[i].axvline(5, color="red", linestyle="--", linewidth=0.8)
for j in range(len(RATIOS), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("c1_false_alarm_histograms.png", dpi=120)
plt.show()
print()
print("saved c1_false_alarm_histograms.png")