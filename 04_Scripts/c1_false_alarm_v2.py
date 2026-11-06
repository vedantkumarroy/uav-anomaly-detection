import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

WARMUP_S = 30.0
PX4_THRESHOLD = 1.0


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


df = pd.read_csv(ANALYSIS_CSV)
print(f"loaded {len(df)}")
print(f"warm-up exclusion: first {WARMUP_S} seconds of each flight")
print()

per_flight = []
for i, row in df.iterrows():
    logdir = logdir_for(row["log_id"])
    if logdir is None:
        continue
    p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
    if not os.path.exists(p):
        continue
    t = pq.read_table(p)
    if "timestamp" not in t.column_names:
        continue
    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if len(ts) < 2:
        continue
    t0 = ts[0]
    keep_mask = (ts - t0) / 1e6 >= WARMUP_S
    if keep_mask.sum() < 10:
        continue

    rec = {"log_id": row["log_id"], "band": row["band"]}
    any_flag = False
    for ratio in RATIOS:
        if ratio not in t.column_names:
            continue
        arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
        arr = arr[keep_mask]
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        rate = float((finite > PX4_THRESHOLD).mean())
        rec[f"rate_{ratio}"] = rate
        if rate > 0.05:
            any_flag = True
    rec["any_over_5pct"] = any_flag
    per_flight.append(rec)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)}")

pdf = pd.DataFrame(per_flight)
pdf.to_csv("c1_warmup_excluded.csv", index=False)

n_total = len(pdf)
n_over = int(pdf["any_over_5pct"].sum())
print()
print("=" * 60)
print(f"C1 headline with {WARMUP_S:.0f}s warm-up excluded")
print("=" * 60)
print()
print(f"  flights:                  {n_total}")
print(f"  over-flagged (>5%):       {n_over}")
print(f"  headline false alarm rate: {100*n_over/n_total:.1f}%")
print()

# for comparison, reprint the naive result
print("Comparison:")
print(f"  naive (all samples):      21.1%")
print(f"  warm-up excluded:         {100*n_over/n_total:.1f}%")
print(f"  difference:               {21.1 - 100*n_over/n_total:.1f} pts (warm-up artifact)")

# also break down by ratio
print()
print("=== Per-ratio false alarm rate (warm-up excluded) ===")
print(f"{'ratio':>18s}  {'mean_rate':>10s}  {'n>5%':>6s}  {'%flights':>9s}")
for ratio in RATIOS:
    col = f"rate_{ratio}"
    if col not in pdf.columns:
        continue
    vals = pdf[col].dropna()
    n = len(vals)
    gt5 = int((vals > 0.05).sum())
    print(f"{ratio:>18s}  {vals.mean():>10.4f}  {gt5:>6d}  {100*gt5/n:>8.1f}%")