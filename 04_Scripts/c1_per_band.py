import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

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

def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


df = pd.read_csv(ANALYSIS_CSV)
print(f"loaded {len(df)}")

rows = []
for i, row in df.iterrows():
    logdir = logdir_for(row["log_id"])
    if logdir is None:
        continue
    p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
    if not os.path.exists(p):
        continue
    t = pq.read_table(p)
    band = row["band"]
    for ratio in RATIOS:
        if ratio not in t.column_names:
            continue
        arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        flag_rate = float((finite > 1.0).mean())
        rows.append({"log_id": row["log_id"], "band": band, "ratio": ratio,
                     "flag_rate": flag_rate})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)}")

rdf = pd.DataFrame(rows)
print(f"total rows: {len(rdf)}")

# aggregate per band x ratio
print()
print("=" * 78)
print("C1: false alarm rate by firmware band (threshold = 1.0)")
print("=" * 78)
print()
print(f"{'ratio':>18s}  {'band':>14s}  {'n':>5s}  {'mean_rate':>10s}  {'>5%':>8s}  {'CI(>5%)':>20s}")

summary = []
for ratio in RATIOS:
    for band in ["v1.12-1.13", "v1.14-1.15", "v1.16+"]:
        sub = rdf[(rdf["ratio"] == ratio) & (rdf["band"] == band)]
        if len(sub) < 5:
            continue
        n = len(sub)
        mean = sub["flag_rate"].mean()
        gt5 = int((sub["flag_rate"] > 0.05).sum())
        lo, hi = wilson_ci(gt5, n)
        print(f"{ratio:>18s}  {band:>14s}  {n:>5d}  {mean:>10.4f}  {gt5:>3d}/{n:<3d}  "
              f"[{lo:.3f}, {hi:.3f}]")
        summary.append({
            "ratio": ratio, "band": band, "n": n,
            "mean_rate": mean, "n_gt5": gt5, "frac_gt5": gt5/n,
            "ci_low": lo, "ci_high": hi,
        })
    print()

sdf = pd.DataFrame(summary)
sdf.to_csv("c1_per_band.csv", index=False)
print("saved c1_per_band.csv")