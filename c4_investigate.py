"""
Investigate why ratio-based detectors miss motor imbalance.

Compare the innovation ratios of:
  1. A missed genuine anomaly with severe motor imbalance (c1506e1f)
  2. The one caught anomaly with global saturation and balanced motors (696590b3)
  3. A typical healthy flight

If the ratio distributions look similar across cases 1 and 3, the ratio signal
does not carry imbalance information. That is the answer.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

CASES = {
    "missed_imbalance_c1506e1f": "c1506e1f-9ee3-4640-b073-1483605b9e32",
    "missed_imbalance_eb456400": "eb456400-8803-4062-b937-ea4b4320bb59",
    "caught_saturation_696590b3": "696590b3-101b-4aa4-820d-d67b22eecea3",
}


def logdir_for(log_id):
    for base in ["pilot_2270", "pilot_extra"]:
        p = os.path.join(base, "logs", log_id)
        if os.path.isdir(p):
            return p
    return None


def ratio_stats(log_id):
    logdir = logdir_for(log_id)
    if logdir is None:
        return None
    p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
    if not os.path.exists(p):
        return None
    t = pq.read_table(p)
    stats = {}
    for r in RATIOS:
        if r not in t.column_names:
            continue
        arr = t[r].to_numpy(zero_copy_only=False).astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        stats[r] = {
            "n": len(finite),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max()),
            "frac_above_1": float((finite > 1.0).mean()),
            "frac_above_5": float((finite > 5.0).mean()),
        }
    return stats


print("=" * 110)
print("Comparing ratio distributions: missed imbalance vs caught saturation")
print("=" * 110)
print()

results = {}
for name, log_id in CASES.items():
    results[name] = ratio_stats(log_id)
    print(f"=== {name} ({log_id[:20]}) ===")
    if results[name] is None:
        print("  no data")
        continue
    for r, s in results[name].items():
        print(f"  {r:>18s}: n={s['n']:>5d}  median={s['median']:.4f}  "
              f"p95={s['p95']:.4f}  max={s['max']:.4f}  frac>1={s['frac_above_1']:.3f}")
    print()

# Compute how many ratios differ substantially between the missed imbalance
# and the caught saturation
print("=" * 110)
print("Cross-comparison: which ratios discriminate the two cases?")
print("=" * 110)
print()

missed_key = "missed_imbalance_c1506e1f"
caught_key = "caught_saturation_696590b3"
missed = results[missed_key]
caught = results[caught_key]

if missed and caught:
    print(f"{'ratio':>18s}  {'missed_frac>1':>14s}  {'caught_frac>1':>14s}  {'difference':>12s}")
    for r in RATIOS:
        if r not in missed or r not in caught:
            continue
        m = missed[r]["frac_above_1"]
        c = caught[r]["frac_above_1"]
        diff = m - c
        marker = "  <-- discriminates" if abs(diff) > 0.05 else ""
        print(f"{r:>18s}  {m:>14.3f}  {c:>14.3f}  {diff:>+12.3f}{marker}")

print()
print("Interpretation:")
print("  If most ratios have similar frac_above_1 across both cases, the ratios")
print("  do not discriminate between motor imbalance and global saturation.")
print("  If only 1 or 2 ratios differ, the detector relies on those.")
print()

# Also compute the same statistics for a batch of healthy flights as a reference
print("=" * 110)
print("Reference: healthy flights (Group C, top 5 not in genuine list)")
print("=" * 110)
print()

df = pd.read_csv("c4_handlabel_final.csv")
healthy = df[df["hand_label"] == "not_genuine"].head(5)

healthy_stats = []
for _, row in healthy.iterrows():
    s = ratio_stats(row["log_id"])
    if s is not None:
        healthy_stats.append(s)

if healthy_stats:
    print(f"Across {len(healthy_stats)} healthy flights:")
    print(f"{'ratio':>18s}  {'mean_frac>1':>14s}  {'std':>10s}")
    for r in RATIOS:
        vals = [h[r]["frac_above_1"] for h in healthy_stats if r in h]
        if len(vals) == 0:
            continue
        print(f"{r:>18s}  {np.mean(vals):>14.4f}  {np.std(vals):>10.4f}")