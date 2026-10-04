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

THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 10.0]


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

# For each threshold, compute fraction of flights with any ratio above threshold in >5% samples
print()
print("=" * 78)
print("C1 sensitivity: fraction of flights over-flagged at different ratio thresholds")
print("=" * 78)
print()

for thr in THRESHOLDS:
    print(f"--- threshold = {thr} ---")
    n_over = 0
    for i, row in df.iterrows():
        logdir = logdir_for(row["log_id"])
        if logdir is None:
            continue
        p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        over = False
        for ratio in RATIOS:
            if ratio not in t.column_names:
                continue
            arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            if (finite > thr).mean() > 0.05:
                over = True
                break
        if over:
            n_over += 1
    frac = n_over / len(df)
    print(f"  flights over-flagged (>5% samples): {n_over} of {len(df)} ({100*frac:.1f}%)")