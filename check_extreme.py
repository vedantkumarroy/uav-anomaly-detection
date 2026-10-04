import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# find logs with extreme ratio scores
df = pd.read_csv("analysis_set_with_scores.csv")

for ratio in ["gps_hpos[0]", "gps_hpos[1]", "gps_vpos"]:
    col = f"score_{ratio}"
    if col not in df.columns:
        continue
    extreme = df[df[col] > 1e9]
    print(f"=== {ratio} ===")
    print(f"  logs with score > 1e9: {len(extreme)}")
    for _, row in extreme.head(3).iterrows():
        print(f"    log_id: {row['log_id']}, score: {row[col]:.4e}, band: {row['band']}")
    print()

# pick one and inspect raw data
target_log = None
for ratio in ["gps_hpos[1]"]:
    col = f"score_{ratio}"
    extreme = df[df[col] > 1e9]
    if len(extreme) > 0:
        target_log = extreme.iloc[0]["log_id"]
        target_ratio = ratio
        break

if target_log is None:
    print("no extreme logs found")
    exit()

print(f"=== inspecting log {target_log} ratio {target_ratio} ===")

for base in ["pilot_2270", "pilot_extra"]:
    logdir = os.path.join(base, "logs", target_log)
    if not os.path.isdir(logdir):
        continue
    p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
    if not os.path.exists(p):
        continue
    t = pq.read_table(p)
    if target_ratio not in t.column_names:
        continue
    arr = t[target_ratio].to_numpy(zero_copy_only=False).astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        continue

    print(f"  file: {p}")
    print(f"  n samples: {len(arr)}")
    print(f"  finite samples: {len(finite)}")
    print(f"  min: {finite.min():.6e}")
    print(f"  max: {finite.max():.6e}")
    print(f"  median: {np.median(finite):.6e}")

    # show top 10 values
    top = np.sort(finite)[-10:]
    print(f"  top 10 values:")
    for v in top[::-1]:
        print(f"    {v:.6e}")

    # check if any are exactly 1e13, 1e15, etc.
    print()
    print("  checking for sentinel values:")
    for sentinel in [1e12, 1e13, 1e14, 1e15, 1e16, 1e20]:
        count = int((finite == sentinel).sum())
        if count > 0:
            print(f"    exact matches for {sentinel:.0e}: {count}")

    # count values > 1e10
    n_large = int((finite > 1e10).sum())
    print(f"  values > 1e10: {n_large}")
    n_near_inf = int((finite > 1e30).sum())
    print(f"  values > 1e30: {n_near_inf}")
    break