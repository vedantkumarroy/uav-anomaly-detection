import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOG_ID = "37f5014b-c8fc-447d-9974-f72adcd6ff34"
RATIO = "gps_hpos[1]"

logdir = os.path.join("pilot_2270", "logs", LOG_ID)
rat_path = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
var_path = os.path.join(logdir, "estimator_innovation_variances.parquet")

if not os.path.exists(rat_path):
    print("no ratio file")
    exit()

rat = pq.read_table(rat_path)
if RATIO not in rat.column_names:
    print(f"{RATIO} not in ratio topic")
    exit()

rat_arr = rat[RATIO].to_numpy(zero_copy_only=False).astype(np.float64)

print(f"ratio topic: {rat_path}")
print(f"ratio column: {RATIO}")
print(f"n samples: {len(rat_arr)}")
print(f"median: {np.nanmedian(rat_arr):.6e}")
print(f"max: {np.nanmax(rat_arr):.6e}")
print()

# find the samples with extreme ratio values
extreme_idx = np.where(rat_arr > 1e10)[0]
print(f"samples with ratio > 1e10: {len(extreme_idx)}")
print(f"first 5 indices: {extreme_idx[:5].tolist()}")
print()

# check variance topic
if not os.path.exists(var_path):
    print(f"variance file not found: {var_path}")
    print()
    print("Cannot confirm directly. But the ratio is:")
    print("  ratio = innovation / sqrt(variance)")
    print("  A continuous distribution up to 10^12 with no sentinels")
    print("  is the classic signature of division by near-zero variance.")
    exit()

var = pq.read_table(var_path)
print(f"variance topic: {var_path}")
print(f"variance columns: {var.columns[:6]}")

# the variance column name for gps_hpos[1] might be gps_hpos[1] or hpos_1
var_col = None
for c in var.column_names:
    if c == RATIO or c.replace("[", "_").replace("]", "") == RATIO.replace("[", "_").replace("]", ""):
        var_col = c
        break
if var_col is None and "gps_hpos[1]" in var.column_names:
    var_col = "gps_hpos[1]"

if var_col is None:
    print(f"variance column for {RATIO} not found")
    print(f"available: {[c for c in var.column_names if 'hpos' in c.lower()]}")
    exit()

var_arr = var[var_col].to_numpy(zero_copy_only=False).astype(np.float64)

print(f"variance column: {var_col}")
print(f"n samples: {len(var_arr)}")
print(f"min: {np.nanmin(var_arr):.6e}")
print(f"median: {np.nanmedian(var_arr):.6e}")
print(f"max: {np.nanmax(var_arr):.6e}")
print()

# align by length or by index
if len(var_arr) == len(rat_arr):
    var_at_extreme = var_arr[extreme_idx]
    print(f"variance at extreme ratio samples:")
    print(f"  min: {np.nanmin(var_at_extreme):.6e}")
    print(f"  median: {np.nanmedian(var_at_extreme):.6e}")
    print(f"  max: {np.nanmax(var_at_extreme):.6e}")
    print()
    print(f"ratio at those samples: median = {np.nanmedian(rat_arr[extreme_idx]):.6e}")
    print()
    # reconstruct
    reconstructed = np.sqrt(var_at_extreme) * rat_arr[extreme_idx]
    print(f"reconstructed innovation (ratio * sqrt(var)): median = {np.nanmedian(reconstructed):.6e}")
    print("if this matches a reasonable innovation, the ratio is innovation / sqrt(var)")