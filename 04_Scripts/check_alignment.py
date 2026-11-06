import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOG_ID = "37f5014b-c8fc-447d-9974-f72adcd6ff34"
COL = "gps_hpos[1]"

logdir = os.path.join("pilot_2270", "logs", LOG_ID)

rat = pq.read_table(os.path.join(logdir, "estimator_innovation_test_ratios.parquet"))
inn = pq.read_table(os.path.join(logdir, "estimator_innovations.parquet"))
var = pq.read_table(os.path.join(logdir, "estimator_innovation_variances.parquet"))

print("topic shapes:")
print(f"  ratios: n={rat.num_rows}, cols={len(rat.column_names)}")
print(f"  innovations: n={inn.num_rows}, cols={len(inn.column_names)}")
print(f"  variances: n={var.num_rows}, cols={len(var.column_names)}")
print()

print(f"ratios columns: {rat.column_names[:5]} ...")
print(f"innovations columns: {inn.column_names[:5]} ...")
print(f"variances columns: {var.column_names[:5]} ...")
print()

# check timestamp alignment
rat_ts = rat["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
inn_ts = inn["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
var_ts = var["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)

print(f"timestamp ranges:")
print(f"  ratios:    {rat_ts[0]} to {rat_ts[-1]}")
print(f"  innovations: {inn_ts[0]} to {inn_ts[-1]}")
print(f"  variances: {var_ts[0]} to {var_ts[-1]}")
print()

# are they byte-identical?
if len(rat_ts) == len(inn_ts):
    print(f"same number of samples (ratios=innovations)")
    diff = np.abs(rat_ts - inn_ts)
    print(f"  max |ts_ratio - ts_innov| = {diff.max()} microseconds")
else:
    print(f"different sample counts: ratios={len(rat_ts)}, innovations={len(inn_ts)}")

if len(rat_ts) == len(var_ts):
    print(f"same number of samples (ratios=variances)")
    diff = np.abs(rat_ts - var_ts)
    print(f"  max |ts_ratio - ts_var| = {diff.max()} microseconds")
else:
    print(f"different sample counts: ratios={len(rat_ts)}, variances={len(var_ts)}")

print()

# if aligned by timestamp, compute ratio for a few samples
if COL in rat.column_names and COL in inn.column_names and COL in var.column_names:
    r = rat[COL].to_numpy(zero_copy_only=False).astype(np.float64)
    i = inn[COL].to_numpy(zero_copy_only=False).astype(np.float64)
    v = var[COL].to_numpy(zero_copy_only=False).astype(np.float64)

    n = min(len(r), len(i), len(v))
    print(f"comparing first {n} samples (assuming index alignment)")
    print()

    # find 5 samples with highest ratio
    idx = np.argsort(-r[:n])[:5]
    print(f"{'idx':>5s}  {'ratio':>14s}  {'innovation':>14s}  {'variance':>14s}  "
          f"{'innov/sqrt(var)':>18s}  {'ratio * sqrt(var)':>18s}")
    for k in idx:
        sqrt_v = np.sqrt(v[k]) if v[k] > 0 else np.nan
        computed = i[k] / sqrt_v if sqrt_v and not np.isnan(sqrt_v) else np.nan
        reconstructed = r[k] * sqrt_v if not np.isnan(sqrt_v) else np.nan
        print(f"{k:>5d}  {r[k]:>14.4e}  {i[k]:>14.4e}  {v[k]:>14.4e}  "
              f"{computed:>18.4e}  {reconstructed:>18.4e}")

    print()
    print("If 'computed' matches 'ratio', the relationship is ratio = innov / sqrt(var).")
    print("If 'reconstructed' matches 'innovation', same relationship, verified.")