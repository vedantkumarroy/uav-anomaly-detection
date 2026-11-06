import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

df = pd.read_csv("pilot_2270_index_original.csv")
sus = df[(df["actuator_source"] == "actuator_motors") & (df["sat_frac"].isna())]
print(f"{len(sus)} logs with actuator_motors but NaN sat_frac")
print()

LOGDIR = "pilot_2270/logs"

for _, row in sus.head(5).iterrows():
    log_id = row["log_id"]
    logdir = os.path.join(LOGDIR, log_id)
    print(f"=== {log_id} ===")

    # check actuator_motors
    p = os.path.join(logdir, "actuator_motors.parquet")
    if not os.path.exists(p):
        print("  actuator_motors.parquet not present")
        continue
    t = pq.read_table(p)
    cols = [c for c in t.column_names if c.startswith("control[")]
    print(f"  control columns: {len(cols)}")
    for c in cols[:4]:
        arr = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
        n_nan = int(np.isnan(arr).sum())
        n_total = len(arr)
        n_nonzero = int(np.sum((~np.isnan(arr)) & (arr != 0)))
        print(f"    {c}: n={n_total}, n_nan={n_nan}, n_nonzero={n_nonzero}")
        if n_nonzero > 0:
            finite = arr[~np.isnan(arr)]
            print(f"      range: [{finite.min():.4f}, {finite.max():.4f}]")

    # check vehicle_land_detected
    p2 = os.path.join(logdir, "vehicle_land_detected.parquet")
    if not os.path.exists(p2):
        print("  vehicle_land_detected.parquet not present")
        print()
        continue
    t2 = pq.read_table(p2)
    if "landed" in t2.column_names:
        landed = t2["landed"].to_numpy(zero_copy_only=False).astype(bool)
        print(f"  landed: n={len(landed)}, n_False(airborne)={int((~landed).sum())}, n_True(ground)={int(landed.sum())}")
    else:
        print("  landed field not present")
    print()