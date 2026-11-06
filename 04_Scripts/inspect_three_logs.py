import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
LOGS = [
    "55bc60e0-36cf-4b03-b47b-56804a060a4e",
    "9a387907-ccd8-45ff-889e-06b34f9c9b65",
    "f386caa8-c372-40e0-b1c5-4687f8d614b7",
]


def load(t, name):
    if name not in t.column_names:
        return None
    return t[name].to_numpy(zero_copy_only=False)


for log_id in LOGS:
    logdir = os.path.join(LOGDIR, log_id)
    print(f"\n================ {log_id} ================")

    p_mot = os.path.join(logdir, "actuator_motors.parquet")
    if not os.path.exists(p_mot):
        print("  no actuator_motors.parquet")
        continue

    t = pq.read_table(p_mot)
    print(f"  columns: {t.column_names}")
    if "reversible_flags" in t.column_names:
        rf = load(t, "reversible_flags")
        print(f"  reversible_flags unique: {sorted(set(rf.tolist()))}")
    else:
        print("  reversible_flags: not present")

    ctrl_cols = sorted([c for c in t.column_names if c.startswith("control[")])
    active_cols = []
    for c in ctrl_cols:
        arr = load(t, c).astype(np.float64)
        if np.isnan(arr).all():
            continue
        if np.nanmax(np.abs(arr)) == 0:
            continue
        active_cols.append(c)

    print(f"  active control cols: {len(active_cols)} -> {active_cols}")
    for c in active_cols[:6]:
        arr = load(t, c).astype(np.float64)
        finite = arr[~np.isnan(arr)]
        if len(finite) == 0:
            print(f"    {c}: all NaN")
            continue
        print(f"    {c}: min={finite.min():.4f} max={finite.max():.4f} mean={finite.mean():.4f}")

    # land detector
    p_ld = os.path.join(logdir, "vehicle_land_detected.parquet")
    if os.path.exists(p_ld):
        t_ld = pq.read_table(p_ld)
        if "landed" in t_ld.column_names:
            ld = load(t_ld, "landed").astype(bool)
            print(f"  landed: air={int((~ld).sum())} ground={int(ld.sum())} of {len(ld)}")

    # airframe / SYS_AUTOSTART may be in a parameters table if we saved one
    p_par = os.path.join(logdir, "initial_parameters.parquet")
    if os.path.exists(p_par):
        t_par = pq.read_table(p_par)
        print(f"  parameters columns: {t_par.column_names}")
    else:
        print(f"  initial_parameters.parquet: not present")