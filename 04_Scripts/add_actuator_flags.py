import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
df = pd.read_csv("pilot_2270_index.csv")

reversible_flags_list = []
active_ctrl_count_list = []

for _, row in df.iterrows():
    log_id = row["log_id"]
    logdir = os.path.join(LOGDIR, log_id)

    rev = np.nan
    n_active = np.nan

    p = os.path.join(logdir, "actuator_motors.parquet")
    if os.path.exists(p):
        t = pq.read_table(p)
        if "reversible_flags" in t.column_names:
            rf = t["reversible_flags"].to_numpy(zero_copy_only=False)
            if len(rf) > 0:
                rev = int(rf[0])
        ctrl_cols = sorted([c for c in t.column_names if c.startswith("control[")])
        n = 0
        for c in ctrl_cols:
            arr = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
            if np.isnan(arr).all():
                continue
            if np.nanmax(np.abs(arr)) == 0:
                continue
            n += 1
        n_active = n

    reversible_flags_list.append(rev)
    active_ctrl_count_list.append(n_active)

df["reversible_flags"] = reversible_flags_list
df["n_active_control_channels"] = active_ctrl_count_list

df.to_csv("pilot_2270_index.csv", index=False)
print("saved")

print()
print("=== reversible_flags distribution ===")
print(df["reversible_flags"].value_counts(dropna=False))

print()
print("=== n_active_control_channels distribution ===")
print(df["n_active_control_channels"].value_counts(dropna=False))

print()
print("=== cross-tab: active channels vs SYS_AUTOSTART class ===")
print(pd.crosstab(df["n_active_control_channels"], df["actuator_source"], dropna=False))