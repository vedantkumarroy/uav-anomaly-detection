import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
df = pd.read_csv("pilot_2270_index.csv")

un = df[df["sat_frac"].isna()].copy()
print(f"{len(un)} unmeasurable logs")

reasons = []
for _, row in un.iterrows():
    log_id = row["log_id"]
    logdir = os.path.join(LOGDIR, log_id)

    p_ld = os.path.join(logdir, "vehicle_land_detected.parquet")
    if not os.path.exists(p_ld):
        reasons.append("no_land_topic")
        continue

    t = pq.read_table(p_ld)
    if "landed" not in t.column_names:
        reasons.append("no_landed_field")
        continue
    landed = t["landed"].to_numpy(zero_copy_only=False).astype(bool)
    n_air = int((~landed).sum())

    src = row["actuator_source"]
    pwm = row["pwm_source"]

    if n_air < 2:
        reasons.append(f"ground_only_{src}")
    elif src == "actuator_outputs" and pwm == "none":
        reasons.append("no_pwm_params")
    elif src == "none":
        reasons.append("no_actuator_topic")
    else:
        reasons.append("unknown")

un["skip_reason"] = reasons
print()
print("skip_reason counts:")
print(un["skip_reason"].value_counts())

# merge back into full df
df["skip_reason"] = ""
df.loc[un.index, "skip_reason"] = un["skip_reason"]

df.to_csv("pilot_2270_index.csv", index=False)
print()
print("saved pilot_2270_index.csv")