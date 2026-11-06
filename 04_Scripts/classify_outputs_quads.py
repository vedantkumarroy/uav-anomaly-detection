import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
df = pd.read_csv("pilot_2270_index.csv")

outputs_logs = df[df["actuator_source"] == "actuator_outputs"].copy()
print(f"{len(outputs_logs)} actuator_outputs logs")

results = []
for _, row in outputs_logs.iterrows():
    log_id = row["log_id"]
    logdir = os.path.join(LOGDIR, log_id)
    p = os.path.join(logdir, "actuator_outputs.parquet")

    n_active_outputs = 0
    noutputs_declared = None

    if os.path.exists(p):
        t = pq.read_table(p)
        if "noutputs" in t.column_names:
            try:
                noutputs_declared = int(np.nanmax(t["noutputs"].to_numpy(zero_copy_only=False)))
            except Exception:
                pass
        cols = sorted([c for c in t.column_names if c.startswith("output[")])
        active = []
        for c in cols:
            arr = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
            if np.nanmax(np.abs(arr)) == 0:
                continue
            active.append(c)
        n_active_outputs = len(active)

    results.append({
        "log_id": log_id,
        "n_active_outputs": n_active_outputs,
        "noutputs_declared": noutputs_declared,
        "is_quad_from_outputs": n_active_outputs == 4,
    })

rdf = pd.DataFrame(results)
print()
print("=== active outputs distribution ===")
print(rdf["n_active_outputs"].value_counts().sort_index())
print()
print("=== is_quad_from_outputs counts ===")
print(rdf["is_quad_from_outputs"].value_counts())

# save
rdf.to_csv("classify_outputs_quads.csv", index=False)
print("\nsaved classify_outputs_quads.csv")

# merge back into master index
df = df.merge(rdf[["log_id", "is_quad_from_outputs"]], on="log_id", how="left")
df["is_quadrotor_final"] = df["is_quadrotor"]
mask = df["actuator_source"] == "actuator_outputs"
df.loc[mask, "is_quadrotor_final"] = df.loc[mask, "is_quad_from_outputs"]
df.to_csv("pilot_2270_index.csv", index=False)
print("updated pilot_2270_index.csv")

print()
print("=== final is_quadrotor_final counts ===")
print(df["is_quadrotor_final"].value_counts(dropna=False))