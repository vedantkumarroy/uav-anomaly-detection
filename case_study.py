import numpy as np
import pandas as pd

INDEX_CSV = "pilot_2270_index.csv"
GROUPS_CSV = "analysis_set_with_scores.csv"

idx = pd.read_csv(INDEX_CSV)
groups = pd.read_csv(GROUPS_CSV)

# drop group from idx if it exists, so the merge does not clash
if "group" in idx.columns:
    idx = idx.drop(columns=["group"])

g = groups[["log_id", "group"]].drop_duplicates()

d = idx.merge(g, on="log_id", how="inner")
print(f"analysis logs: {len(d)}")

counts = np.zeros(len(d), dtype=int)
counts += (d["failsafe_frac"] > 0).astype(int).values
counts += (d["nav_state_forbidden_frac"] > 0).astype(int).values
counts += (d["filter_fault_flags_frac"] > 0).astype(int).values
counts += (d["landed_at_end"] == False).astype(int).values
counts += (d["sat_frac"] > 0.05).astype(int).values
d["n_indicators"] = counts

top = d[d["group"] == "C"].sort_values("n_indicators", ascending=False).head(15)
cols = ["log_id", "band", "n_indicators",
        "failsafe_frac", "nav_state_forbidden_frac",
        "filter_fault_flags_frac", "landed_at_end", "sat_frac",
        "sat_max_run_s", "motor_spread_p95"]
print("=== Top 15 high-anomaly logs in Group C ===")
print(top[cols].to_string(index=False))
top.to_csv("case_study_top15.csv", index=False)
print()
print("saved case_study_top15.csv")