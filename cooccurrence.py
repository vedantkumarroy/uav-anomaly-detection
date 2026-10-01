import numpy as np
import pandas as pd
import itertools

idx = pd.read_csv("pilot_2270_index.csv")
groups = pd.read_csv("analysis_set_with_groups.csv")

# drop group from idx if present, to avoid merge collision
if "group" in idx.columns:
    idx = idx.drop(columns=["group"])

# get only log_id and group from groups
g = groups[["log_id", "group"]].drop_duplicates()

d = idx.merge(g, on="log_id", how="inner")
print(f"analysis logs: {len(d)}")

d["ind_failsafe"] = d["failsafe_frac"] > 0
d["ind_nav"] = d["nav_state_forbidden_frac"] > 0
d["ind_fault"] = d["filter_fault_flags_frac"] > 0
d["ind_landing"] = d["landed_at_end"] == False
d["ind_sat"] = d["sat_frac"] > 0.05

ind_cols = ["ind_failsafe", "ind_nav", "ind_fault", "ind_landing", "ind_sat"]

print()
print("Per-indicator counts:")
for c in ind_cols:
    print(f"  {c}: {int(d[c].sum())} of {len(d)} ({100*d[c].mean():.1f}%)")

print()
print("Pairwise co-occurrence:")
for a, b in itertools.combinations(ind_cols, 2):
    both = int((d[a] & d[b]).sum())
    if both > 0:
        print(f"  {a} + {b}: {both}")

print()
print("n_indicators distribution:")
counts = np.zeros(len(d), dtype=int)
for c in ind_cols:
    counts += d[c].astype(int).values
d["n_indicators"] = counts
print(d["n_indicators"].value_counts().sort_index().to_string())

d[["log_id", "band", "group", "n_indicators"] + ind_cols].to_csv(
    "cooccurrence_table.csv", index=False)
print()
print("saved cooccurrence_table.csv")