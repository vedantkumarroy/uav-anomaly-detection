import numpy as np
import pandas as pd

INPUT_CSV = "analysis_set_with_log_scores.csv"
INDEX_CSV = "pilot_2270_index.csv"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

ALPHA = 0.05


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


df = pd.read_csv(INPUT_CSV)
idx = pd.read_csv(INDEX_CSV)

extra = idx[["log_id", "failsafe_frac", "nav_state_forbidden_frac",
             "filter_fault_flags_frac", "landed_at_end", "sat_frac"]].copy()
extra = extra.rename(columns={
    "failsafe_frac": "failsafe_frac_idx",
    "nav_state_forbidden_frac": "nav_state_forbidden_frac_idx",
    "filter_fault_flags_frac": "filter_fault_flags_frac_idx",
    "landed_at_end": "landed_at_end_idx",
    "sat_frac": "sat_frac_idx",
})
df = df.merge(extra, on="log_id", how="left")

counts = np.zeros(len(df), dtype=int)
counts += (df["failsafe_frac_idx"] > 0).astype(int).values
counts += (df["nav_state_forbidden_frac_idx"] > 0).astype(int).values
counts += (df["filter_fault_flags_frac_idx"] > 0).astype(int).values
counts += (df["landed_at_end_idx"] == False).astype(int).values
counts += (df["sat_frac_idx"] > 0.05).astype(int).values
df["n_indicators"] = counts

B_mask = df["group"].values == "B"
C_mask = df["group"].values == "C"
labels_C = df["n_indicators"].values[C_mask]

print(f"{'ratio':>18s}  {'n_cal':>6s}  {'n_test':>7s}  {'coverage':>10s}  "
      f"{'det_n2':>8s}  {'det_n3':>8s}")
results = []
for r in RATIOS:
    col = f"score_log_{r}"
    cal = df[B_mask][col].dropna().values
    test = df[C_mask][col].dropna().values
    if len(cal) < 10 or len(test) < 10:
        continue
    thr = conformal_threshold(cal, ALPHA)
    flags = test > thr
    cov = 1 - flags.mean()
    # detection power (the log must be in C with a valid score)
    c_idx = df[C_mask].index[df[C_mask][col].notna()].values
    c_labels = df.loc[c_idx, "n_indicators"].values
    hi2 = c_labels >= 2
    hi3 = c_labels >= 3
    det2 = flags[hi2].mean() if hi2.sum() > 0 else 0
    det3 = flags[hi3].mean() if hi3.sum() > 0 else 0
    results.append({
        "ratio": r,
        "n_cal": len(cal),
        "n_test": len(test),
        "coverage": cov,
        "detection_n2": det2,
        "detection_n3": det3,
    })
    print(f"{r:>18s}  {len(cal):>6d}  {len(test):>7d}  {cov:>10.4f}  "
          f"{det2:>8.4f}  {det3:>8.4f}")

rdf = pd.DataFrame(results)
rdf.to_csv("per_ratio_detection_power.csv", index=False)
print()
print("saved per_ratio_detection_power.csv")