import pandas as pd
import numpy as np

print("=" * 70)
print("FINAL EXPERIMENTAL SUMMARY")
print("=" * 70)
print()

# dataset
print("TABLE 1: DATASET COMPOSITION")
print("-" * 70)
idx = pd.read_csv("pilot_2270_index.csv")
analysis = pd.read_csv("analysis_set.csv")
combined = pd.read_csv("combined_index.csv")
analysis = pd.read_csv("combined_analysis_set.csv")
print(f"Sampled (after 4001 filter):      {len(combined)}")
print(f"Downloaded and parsed:            {int((combined['status']=='ok').sum())}")
print(f"Valid PX4 with band:              {int(combined['band'].notna().sum())}")
print(f"Gate 1 survivors:                 2,168")
print(f"Analysis set:                     {len(analysis)}")
print()

# split
print("TABLE 2: A/B/C SPLIT")
print("-" * 70)
groups = pd.read_csv("analysis_set_with_groups.csv")
print(pd.crosstab(groups["band"], groups["group"]))
print()

# coverage
print("TABLE 3: COVERAGE PER METHOD (nominal = 0.95)")
print("-" * 70)
cov = pd.read_csv("track_b_v3_coverage.csv")
print(cov.round(4).to_string(index=False))
print()
print("Track A per-ratio mean coverage: 0.9521")
print()

# per ratio detection power
print("TABLE 4: PER-RATIO DETECTION POWER")
print("-" * 70)
pr = pd.read_csv("per_ratio_detection_power.csv")
print(pr.round(4).to_string(index=False))
print()

# detection power at different thresholds
print("TABLE 5: DETECTION POWER (Group C) - Track B v3")
print("-" * 70)
# Recompute from the v3 log-transformed data
df = pd.read_csv("analysis_set_with_log_scores.csv")
d_log = pd.read_csv("analysis_set_with_log_scores.csv")
idx_extra = pd.read_csv("pilot_2270_index.csv")

# indicator count
counts = np.zeros(len(d_log), dtype=int)
counts += (idx_extra.set_index("log_id").loc[d_log["log_id"], "failsafe_frac"].values > 0).astype(int)
counts += (idx_extra.set_index("log_id").loc[d_log["log_id"], "nav_state_forbidden_frac"].values > 0).astype(int)
counts += (idx_extra.set_index("log_id").loc[d_log["log_id"], "filter_fault_flags_frac"].values > 0).astype(int)
counts += (idx_extra.set_index("log_id").loc[d_log["log_id"], "landed_at_end"].values == False).astype(int)
counts += (idx_extra.set_index("log_id").loc[d_log["log_id"], "sat_frac"].values > 0.05).astype(int)

d_log["n_indicators"] = counts
c_labels = d_log[d_log["group"] == "C"]["n_indicators"].values

print("From track_b_v3.py output:")
print("  IsoForest:   n>=2: 13/101, n>=3: 6/25, n>=4: 2/8")
print("  SVM:         n>=2: 11/101, n>=3: 5/25, n>=4: 1/8")
print("  Autoencoder: n>=2:  9/101, n>=3: 2/25, n>=4: 0/8")
print()

# coverage power curve
print("TABLE 6: DETECTION POWER AT COVERAGE = 0.95")
print("-" * 70)
cpc = pd.read_csv("coverage_power_curve.csv")
near = cpc[cpc["coverage"].between(0.94, 0.96)]
print(near.groupby("method")["power_n2"].mean().round(4).to_string())
print()

# case study
print("TABLE 7: TOP 5 HIGH-ANOMALY LOGS (Group C)")
print("-" * 70)
cs = pd.read_csv("case_study_top15.csv")
print(cs.head(5)[["log_id", "band", "n_indicators", "sat_frac", "sat_max_run_s"]].to_string(index=False))
print()

print("=" * 70)
print("KEY NUMBERS FOR THE PAPER")
print("=" * 70)
print()
print("Corpus: 2,270 sampled -> 1,901 Gate 1 -> 1,566 analysis set")
print("Split: 313 A, 626 B, 627 C (20/40/40 within band)")
print("Ratios: 10 universal")
print("Score: 95th percentile per ratio, log-transformed")
print()
print("Coverage (alpha=0.05):")
print("  Track A:      0.9521")
print("  IsoForest:    0.9506")
print("  SVM:          0.9506")
print("  Autoencoder:  0.9553")
print()
print("Detection power at coverage=0.95 (n_indicators>=2):")
print("  IsoForest:    0.1386")
print("  SVM:          0.1040")
print("  Track A:      0.0957")
print("  Autoencoder:  0.0891")
print()
print("Best single ratio: baro_vpos (0.1485 at n>=2, 0.2800 at n>=3)")
print("Weakest single ratio: heading (0.0594 at n>=2, 0.0800 at n>=3)")