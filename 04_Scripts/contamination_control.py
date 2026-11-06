"""
Contamination control.

The clean-B result (379 flights) showed coverage drop of 1.5 pp vs full-B (656).
We need to separate two effects:
  1. Real contamination (tripped flights inflate the threshold)
  2. Reduced sample size (379 vs 656)

Control: pick 379 random flights from the full B set, repeat 200 times,
compute coverage. Compare with clean-B.
"""

import os
import numpy as np
import pandas as pd

ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
INDEX_CSV = "combined_index.csv"

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
N_RANDOM = 200
N_CLEAN = 379


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    idx = pd.read_csv(INDEX_CSV)

    cols = ["log_id", "failsafe_frac", "nav_state_forbidden_frac",
            "filter_fault_flags_frac", "landed_at_end", "sat_frac"]
    extra = idx[cols].copy()
    df = df.merge(extra, on="log_id", how="left", suffixes=("", "_idx"))

    tripped = (
        (df["failsafe_frac"] > 0) |
        (df["nav_state_forbidden_frac"] > 0) |
        (df["filter_fault_flags_frac"] > 0) |
        (df["landed_at_end"] == False) |
        (df["sat_frac"] > 0.05)
    )

    B_mask = df["group"] == "B"
    C_mask = df["group"] == "C"
    B_clean = B_mask & ~tripped

    rng = np.random.default_rng(42)

    print("=" * 82)
    print("Contamination control: clean-B vs random-B of same size")
    print("=" * 82)
    print()
    print(f"n_cal_full  = {int(B_mask.sum())}")
    print(f"n_cal_clean = {int(B_clean.sum())}")
    print(f"n_cal_random = {N_CLEAN} (200 iterations)")
    print()
    print(f"{'ratio':>18s}  {'full_B':>10s}  {'clean_B':>10s}  {'rand_B':>10s}  "
          f"{'clean-full':>12s}  {'rand-full':>10s}")

    results = []
    for ratio in RATIOS:
        col = f"score_log_{ratio}"

        # Full
        cal_full = df.loc[B_mask, col].dropna().values
        test = df.loc[C_mask, col].dropna().values
        if len(cal_full) < 10 or len(test) < 10:
            continue
        thr_full = conformal_threshold(cal_full, ALPHA)
        cov_full = float(np.mean(test <= thr_full))

        # Clean
        cal_clean = df.loc[B_clean, col].dropna().values
        thr_clean = conformal_threshold(cal_clean, ALPHA)
        cov_clean = float(np.mean(test <= thr_clean))

        # Random subsets of size N_CLEAN
        full_idx = df.index[B_mask].tolist()
        rand_covs = []
        for _ in range(N_RANDOM):
            sample_idx = rng.choice(full_idx, size=min(N_CLEAN, len(full_idx)), replace=False)
            cal_rand = df.loc[sample_idx, col].dropna().values
            if len(cal_rand) < 10:
                continue
            thr_rand = conformal_threshold(cal_rand, ALPHA)
            rand_covs.append(float(np.mean(test <= thr_rand)))
        cov_rand = np.mean(rand_covs) if rand_covs else np.nan

        diff_clean = cov_clean - cov_full
        diff_rand = cov_rand - cov_full

        print(f"{ratio:>18s}  {cov_full:>10.4f}  {cov_clean:>10.4f}  {cov_rand:>10.4f}  "
              f"{diff_clean:>+12.4f}  {diff_rand:>+10.4f}")
        results.append({
            "ratio": ratio,
            "coverage_full": cov_full,
            "coverage_clean": cov_clean,
            "coverage_random": cov_rand,
            "diff_clean": diff_clean,
            "diff_random": diff_rand,
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv("contamination_control.csv", index=False)

    print()
    print("=" * 82)
    print("Summary")
    print("=" * 82)
    print()
    print(f"Mean coverage full-B:         {rdf['coverage_full'].mean():.4f}")
    print(f"Mean coverage clean-B:        {rdf['coverage_clean'].mean():.4f}")
    print(f"Mean coverage random-B:       {rdf['coverage_random'].mean():.4f}")
    print()
    print(f"Mean diff (clean - full):     {rdf['diff_clean'].mean():+.4f}")
    print(f"Mean diff (random - full):    {rdf['diff_random'].mean():+.4f}")
    print()
    print("Interpretation:")
    contamination_effect = rdf['diff_clean'].mean() - rdf['diff_random'].mean()
    print(f"  If random-B is the same as clean-B, the effect is sample size.")
    print(f"  If random-B is closer to full-B, the effect is contamination.")
    print(f"  Difference of differences: {contamination_effect:+.4f}")
    if abs(contamination_effect) > 0.005:
        print("  Contamination is a real effect at the 0.5 pp level.")
    else:
        print("  Contamination effect is small. Sample size explains most of it.")

    print()
    print("saved contamination_control.csv")


if __name__ == "__main__":
    main()