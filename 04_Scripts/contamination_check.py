"""
Contamination check.

Remove flights that trip any of the 5 indicators from the calibration set.
Recompute coverage on Group C.

If coverage moves by less than ~1 percentage point, the calibration set was
already clean enough and the unlabeled setup is defensible.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

    # merge indicator columns
    if "failsafe_frac" in idx.columns:
        cols = ["log_id", "failsafe_frac", "nav_state_forbidden_frac",
                "filter_fault_flags_frac", "landed_at_end", "sat_frac"]
        extra = idx[cols].copy()
        df = df.merge(extra, on="log_id", how="left", suffixes=("", "_idx"))

    print(f"loaded {len(df)}")

    # tripped = any indicator non-zero or landed_at_end False or sat_frac > 0.05
    tripped = (
        (df["failsafe_frac"] > 0) |
        (df["nav_state_forbidden_frac"] > 0) |
        (df["filter_fault_flags_frac"] > 0) |
        (df["landed_at_end"] == False) |
        (df["sat_frac"] > 0.05)
    )
    print(f"flights tripping at least one indicator: {int(tripped.sum())} of {len(df)}")

    # baseline: full B cal, C test
    B_mask = df["group"] == "B"
    C_mask = df["group"] == "C"

    # contaminated B and C
    B_contam_mask = B_mask & tripped
    C_contam_mask = C_mask & tripped
    print(f"  Group B: {int(B_mask.sum())} total, {int(B_contam_mask.sum())} tripped")
    print(f"  Group C: {int(C_mask.sum())} total, {int(C_contam_mask.sum())} tripped")
    print()

    results = []
    print("=" * 82)
    print("Coverage with and without contamination in the calibration set")
    print("=" * 82)
    print()
    print(f"{'ratio':>18s}  {'full_B':>10s}  {'clean_B':>10s}  {'diff':>8s}  "
          f"{'n_cal_full':>10s}  {'n_cal_clean':>11s}")

    for ratio in RATIOS:
        col = f"score_log_{ratio}"

        # full
        cal_full = df.loc[B_mask, col].dropna().values
        test = df.loc[C_mask, col].dropna().values
        if len(cal_full) < 10 or len(test) < 10:
            continue
        thr_full = conformal_threshold(cal_full, ALPHA)
        cov_full = float(np.mean(test <= thr_full))

        # clean
        B_clean = B_mask & ~tripped
        cal_clean = df.loc[B_clean, col].dropna().values
        if len(cal_clean) < 10:
            continue
        thr_clean = conformal_threshold(cal_clean, ALPHA)
        cov_clean = float(np.mean(test <= thr_clean))

        diff = cov_clean - cov_full
        print(f"{ratio:>18s}  {cov_full:>10.4f}  {cov_clean:>10.4f}  "
              f"{diff:>+8.4f}  {len(cal_full):>10d}  {len(cal_clean):>11d}")
        results.append({
            "ratio": ratio,
            "n_cal_full": len(cal_full),
            "n_cal_clean": len(cal_clean),
            "coverage_full": cov_full,
            "coverage_clean": cov_clean,
            "diff": diff,
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv("contamination_check.csv", index=False)
    print()
    print(f"Mean coverage (full B):  {rdf['coverage_full'].mean():.4f}")
    print(f"Mean coverage (clean B): {rdf['coverage_clean'].mean():.4f}")
    print(f"Mean difference:         {rdf['diff'].mean():+.4f}")
    print()
    print("Interpretation:")
    if abs(rdf['diff'].mean()) < 0.01:
        print("  Mean difference is less than 1 percentage point.")
        print("  The calibration set was already clean enough.")
        print("  The unlabeled setup is defensible.")
    else:
        print("  Mean difference is larger than 1 percentage point.")
        print("  Contamination in the calibration set matters.")
        print("  Reviewer concern is valid.")
    print()
    print("saved contamination_check.csv")


if __name__ == "__main__":
    main()