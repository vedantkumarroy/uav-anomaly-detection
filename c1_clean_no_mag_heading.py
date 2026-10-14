"""
c1_clean_no_mag_heading.py

Sir's item 5. The clean-flight C1 rate is 15.57 percent.
The per-ratio table shows mag_field[0] at 8.47 percent and heading at 6.58
percent. All other eight ratios are 1.88 percent or less.

Compute the union over-flag rate with mag_field[0] and heading removed.
If it falls to 3 to 4 percent, the gate scheme is generally correct and
the magnetometer and heading gates need recalibration.
"""

import numpy as np
import pandas as pd

C1_CSV = "c1_warmup_excluded.csv"
ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
SATFRAC_CSV = "analysis_set_with_satfrac.csv"

RATIOS_ALL = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

DROP = {"mag_field[0]", "heading"}

_lines = []
def log(s=""):
    print(s)
    _lines.append(s)


def load_analysis():
    a = pd.read_csv(ANALYSIS_CSV)
    if "sat_frac" in a.columns:
        a["_sat_frac"] = a["sat_frac"]
    elif "sat_frac_recomputed" in a.columns:
        a["_sat_frac"] = a["sat_frac_recomputed"]
    else:
        s = pd.read_csv(SATFRAC_CSV)
        col = "sat_frac_recomputed" if "sat_frac_recomputed" in s.columns else "sat_frac"
        a = a.merge(s[["log_id", col]].rename(columns={col: "_sat_frac"}),
                    on="log_id", how="left")
    return a


def compute_tripped(df):
    return (
        (df["failsafe_frac"] > 0)
        | (df["nav_state_forbidden_frac"] > 0)
        | (df["filter_fault_flags_frac"] > 0)
        | (df["landed_at_end"] == False)  # noqa: E712
        | (df["_sat_frac"] > 0.05)
    )


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def main():
    c1 = pd.read_csv(C1_CSV)
    analysis = load_analysis()
    analysis["_tripped"] = compute_tripped(analysis).values
    merged = c1.merge(analysis[["log_id", "_tripped"]], on="log_id", how="left")
    merged["_tripped"] = merged["_tripped"].fillna(True)

    clean = merged[~merged["_tripped"]].copy()
    n_clean = len(clean)
    log(f"clean flights: {n_clean}")

    # Union rate with all 10 ratios
    n_over_all = int(clean["any_over_5pct"].sum())
    lo_all, hi_all = wilson(n_over_all, n_clean)

    # Union rate with mag_field[0] and heading removed
    keep_ratios = [r for r in RATIOS_ALL if r not in DROP]
    log()
    log(f"keeping {len(keep_ratios)} ratios: {keep_ratios}")

    any_no_mag_head = np.zeros(n_clean, dtype=bool)
    for r in keep_ratios:
        col = f"rate_{r}"
        if col not in clean.columns:
            log(f"  WARN: {col} missing")
            continue
        any_no_mag_head |= (clean[col].fillna(0.0).values > 0.05)
    n_over_removed = int(any_no_mag_head.sum())
    lo_rem, hi_rem = wilson(n_over_removed, n_clean)

    log()
    log("C1 union over-flag rate, clean flights:")
    log(f"  all 10 ratios:                    {n_over_all}/{n_clean} = {n_over_all/n_clean:.4f}  [{lo_all:.4f}, {hi_all:.4f}]")
    log(f"  mag_field[0] and heading removed: {n_over_removed}/{n_clean} = {n_over_removed/n_clean:.4f}  [{lo_rem:.4f}, {hi_rem:.4f}]")

    # Contribution analysis
    log()
    log("individual ratio over-flag rate on clean flights:")
    for r in RATIOS_ALL:
        col = f"rate_{r}"
        if col not in clean.columns:
            continue
        k = int((clean[col].fillna(0.0).values > 0.05).sum())
        log(f"  {r:>16s}  {k:>4d}/{n_clean}  = {k/n_clean:.4f}")

    # Save
    pd.DataFrame([
        {"subset": "clean_all_10", "n": n_clean, "n_over": n_over_all,
         "rate": n_over_all / n_clean, "lo": lo_all, "hi": hi_all},
        {"subset": "clean_no_mag_heading", "n": n_clean, "n_over": n_over_removed,
         "rate": n_over_removed / n_clean, "lo": lo_rem, "hi": hi_rem},
    ]).to_csv("c1_clean_no_mag_heading.csv", index=False)

    # Also compute clean vs dirty CI overlap check
    dirty = merged[merged["_tripped"]]
    n_dirty = len(dirty)
    n_over_dirty = int(dirty["any_over_5pct"].sum())
    lo_d, hi_d = wilson(n_over_dirty, n_dirty)

    log()
    log("clean vs dirty CI overlap:")
    log(f"  clean:  {n_over_all}/{n_clean} = {n_over_all/n_clean:.4f}  [{lo_all:.4f}, {hi_all:.4f}]")
    log(f"  dirty:  {n_over_dirty}/{n_dirty} = {n_over_dirty/n_dirty:.4f}  [{lo_d:.4f}, {hi_d:.4f}]")
    log(f"  intervals overlap: {not (hi_all < lo_d or hi_d < lo_all)}")

    with open("c1_clean_no_mag_heading_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))

    log()
    log("saved c1_clean_no_mag_heading.csv and .txt")


if __name__ == "__main__":
    main()
