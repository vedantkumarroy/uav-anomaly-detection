"""
c1_clean.py

Sir's correction. The C1 headline of 18.9 percent was measured on all 1,627
analysis flights. About 40 percent of those trip a runtime indicator, so
they are not "healthy" by the indicator definition.

This recomputes C1 on the clean subset only. Reports both numbers.
"""

import numpy as np
import pandas as pd

C1_CSV = "c1_warmup_excluded.csv"
ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
SATFRAC_CSV = "analysis_set_with_satfrac.csv"

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


def main():
    c1 = pd.read_csv(C1_CSV)
    log(f"c1_warmup_excluded rows: {len(c1)}")

    analysis = load_analysis()
    tripped = compute_tripped(analysis)
    analysis["_tripped"] = tripped.values

    merged = c1.merge(analysis[["log_id", "_tripped"]], on="log_id", how="left")
    log(f"merged rows: {len(merged)}")
    log(f"merged missing tripped: {int(merged['_tripped'].isna().sum())}")

    merged["_tripped"] = merged["_tripped"].fillna(True)

    n_total = len(merged)
    n_clean = int((~merged["_tripped"]).sum())
    n_dirty = int(merged["_tripped"].sum())

    log()
    log("counts:")
    log(f"  total:  {n_total}")
    log(f"  clean:  {n_clean}")
    log(f"  dirty:  {n_dirty}")

    n_over_all = int(merged["any_over_5pct"].sum())
    n_over_clean = int(merged.loc[~merged["_tripped"], "any_over_5pct"].sum())
    n_over_dirty = int(merged.loc[merged["_tripped"], "any_over_5pct"].sum())

    log()
    log("C1 headline (warm-up excluded, 30 s):")
    log(f"  all flights:    {n_over_all}/{n_total} = {n_over_all/n_total:.4f}")
    log(f"  clean only:     {n_over_clean}/{n_clean} = {n_over_clean/n_clean:.4f}")
    log(f"  dirty only:     {n_over_dirty}/{n_dirty} = {n_over_dirty/n_dirty:.4f}")

    # Wilson 95 percent CI
    def wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (centre - half, centre + half)

    a_lo, a_hi = wilson(n_over_all, n_total)
    c_lo, c_hi = wilson(n_over_clean, n_clean)
    d_lo, d_hi = wilson(n_over_dirty, n_dirty)

    log()
    log("95% Wilson CI:")
    log(f"  all flights:    [{a_lo:.4f}, {a_hi:.4f}]")
    log(f"  clean only:     [{c_lo:.4f}, {c_hi:.4f}]")
    log(f"  dirty only:     [{d_lo:.4f}, {d_hi:.4f}]")

    # Per-ratio breakdown on clean flights
    log()
    log("per-ratio over-flag rate (rate > 5% of samples), clean flights only:")
    rate_cols = [c for c in merged.columns if c.startswith("rate_")]
    for col in sorted(rate_cols):
        ratio = col[len("rate_"):]
        vals = merged.loc[~merged["_tripped"], col].dropna().values
        if len(vals) == 0:
            continue
        frac_over = float((vals > 0.05).mean())
        log(f"  {ratio:>16s}  mean_rate={vals.mean():.4f}  over_5pct_frac={frac_over:.4f}")

    pd.DataFrame([
        {"subset": "all", "n": n_total, "n_over": n_over_all,
         "rate": n_over_all / n_total, "lo": a_lo, "hi": a_hi},
        {"subset": "clean", "n": n_clean, "n_over": n_over_clean,
         "rate": n_over_clean / n_clean, "lo": c_lo, "hi": c_hi},
        {"subset": "dirty", "n": n_dirty, "n_over": n_over_dirty,
         "rate": n_over_dirty / n_dirty, "lo": d_lo, "hi": d_hi},
    ]).to_csv("c1_clean.csv", index=False)

    with open("c1_clean_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))

    log()
    log("saved c1_clean.csv and c1_clean_summary.txt")


if __name__ == "__main__":
    main()
