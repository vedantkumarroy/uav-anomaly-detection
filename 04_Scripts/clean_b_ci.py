"""
clean_b_ci.py

Two jobs:
  1. Diagnose the clean-B count (374 vs 379) by printing the indicator
     breakdown on Group B.
  2. Recompute C2 and C3 cluster-robust bootstrap CIs on BOTH contaminated
     B (all) and clean B (indicator-tripping removed).

Outputs:
  c2_clean_ci.csv
  c3_clean_ci.csv
  clean_b_ci_summary.txt
"""

import numpy as np
import pandas as pd

ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
SATFRAC_CSV = "analysis_set_with_satfrac.csv"
REGIME_CSV = "regime_segments.csv"

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
N_BOOT = 2000
RNG_SEED = 0

_lines = []
def log(s=""):
    print(s)
    _lines.append(s)


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def load_analysis():
    a = pd.read_csv(ANALYSIS_CSV)
    log(f"analysis_set_with_log_scores: shape={a.shape}, unique log_id={a['log_id'].nunique()}")
    if "sat_frac" in a.columns:
        a["_sat_frac"] = a["sat_frac"]
    elif "sat_frac_recomputed" in a.columns:
        a["_sat_frac"] = a["sat_frac_recomputed"]
    else:
        s = pd.read_csv(SATFRAC_CSV)
        log(f"satfrac file: shape={s.shape}, columns={list(s.columns)}")
        if "sat_frac_recomputed" in s.columns:
            sat_col = "sat_frac_recomputed"
        elif "sat_frac" in s.columns:
            sat_col = "sat_frac"
        else:
            raise SystemExit("No sat_frac column found anywhere")
        s2 = s[["log_id", sat_col]].rename(columns={sat_col: "_sat_frac"})
        a = a.merge(s2, on="log_id", how="left")
    return a


def compute_tripped(df):
    return (
        (df["failsafe_frac"] > 0)
        | (df["nav_state_forbidden_frac"] > 0)
        | (df["filter_fault_flags_frac"] > 0)
        | (df["landed_at_end"] == False)  # noqa: E712
        | (df["_sat_frac"] > 0.05)
    )


def cluster_bootstrap_mean(M, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan
    point = float(M.mean())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = M[idx].mean()
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def coverage_with_ci(cal_df, test_df, prefix, min_cal, min_test):
    ratio_maps = []
    for r in RATIOS:
        col = f"{prefix}{r}"
        if col not in cal_df.columns or col not in test_df.columns:
            continue
        c = cal_df[col].dropna().values
        if len(c) < min_cal:
            continue
        thr = conformal_threshold(c, ALPHA)
        if np.isnan(thr):
            continue
        per_flight = {}
        for log_id, g in test_df.groupby("log_id"):
            arr = g[col].dropna().values
            if len(arr) > 0:
                per_flight[log_id] = float(np.mean(arr <= thr))
        if len(per_flight) < min_test:
            continue
        ratio_maps.append(per_flight)

    if not ratio_maps:
        return np.nan, np.nan, np.nan

    common = None
    for d in ratio_maps:
        s = set(d.keys())
        common = s if common is None else common & s
    common = sorted(common)
    if len(common) < min_test:
        return np.nan, np.nan, np.nan

    M = np.array([[d[i] for d in ratio_maps] for i in common])
    return cluster_bootstrap_mean(M)


def c2_clean_ci(analysis):
    log()
    log("=" * 78)
    log("C2: regime coverage, contaminated vs clean B")
    log("=" * 78)

    seg = pd.read_csv(REGIME_CSV)
    tripped_ids = set(analysis.loc[compute_tripped(analysis), "log_id"].values)
    seg["tripped"] = seg["log_id"].isin(tripped_ids)

    rows = []
    for regime in ["hover", "translation", "descent"]:
        sub = seg[seg["regime"] == regime]
        cal_all = sub[sub["group"] == "B"]
        cal_clean = cal_all[~cal_all["tripped"]]
        test = sub[sub["group"] == "C"]

        cp, clo, chi = coverage_with_ci(cal_all, test, "score_", 20, 20)
        np_, nlo, nhi = coverage_with_ci(cal_clean, test, "score_", 20, 20)

        log()
        log(f"{regime}")
        log(f"   contaminated: {cp:.4f} [{clo:.4f}, {chi:.4f}]  n_cal={len(cal_all)}")
        log(f"   clean:        {np_:.4f} [{nlo:.4f}, {nhi:.4f}]  n_cal={len(cal_clean)}")
        log(f"   nominal 0.95 inside contaminated CI: {clo <= 0.95 <= chi}")
        log(f"   nominal 0.95 inside clean CI:        {nlo <= 0.95 <= nhi}")

        rows.append({
            "regime": regime,
            "contaminated_point": cp,
            "contaminated_lo": clo,
            "contaminated_hi": chi,
            "clean_point": np_,
            "clean_lo": nlo,
            "clean_hi": nhi,
            "n_cal_contaminated": len(cal_all),
            "n_cal_clean": len(cal_clean),
        })

    pd.DataFrame(rows).to_csv("c2_clean_ci.csv", index=False)
    log()
    log("saved c2_clean_ci.csv")


def c3_clean_ci(analysis):
    log()
    log("=" * 78)
    log("C3: cross-band coverage, contaminated vs clean B")
    log("=" * 78)

    df = analysis.copy()
    trip_series = compute_tripped(df)

    cal_mask = (df["group"] == "B") & (df["band"].isin(["v1.12-1.13", "v1.14-1.15"]))
    test_mask = (df["group"] == "C") & (df["band"] == "v1.16+")

    cal_all = df[cal_mask].copy()
    cal_clean = df[cal_mask & (~trip_series)].copy()
    test = df[test_mask].copy()

    log()
    log(f"cal contaminated: {len(cal_all)}")
    log(f"cal clean:        {len(cal_clean)}")
    log(f"test:             {len(test)}")

    cp, clo, chi = coverage_with_ci(cal_all, test, "score_log_", 10, 10)
    np_, nlo, nhi = coverage_with_ci(cal_clean, test, "score_log_", 10, 10)

    log()
    log(f"cross-band contaminated: {cp:.4f} [{clo:.4f}, {chi:.4f}]  nominal inside: {clo <= 0.95 <= chi}")
    log(f"cross-band clean:        {np_:.4f} [{nlo:.4f}, {nhi:.4f}]  nominal inside: {nlo <= 0.95 <= nhi}")

    pd.DataFrame([
        {"setting": "contaminated", "point": cp, "lo": clo, "hi": chi, "n_cal": len(cal_all)},
        {"setting": "clean", "point": np_, "lo": nlo, "hi": nhi, "n_cal": len(cal_clean)},
    ]).to_csv("c3_clean_ci.csv", index=False)
    log()
    log("saved c3_clean_ci.csv")


def main():
    analysis = load_analysis()

    log()
    log("=" * 78)
    log("Diagnostic: clean-B count")
    log("=" * 78)
    B = analysis[analysis["group"] == "B"].copy()
    trip = compute_tripped(B)
    log()
    log(f"B total: {len(B)}")
    log(f"tripped: {int(trip.sum())}")
    log(f"clean:   {int((~trip).sum())}")
    log()
    log("indicator breakdown (count of B flights where indicator trips):")
    log(f"  failsafe_frac > 0:               {int((B['failsafe_frac'] > 0).sum())}")
    log(f"  nav_state_forbidden_frac > 0:    {int((B['nav_state_forbidden_frac'] > 0).sum())}")
    log(f"  filter_fault_flags_frac > 0:     {int((B['filter_fault_flags_frac'] > 0).sum())}")
    log(f"  landed_at_end == False:          {int((B['landed_at_end'] == False).sum())}")
    log(f"  _sat_frac > 0.05:                {int((B['_sat_frac'] > 0.05).sum())}")
    log(f"  _sat_frac is NaN:                {int(B['_sat_frac'].isna().sum())}")

    c2_clean_ci(analysis)
    c3_clean_ci(analysis)

    with open("clean_b_ci_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print()
    print("saved clean_b_ci_summary.txt")


if __name__ == "__main__":
    main()
