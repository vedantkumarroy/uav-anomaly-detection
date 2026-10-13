"""
cell2_clean.py

Sir's correction. Three cells:

  cell 1   dirty B, dirty C    valid
  cell 2   clean B, clean C    valid, the paper experiment
  cell 3   clean B, dirty C    invalid, drop

This script computes cell 1 and cell 2 for C2 and C3, using flight-block
bootstrap CIs.

Group B: all 650 (dirty) or 374 (clean).
Group C: filtered by the same indicator rule.

Outputs:
  c2_cell2.csv
  c3_cell2.csv
  cell2_summary.txt
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
        return np.nan, np.nan, np.nan, 0, 0

    common = None
    for d in ratio_maps:
        s = set(d.keys())
        common = s if common is None else common & s
    common = sorted(common)
    if len(common) < min_test:
        return np.nan, np.nan, np.nan, 0, 0

    M = np.array([[d[i] for d in ratio_maps] for i in common])
    pt, lo, hi = cluster_bootstrap_mean(M)
    return pt, lo, hi, M.shape[0], M.shape[1]


def run_c2(analysis):
    log()
    log("=" * 78)
    log("C2: regime coverage, cell 1 (dirty/dirty) vs cell 2 (clean/clean)")
    log("=" * 78)

    seg = pd.read_csv(REGIME_CSV)
    tripped_ids = set(analysis.loc[compute_tripped(analysis), "log_id"].values)
    seg["tripped"] = seg["log_id"].isin(tripped_ids)

    rows = []
    for regime in ["hover", "translation", "descent"]:
        sub = seg[seg["regime"] == regime]
        b_all = sub[sub["group"] == "B"]
        c_all = sub[sub["group"] == "C"]
        b_clean = b_all[~b_all["tripped"]]
        c_clean = c_all[~c_all["tripped"]]

        p1, l1, h1, nf1, nr1 = coverage_with_ci(b_all, c_all, "score_", 20, 20)
        p2, l2, h2, nf2, nr2 = coverage_with_ci(b_clean, c_clean, "score_", 20, 20)

        log()
        log(f"{regime}")
        log(f"  cell 1 (dirty B, dirty C): {p1:.4f} [{l1:.4f}, {h1:.4f}]  "
            f"n_cal={len(b_all)}  n_test={len(c_all)}  "
            f"test_flights={nf1}  ratios={nr1}")
        log(f"  cell 2 (clean B, clean C): {p2:.4f} [{l2:.4f}, {h2:.4f}]  "
            f"n_cal={len(b_clean)}  n_test={len(c_clean)}  "
            f"test_flights={nf2}  ratios={nr2}")
        log(f"  0.95 inside cell 1 CI: {l1 <= 0.95 <= h1}")
        log(f"  0.95 inside cell 2 CI: {l2 <= 0.95 <= h2}")

        rows.append({
            "regime": regime,
            "cell1_point": p1, "cell1_lo": l1, "cell1_hi": h1,
            "cell1_n_cal": len(b_all), "cell1_n_test": len(c_all),
            "cell2_point": p2, "cell2_lo": l2, "cell2_hi": h2,
            "cell2_n_cal": len(b_clean), "cell2_n_test": len(c_clean),
        })

    pd.DataFrame(rows).to_csv("c2_cell2.csv", index=False)
    log()
    log("saved c2_cell2.csv")


def run_c3(analysis):
    log()
    log("=" * 78)
    log("C3: cross-band coverage, cell 1 vs cell 2")
    log("=" * 78)

    df = analysis.copy()
    trip = compute_tripped(df)

    cal_band = df["band"].isin(["v1.12-1.13", "v1.14-1.15"])
    test_band = df["band"] == "v1.16+"

    b_all = df[(df["group"] == "B") & cal_band]
    c_all = df[(df["group"] == "C") & test_band]
    b_clean = df[(df["group"] == "B") & cal_band & (~trip)]
    c_clean = df[(df["group"] == "C") & test_band & (~trip)]

    log()
    log(f"B cal all:    {len(b_all)}")
    log(f"B cal clean:  {len(b_clean)}")
    log(f"C test all:   {len(c_all)}")
    log(f"C test clean: {len(c_clean)}")

    p1, l1, h1, nf1, nr1 = coverage_with_ci(b_all, c_all, "score_log_", 10, 10)
    p2, l2, h2, nf2, nr2 = coverage_with_ci(b_clean, c_clean, "score_log_", 10, 10)

    log()
    log(f"cell 1 (dirty B, dirty C): {p1:.4f} [{l1:.4f}, {h1:.4f}]  "
        f"test_flights={nf1}  ratios={nr1}  nominal inside: {l1 <= 0.95 <= h1}")
    log(f"cell 2 (clean B, clean C): {p2:.4f} [{l2:.4f}, {h2:.4f}]  "
        f"test_flights={nf2}  ratios={nr2}  nominal inside: {l2 <= 0.95 <= h2}")

    pd.DataFrame([
        {"cell": 1, "point": p1, "lo": l1, "hi": h1,
         "n_cal": len(b_all), "n_test": len(c_all)},
        {"cell": 2, "point": p2, "lo": l2, "hi": h2,
         "n_cal": len(b_clean), "n_test": len(c_clean)},
    ]).to_csv("c3_cell2.csv", index=False)
    log()
    log("saved c3_cell2.csv")


def main():
    analysis = load_analysis()
    log(f"analysis rows: {len(analysis)}")

    all_trip = compute_tripped(analysis)
    log()
    log("tripped totals across A+B+C:")
    log(f"  total:  {len(analysis)}")
    log(f"  tripped: {int(all_trip.sum())}")
    log(f"  clean:   {int((~all_trip).sum())}")

    for g in ["A", "B", "C"]:
        m = analysis["group"] == g
        t = (all_trip & m).sum()
        log(f"  {g}: total={int(m.sum())}  tripped={int(t)}  clean={int((m & ~all_trip).sum())}")

    run_c2(analysis)
    run_c3(analysis)

    with open("cell2_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print()
    print("saved cell2_summary.txt")


if __name__ == "__main__":
    main()
