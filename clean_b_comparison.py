"""
Clean-B vs contaminated-B comparison for C2, C3, C4.

Contaminated B: all 650 flights
Clean B: exclude flights that trip any of the 5 indicators
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

ANALYSIS_CSV = "analysis_set_with_satfrac.csv"
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


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def is_tripped(row):
    return (
        (row.get("failsafe_frac", 0) > 0) or
        (row.get("nav_state_forbidden_frac", 0) > 0) or
        (row.get("filter_fault_flags_frac", 0) > 0) or
        (row.get("landed_at_end", None) == False) or
        (row.get("sat_frac", 0) > 0.05)
    )


def c2_comparison():
    df = pd.read_csv(REGIME_CSV)
    print("=== C2: regime coverage, contaminated vs clean B ===")
    print()

    regimes = ["hover", "translation", "descent"]
    results = []

    for cal_label, filter_contam in [("contaminated", False), ("clean", True)]:
        for regime in regimes:
            sub = df[df["regime"] == regime]
            cal = sub[sub["group"] == "B"].copy()
            test = sub[sub["group"] == "C"]

            if filter_contam:
                # filter out tripped calibration segments
                cal = cal[~cal.apply(is_tripped, axis=1)]

            covs = []
            for r in RATIOS:
                col = f"score_{r}"
                c = cal[col].dropna().values
                t = test[col].dropna().values
                if len(c) < 20 or len(t) < 20:
                    continue
                thr = conformal_threshold(c, ALPHA)
                covs.append(float(np.mean(t <= thr)))
            mean_cov = np.mean(covs) if covs else np.nan
            results.append({
                "claim": "C2", "cal_set": cal_label, "regime": regime,
                "n_cal": len(cal), "coverage": mean_cov,
            })
            print(f"  {cal_label:>13s}  {regime:>12s}  n_cal={len(cal):>5d}  "
                  f"coverage={mean_cov:.4f}")
    return pd.DataFrame(results)


def c3_comparison():
    df = pd.read_csv(ANALYSIS_CSV)
    print()
    print("=== C3: cross-band coverage, contaminated vs clean B ===")
    print()

    cal_mask = (df["group"] == "B") & (df["band"].isin(["v1.12-1.13", "v1.14-1.15"]))
    test_mask = (df["group"] == "C") & (df["band"] == "v1.16+")

    cal_all = df[cal_mask].copy()
    test = df[test_mask].copy()

    results = []
    for cal_label, filter_contam in [("contaminated", False), ("clean", True)]:
        cal = cal_all.copy()
        if filter_contam:
            cal = cal[~cal.apply(is_tripped, axis=1)]

        covs = []
        for r in RATIOS:
            col = f"score_log_{r}"
            c = cal[col].dropna().values
            t = test[col].dropna().values
            if len(c) < 10 or len(t) < 10:
                continue
            thr = conformal_threshold(c, ALPHA)
            covs.append(float(np.mean(t <= thr)))
        mean_cov = np.mean(covs) if covs else np.nan
        results.append({
            "claim": "C3", "cal_set": cal_label,
            "n_cal": len(cal), "coverage": mean_cov,
        })
        print(f"  {cal_label:>13s}  n_cal={len(cal):>5d}  coverage={mean_cov:.4f}")
    return pd.DataFrame(results)


def c4_comparison():
    df = pd.read_csv(ANALYSIS_CSV)
    print()
    print("=== C4: detection power, contaminated vs clean B ===")
    print()

    score_cols = [f"score_log_{r}" for r in RATIOS]
    X = df[score_cols].values.astype(np.float64)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = medians[j] if not np.isnan(medians[j]) else 0.0

    A_mask = df["group"].values == "A"
    B_mask = df["group"].values == "B"
    C_mask = df["group"].values == "C"

    X_A = X[A_mask]
    X_B = X[B_mask]
    X_C = X[C_mask]

    iso = IsolationForest(n_estimators=200, random_state=0)
    iso.fit(X_A)
    iso_cal = -iso.score_samples(X_B)
    iso_test = -iso.score_samples(X_C)

    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(X_A)
    svm_cal = -svm.decision_function(X_B)
    svm_test = -svm.decision_function(X_C)

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_A)
    Xb = scaler.transform(X_B)
    Xc = scaler.transform(X_C)
    ae = MLPRegressor(hidden_layer_sizes=(5,), activation="relu", solver="adam",
                      learning_rate_init=0.01, max_iter=2000, random_state=0)
    ae.fit(Xt, Xt)
    ae_cal = np.mean((ae.predict(Xb) - Xb) ** 2, axis=1)
    ae_test = np.mean((ae.predict(Xc) - Xc) ** 2, axis=1)

    B_df = df[B_mask].reset_index(drop=True)
    tripped_b = B_df.apply(is_tripped, axis=1).values
    clean_mask = ~tripped_b

    sat_C = df.loc[C_mask, "sat_frac_recomputed"].values
    n_top = int(0.10 * np.sum(~np.isnan(sat_C)))
    order = np.argsort(-np.nan_to_num(sat_C, nan=-1.0))
    is_top = np.zeros(len(sat_C), dtype=bool)
    is_top[order[:n_top]] = True

    results = []
    print(f"{'detector':>18s}  {'cal_set':>13s}  {'threshold':>10s}  {'k/n_top':>10s}  {'power':>8s}")
    for name, cal, test in [
        ("IsolationForest", iso_cal, iso_test),
        ("OneClassSVM", svm_cal, svm_test),
        ("Autoencoder", ae_cal, ae_test),
    ]:
        for cal_label, mask in [("contaminated", np.ones(len(cal), dtype=bool)),
                                ("clean", clean_mask)]:
            cal_sub = cal[mask]
            thr = conformal_threshold(cal_sub, ALPHA)
            flags = test > thr
            k = int(flags[is_top].sum())
            power = k / n_top if n_top > 0 else np.nan
            results.append({
                "claim": "C4", "detector": name, "cal_set": cal_label,
                "threshold": thr, "k": k, "n_top": n_top, "power": power,
            })
            print(f"{name:>18s}  {cal_label:>13s}  {thr:>10.4f}  "
                  f"{k:>3d}/{n_top:<5d}  {power:>8.4f}")
    return pd.DataFrame(results)


def main():
    c2 = c2_comparison()
    c3 = c3_comparison()
    c4 = c4_comparison()

    all_df = pd.concat([c2, c3, c4], ignore_index=True)
    all_df.to_csv("clean_b_comparison.csv", index=False)
    print()
    print("saved clean_b_comparison.csv")


if __name__ == "__main__":
    main()