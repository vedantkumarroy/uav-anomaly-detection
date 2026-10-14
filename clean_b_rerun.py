"""
Rerun C1, C2, C3, C4 with clean-B calibration.

Clean-B = Group B with indicator-tripping flights removed.
Produces comparison table: contaminated vs clean, every claim.
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

ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
INDEX_CSV = "combined_index.csv"
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
PX4_THRESHOLD = 1.0
TOP_FRAC = 0.10


def logdir_for(log_id):
    for base in ["pilot_2270", "pilot_extra"]:
        p = os.path.join(base, "logs", log_id)
        if os.path.isdir(p):
            return p
    return None


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    idx = pd.read_csv(INDEX_CSV)

    # merge indicators if not present
    if "failsafe_frac" not in df.columns:
        extra = idx[["log_id", "failsafe_frac", "nav_state_forbidden_frac",
                     "filter_fault_flags_frac", "landed_at_end", "sat_frac"]].copy()
        df = df.merge(extra, on="log_id", how="left")

    tripped = (
        (df["failsafe_frac"] > 0) |
        (df["nav_state_forbidden_frac"] > 0) |
        (df["filter_fault_flags_frac"] > 0) |
        (df["landed_at_end"] == False) |
        (df["sat_frac"] > 0.05)
    )
    df["tripped"] = tripped

    B_mask = df["group"].values == "B"
    C_mask = df["group"].values == "C"

    B_clean = B_mask & ~tripped.values
    B_contam = B_mask

    print(f"Group B: {B_mask.sum()}, of which tripped: {(B_mask & tripped.values).sum()}")
    print(f"Clean B: {B_clean.sum()}")
    print(f"Group C: {C_mask.sum()}")
    print()

    # ===================================================================
    # C1: false alarm rate under clean-B and contaminated-B
    # ===================================================================
    print("=" * 78)
    print("C1: false alarm rate (contaminated-B vs clean-B)")
    print("=" * 78)
    print()
    print("C1 does not use B for calibration. It is a per-flight threshold")
    print("measurement on Group C. Result is identical under either B filter.")
    print()
    print("C1 headline (all samples, threshold 1.0): 21.1% of flights over-flagged")
    print("C1 headline (30s warm-up excluded):       18.9% of flights over-flagged")
    print()

    # ===================================================================
    # C2: regime coverage with clean-B calibration
    # ===================================================================
    print("=" * 78)
    print("C2: regime coverage (contaminated-B vs clean-B)")
    print("=" * 78)
    print()

    reg = pd.read_csv(REGIME_CSV)

    # merge tripped flag by log_id
    reg = reg.merge(df[["log_id", "tripped"]], on="log_id", how="left", suffixes=("", "_y"))
    if "tripped_y" in reg.columns:
        reg["tripped"] = reg["tripped_y"].fillna(reg["tripped"])
        reg = reg.drop(columns=["tripped_y"])

    print(f"{'regime':>14s}  {'setting':>12s}  {'coverage':>10s}")
    for regime in ["hover", "translation", "descent"]:
        sub = reg[reg["regime"] == regime]
        cal_contam = sub[sub["group"] == "B"]
        cal_clean = sub[(sub["group"] == "B") & (~sub["tripped"])]
        test = sub[sub["group"] == "C"]

        for setting, cal in [("contaminated", cal_contam), ("clean", cal_clean)]:
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
            print(f"{regime:>14s}  {setting:>12s}  {mean_cov:>10.4f}")
    print()

    # ===================================================================
    # C3: cross-band coverage with clean-B calibration
    # ===================================================================
    print("=" * 78)
    print("C3: cross-band coverage (contaminated-B vs clean-B)")
    print("=" * 78)
    print()

    cal_contam = df[(df["group"] == "B") & (df["band"].isin(["v1.12-1.13", "v1.14-1.15"]))]
    cal_clean = cal_contam[~cal_contam["tripped"]]
    test = df[(df["group"] == "C") & (df["band"] == "v1.16+")]

    print(f"cal contaminated: {len(cal_contam)}")
    print(f"cal clean:        {len(cal_clean)}")
    print(f"test:             {len(test)}")
    print()

    for setting, cal in [("contaminated", cal_contam), ("clean", cal_clean)]:
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
        print(f"  {setting:>12s}: mean cross-band coverage = {mean_cov:.4f}")
    print()

    # ===================================================================
    # C4: detection power with clean-B calibration
    # ===================================================================
    print("=" * 78)
    print("C4: detection power (contaminated-B vs clean-B)")
    print("=" * 78)
    print()

    sat_col = "sat_frac"
    if sat_col not in df.columns:
        sat_col = "sat_frac_recomputed"

    score_cols = [f"score_log_{r}" for r in RATIOS]
    X = df[score_cols].values.astype(np.float64)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = medians[j] if not np.isnan(medians[j]) else 0.0

    A_mask = df["group"].values == "A"
    X_A = X[A_mask]
    X_C = X[C_mask]

    print("training detectors (unchanged, always on A) ...")
    iso = IsolationForest(n_estimators=200, random_state=0)
    iso.fit(X_A)
    iso_test = -iso.score_samples(X_C)

    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(X_A)
    svm_test = -svm.decision_function(X_C)

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_A)
    Xc = scaler.transform(X_C)
    ae = MLPRegressor(hidden_layer_sizes=(5,), activation="relu", solver="adam",
                      learning_rate_init=0.01, max_iter=2000, random_state=0)
    ae.fit(Xt, Xt)
    ae_test = np.mean((ae.predict(Xc) - Xc) ** 2, axis=1)

    # sat_frac in C
    sat_C = df.loc[C_mask, sat_col].values
    n_C = len(sat_C)
    n_top = int(TOP_FRAC * np.sum(~np.isnan(sat_C)))
    order = np.argsort(-np.nan_to_num(sat_C, nan=-1.0))
    is_top = np.zeros(n_C, dtype=bool)
    is_top[order[:n_top]] = True

    for setting, cal_mask in [("contaminated", B_mask), ("clean", B_clean)]:
        X_B = X[cal_mask]
        # refit scaling for AE on the respective calibration set
        scaler_s = StandardScaler()
        scaler_s.fit(X_A)
        Xb_s = scaler_s.transform(X_B)
        ae_cal_s = np.mean((ae.predict(Xb_s) - Xb_s) ** 2, axis=1)

        # refit iso / svm on A, use B for conformal threshold
        iso_cal_s = -iso.score_samples(X_B)
        svm_cal_s = -svm.decision_function(X_B)

        results = []
        for name, cal_scores, test_scores in [
            ("IsoForest", iso_cal_s, iso_test),
            ("SVM", svm_cal_s, svm_test),
            ("Autoencoder", ae_cal_s, ae_test),
        ]:
            thr = conformal_threshold(cal_scores, ALPHA)
            flags = test_scores > thr
            k = int(flags[is_top].sum())
            rate = k / n_top if n_top > 0 else np.nan
            results.append({"method": name, "k": k, "n": n_top, "rate": rate})

        print(f"--- {setting} calibration ---")
        for r in results:
            print(f"  {r['method']:>14s}: {r['k']:>3d}/{r['n']:<4d}  ({r['rate']*100:>5.1f}%)")
        print()


if __name__ == "__main__":
    main()