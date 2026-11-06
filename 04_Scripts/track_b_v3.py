import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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


def conformal_threshold(cal_scores, alpha):
    s = np.sort(cal_scores)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def main():
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

    ratio_cols = [f"score_log_{r}" for r in RATIOS]
    X = df[ratio_cols].values.astype(np.float64)

    # impute NaN
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j]

    print(f"feature matrix shape: {X.shape}")
    print(f"range: min={X.min():.4f} max={X.max():.4f} median={np.median(X):.4f}")

    A_mask = df["group"].values == "A"
    B_mask = df["group"].values == "B"
    C_mask = df["group"].values == "C"

    X_A = X[A_mask]
    X_B = X[B_mask]
    X_C = X[C_mask]

    labels_C = df["n_indicators"].values[C_mask]

    # IsoForest
    iso = IsolationForest(n_estimators=200, random_state=0)
    iso.fit(X_A)
    iso_cal = -iso.score_samples(X_B)
    iso_test = -iso.score_samples(X_C)
    iso_thr = conformal_threshold(iso_cal, ALPHA)
    iso_flags = iso_test > iso_thr

    # SVM
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(X_A)
    svm_cal = -svm.decision_function(X_B)
    svm_test = -svm.decision_function(X_C)
    svm_thr = conformal_threshold(svm_cal, ALPHA)
    svm_flags = svm_test > svm_thr

    # AE
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_A)
    Xb = scaler.transform(X_B)
    Xc = scaler.transform(X_C)
    ae = MLPRegressor(hidden_layer_sizes=(5,), activation="relu", solver="adam",
                      learning_rate_init=0.01, max_iter=2000, random_state=0)
    ae.fit(Xt, Xt)

    def re(X):
        return np.mean((ae.predict(X) - X) ** 2, axis=1)

    ae_cal = re(Xb)
    ae_test = re(Xc)
    ae_thr = conformal_threshold(ae_cal, ALPHA)
    ae_flags = ae_test > ae_thr

    print()
    print("=== Coverage ===")
    print(f"IsoForest:   {1 - iso_flags.mean():.4f}")
    print(f"SVM:         {1 - svm_flags.mean():.4f}")
    print(f"Autoencoder: {1 - ae_flags.mean():.4f}")

    print()
    print("=== Detection power ===")
    print(f"{'thr':>4s}  {'n':>5s}  {'iso':>10s}  {'svm':>10s}  {'ae':>10s}")
    for thr in [1, 2, 3, 4]:
        n = int((labels_C >= thr).sum())
        if n == 0:
            continue
        k_iso = int(iso_flags[labels_C >= thr].sum())
        k_svm = int(svm_flags[labels_C >= thr].sum())
        k_ae = int(ae_flags[labels_C >= thr].sum())
        print(f"{thr:>4d}  {n:>5d}  {k_iso:>3d}/{n:<6d}  {k_svm:>3d}/{n:<6d}  {k_ae:>3d}/{n:<6d}")

    out = pd.DataFrame({
        "method": ["IsolationForest", "OneClassSVM", "Autoencoder"],
        "coverage": [1 - iso_flags.mean(), 1 - svm_flags.mean(), 1 - ae_flags.mean()],
    })
    out.to_csv("track_b_v3_coverage.csv", index=False)


if __name__ == "__main__":
    main()