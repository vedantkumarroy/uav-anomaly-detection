import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "analysis_set_with_scores.csv"
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


def prepare_features(df):
    cols = [f"score_{r}" for r in RATIOS]
    X = df[cols].values.astype(np.float64)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j]
    return X


def compute_indicator_count(df):
    """Count of five indicators tripped per log."""
    counts = np.zeros(len(df), dtype=int)
    counts += (df["failsafe_frac"] > 0).astype(int).values
    counts += (df["nav_state_forbidden_frac"] > 0).astype(int).values
    counts += (df["filter_fault_flags_frac"] > 0).astype(int).values
    counts += (df["landed_at_end"] == False).astype(int).values
    counts += (df["sat_frac"] > 0.05).astype(int).values  # threshold for saturation
    return counts


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"loaded: {len(df)} logs")

    df["n_indicators"] = compute_indicator_count(df)
    print()
    print("=== n_indicators distribution (analysis set) ===")
    print(df["n_indicators"].value_counts().sort_index())

    X = prepare_features(df)

    A_mask = df["group"].values == "A"
    B_mask = df["group"].values == "B"
    C_mask = df["group"].values == "C"

    X_A = X[A_mask]
    X_B = X[B_mask]
    X_C = X[C_mask]

    df_C = df[C_mask].copy()
    df_C["n_indicators"] = df["n_indicators"].values[C_mask]

    # Track A: conformal on ratio scores directly (the "max z-score across ratios" proxy)
    # We'll use the raw composite score = max over ratios of the score
    score_cols = [f"score_{r}" for r in RATIOS]
    # normalize each ratio score by its calibration median and IQR
    A_scores = df[A_mask][score_cols].values
    B_scores = df[B_mask][score_cols].values
    C_scores = df[C_mask][score_cols].values

    medians = np.nanmedian(A_scores, axis=0)
    iqrs = np.nanpercentile(A_scores, 75, axis=0) - np.nanpercentile(A_scores, 25, axis=0)
    iqrs = np.where(iqrs == 0, 1e-9, iqrs)

    def normalize(X):
        Xn = (X - medians) / iqrs
        return np.nanmax(Xn, axis=1)  # max across ratios

    cal_scores = normalize(B_scores)
    test_scores = normalize(C_scores)

    track_a_threshold = conformal_threshold(cal_scores, ALPHA)
    track_a_flags = test_scores > track_a_threshold
    df_C["flag_track_a"] = track_a_flags

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, random_state=0)
    iso.fit(X_A)
    iso_cal = -iso.score_samples(X_B)
    iso_test = -iso.score_samples(X_C)
    iso_thr = conformal_threshold(iso_cal, ALPHA)
    df_C["flag_iso"] = iso_test > iso_thr

    # One-Class SVM
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(X_A)
    svm_cal = -svm.decision_function(X_B)
    svm_test = -svm.decision_function(X_C)
    svm_thr = conformal_threshold(svm_cal, ALPHA)
    df_C["flag_svm"] = svm_test > svm_thr

    # Autoencoder
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
    df_C["flag_ae"] = ae_test > ae_thr

    # cross-tab coverage of anomalies
    print()
    print("=== Flag counts by n_indicators (Group C) ===")
    print()
    print(f"{'n_indicators':>14s}  {'total':>6s}  {'track_a':>8s}  {'iso':>6s}  {'svm':>6s}  {'ae':>6s}")
    for n in sorted(df_C["n_indicators"].unique()):
        sub = df_C[df_C["n_indicators"] == n]
        print(f"{n:>14d}  {len(sub):>6d}  {int(sub['flag_track_a'].sum()):>8d}  "
              f"{int(sub['flag_iso'].sum()):>6d}  {int(sub['flag_svm'].sum()):>6d}  "
              f"{int(sub['flag_ae'].sum()):>6d}")

    print()
    print("=== Detection power on high-anomaly logs ===")
    print("(logs with n_indicators >= 2)")
    hi = df_C[df_C["n_indicators"] >= 2]
    print(f"n high: {len(hi)}")
    for col in ["flag_track_a", "flag_iso", "flag_svm", "flag_ae"]:
        print(f"  {col}: {int(hi[col].sum())} of {len(hi)} detected "
              f"({100*hi[col].mean():.1f}%)")

    df_C.to_csv("detection_power_c.csv", index=False)
    print("\nsaved detection_power_c.csv")


if __name__ == "__main__":
    main()