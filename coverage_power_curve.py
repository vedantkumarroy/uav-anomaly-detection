import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j] if not np.isnan(medians[j]) else 0.0

    A_mask = df["group"].values == "A"
    B_mask = df["group"].values == "B"
    C_mask = df["group"].values == "C"

    X_A = X[A_mask]
    X_B = X[B_mask]
    X_C = X[C_mask]
    labels_C = df["n_indicators"].values[C_mask]

    # Train once
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

    def re(X):
        return np.mean((ae.predict(X) - X) ** 2, axis=1)

    ae_cal = re(Xb)
    ae_test = re(Xc)

    # Track A (conformal max-ratio)
    A_scores = df[A_mask][ratio_cols].values
    B_scores = df[B_mask][ratio_cols].values
    C_scores = df[C_mask][ratio_cols].values
    med = np.nanmedian(A_scores, axis=0)
    iqr = np.nanpercentile(A_scores, 75, axis=0) - np.nanpercentile(A_scores, 25, axis=0)
    iqr = np.where(iqr == 0, 1e-9, iqr)
    ta_cal = np.nanmax((B_scores - med) / iqr, axis=1)
    ta_test = np.nanmax((C_scores - med) / iqr, axis=1)

    # Sweep alpha
    alphas = np.linspace(0.01, 0.30, 30)
    results = []
    for a in alphas:
        for name, cal, test in [
            ("TrackA", ta_cal, ta_test),
            ("IsoForest", iso_cal, iso_test),
            ("OneClassSVM", svm_cal, svm_test),
            ("Autoencoder", ae_cal, ae_test),
        ]:
            thr = conformal_threshold(cal, a)
            flags = test > thr
            coverage = 1 - flags.mean()
            # detection power at n>=2
            hi = labels_C >= 2
            if hi.sum() > 0:
                power = flags[hi].mean()
            else:
                power = 0.0
            results.append({"alpha": a, "method": name,
                            "coverage": coverage, "power_n2": power})

    rdf = pd.DataFrame(results)
    rdf.to_csv("coverage_power_curve.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    for name in ["TrackA", "IsoForest", "OneClassSVM", "Autoencoder"]:
        sub = rdf[rdf["method"] == name]
        plt.plot(sub["coverage"], sub["power_n2"], marker="o", markersize=3, label=name)
    plt.xlabel("Achieved coverage")
    plt.ylabel("Detection power (n_indicators >= 2)")
    plt.title("Coverage vs detection power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_power_curve.png", dpi=120)
    plt.show()
    print("saved coverage_power_curve.csv and coverage_power_curve.png")

    # Print summary
    print()
    print("=== Detection power at coverage = 0.95 ===")
    closest = rdf[rdf["coverage"].between(0.94, 0.96)]
    print(closest.pivot_table(index="method", values="power_n2", aggfunc="mean"))


if __name__ == "__main__":
    main()