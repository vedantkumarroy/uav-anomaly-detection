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
    # impute NaN with column median
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j]
    return X


def isolation_forest_detector(X_train, X_cal, X_test, seed=0):
    model = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    model.fit(X_train)
    # score_samples returns negative values; we want higher = more anomalous
    return -model.score_samples(X_cal), -model.score_samples(X_test)


def oneclass_svm_detector(X_train, X_cal, X_test, seed=0):
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    model.fit(X_train)
    # decision_function is signed; higher = more normal. Flip it.
    return -model.decision_function(X_cal), -model.decision_function(X_test)


def autoencoder_detector(X_train, X_cal, X_test, seed=0):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xc = scaler.transform(X_cal)
    Xe = scaler.transform(X_test)

    # small bottleneck: 10 -> 5 -> 10
    model = MLPRegressor(
        hidden_layer_sizes=(5,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=2000,
        random_state=seed,
    )
    model.fit(Xt, Xt)

    def recon_error(X):
        pred = model.predict(X)
        return np.mean((pred - X) ** 2, axis=1)

    return recon_error(Xc), recon_error(Xe)


def evaluate(detector_name, cal_scores, test_scores):
    threshold = conformal_threshold(cal_scores, ALPHA)
    coverage = float(np.mean(test_scores <= threshold))
    return {
        "detector": detector_name,
        "n_cal": len(cal_scores),
        "n_test": len(test_scores),
        "threshold": threshold,
        "coverage": coverage,
    }


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"loaded: {len(df)} logs")

    # assign A/B/C if not already
    if "group" not in df.columns:
        print("ERROR: need group column")
        return

    X = prepare_features(df)
    print(f"feature matrix shape: {X.shape}")

    # split
    A_mask = df["group"].values == "A"
    B_mask = df["group"].values == "B"
    C_mask = df["group"].values == "C"

    X_A = X[A_mask]
    X_B = X[B_mask]
    X_C = X[C_mask]

    print(f"A: {len(X_A)}, B: {len(X_B)}, C: {len(X_C)}")

    results = []

    # Isolation Forest
    print("\nrunning Isolation Forest ...")
    cal, test = isolation_forest_detector(X_A, X_B, X_C)
    results.append(evaluate("IsolationForest", cal, test))

    # One-Class SVM
    print("running One-Class SVM ...")
    cal, test = oneclass_svm_detector(X_A, X_B, X_C)
    results.append(evaluate("OneClassSVM", cal, test))

    # Autoencoder
    print("running Autoencoder (MLPRegressor) ...")
    cal, test = autoencoder_detector(X_A, X_B, X_C)
    results.append(evaluate("Autoencoder", cal, test))

    # Track A (conformal only, per-ratio coverage already computed)
    # Load the pooled track A results for comparison
    try:
        ta = pd.read_csv("coverage_per_ratio_pooled.csv")
        track_a_mean = ta["coverage"].mean()
        results.append({
            "detector": "TrackA_conformal_per_ratio_mean",
            "n_cal": int(ta["n_cal"].mean()),
            "n_test": int(ta["n_test"].mean()),
            "threshold": float("nan"),
            "coverage": float(track_a_mean),
        })
    except Exception:
        pass

    rdf = pd.DataFrame(results)
    print()
    print("=== Coverage summary ===")
    print(rdf.round(4).to_string(index=False))

    rdf.to_csv("track_b_coverage.csv", index=False)
    print("\nsaved track_b_coverage.csv")


if __name__ == "__main__":
    main()