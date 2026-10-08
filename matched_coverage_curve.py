"""
Matched-coverage curve.

Sweep alpha from 0.01 to 0.30. For each alpha, compute the
conformal threshold on Group B and measure:
  - achieved false alarm rate on Group C
  - detection power on the top-10% sat_frac flights in Group C
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ANALYSIS_CSV = "analysis_set_with_satfrac.csv"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

ALPHAS = np.linspace(0.01, 0.30, 30)
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


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"loaded: {len(df)}")
    print(f"columns include sat_frac_recomputed: {'sat_frac_recomputed' in df.columns}")

    sat_col = "sat_frac_recomputed"
    print(f"sat_frac non-null: {df[sat_col].notna().sum()} of {len(df)}")

    # features
    score_cols = [f"score_log_{r}" for r in RATIOS]
    X = df[score_cols].values.astype(np.float64)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = medians[j] if not np.isnan(medians[j]) else 0.0

    A_mask = (df["group"].values == "A")
    B_mask = (df["group"].values == "B")
    C_mask = (df["group"].values == "C")

    print(f"A: {A_mask.sum()}, B: {B_mask.sum()}, C: {C_mask.sum()}")

    X_A = X[A_mask]
    X_B = X[B_mask]
    X_C = X[C_mask]

    print("training detectors ...")
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

    # PX4 scores
    print("computing PX4 scores ...")
    px4_cal = []
    px4_test = []
    for group_mask, target in [(B_mask, px4_cal), (C_mask, px4_test)]:
        for i in np.where(group_mask)[0]:
            row = df.iloc[i]
            logdir = logdir_for(row["log_id"])
            if logdir is None:
                target.append(np.nan)
                continue
            p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
            if not os.path.exists(p):
                target.append(np.nan)
                continue
            t = pq.read_table(p)
            any_flag = np.zeros(len(t), dtype=bool)
            for r in RATIOS:
                if r not in t.column_names:
                    continue
                arr = t[r].to_numpy(zero_copy_only=False).astype(np.float64)
                finite = np.isfinite(arr)
                flags = np.zeros(len(arr), dtype=bool)
                flags[finite] = arr[finite] > 1.0
                any_flag |= flags
            target.append(float(any_flag.mean()))

    px4_cal = np.array(px4_cal)
    px4_test = np.array(px4_test)

    # top-10% sat_frac in C
    sat_C = df.loc[C_mask, sat_col].values
    n_C = len(sat_C)
    n_valid = int(np.sum(~np.isnan(sat_C)))
    n_top = int(TOP_FRAC * n_valid)
    order = np.argsort(-np.nan_to_num(sat_C, nan=-1.0))
    is_top = np.zeros(n_C, dtype=bool)
    is_top[order[:n_top]] = True
    print(f"C flights: {n_C}, valid sat_frac: {n_valid}, top-{int(TOP_FRAC*100)}%: {n_top}")

    # sweep
    detectors = [
        ("IsolationForest", iso_cal, iso_test),
        ("OneClassSVM", svm_cal, svm_test),
        ("Autoencoder", ae_cal, ae_test),
        ("PX4_threshold", px4_cal, px4_test),
    ]

    rows = []
    for name, cal, test in detectors:
        cal_clean = cal[~np.isnan(cal)]
        valid = ~np.isnan(test)
        for alpha in ALPHAS:
            thr = conformal_threshold(cal_clean, alpha)
            flags = np.zeros(n_C, dtype=bool)
            flags[valid] = test[valid] > thr
            fa_rate = float(flags.mean())
            k = int(flags[is_top].sum())
            power = k / n_top if n_top > 0 else np.nan
            rows.append({
                "detector": name,
                "alpha": alpha,
                "fa_rate": fa_rate,
                "detection_power": power,
                "k": k,
                "n_top": n_top,
            })

    rdf = pd.DataFrame(rows)
    rdf.to_csv("matched_coverage_curve.csv", index=False)
    print()
    print("saved matched_coverage_curve.csv")

    # figure
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "IsolationForest": "steelblue",
        "OneClassSVM": "seagreen",
        "Autoencoder": "indianred",
        "PX4_threshold": "black",
    }
    markers = {
        "IsolationForest": "o",
        "OneClassSVM": "s",
        "Autoencoder": "^",
        "PX4_threshold": "x",
    }
    for name in ["IsolationForest", "OneClassSVM", "Autoencoder", "PX4_threshold"]:
        sub = rdf[rdf["detector"] == name].sort_values("fa_rate")
        ax.plot(sub["fa_rate"] * 100, sub["detection_power"] * 100,
                marker=markers[name], markersize=5, linewidth=1.5,
                color=colors[name], label=name)

    ax.axvline(5.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Achieved false alarm rate (%)", fontsize=12)
    ax.set_ylabel("Detection power on top-10% sat_frac flights (%)", fontsize=12)
    ax.set_title("Matched-coverage curve: detection power vs false alarm rate", fontsize=13)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("matched_coverage_curve.png", dpi=150)
    plt.show()
    print("saved matched_coverage_curve.png")

    # print at alpha=0.05
    print()
    print("=== At alpha = 0.05 ===")
    at_005 = rdf[np.abs(rdf["alpha"] - 0.05) < 0.01]
    print(at_005[["detector", "fa_rate", "k", "n_top", "detection_power"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()