"""
c4_cells_figure.py

C4 detection power under contaminated vs clean calibration.
Same Group C test set (top 10 percent sat_frac).
Fitted detectors trained on A both times.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
SATFRAC_CSV = "analysis_set_with_satfrac.csv"

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


def load_analysis():
    a = pd.read_csv(ANALYSIS_CSV)
    if "sat_frac" not in a.columns:
        s = pd.read_csv(SATFRAC_CSV)
        col = "sat_frac_recomputed" if "sat_frac_recomputed" in s.columns else "sat_frac"
        a = a.merge(s[["log_id", col]].rename(columns={col: "sat_frac"}),
                    on="log_id", how="left")
    return a


def compute_tripped(df):
    return (
        (df["failsafe_frac"] > 0)
        | (df["nav_state_forbidden_frac"] > 0)
        | (df["filter_fault_flags_frac"] > 0)
        | (df["landed_at_end"] == False)
        | (df["sat_frac"] > 0.05)
    )


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def main():
    df = load_analysis()
    df["_tripped"] = compute_tripped(df).values

    score_cols = [f"score_log_{r}" for r in RATIOS]
    X = df[score_cols].values.astype(np.float64)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = medians[j] if not np.isnan(medians[j]) else 0.0

    A_mask = (df["group"].values == "A")
    B_all_mask = (df["group"].values == "B")
    B_clean_mask = B_all_mask & (~df["_tripped"].values)
    C_mask = (df["group"].values == "C")

    X_A = X[A_mask]
    X_B_all = X[B_all_mask]
    X_B_clean = X[B_clean_mask]
    X_C = X[C_mask]

    print(f"A train: {A_mask.sum()}")
    print(f"B all:   {B_all_mask.sum()}")
    print(f"B clean: {B_clean_mask.sum()}")
    print(f"C test:  {C_mask.sum()}")

    # Top 10% sat_frac within C
    sat_C = df.loc[C_mask, "sat_frac"].values
    n_C = len(sat_C)
    n_top = int(0.10 * np.sum(~np.isnan(sat_C)))
    order = np.argsort(-np.nan_to_num(sat_C, nan=-1.0))
    is_top = np.zeros(n_C, dtype=bool)
    is_top[order[:n_top]] = True
    print(f"top-10% sat_frac flights: {int(is_top.sum())}")

    # Fit on A
    iso = IsolationForest(n_estimators=200, random_state=0).fit(X_A)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_A)

    scaler = StandardScaler().fit(X_A)
    Xt = scaler.transform(X_A)
    Xb_all = scaler.transform(X_B_all)
    Xb_clean = scaler.transform(X_B_clean)
    Xc = scaler.transform(X_C)
    ae = MLPRegressor(hidden_layer_sizes=(5,), activation="relu", solver="adam",
                      learning_rate_init=0.01, max_iter=2000, random_state=0)
    ae.fit(Xt, Xt)

    iso_cal_all = -iso.score_samples(X_B_all)
    iso_cal_clean = -iso.score_samples(X_B_clean)
    iso_test = -iso.score_samples(X_C)

    svm_cal_all = -svm.decision_function(X_B_all)
    svm_cal_clean = -svm.decision_function(X_B_clean)
    svm_test = -svm.decision_function(X_C)

    ae_cal_all = np.mean((ae.predict(Xb_all) - Xb_all) ** 2, axis=1)
    ae_cal_clean = np.mean((ae.predict(Xb_clean) - Xb_clean) ** 2, axis=1)
    ae_test = np.mean((ae.predict(Xc) - Xc) ** 2, axis=1)

    detectors = [
        ("IsolationForest", iso_cal_all, iso_cal_clean, iso_test),
        ("OneClassSVM", svm_cal_all, svm_cal_clean, svm_test),
        ("Autoencoder", ae_cal_all, ae_cal_clean, ae_test),
    ]

    results = []
    print()
    print(f"{'detector':>18s}  {'contaminated':>14s}  {'clean':>14s}")
    for name, cal_all, cal_clean, test in detectors:
        thr_all = conformal_threshold(cal_all, ALPHA)
        thr_clean = conformal_threshold(cal_clean, ALPHA)
        k_all = int((test[is_top] > thr_all).sum())
        k_clean = int((test[is_top] > thr_clean).sum())
        n = int(is_top.sum())
        lo_all, hi_all = wilson(k_all, n)
        lo_clean, hi_clean = wilson(k_clean, n)
        print(f"{name:>18s}  {k_all:>3d}/{n:<3d} ({k_all/n:.3f})  "
              f"{k_clean:>3d}/{n:<3d} ({k_clean/n:.3f})")
        results.append({
            "detector": name, "n_top": n,
            "k_contaminated": k_all, "power_contaminated": k_all / n,
            "lo_contaminated": lo_all, "hi_contaminated": hi_all,
            "k_clean": k_clean, "power_clean": k_clean / n,
            "lo_clean": lo_clean, "hi_clean": hi_clean,
        })

    pd.DataFrame(results).to_csv("c4_cells_table.csv", index=False)
    print()
    print("saved c4_cells_table.csv")

    # Figure
    rdf = pd.DataFrame(results)
    x = np.arange(len(rdf))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    p1 = rdf["power_contaminated"].values
    p2 = rdf["power_clean"].values
    e1 = np.vstack([p1 - rdf["lo_contaminated"].values,
                    rdf["hi_contaminated"].values - p1])
    e2 = np.vstack([p2 - rdf["lo_clean"].values,
                    rdf["hi_clean"].values - p2])
    ax.bar(x - w/2, p1, w, yerr=e1, capsize=5, label="Contaminated B (650)",
           color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, p2, w, yerr=e2, capsize=5, label="Clean B (374)",
           color="#DD8452", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(rdf["detector"])
    ax.set_ylabel("Detection power (top-10% sat_frac)")
    ax.set_title("C4: detection power, contaminated vs clean calibration\n"
                 "Same Group C test set (n=%d)" % int(rdf["n_top"].iloc[0]))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("c4_cells_figure.png", dpi=150, bbox_inches="tight")
    print("saved c4_cells_figure.png")


if __name__ == "__main__":
    main()
