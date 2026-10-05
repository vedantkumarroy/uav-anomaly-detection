import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ANALYSIS_CSV = "analysis_set_with_log_scores.csv"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

MOTOR_SAT_THRESHOLD = 0.95


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


def compute_sat_frac(logdir):
    """Compute sat_frac from actuator_motors or actuator_outputs Parquet."""
    motors_p = os.path.join(logdir, "actuator_motors.parquet")
    outputs_p = os.path.join(logdir, "actuator_outputs.parquet")

    if os.path.exists(motors_p):
        t = pq.read_table(motors_p)
        cols = [c for c in t.column_names if c.startswith("control[")]
        active = []
        for c in cols:
            arr = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
            if np.isnan(arr).all():
                continue
            if np.nanmax(np.abs(arr)) == 0:
                continue
            active.append(c)
        if not active:
            return np.nan
        arr = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in active])
        mx = np.nanmax(arr, axis=1)
        sat = mx >= MOTOR_SAT_THRESHOLD
        sat = sat[~np.isnan(mx)]
        if len(sat) == 0:
            return np.nan
        return float(sat.mean())

    if os.path.exists(outputs_p):
        t = pq.read_table(outputs_p)
        noutputs = t["noutputs"].to_numpy(zero_copy_only=False).astype(np.int64) if "noutputs" in t.column_names else None
        cols = [c for c in t.column_names if c.startswith("output[")]
        if noutputs is not None and len(noutputs) > 0:
            n = int(np.nanmax(noutputs))
            cols = [c for c in cols if int(c.split("[")[1].rstrip("]")) < n]
        active = []
        for c in cols:
            arr = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
            if np.nanmax(np.abs(arr)) == 0:
                continue
            active.append(c)
        if not active:
            return np.nan
        # assume PWM 1000-2000 range
        arr = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in active])
        norm = np.clip((arr - 1000.0) / (2000.0 - 1000.0), 0.0, 1.0)
        mx = np.nanmax(norm, axis=1)
        sat = mx >= MOTOR_SAT_THRESHOLD
        sat = sat[~np.isnan(mx)]
        if len(sat) == 0:
            return np.nan
        return float(sat.mean())

    return np.nan


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
    print(f"loaded: {len(df)}")

    # compute sat_frac per log
    print("computing sat_frac from Parquet ...")
    sats = []
    for i, row in df.iterrows():
        logdir = logdir_for(row["log_id"])
        if logdir is None:
            sats.append(np.nan)
            continue
        try:
            sats.append(compute_sat_frac(logdir))
        except Exception:
            sats.append(np.nan)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}")

    df["sat_frac_recomputed"] = sats
    df.to_csv("analysis_set_with_satfrac.csv", index=False)
    print(f"\nsaved analysis_set_with_satfrac.csv")
    print(f"sat_frac non-null: {df['sat_frac_recomputed'].notna().sum()} of {len(df)}")

    # features
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

    # detectors
    print("\ntraining detectors ...")
    iso = IsolationForest(n_estimators=200, random_state=0)
    iso.fit(X_A)
    iso_cal = -iso.score_samples(X_B)
    iso_test = -iso.score_samples(X_C)
    iso_thr = conformal_threshold(iso_cal, 0.05)
    iso_flags = iso_test > iso_thr

    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(X_A)
    svm_cal = -svm.decision_function(X_B)
    svm_test = -svm.decision_function(X_C)
    svm_thr = conformal_threshold(svm_cal, 0.05)
    svm_flags = svm_test > svm_thr

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_A)
    Xb = scaler.transform(X_B)
    Xc = scaler.transform(X_C)
    ae = MLPRegressor(hidden_layer_sizes=(5,), activation="relu", solver="adam",
                      learning_rate_init=0.01, max_iter=2000, random_state=0)
    ae.fit(Xt, Xt)
    ae_cal = np.mean((ae.predict(Xb) - Xb) ** 2, axis=1)
    ae_test = np.mean((ae.predict(Xc) - Xc) ** 2, axis=1)
    ae_thr = conformal_threshold(ae_cal, 0.05)
    ae_flags = ae_test > ae_thr

    # sort C by sat_frac descending
    sat_C = df.loc[C_mask, "sat_frac_recomputed"].values
    order = np.argsort(-np.nan_to_num(sat_C, nan=-1.0))

    print()
    print("=" * 92)
    print("C4: Detection power by sat_frac ranking")
    print("=" * 92)
    print()

    n_C = len(sat_C)
    thresholds = {
        "top_5pct": int(0.05 * n_C),
        "top_10pct": int(0.10 * n_C),
        "top_20pct": int(0.20 * n_C),
        "top_50pct": int(0.50 * n_C),
    }

    print(f"{'bucket':>12s}  {'n':>5s}  {'method':>14s}  {'k/n':>10s}  "
          f"{'rate':>8s}  {'CI_low':>8s}  {'CI_high':>8s}")

    results = []
    for bucket_name, n_top in thresholds.items():
        top_idx = order[:n_top]
        for method, flags in [("IsoForest", iso_flags),
                              ("SVM", svm_flags),
                              ("Autoencoder", ae_flags)]:
            selected = flags[top_idx]
            k = int(selected.sum())
            n = len(selected)
            rate = k / n if n > 0 else np.nan
            lo, hi = wilson_ci(k, n)
            print(f"{bucket_name:>12s}  {n:>5d}  {method:>14s}  {k:>4d}/{n:<5d}  "
                  f"{rate:>8.4f}  {lo:>8.4f}  {hi:>8.4f}")
            results.append({
                "bucket": bucket_name, "n": n, "method": method,
                "n_detected": k, "rate": rate, "ci_low": lo, "ci_high": hi,
            })
        print()

    rdf = pd.DataFrame(results)
    rdf.to_csv("c4_satfrac_ranking.csv", index=False)

    print("Comparison: n_indicators >= 2 (from prior run):")
    print("  IsoForest:   13/94 (13.8%)  CI=[0.083, 0.222]")
    print("  SVM:         14/94 (14.9%)  CI=[0.091, 0.235]")
    print("  Autoencoder: 15/94 (16.0%)  CI=[0.099, 0.247]")
    print()
    print("Top-10% sat_frac (approximately same sample size):")
    sub = rdf[rdf["bucket"] == "top_10pct"]
    for _, r in sub.iterrows():
        print(f"  {r['method']:>14s}: {r['n_detected']:>3d}/{r['n']:<5d} "
              f"({r['rate']*100:.1f}%)  CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


if __name__ == "__main__":
    main()