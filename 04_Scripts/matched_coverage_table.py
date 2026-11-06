"""
Matched-coverage comparison table.

Four detectors on the same test set (Group C):
  1. Isolation Forest
  2. One-Class SVM
  3. Autoencoder
  4. PX4 threshold (ratio > 1.0, no calibration)

Each is calibrated to the same false alarm rate. Detection power measured
against the top-10% sat_frac flights.
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
COMBINED_CSV = "combined_index.csv"

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
MOTOR_SAT_THRESHOLD = 0.95


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def compute_sat_frac(logdir):
    """Compute per-flight sat_frac from actuator_motors or actuator_outputs."""
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
        arr = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in active])
        norm = np.clip((arr - 1000.0) / 1000.0, 0.0, 1.0)
        mx = np.nanmax(norm, axis=1)
        sat = mx >= MOTOR_SAT_THRESHOLD
        sat = sat[~np.isnan(mx)]
        if len(sat) == 0:
            return np.nan
        return float(sat.mean())
    return np.nan


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"analysis set: {len(df)}")

    # Compute sat_frac for every log (or reuse from prior run if the file exists)
    satfrac_path = "analysis_set_with_satfrac.csv"
    if os.path.exists(satfrac_path):
        sf = pd.read_csv(satfrac_path)
        df = df.merge(sf[["log_id", "sat_frac_recomputed"]], on="log_id", how="left")
        print("loaded sat_frac from cached file")
    else:
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
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(df)}")
        df["sat_frac_recomputed"] = sats
        df.to_csv("analysis_set_with_satfrac_matched.csv", index=False)

    # Feature matrix
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

    # Train detectors
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

    # PX4 threshold: use ratio > 1.0 directly.
    # Score = fraction of samples with any ratio > 1.0. Higher = more anomalous.
    px4_scores_C = []
    px4_scores_B = []
    for group_mask, target_list in [(B_mask, px4_scores_B), (C_mask, px4_scores_C)]:
        for i in np.where(group_mask)[0]:
            row = df.iloc[i]
            logdir = logdir_for(row["log_id"])
            if logdir is None:
                target_list.append(np.nan)
                continue
            p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
            if not os.path.exists(p):
                target_list.append(np.nan)
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
            target_list.append(float(any_flag.mean()))

    px4_cal = np.array([s for s in px4_scores_B if not np.isnan(s)])
    px4_test = np.array([s for s in px4_scores_C if not np.isnan(s)])

    # Filter C for sat_frac labels
    satfrac_C = df.loc[C_mask, "sat_frac_recomputed"].values

    # Top 10% sat_frac in C
    sat_valid_mask = ~np.isnan(satfrac_C)
    n_valid = sat_valid_mask.sum()
    n_top = int(0.10 * n_valid)
    order = np.argsort(-np.nan_to_num(satfrac_C, nan=-1.0))
    top_idx = order[:n_top]

    # For each detector, use the top-10% bucket as the "should detect" set
    print()
    print("=" * 92)
    print("Matched-coverage comparison at alpha = 0.05")
    print("=" * 92)
    print()
    print("Each method calibrated to 5% false alarm rate on Group B.")
    print("Test set: Group C. Detection power target: top 10% sat_frac flights.")
    print()

    # NOTE: the top_idx is index into C positions. But scores arrays are aligned to C rows.
    # We need to be careful: px4_test has fewer entries because NaNs were dropped.
    # Handle this by aligning scores to C rows.

    # Create aligned arrays for C
    n_C = C_mask.sum()
    iso_C_full = np.full(n_C, np.nan)
    iso_C_full[:] = iso_test  # same length
    svm_C_full = np.full(n_C, np.nan)
    svm_C_full[:] = svm_test
    ae_C_full = np.full(n_C, np.nan)
    ae_C_full[:] = ae_test

    # px4 needs alignment
    px4_C_full = np.full(n_C, np.nan)
    j = 0
    for i in range(n_C):
        if not np.isnan(px4_scores_C[i]) if i < len(px4_scores_C) else True:
            pass
    # simpler: fill in order
    valid_px4_C = [s for s in px4_scores_C if not np.isnan(s)]
    px4_C_full = np.array(valid_px4_C + [np.nan] * (n_C - len(valid_px4_C)))

    detectors = [
        ("IsolationForest", iso_cal, iso_C_full),
        ("OneClassSVM", svm_cal, svm_C_full),
        ("Autoencoder", ae_cal, ae_C_full),
        ("PX4_threshold", px4_cal, px4_C_full),
    ]

    rows = []
    for name, cal, test in detectors:
        cal_clean = cal[~np.isnan(cal)]
        test_clean = test[~np.isnan(test)]
        if len(cal_clean) < 10:
            continue
        thr = conformal_threshold(cal_clean, ALPHA)
        flags = test_clean > thr
        n_flag = int(flags.sum())
        n_test = len(test_clean)
        fa_rate = n_flag / n_test
        ci_fa = wilson_ci(n_flag, n_test)
        rows.append({
            "detector": name,
            "n_cal": len(cal_clean),
            "n_test": n_test,
            "threshold": thr,
            "flags": n_flag,
            "false_alarm_rate": fa_rate,
            "fa_ci_low": ci_fa[0],
            "fa_ci_high": ci_fa[1],
        })

    rdf = pd.DataFrame(rows)
    print(rdf.round(4).to_string(index=False))
    print()

    # Detection power on top-10% sat_frac bucket
    # Need to align top_idx with the position of each detector's score after NaN removal
    print("Detection power on top-10% sat_frac flights (n={}):".format(n_top))
    print()

    # Build a boolean "is_top" array of length n_C
    is_top = np.zeros(n_C, dtype=bool)
    is_top[top_idx] = True

    print(f"{'detector':>18s}  {'k/n':>10s}  {'rate':>8s}  {'95% CI':>20s}")
    power_rows = []
    for name, cal, test in detectors:
        cal_clean = cal[~np.isnan(cal)]
        thr = conformal_threshold(cal_clean, ALPHA)
        # scores aligned to C rows, NaN allowed
        flags = test > thr
        flags[np.isnan(test)] = False
        k = int(flags[is_top].sum())
        n = int(is_top.sum())
        rate = k / n if n > 0 else np.nan
        lo, hi = wilson_ci(k, n)
        print(f"{name:>18s}  {k:>3d}/{n:<5d}  {rate:>8.4f}  [{lo:.3f}, {hi:.3f}]")
        power_rows.append({
            "detector": name, "n_top": n, "n_detected": k,
            "detection_rate": rate, "ci_low": lo, "ci_high": hi,
        })

    print()
    pdf = pd.DataFrame(power_rows)
    pdf.to_csv("matched_coverage_detection_power.csv", index=False)
    rdf.to_csv("matched_coverage_false_alarm.csv", index=False)
    print("saved matched_coverage_false_alarm.csv")
    print("saved matched_coverage_detection_power.csv")


if __name__ == "__main__":
    main()