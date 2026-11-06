"""
C4 hand-label protocol.

Take the top 50 flights by sat_frac in Group C. For each, extract motor
time series and produce a worksheet for hand classification.

Output: c4_handlabel_worksheet.csv
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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

ALPHA = 0.05
MOTOR_SAT_THRESHOLD = 0.95
N_TOP = 50


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def longest_run_seconds(bools, ts_s):
    if len(bools) == 0:
        return 0.0
    best = 0.0
    start = None
    for i, v in enumerate(bools):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dur = ts_s[i] - ts_s[start]
            if dur > best:
                best = dur
            start = None
    if start is not None:
        dur = ts_s[-1] - ts_s[start]
        if dur > best:
            best = dur
    return best


def extract_motor_stats(logdir):
    out = {
        "source": "none",
        "n_motors": 0,
        "motor_means": [],
        "motor_stds": [],
        "motor_maxes": [],
        "sat_frac": np.nan,
        "sat_max_run_s": np.nan,
        "spread_p95": np.nan,
        "reversible_flags": 0,
    }

    motors_p = os.path.join(logdir, "actuator_motors.parquet")
    outputs_p = os.path.join(logdir, "actuator_outputs.parquet")

    if os.path.exists(motors_p):
        t = pq.read_table(motors_p)
        out["source"] = "actuator_motors"
        if "reversible_flags" in t.column_names:
            try:
                rf = t["reversible_flags"].to_numpy(zero_copy_only=False)
                out["reversible_flags"] = int(rf[0]) if len(rf) > 0 else 0
            except Exception:
                pass
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
            return out
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        arr = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in active])
        out["n_motors"] = len(active)
    elif os.path.exists(outputs_p):
        t = pq.read_table(outputs_p)
        out["source"] = "actuator_outputs"
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
            return out
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        arr = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in active])
        arr = np.clip((arr - 1000.0) / 1000.0, 0.0, 1.0)
        out["n_motors"] = len(active)
    else:
        return out

    # airborne mask
    ld_path = os.path.join(logdir, "vehicle_land_detected.parquet")
    if os.path.exists(ld_path):
        ld = pq.read_table(ld_path)
        if "landed" in ld.column_names and "timestamp" in ld.column_names:
            ld_ts = ld["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
            ld_landed = ld["landed"].to_numpy(zero_copy_only=False).astype(bool)
            idx = np.searchsorted(ld_ts, ts, side="right") - 1
            idx = np.clip(idx, 0, len(ld_landed) - 1)
            mask = ~ld_landed[idx]
        else:
            mask = np.ones(len(ts), dtype=bool)
    else:
        mask = np.ones(len(ts), dtype=bool)

    if mask.sum() < 2:
        return out

    arr_air = arr[mask]
    ts_air = ts[mask]
    all_nan = np.isnan(arr_air).all(axis=1)
    arr_valid = arr_air[~all_nan]
    ts_valid = ts_air[~all_nan]
    if len(arr_valid) == 0:
        return out

    mx = np.nanmax(arr_valid, axis=1)
    mn = np.nanmin(arr_valid, axis=1)
    sat = mx >= MOTOR_SAT_THRESHOLD

    out["sat_frac"] = float(sat.mean())
    out["sat_max_run_s"] = longest_run_seconds(sat, (ts_valid - ts_valid[0]) / 1e6)
    spread = mx - mn
    spread = spread[np.isfinite(spread)]
    if len(spread) > 0:
        out["spread_p95"] = float(np.percentile(spread, 95))

    for i in range(arr_valid.shape[1]):
        col = arr_valid[:, i]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            out["motor_means"].append(np.nan)
            out["motor_stds"].append(np.nan)
            out["motor_maxes"].append(np.nan)
        else:
            out["motor_means"].append(float(col.mean()))
            out["motor_stds"].append(float(col.std()))
            out["motor_maxes"].append(float(col.max()))

    return out


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"loaded: {len(df)}")
    print(f"columns: {df.columns.tolist()[:10]} ...")

    # Restrict to Group C
    C = df[df["group"] == "C"].copy()
    print(f"Group C: {len(C)}")
    print()

    # compute motor stats
    print("computing motor stats for Group C flights ...")
    rows = []
    for i, row in C.iterrows():
        logdir = logdir_for(row["log_id"])
        if logdir is None:
            continue
        stats = extract_motor_stats(logdir)
        rec = {
            "log_id": row["log_id"],
            "band": row["band"],
            "sat_frac": stats["sat_frac"],
            "sat_max_run_s": stats["sat_max_run_s"],
            "spread_p95": stats["spread_p95"],
            "n_motors": stats["n_motors"],
            "actuator_source": stats["source"],
            "reversible_flags": stats["reversible_flags"],
            "failsafe_frac": row.get("failsafe_frac", np.nan),
            "nav_state_forbidden_frac": row.get("nav_state_forbidden_frac", np.nan),
            "filter_fault_flags_frac": row.get("filter_fault_flags_frac", np.nan),
            "landed_at_end": row.get("landed_at_end", None),
        }
        for j, m in enumerate(stats["motor_means"]):
            rec[f"motor_{j+1}_mean"] = m
            rec[f"motor_{j+1}_std"] = stats["motor_stds"][j]
            rec[f"motor_{j+1}_max"] = stats["motor_maxes"][j]
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(C)}")

    mdf = pd.DataFrame(rows)
    mdf = mdf.dropna(subset=["sat_frac"]).sort_values("sat_frac", ascending=False).reset_index(drop=True)
    print(f"\nflights with measurable sat_frac: {len(mdf)}")

    top = mdf.head(N_TOP).copy()
    top["rank"] = range(1, len(top) + 1)

    # detector flags
    print("\nrecomputing detector flags ...")
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
    iso_thr = conformal_threshold(iso_cal, ALPHA)

    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(X_A)
    svm_cal = -svm.decision_function(X_B)
    svm_test = -svm.decision_function(X_C)
    svm_thr = conformal_threshold(svm_cal, ALPHA)

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_A)
    Xb = scaler.transform(X_B)
    Xc = scaler.transform(X_C)
    ae = MLPRegressor(hidden_layer_sizes=(5,), activation="relu", solver="adam",
                      learning_rate_init=0.01, max_iter=2000, random_state=0)
    ae.fit(Xt, Xt)
    ae_cal = np.mean((ae.predict(Xb) - Xb) ** 2, axis=1)
    ae_test = np.mean((ae.predict(Xc) - Xc) ** 2, axis=1)
    ae_thr = conformal_threshold(ae_cal, ALPHA)

    # map log_id to flags
    flags_by_log = {}
    C_log_ids = df.loc[C_mask, "log_id"].values
    for i, log_id in enumerate(C_log_ids):
        flags_by_log[log_id] = {
            "flag_iso": bool(iso_test[i] > iso_thr),
            "flag_svm": bool(svm_test[i] > svm_thr),
            "flag_ae": bool(ae_test[i] > ae_thr),
        }

    for i, row in top.iterrows():
        f = flags_by_log.get(row["log_id"], {})
        top.at[i, "flag_iso"] = f.get("flag_iso", False)
        top.at[i, "flag_svm"] = f.get("flag_svm", False)
        top.at[i, "flag_ae"] = f.get("flag_ae", False)
        top.at[i, "flag_count"] = int(f.get("flag_iso", False)) + int(f.get("flag_svm", False)) + int(f.get("flag_ae", False))
        top.at[i, "hand_label"] = ""

    top.to_csv("c4_handlabel_worksheet.csv", index=False)
    print(f"\nsaved c4_handlabel_worksheet.csv ({len(top)} flights)")

    print()
    print("=" * 122)
    print(f"C4 hand-label worksheet: top {N_TOP} sat_frac flights in Group C")
    print("=" * 122)
    print()
    show_cols = ["rank", "log_id", "band", "sat_frac", "sat_max_run_s", "spread_p95",
                 "n_motors", "flag_iso", "flag_svm", "flag_ae", "flag_count"]
    print(top[show_cols].to_string(index=False))

    print()
    print("=" * 122)
    print("Motor means per flight")
    print("=" * 122)
    motor_cols = sorted([c for c in top.columns if c.endswith("_mean") and "motor_" in c],
                        key=lambda c: int(c.split("_")[1]))
    print(top[["rank", "log_id", "band"] + motor_cols].to_string(index=False))


if __name__ == "__main__":
    main()