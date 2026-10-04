import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

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

# regime thresholds in m/s
HOVER_H_SPEED_MAX = 0.5
HOVER_V_SPEED_MAX = 0.5
TRANSLATION_H_SPEED_MIN = 0.5
DESCENT_V_SPEED_MIN = 0.5

# min segment length in seconds
MIN_SEGMENT_S = 5.0


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


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def detect_regime_segments(local_pos_ts, vx, vy, vz):
    """Return list of (start_ts, end_ts, regime) segments."""
    n = len(local_pos_ts)
    if n < 10:
        return []

    h_speed = np.sqrt(vx**2 + vy**2)
    v_speed = np.abs(vz)

    regimes = np.full(n, "unknown", dtype=object)
    hover = (h_speed < HOVER_H_SPEED_MAX) & (v_speed < HOVER_V_SPEED_MAX)
    descent = vz > DESCENT_V_SPEED_MIN
    translation = (h_speed >= TRANSLATION_H_SPEED_MIN) & ~descent

    regimes[hover] = "hover"
    regimes[descent] = "descent"
    regimes[translation] = "translation"

    # find runs
    segments = []
    start = 0
    for i in range(1, n):
        if regimes[i] != regimes[i - 1]:
            duration_s = (local_pos_ts[i - 1] - local_pos_ts[start]) / 1e6
            if duration_s >= MIN_SEGMENT_S and regimes[start] != "unknown":
                segments.append((local_pos_ts[start], local_pos_ts[i - 1], regimes[start]))
            start = i
    # last segment
    duration_s = (local_pos_ts[-1] - local_pos_ts[start]) / 1e6
    if duration_s >= MIN_SEGMENT_S and regimes[start] != "unknown":
        segments.append((local_pos_ts[start], local_pos_ts[-1], regimes[start]))

    return segments


def segment_scores(ratio_ts, ratio_arrays, seg_start, seg_end, percentile=95):
    """For a segment window, compute p95 of each ratio."""
    mask = (ratio_ts >= seg_start) & (ratio_ts <= seg_end)
    if mask.sum() < 10:
        return None
    out = {}
    for ratio in RATIOS:
        arr = ratio_arrays.get(ratio)
        if arr is None:
            out[ratio] = np.nan
            continue
        window = arr[mask]
        window = window[np.isfinite(window)]
        if len(window) == 0:
            out[ratio] = np.nan
        else:
            out[ratio] = float(np.percentile(window, percentile))
    return out


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"loaded: {len(df)}")

    all_segments = []

    for i, row in df.iterrows():
        log_id = row["log_id"]
        logdir = logdir_for(log_id)
        if logdir is None:
            continue

        lp_path = os.path.join(logdir, "vehicle_local_position.parquet")
        rat_path = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
        if not os.path.exists(lp_path) or not os.path.exists(rat_path):
            continue

        try:
            lp = pq.read_table(lp_path)
            if "timestamp" not in lp.column_names:
                continue
            vx = lp["vx"].to_numpy(zero_copy_only=False).astype(np.float64)
            vy = lp["vy"].to_numpy(zero_copy_only=False).astype(np.float64)
            vz = lp["vz"].to_numpy(zero_copy_only=False).astype(np.float64)
            lp_ts = lp["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        except Exception:
            continue

        segments = detect_regime_segments(lp_ts, vx, vy, vz)
        if not segments:
            continue

        try:
            rat = pq.read_table(rat_path)
            rat_ts = rat["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
            ratio_arrays = {}
            for r in RATIOS:
                if r in rat.column_names:
                    arr = rat[r].to_numpy(zero_copy_only=False).astype(np.float64)
                    ratio_arrays[r] = np.log1p(np.maximum(arr, 0))
        except Exception:
            continue

        for seg_start, seg_end, regime in segments:
            scores = segment_scores(rat_ts, ratio_arrays, seg_start, seg_end)
            if scores is None:
                continue
            rec = {
                "log_id": log_id,
                "band": row["band"],
                "group": row["group"],
                "regime": regime,
                "seg_start": seg_start,
                "seg_end": seg_end,
                "seg_duration_s": (seg_end - seg_start) / 1e6,
            }
            rec.update({f"score_{r}": scores[r] for r in RATIOS})
            all_segments.append(rec)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}")

    sdf = pd.DataFrame(all_segments)
    print()
    print(f"total segments: {len(sdf)}")
    print(f"segments by regime:")
    print(sdf["regime"].value_counts().to_string())
    print()
    print(f"segments by regime x band:")
    print(pd.crosstab(sdf["regime"], sdf["band"]).to_string())

    sdf.to_csv("regime_segments.csv", index=False)
    print("\nsaved regime_segments.csv")

    # calibrate and test per regime
    print()
    print("=" * 78)
    print("Regime-level C2: coverage per regime")
    print("=" * 78)
    print()
    print(f"{'regime':>14s}  {'ratio':>18s}  {'n_cal':>6s}  {'n_test':>7s}  "
          f"{'threshold':>10s}  {'coverage':>10s}")

    results = []
    for regime in ["hover", "translation", "descent"]:
        sub = sdf[sdf["regime"] == regime]
        if len(sub) < 50:
            continue
        cal = sub[sub["group"] == "B"]
        test = sub[sub["group"] == "C"]
        for ratio in RATIOS:
            col = f"score_{ratio}"
            c = cal[col].dropna().values
            t = test[col].dropna().values
            if len(c) < 20 or len(t) < 20:
                continue
            thr = conformal_threshold(c, ALPHA)
            cov = float(np.mean(t <= thr))
            results.append({
                "regime": regime, "ratio": ratio,
                "n_cal": len(c), "n_test": len(t),
                "threshold": thr, "coverage": cov,
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv("regime_c2_coverage.csv", index=False)

    print()
    print("=== Mean coverage per regime (across ratios) ===")
    print()
    for regime in ["hover", "translation", "descent"]:
        sub = rdf[rdf["regime"] == regime]
        if len(sub) == 0:
            continue
        mean_cov = sub["coverage"].mean()
        lo, hi = wilson_ci(int((sub["coverage"] * sub["n_test"]).sum()), sub["n_test"].sum())
        print(f"  {regime:>14s}: mean={mean_cov:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
              f"nominal_in={'yes' if lo <= 0.95 <= hi else 'NO'}")

    print()
    print("=== Compare to pooled calibration ===")
    pooled_cal = sdf[sdf["group"] == "B"]
    pooled_test = sdf[sdf["group"] == "C"]
    for ratio in RATIOS:
        col = f"score_{ratio}"
        c = pooled_cal[col].dropna().values
        t = pooled_test[col].dropna().values
        if len(c) < 20 or len(t) < 20:
            continue
        thr = conformal_threshold(c, ALPHA)
        cov = float(np.mean(t <= thr))
        print(f"  pooled, {ratio:>18s}: {cov:.4f}  (n_cal={len(c)}, n_test={len(t)})")


if __name__ == "__main__":
    main()