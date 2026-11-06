import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

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


def conformal_threshold(scores, alpha, weights=None):
    """Weighted split conformal. weights = None => unweighted."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return np.nan
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    # sort by score
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    # find the smallest score where cumulative weight >= 1 - alpha
    target = 1 - alpha
    idx = np.searchsorted(cum, target, side="left")
    if idx >= n:
        idx = n - 1
    return float(s_sorted[idx])


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"analysis set: {len(df)}")
    print()

    # log-transformed columns
    score_cols = [f"score_log_{r}" for r in RATIOS]

    # --- split by band ---
    # calibration: B logs from v1.12-1.13 and v1.14-1.15
    cal_mask = (df["group"] == "B") & (df["band"].isin(["v1.12-1.13", "v1.14-1.15"]))
    # test: C logs from v1.16+
    test_mask = (df["group"] == "C") & (df["band"] == "v1.16+")

    cal = df[cal_mask].copy()
    test = df[test_mask].copy()

    print(f"calibration (B, v1.12-1.15): {len(cal)}")
    print(f"test (C, v1.16+):           {len(test)}")
    print()

    # also a reference: in-band calibration and test
    # B v1.16+ calibrate, C v1.16+ test
    cal_inband = df[(df["group"] == "B") & (df["band"] == "v1.16+")]
    test_inband = df[(df["group"] == "C") & (df["band"] == "v1.16+")]
    print(f"in-band calibration (B v1.16+): {len(cal_inband)}")
    print(f"in-band test (C v1.16+):        {len(test_inband)}")
    print()

    # --- experiment 1: standard conformal, cross-band ---
    print("=" * 70)
    print("C3: Standard conformal, calibration v1.12-1.15, test v1.16+")
    print("=" * 70)
    print()
    print(f"{'ratio':>18s}  {'n_cal':>6s}  {'n_test':>7s}  {'threshold':>10s}  {'coverage':>10s}")

    cross_results = []
    for ratio in RATIOS:
        col = f"score_log_{ratio}"
        c = cal[col].dropna().values
        t = test[col].dropna().values
        if len(c) < 10 or len(t) < 10:
            continue
        thr = conformal_threshold(c, ALPHA)
        cov = float(np.mean(t <= thr))
        print(f"{ratio:>18s}  {len(c):>6d}  {len(t):>7d}  {thr:>10.4f}  {cov:>10.4f}")
        cross_results.append({
            "ratio": ratio, "n_cal": len(c), "n_test": len(t),
            "threshold": thr, "coverage": cov,
        })

    cross = pd.DataFrame(cross_results)
    cross_mean = cross["coverage"].mean()
    print()
    print(f"mean cross-band coverage: {cross_mean:.4f}")
    print()

    # --- experiment 2: in-band reference ---
    print("=" * 70)
    print("Reference: in-band conformal (B v1.16+, C v1.16+)")
    print("=" * 70)
    print()
    inband_results = []
    for ratio in RATIOS:
        col = f"score_log_{ratio}"
        c = cal_inband[col].dropna().values
        t = test_inband[col].dropna().values
        if len(c) < 10 or len(t) < 10:
            continue
        thr = conformal_threshold(c, ALPHA)
        cov = float(np.mean(t <= thr))
        inband_results.append({
            "ratio": ratio, "n_cal": len(c), "n_test": len(t),
            "threshold": thr, "coverage": cov,
        })
    inband = pd.DataFrame(inband_results)
    print(f"mean in-band coverage: {inband['coverage'].mean():.4f}")
    print()

    # --- experiment 3: weighted conformal for repair ---
    print("=" * 70)
    print("Weighted conformal repair")
    print("=" * 70)
    print()
    # weights: we want calibration samples from v1.14-1.15 to count more
    # if the test is v1.16+, since v1.14-1.15 is closer to v1.16+ than v1.12-1.13
    # simple approach: weight v1.14-1.15 by 2x, v1.12-1.13 by 0.5x
    # a more principled approach uses the density ratio, but we do not have it
    print("Weighting scheme: v1.14-1.15 samples weighted 2x, v1.12-1.13 weighted 0.5x")
    print()
    weights = np.where(cal["band"] == "v1.14-1.15", 2.0, 0.5)
    weighted_results = []
    for ratio in RATIOS:
        col = f"score_log_{ratio}"
        mask = cal[col].notna()
        c = cal.loc[mask, col].values
        w = weights[mask.values]
        t = test[col].dropna().values
        if len(c) < 10 or len(t) < 10:
            continue
        thr = conformal_threshold(c, ALPHA, weights=w)
        cov = float(np.mean(t <= thr))
        weighted_results.append({
            "ratio": ratio, "n_cal": len(c), "n_test": len(t),
            "threshold": thr, "coverage": cov,
        })
    weighted = pd.DataFrame(weighted_results)
    print(f"{'ratio':>18s}  {'n_cal':>6s}  {'n_test':>7s}  {'threshold':>10s}  {'coverage':>10s}")
    for _, r in weighted.iterrows():
        print(f"{r['ratio']:>18s}  {int(r['n_cal']):>6d}  {int(r['n_test']):>7d}  {r['threshold']:>10.4f}  {r['coverage']:>10.4f}")
    print()
    print(f"mean weighted coverage: {weighted['coverage'].mean():.4f}")
    print()

    # save
    cross.to_csv("c3_cross_band.csv", index=False)
    inband.to_csv("c3_inband.csv", index=False)
    weighted.to_csv("c3_weighted.csv", index=False)
    print("saved c3_cross_band.csv, c3_inband.csv, c3_weighted.csv")

    # figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(RATIOS))
    width = 0.27
    cross_vals = [cross[cross["ratio"] == r]["coverage"].values[0] if len(cross[cross["ratio"] == r]) > 0 else np.nan for r in RATIOS]
    inband_vals = [inband[inband["ratio"] == r]["coverage"].values[0] if len(inband[inband["ratio"] == r]) > 0 else np.nan for r in RATIOS]
    weighted_vals = [weighted[weighted["ratio"] == r]["coverage"].values[0] if len(weighted[weighted["ratio"] == r]) > 0 else np.nan for r in RATIOS]
    ax.bar(x - width, cross_vals, width, label="Cross-band (v1.12-1.15 cal)", color="indianred")
    ax.bar(x, inband_vals, width, label="In-band reference (v1.16+ cal)", color="steelblue")
    ax.bar(x + width, weighted_vals, width, label="Weighted conformal repair", color="seagreen")
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(RATIOS, rotation=30, ha="right")
    ax.set_ylabel("Coverage on v1.16+ test set")
    ax.set_title("C3: Cross-band distribution shift and repair")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("c3_shift_figure.png", dpi=150)
    plt.show()
    print("saved c3_shift_figure.png")


if __name__ == "__main__":
    main()