import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEGMENTS_CSV = "regime_segments.csv"

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
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def weighted_conformal_threshold(scores, weights, alpha):
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(w)
    s = s[mask]
    w = w[mask]
    if len(s) == 0 or w.sum() <= 0:
        return np.nan
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    w_norm = w_sorted / w_sorted.sum()
    cum = np.cumsum(w_norm)
    target = 1 - alpha
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(s_sorted) - 1)
    return float(s_sorted[idx])


def cluster_bootstrap_weighted(seg, n_boot=N_BOOTSTRAP):
    cal_seg = seg[(seg["group"] == "B") & (seg["band"].isin(["v1.12-1.13", "v1.14-1.15"]))].copy()
    test_seg = seg[(seg["group"] == "C") & (seg["band"] == "v1.16+")].copy()

    # weights for calibration: v1.14-1.15 weighted 2, v1.12-1.13 weighted 0.5
    cal_seg["w"] = cal_seg["band"].apply(lambda b: 2.0 if b == "v1.14-1.15" else 0.5)

    cal_flights = cal_seg["log_id"].unique().tolist()
    test_flights = test_seg["log_id"].unique().tolist()
    cal_groups = cal_seg.groupby("log_id").indices
    test_groups = test_seg.groupby("log_id").indices

    boot = []
    for b in range(n_boot):
        samp_cal = RNG.choice(cal_flights, size=len(cal_flights), replace=True)
        samp_test = RNG.choice(test_flights, size=len(test_flights), replace=True)
        cal_idx = np.concatenate([cal_groups[f] for f in samp_cal])
        test_idx = np.concatenate([test_groups[f] for f in samp_test])
        cal_sub = cal_seg.iloc[cal_idx]
        test_sub = test_seg.iloc[test_idx]

        covs = []
        for r in RATIOS:
            col = f"score_{r}"
            if col not in cal_sub.columns or col not in test_sub.columns:
                continue
            s_cal = cal_sub[col].values
            w_cal = cal_sub["w"].values
            s_test = test_sub[col].dropna().values
            if len(s_test) < 5:
                continue
            thr = weighted_conformal_threshold(s_cal, w_cal, ALPHA)
            if np.isnan(thr):
                continue
            covs.append(float(np.mean(s_test <= thr)))
        if covs:
            boot.append(np.mean(covs))
    return np.array(boot)


seg = pd.read_csv(SEGMENTS_CSV)
print(f"loaded {len(seg)} segments")

# point estimate for weighted
cal_seg = seg[(seg["group"] == "B") & (seg["band"].isin(["v1.12-1.13", "v1.14-1.15"]))].copy()
test_seg = seg[(seg["group"] == "C") & (seg["band"] == "v1.16+")].copy()
cal_seg["w"] = cal_seg["band"].apply(lambda b: 2.0 if b == "v1.14-1.15" else 0.5)

covs_point = []
for r in RATIOS:
    col = f"score_{r}"
    if col not in cal_seg.columns:
        continue
    s_cal = cal_seg[col].values
    w_cal = cal_seg["w"].values
    s_test = test_seg[col].dropna().values
    if len(s_test) < 5:
        continue
    thr = weighted_conformal_threshold(s_cal, w_cal, ALPHA)
    if np.isnan(thr):
        continue
    covs_point.append(float(np.mean(s_test <= thr)))

point = np.mean(covs_point)
print(f"\nweighted conformal point estimate: {point:.4f}")

print("\nrunning block bootstrap (2000 iterations) ...")
boot = cluster_bootstrap_weighted(seg)
lo = np.percentile(boot, 2.5)
hi = np.percentile(boot, 97.5)
nominal_in = "yes" if lo <= 0.95 <= hi else "NO"

print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
print(f"  width:  {hi - lo:.4f}")
print(f"  nominal in: {nominal_in}")

# comparison chart with previous results
fig, ax = plt.subplots(figsize=(10, 6))
labels = ["Cross-band\n(no repair)", "Weighted\nconformal", "In-band\nreference"]
points = [0.9145, point, 0.9324]
lows = [0.8790, lo, 0.8935]
highs = [0.9432, hi, 0.9578]
colors = ["indianred", "seagreen", "steelblue"]
x = np.arange(len(labels))
err_low = [points[i] - lows[i] for i in range(3)]
err_high = [highs[i] - points[i] for i in range(3)]
ax.bar(x, points, yerr=[err_low, err_high], capsize=12, color=colors, edgecolor="black", alpha=0.85)
ax.axhline(0.95, color="black", linestyle="--", linewidth=1.4, label="Nominal 0.95")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Mean coverage on v1.16+ test segments", fontsize=11)
ax.set_title("C3: cluster-robust CI, cross-band vs weighted repair vs in-band", fontsize=13)
ax.set_ylim(0.86, 0.99)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, m in enumerate(points):
    ax.text(i, m + 0.005, f"{m:.4f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("c3_weighted_segments_figure.png", dpi=150)
plt.show()
print("\nsaved c3_weighted_segments_figure.png")

pd.DataFrame([{"method": "cross_band", "point": 0.9145, "ci_low": 0.8790, "ci_high": 0.9432},
              {"method": "weighted_conformal", "point": point, "ci_low": lo, "ci_high": hi},
              {"method": "in_band_reference", "point": 0.9324, "ci_low": 0.8935, "ci_high": 0.9578}]
             ).to_csv("c3_segments_final.csv", index=False)
print("saved c3_segments_final.csv")