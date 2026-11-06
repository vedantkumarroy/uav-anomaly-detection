"""
Alpha sensitivity of C2 and C3.

For each alpha in {0.01, 0.05, 0.10, 0.20}:
  - Recompute regime-level coverage (C2)
  - Recompute cross-band and weighted conformal coverage (C3)
  - Report whether the calibration gap holds across confidence levels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_CSV = "regime_segments.csv"
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

ALPHAS = [0.01, 0.05, 0.10, 0.20]


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def weighted_conformal_threshold(scores, weights, alpha):
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(w)
    s, w = s[mask], w[mask]
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


def c2_alpha_sweep():
    df = pd.read_csv(REGIME_CSV)
    print("=" * 78)
    print("C2: regime coverage across alphas")
    print("=" * 78)
    print()

    regimes = ["hover", "translation", "descent"]
    results = []
    print(f"{'alpha':>6s}  " + "".join(f"{r:>14s}" for r in regimes))
    for alpha in ALPHAS:
        line = f"{alpha:>6.2f}  "
        for regime in regimes:
            sub = df[df["regime"] == regime]
            cal = sub[sub["group"] == "B"]
            test = sub[sub["group"] == "C"]
            covs = []
            for r in RATIOS:
                col = f"score_{r}"
                c = cal[col].dropna().values
                t = test[col].dropna().values
                if len(c) < 20 or len(t) < 20:
                    continue
                thr = conformal_threshold(c, alpha)
                covs.append(float(np.mean(t <= thr)))
            mean_cov = np.mean(covs) if covs else np.nan
            results.append({"alpha": alpha, "regime": regime, "coverage": mean_cov})
            line += f"{mean_cov:>14.4f}"
        print(line)

    return pd.DataFrame(results)


def c3_alpha_sweep():
    df = pd.read_csv(ANALYSIS_CSV)
    print()
    print("=" * 78)
    print("C3: cross-band and weighted coverage across alphas")
    print("=" * 78)
    print()

    cal_mask = (df["group"] == "B") & (df["band"].isin(["v1.12-1.13", "v1.14-1.15"]))
    test_mask = (df["group"] == "C") & (df["band"] == "v1.16+")

    cal = df[cal_mask].copy()
    test = df[test_mask].copy()
    cal["w"] = cal["band"].apply(lambda b: 2.0 if b == "v1.14-1.15" else 0.5)

    results = []
    print(f"{'alpha':>6s}  {'cross':>10s}  {'weighted':>10s}")
    for alpha in ALPHAS:
        cross_covs = []
        weighted_covs = []
        for r in RATIOS:
            col = f"score_log_{r}"
            c = cal[col].dropna().values
            t = test[col].dropna().values
            if len(c) < 10 or len(t) < 10:
                continue

            thr_cross = conformal_threshold(c, alpha)
            cross_covs.append(float(np.mean(t <= thr_cross)))

            # weighted
            mask = cal[col].notna()
            c_w = cal.loc[mask, col].values
            w_w = cal.loc[mask, "w"].values
            thr_w = weighted_conformal_threshold(c_w, w_w, alpha)
            if not np.isnan(thr_w):
                weighted_covs.append(float(np.mean(t <= thr_w)))

        cross_mean = np.mean(cross_covs) if cross_covs else np.nan
        weighted_mean = np.mean(weighted_covs) if weighted_covs else np.nan
        results.append({"alpha": alpha, "cross": cross_mean, "weighted": weighted_mean})
        print(f"{alpha:>6.2f}  {cross_mean:>10.4f}  {weighted_mean:>10.4f}")

    return pd.DataFrame(results)


def main():
    c2 = c2_alpha_sweep()
    c3 = c3_alpha_sweep()

    c2.to_csv("c2_alpha_sweep.csv", index=False)
    c3.to_csv("c3_alpha_sweep.csv", index=False)
    print()
    print("saved c2_alpha_sweep.csv, c3_alpha_sweep.csv")

    # figure: two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: C2 regime coverage
    ax = axes[0]
    for regime in ["hover", "translation", "descent"]:
        sub = c2[c2["regime"] == regime]
        ax.plot(sub["alpha"], sub["coverage"], marker="o", label=regime)
    ax.plot([0.01, 0.20], [0.99, 0.80], linestyle="--", color="black", linewidth=1,
            label="nominal (1 - alpha)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Mean coverage")
    ax.set_title("C2: regime coverage across alpha")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B: C3
    ax = axes[1]
    ax.plot(c3["alpha"], c3["cross"], marker="o", color="indianred", label="cross-band")
    ax.plot(c3["alpha"], c3["weighted"], marker="s", color="seagreen", label="weighted")
    ax.plot([0.01, 0.20], [0.99, 0.80], linestyle="--", color="black", linewidth=1,
            label="nominal")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Mean coverage")
    ax.set_title("C3: cross-band and weighted coverage across alpha")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("alpha_sensitivity.png", dpi=150)
    plt.show()
    print("saved alpha_sensitivity.png")


if __name__ == "__main__":
    main()