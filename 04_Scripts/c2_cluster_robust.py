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


def conformal_threshold(cal, alpha):
    s = np.sort(cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def coverage_for_subset(seg_df, cal_mask, test_mask, col):
    cal = seg_df.loc[cal_mask, col].dropna().values
    test = seg_df.loc[test_mask, col].dropna().values
    if len(cal) < 5 or len(test) < 5:
        return np.nan
    thr = conformal_threshold(cal, ALPHA)
    return float(np.mean(test <= thr))


def cluster_bootstrap_coverage(seg, n_boot=N_BOOTSTRAP):
    """Block bootstrap with flight as the block.
    Returns bootstrap distribution of the mean coverage across ratios."""
    flight_groups = seg.groupby("log_id").indices
    flight_ids = list(flight_groups.keys())
    n_flights = len(flight_ids)

    boot_means = []
    for b in range(n_boot):
        sampled_flights = RNG.choice(flight_ids, size=n_flights, replace=True)
        sampled_idx = np.concatenate([flight_groups[f] for f in sampled_flights])
        sampled_seg = seg.iloc[sampled_idx].reset_index(drop=True)
        cal_mask = sampled_seg["group"] == "B"
        test_mask = sampled_seg["group"] == "C"
        covs = []
        for r in RATIOS:
            col = f"score_{r}"
            if col not in sampled_seg.columns:
                continue
            c = coverage_for_subset(sampled_seg, cal_mask, test_mask, col)
            if not np.isnan(c):
                covs.append(c)
        if covs:
            boot_means.append(np.mean(covs))
    return np.array(boot_means)


def main():
    seg = pd.read_csv(SEGMENTS_CSV)
    print(f"loaded {len(seg)} segments from {seg['log_id'].nunique()} flights")
    print()

    print("=" * 80)
    print("C2: Regime coverage with flight-clustered bootstrap intervals")
    print("=" * 80)
    print()
    print(f"{'regime':>14s}  {'n_seg':>7s}  {'n_flights':>10s}  "
          f"{'point':>8s}  {'CI_low':>8s}  {'CI_high':>8s}  {'width':>7s}  {'nominal_in':>10s}")

    rows = []
    for regime in ["hover", "translation", "descent"]:
        sub = seg[seg["regime"] == regime].copy().reset_index(drop=True)
        if len(sub) < 30:
            continue

        cal_mask = sub["group"] == "B"
        test_mask = sub["group"] == "C"

        # point estimate: mean of per-ratio coverage using the actual segments
        covs = []
        for r in RATIOS:
            col = f"score_{r}"
            if col not in sub.columns:
                continue
            c = coverage_for_subset(sub, cal_mask, test_mask, col)
            if not np.isnan(c):
                covs.append(c)
        point = np.mean(covs) if covs else np.nan

        # cluster bootstrap
        boot = cluster_bootstrap_coverage(sub, n_boot=N_BOOTSTRAP)
        if len(boot) == 0:
            continue
        lo = np.percentile(boot, 2.5)
        hi = np.percentile(boot, 97.5)
        width = hi - lo
        nominal_in = "yes" if lo <= 0.95 <= hi else "NO"

        print(f"{regime:>14s}  {len(sub):>7d}  {sub['log_id'].nunique():>10d}  "
              f"{point:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  {width:>7.4f}  {nominal_in:>10s}")

        rows.append({
            "regime": regime,
            "n_segments": len(sub),
            "n_flights": sub["log_id"].nunique(),
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "ci_width": width,
        })

    rdf = pd.DataFrame(rows)
    rdf.to_csv("c2_cluster_robust.csv", index=False)

    # figure
    fig, ax = plt.subplots(figsize=(9, 6))
    regimes = rdf["regime"].tolist()
    means = rdf["point"].tolist()
    lows = rdf["ci_low"].tolist()
    highs = rdf["ci_high"].tolist()
    x = np.arange(len(regimes))
    err_low = [means[i] - lows[i] for i in range(len(regimes))]
    err_high = [highs[i] - means[i] for i in range(len(regimes))]

    ax.bar(x, means, yerr=[err_low, err_high], capsize=12,
           color=["steelblue", "indianred", "seagreen"],
           edgecolor="black", alpha=0.85)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.4, label="Nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=12)
    ax.set_ylabel("Mean coverage (across 10 ratios)", fontsize=11)
    ax.set_title("C2: Regime coverage with flight-clustered bootstrap 95% CI", fontsize=13)
    ax.set_ylim(0.88, 0.99)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m + 0.004, f"{m:.4f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("c2_cluster_robust_figure.png", dpi=150)
    plt.show()
    print()
    print("saved c2_cluster_robust.csv and c2_cluster_robust_figure.png")


if __name__ == "__main__":
    main()