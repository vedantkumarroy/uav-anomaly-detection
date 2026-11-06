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


def cluster_bootstrap_coverage(seg, cal_band_set, test_band, cal_group, test_group, n_boot=N_BOOTSTRAP):
    """Block bootstrap with flight as the block.
    Returns bootstrap distribution of mean coverage across ratios."""
    cal_seg = seg[(seg["group"] == cal_group) & (seg["band"].isin(cal_band_set))]
    test_seg = seg[(seg["group"] == test_group) & (seg["band"] == test_band)]

    # group both by flight for bootstrapping
    cal_flights = cal_seg["log_id"].unique().tolist()
    test_flights = test_seg["log_id"].unique().tolist()

    cal_groups = cal_seg.groupby("log_id").indices
    test_groups = test_seg.groupby("log_id").indices

    boot_means = []
    for b in range(n_boot):
        sampled_cal = RNG.choice(cal_flights, size=len(cal_flights), replace=True)
        sampled_test = RNG.choice(test_flights, size=len(test_flights), replace=True)

        cal_idx = np.concatenate([cal_groups[f] for f in sampled_cal])
        test_idx = np.concatenate([test_groups[f] for f in sampled_test])

        cal_sub = cal_seg.iloc[cal_idx].reset_index(drop=True)
        test_sub = test_seg.iloc[test_idx].reset_index(drop=True)

        covs = []
        for r in RATIOS:
            col = f"score_{r}"
            if col not in cal_sub.columns or col not in test_sub.columns:
                continue
            cal_arr = cal_sub[col].dropna().values
            test_arr = test_sub[col].dropna().values
            if len(cal_arr) < 5 or len(test_arr) < 5:
                continue
            thr = conformal_threshold(cal_arr, ALPHA)
            c = float(np.mean(test_arr <= thr))
            covs.append(c)
        if covs:
            boot_means.append(np.mean(covs))
    return np.array(boot_means)


def main():
    seg = pd.read_csv(SEGMENTS_CSV)
    print(f"loaded {len(seg)} segments from {seg['log_id'].nunique()} flights")
    print()

    cal_bands = ["v1.12-1.13", "v1.14-1.15"]
    test_band = "v1.16+"

    cal_seg = seg[(seg["group"] == "B") & (seg["band"].isin(cal_bands))]
    test_seg = seg[(seg["group"] == "C") & (seg["band"] == test_band)]

    print(f"calibration segments: {len(cal_seg)} from {cal_seg['log_id'].nunique()} flights")
    print(f"test segments:        {len(test_seg)} from {test_seg['log_id'].nunique()} flights")
    print()

    # in-band reference: B v1.16+ calibrate, C v1.16+ test
    inband_cal = seg[(seg["group"] == "B") & (seg["band"] == test_band)]
    print(f"in-band calibration segments: {len(inband_cal)} from {inband_cal['log_id'].nunique()} flights")
    print()

    # cross-band point estimate
    covs_cross = []
    cal_mask = cal_seg.index
    test_mask = test_seg.index
    for r in RATIOS:
        col = f"score_{r}"
        if col not in seg.columns:
            continue
        c = coverage_for_subset(seg, cal_mask, test_mask, col)
        if not np.isnan(c):
            covs_cross.append(c)
    cross_point = np.mean(covs_cross)

    # in-band point estimate
    covs_inband = []
    cal_mask_in = inband_cal.index
    test_mask_in = test_seg.index
    for r in RATIOS:
        col = f"score_{r}"
        if col not in seg.columns:
            continue
        c = coverage_for_subset(seg, cal_mask_in, test_mask_in, col)
        if not np.isnan(c):
            covs_inband.append(c)
    inband_point = np.mean(covs_inband)

    print("=" * 80)
    print("C3 on segments: cluster-robust bootstrap intervals")
    print("=" * 80)
    print()

    # cross-band bootstrap
    print("running cross-band bootstrap (2000 iterations) ...")
    cross_boot = cluster_bootstrap_coverage(seg, cal_bands, test_band, "B", "C")
    cross_lo = np.percentile(cross_boot, 2.5)
    cross_hi = np.percentile(cross_boot, 97.5)
    cross_in = "yes" if cross_lo <= 0.95 <= cross_hi else "NO"

    print(f"  point:       {cross_point:.4f}")
    print(f"  95% CI:      [{cross_lo:.4f}, {cross_hi:.4f}]")
    print(f"  width:       {cross_hi - cross_lo:.4f}")
    print(f"  nominal in:  {cross_in}")
    print()

    # in-band bootstrap
    print("running in-band bootstrap (2000 iterations) ...")
    inband_boot = cluster_bootstrap_coverage(seg, [test_band], test_band, "B", "C")
    inband_lo = np.percentile(inband_boot, 2.5)
    inband_hi = np.percentile(inband_boot, 97.5)
    inband_in = "yes" if inband_lo <= 0.95 <= inband_hi else "NO"

    print(f"  point:       {inband_point:.4f}")
    print(f"  95% CI:      [{inband_lo:.4f}, {inband_hi:.4f}]")
    print(f"  width:       {inband_hi - inband_lo:.4f}")
    print(f"  nominal in:  {inband_in}")
    print()

    # save
    results = pd.DataFrame([
        {"setting": "cross_band", "point": cross_point,
         "ci_low": cross_lo, "ci_high": cross_hi, "width": cross_hi - cross_lo},
        {"setting": "inband_reference", "point": inband_point,
         "ci_low": inband_lo, "ci_high": inband_hi, "width": inband_hi - inband_lo},
    ])
    results.to_csv("c3_segments_cluster.csv", index=False)

    # figure
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Cross-band\n(v1.12-1.15 -> v1.16+)", "In-band reference\n(v1.16+ -> v1.16+)"]
    means = [cross_point, inband_point]
    lows = [cross_lo, inband_lo]
    highs = [cross_hi, inband_hi]
    x = np.arange(2)
    err_low = [means[i] - lows[i] for i in range(2)]
    err_high = [highs[i] - means[i] for i in range(2)]

    ax.bar(x, means, yerr=[err_low, err_high], capsize=12,
           color=["indianred", "steelblue"], edgecolor="black", alpha=0.85)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.4, label="Nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean coverage across 10 ratios", fontsize=11)
    ax.set_title("C3 on segments: cluster-robust bootstrap CI", fontsize=13)
    ax.set_ylim(0.88, 0.99)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m + 0.005, f"{m:.4f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("c3_segments_figure.png", dpi=150)
    plt.show()
    print("saved c3_segments_cluster.csv and c3_segments_figure.png")


if __name__ == "__main__":
    main()