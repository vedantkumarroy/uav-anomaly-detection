import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rdf = pd.read_csv("regime_c2_coverage.csv")
print(f"loaded {len(rdf)} rows")

# aggregate per regime
agg = rdf.groupby("regime").agg(
    mean_cov=("coverage", "mean"),
    n_cal=("n_cal", "sum"),
    n_test=("n_test", "sum"),
).reset_index()

# compute Wilson CI on pooled counts
from scipy import stats

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

for _, row in agg.iterrows():
    k = int(round(row["mean_cov"] * row["n_test"]))
    lo, hi = wilson_ci(k, int(row["n_test"]))
    agg.loc[agg["regime"] == row["regime"], "ci_low"] = lo
    agg.loc[agg["regime"] == row["regime"], "ci_high"] = hi

# also pooled
pooled_cal = rdf["n_cal"].sum()
pooled_test = rdf["n_test"].sum()
pooled_mean = rdf["coverage"].mean()

print()
print("Regime coverage summary:")
for _, row in agg.iterrows():
    print(f"  {row['regime']:>14s}: mean={row['mean_cov']:.4f}  "
          f"CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")

# figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# panel A: bar chart per regime with CIs
ax = axes[0]
regimes = ["hover", "translation", "descent"]
means = [agg[agg["regime"] == r]["mean_cov"].values[0] for r in regimes]
lows = [agg[agg["regime"] == r]["ci_low"].values[0] for r in regimes]
highs = [agg[agg["regime"] == r]["ci_high"].values[0] for r in regimes]

x = np.arange(len(regimes))
err_low = [means[i] - lows[i] for i in range(len(regimes))]
err_high = [highs[i] - means[i] for i in range(len(regimes))]

ax.bar(x, means, yerr=[err_low, err_high], capsize=8, color=["steelblue", "indianred", "seagreen"],
       edgecolor="black", alpha=0.85)
ax.axhline(0.95, color="black", linestyle="--", linewidth=1.2, label="Nominal 0.95")
ax.set_xticks(x)
ax.set_xticklabels(regimes, fontsize=11)
ax.set_ylabel("Mean coverage (across 10 ratios)", fontsize=11)
ax.set_title("A. Regime-level coverage with 95% Wilson CI", fontsize=12)
ax.set_ylim(0.90, 0.98)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

for i, m in enumerate(means):
    ax.text(i, m + 0.003, f"{m:.4f}", ha="center", fontsize=9, fontweight="bold")

# panel B: histogram of per-ratio coverage per regime
ax = axes[1]
regimes_order = ["hover", "translation", "descent"]
colors = {"hover": "steelblue", "translation": "indianred", "descent": "seagreen"}
for regime in regimes_order:
    sub = rdf[rdf["regime"] == regime]
    ax.hist(sub["coverage"], bins=20, alpha=0.6, label=f"{regime} (n={len(sub)})",
            color=colors[regime], edgecolor="black", range=(0.88, 1.0))
ax.axvline(0.95, color="black", linestyle="--", linewidth=1.2, label="Nominal 0.95")
ax.set_xlabel("Per-ratio coverage", fontsize=11)
ax.set_ylabel("Count of ratios", fontsize=11)
ax.set_title("B. Distribution of per-ratio coverage within each regime", fontsize=12)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("c2_regime_figure.png", dpi=150)
plt.show()
print()
print("saved c2_regime_figure.png")

# also build a small per-band breakdown of C2
rdf_pb = rdf.copy()
rdf_pb["band_from_segments"] = None  # placeholder
# save summary
agg.to_csv("c2_regime_summary.csv", index=False)
print("saved c2_regime_summary.csv")