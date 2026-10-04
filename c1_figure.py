import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv("c1_summary.csv")

# sort by frac_gt_5pct
summary = summary.sort_values("frac_gt_5pct", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: mean false alarm rate
ax = axes[0]
ax.barh(summary["ratio"], summary["mean_rate"] * 100, color="steelblue", edgecolor="black")
ax.set_xlabel("Mean false alarm rate (%)")
ax.set_title("A. Mean fraction of samples above PX4 threshold = 1.0")
ax.grid(axis="x", alpha=0.3)
for i, v in enumerate(summary["mean_rate"] * 100):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

# Panel B: fraction of flights with >5% samples flagged
ax = axes[1]
ax.barh(summary["ratio"], summary["frac_gt_5pct"] * 100, color="indianred", edgecolor="black")
ax.set_xlabel("Fraction of flights (%)")
ax.set_title("B. Flights with >5% of samples above PX4 threshold")
ax.grid(axis="x", alpha=0.3)
for i, v in enumerate(summary["frac_gt_5pct"] * 100):
    ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("c1_figure.png", dpi=150)
plt.show()
print("saved c1_figure.png")

# also a simple table
print()
print("=== C1 summary table ===")
print(summary[["ratio", "mean_rate", "p95_rate", "frac_gt_1pct", "frac_gt_5pct", "frac_gt_10pct"]]
      .round(4).to_string(index=False))