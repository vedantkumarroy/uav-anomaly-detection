"""
c3_figure_cells.py

C3 figure. Cell 1 and cell 2 side by side, with 95% CI error bars.
Nominal 0.95 line.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c3 = pd.read_csv("c3_cell2.csv")
print(c3)

labels = ["Cell 1\ncorpus population", "Cell 2\nnominal population"]
points = c3["point"].values
lo = c3["lo"].values
hi = c3["hi"].values
err = np.vstack([points - lo, hi - points])
ns = c3["n_test"].values
colors = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels))
ax.bar(x, points, yerr=err, capsize=8, color=colors,
       edgecolor="black", linewidth=0.6)
ax.axhline(0.95, color="red", linestyle="--", linewidth=1.2, label="nominal 0.95")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Cross-band coverage")
ax.set_ylim(0.86, 1.00)
ax.set_title("C3: cross-band coverage, flight-block bootstrap CIs")
ax.grid(axis="y", alpha=0.3)
for xi, (p, n) in enumerate(zip(points, ns)):
    ax.text(xi, p - 0.005, f"{p:.4f}\nn={n}", ha="center", va="top", fontsize=9)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("c3_cells_figure.png", dpi=150, bbox_inches="tight")
print("saved c3_cells_figure.png")
