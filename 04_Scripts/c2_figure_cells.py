"""
c2_figure_cells.py

C2 figure. Two panels: cell 1 (corpus) and cell 2 (nominal).
Bars per regime, with 95% CI error bars. Nominal 0.95 line.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c2 = pd.read_csv("c2_cell2.csv")
print(c2.columns.tolist())
print(c2)

regimes = ["hover", "translation", "descent"]
x = np.arange(len(regimes))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, cell, colors in [
    (axes[0], 1, ("#4C72B0",)),
    (axes[1], 2, ("#DD8452",)),
]:
    p = c2[f"cell{cell}_point"].values
    lo = c2[f"cell{cell}_lo"].values
    hi = c2[f"cell{cell}_hi"].values
    err = np.vstack([p - lo, hi - p])
    ax.bar(x, p, yerr=err, capsize=6, color=colors[0],
           edgecolor="black", linewidth=0.6)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.2, label="nominal 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylim(0.88, 1.00)
    ax.set_ylabel("Coverage" if cell == 1 else "")
    title = "Cell 1: dirty B, dirty C\n(corpus population)" if cell == 1 \
            else "Cell 2: clean B, clean C\n(nominal population)"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("C2: regime coverage, flight-block bootstrap CIs")
plt.tight_layout()
plt.savefig("c2_cells_figure.png", dpi=150, bbox_inches="tight")
print("saved c2_cells_figure.png")
