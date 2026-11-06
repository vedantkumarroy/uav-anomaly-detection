"""
split_figure.py

Corpus composition by firmware band and group.
Values taken from the internship report, Table 2.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

bands = ["v1.12-1.13", "v1.14-1.15", "v1.16+"]
group_a = [51, 180, 94]
group_b = [102, 360, 188]
group_c = [103, 360, 189]

x = np.arange(len(bands))
w = 0.26

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w, group_a, w, label="Group A (train)", color="#4C72B0",
            edgecolor="black", linewidth=0.6)
b2 = ax.bar(x,     group_b, w, label="Group B (calibrate)", color="#DD8452",
            edgecolor="black", linewidth=0.6)
b3 = ax.bar(x + w, group_c, w, label="Group C (test)", color="#55A868",
            edgecolor="black", linewidth=0.6)

for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 6,
                f"{int(b.get_height())}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_ylabel("Flights")
ax.set_title("Corpus composition by firmware band and group")
ax.set_ylim(0, 410)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("split_figure.png", dpi=300, bbox_inches="tight")
print("saved split_figure.png")
