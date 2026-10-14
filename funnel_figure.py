"""
funnel_figure.py

Screening funnel from streamed logs to the analysis set.
Values taken from the internship report, Table 1.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

stages = [
    "Sampled after 4001 filter",
    "Valid PX4 firmware band",
    "Rate gates (5/10/1.5 Hz)",
    "Duration >= 60 s",
    "Airborne >= 45 s",
    "Dropout <= 0.02",
    "Analysis set",
]
counts = [2770, 2679, 2524, 2510, 2314, 2168, 1627]

fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(stages))[::-1]
bars = ax.barh(y, counts, color="#4C72B0", edgecolor="black", linewidth=0.6)

for i, (c, b) in enumerate(zip(counts, bars)):
    ax.text(c + 25, b.get_y() + b.get_height() / 2,
            f"{c:,}", va="center", fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels(stages)
ax.set_xlabel("Logs retained")
ax.set_xlim(0, 3050)
ax.set_title("Screening funnel from streamed logs to the analysis set")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("funnel_figure.png", dpi=300, bbox_inches="tight")
print("saved funnel_figure.png")
