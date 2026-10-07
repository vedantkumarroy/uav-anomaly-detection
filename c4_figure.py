import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("c4_handlabel_final.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: hand-label distribution
ax = axes[0]
counts = df["hand_label"].value_counts().reindex(["genuine", "borderline", "not_genuine"])
colors = ["indianred", "orange", "steelblue"]
ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
for i, v in enumerate(counts.values):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of flights")
ax.set_title("A. Hand-label distribution (top 50 sat_frac flights)")
ax.grid(axis="y", alpha=0.3)

# Panel B: detector agreement (TP, FP, TN, FN)
ax = axes[1]
genuine = df[df["hand_label"] == "genuine"]
not_genuine = df[df["hand_label"] == "not_genuine"]

tp = int(genuine["flag_any"].sum())
fn = len(genuine) - tp
fp = int(not_genuine["flag_any"].sum())
tn = len(not_genuine) - fp

categories = ["TP", "FN", "FP", "TN"]
values = [tp, fn, fp, tn]
colors2 = ["seagreen", "indianred", "orange", "steelblue"]
bars = ax.bar(categories, values, color=colors2, edgecolor="black")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Number of flights")
ax.set_title(f"B. Detector agreement\nPrecision = {tp/(tp+fp):.2f}, Recall = {tp/(tp+fn):.2f}")
ax.grid(axis="y", alpha=0.3)

# Panel C: scatter of sat_frac vs spread_p95, coloured by hand label and flagged
ax = axes[2]
colors3 = {"genuine": "indianred", "borderline": "orange", "not_genuine": "steelblue"}
for label in ["genuine", "borderline", "not_genuine"]:
    sub = df[df["hand_label"] == label]
    ax.scatter(sub["sat_frac"], sub["spread_p95"], c=colors3[label],
               label=label, s=60, edgecolor="black", alpha=0.8)
# mark flagged flights with X
flagged = df[df["flag_any"]]
ax.scatter(flagged["sat_frac"], flagged["spread_p95"], facecolors="none",
           edgecolors="red", s=200, linewidths=2, marker="s", label="flagged")
ax.set_xlabel("sat_frac")
ax.set_ylabel("motor_spread_p95")
ax.set_title("C. sat_frac vs spread, coloured by hand label")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c4_handlabel_figure.png", dpi=150)
plt.show()
print("saved c4_handlabel_figure.png")