import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ANALYSIS_CSV = "combined_analysis_set.csv"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 10.0]
BANDS = ["v1.12-1.13", "v1.14-1.15", "v1.16+"]


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


df = pd.read_csv(ANALYSIS_CSV)
print(f"loaded {len(df)}")

# Compute per-flight per-ratio per-threshold flag rates
# Store as: dict[(ratio, band)] -> list of flag rates per threshold
from collections import defaultdict
data = defaultdict(lambda: defaultdict(list))  # data[band][ratio] = list of per-flight [rate_1, rate_2, rate_3, rate_5, rate_10]

for i, row in df.iterrows():
    logdir = logdir_for(row["log_id"])
    if logdir is None:
        continue
    p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
    if not os.path.exists(p):
        continue
    t = pq.read_table(p)
    band = row["band"]
    for ratio in RATIOS:
        if ratio not in t.column_names:
            continue
        arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        rates = [float((finite > thr).mean()) for thr in THRESHOLDS]
        data[band][ratio].append(rates)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)}")

# Build matrices: rows = ratios, cols = thresholds
matrices = {}
for band in BANDS:
    m = np.zeros((len(RATIOS), len(THRESHOLDS)))
    for ri, ratio in enumerate(RATIOS):
        arr = np.array(data[band][ratio])
        if arr.size == 0:
            m[ri, :] = np.nan
            continue
        for ti in range(len(THRESHOLDS)):
            m[ri, ti] = float((arr[:, ti] > 0.05).mean()) * 100
    matrices[band] = m

# Save raw matrices
for band in BANDS:
    mdf = pd.DataFrame(matrices[band], index=RATIOS, columns=[f">{t}" for t in THRESHOLDS])
    mdf.to_csv(f"c1_heatmap_{band.replace('.','_')}.csv")

# Figure: 3 panels side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

vmin = 0
vmax = 20

for ax, band in zip(axes, BANDS):
    m = matrices[band]
    im = ax.imshow(m, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels([f"{t}" for t in THRESHOLDS])
    ax.set_yticks(range(len(RATIOS)))
    ax.set_yticklabels(RATIOS, fontsize=9)
    ax.set_xlabel("Threshold")
    ax.set_title(band, fontsize=12)

    for ri in range(len(RATIOS)):
        for ti in range(len(THRESHOLDS)):
            v = m[ri, ti]
            if np.isnan(v):
                ax.text(ti, ri, "-", ha="center", va="center", fontsize=8, color="gray")
            else:
                color = "white" if v > 12 else "black"
                ax.text(ti, ri, f"{v:.1f}", ha="center", va="center", fontsize=8, color=color)

cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Percent of flights with >5% of samples above threshold")

fig.suptitle("C1: PX4 EKF false alarm rate by ratio and threshold", fontsize=14)
plt.tight_layout()
plt.savefig("c1_heatmap.png", dpi=150)
plt.show()
print()
print("saved c1_heatmap.png")

# Print summary
print()
print("=== Heatmap values (percent of flights over-flagged) ===")
for band in BANDS:
    print()
    print(f"--- {band} ---")
    m = matrices[band]
    print(f"{'ratio':>18s}" + "".join(f"{t:>8.1f}" for t in THRESHOLDS))
    for ri, ratio in enumerate(RATIOS):
        line = f"{ratio:>18s}"
        for ti in range(len(THRESHOLDS)):
            v = m[ri, ti]
            line += f"{v:>8.1f}" if not np.isnan(v) else f"{'-':>8s}"
        print(line)