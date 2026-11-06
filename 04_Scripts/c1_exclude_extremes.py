import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

PX4_THRESHOLD = 1.0
FLAG_FRACTION = 0.05
EXTREME_CUTOFFS = [1e6, 1e9, 1e12]


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


def c1_with_cutoff(df, extreme_cutoff):
    n_over = 0
    n_total = 0
    for _, row in df.iterrows():
        logdir = logdir_for(row["log_id"])
        if logdir is None:
            continue
        p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        n_total += 1
        over = False
        for ratio in RATIOS:
            if ratio not in t.column_names:
                continue
            arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            if extreme_cutoff is not None:
                finite = finite[finite <= extreme_cutoff]
            if len(finite) == 0:
                continue
            if (finite > PX4_THRESHOLD).mean() > FLAG_FRACTION:
                over = True
                break
        if over:
            n_over += 1
    return n_over, n_total


df = pd.read_csv(ANALYSIS_CSV)
print(f"loaded {len(df)} flights")
print()
print("=" * 78)
print("C1 after excluding extreme samples")
print("=" * 78)
print()
print(f"{'extreme_cutoff':>16s}  {'n_flights':>10s}  {'n_over':>8s}  {'frac':>8s}")

results = []
configs = [None, 1e12, 1e9, 1e6]
labels = ["no exclusion", "exclude > 1e12", "exclude > 1e9", "exclude > 1e6"]
for config, label in zip(configs, labels):
    n_over, n_total = c1_with_cutoff(df, config)
    frac = n_over / n_total if n_total > 0 else np.nan
    print(f"{label:>16s}  {n_total:>10d}  {n_over:>8d}  {frac:>8.4f}")
    results.append({"config": label, "extreme_cutoff": config if config else 0,
                    "n_flights": n_total, "n_over": n_over, "frac": frac})

rdf = pd.DataFrame(results)
rdf.to_csv("c1_exclude_extremes.csv", index=False)
print()
print("saved c1_exclude_extremes.csv")

# figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(rdf["config"], rdf["frac"] * 100, color="indianred", edgecolor="black", alpha=0.85)
for i, v in enumerate(rdf["frac"] * 100):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Flights over-flagged (%)", fontsize=11)
ax.set_title("C1: effect of excluding extreme ratio samples", fontsize=13)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 25)
plt.tight_layout()
plt.savefig("c1_exclude_extremes_figure.png", dpi=150)
plt.show()
print("saved c1_exclude_extremes_figure.png")