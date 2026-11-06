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

# warm-up exclusion windows to test
WARMUP_WINDOWS = [0, 5, 10, 20, 30]
FLAG_FRACTION = 0.05
PX4_THRESHOLD = 1.0


def logdir_for(log_id):
    p1 = os.path.join("pilot_2270", "logs", log_id)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join("pilot_extra", "logs", log_id)
    if os.path.isdir(p2):
        return p2
    return None


def c1_for_window(df, warmup_s, flag_fraction=0.05):
    """Compute fraction of flights over-flagged with warm-up exclusion."""
    n_over = 0
    n_total = 0
    for i, row in df.iterrows():
        logdir = logdir_for(row["log_id"])
        if logdir is None:
            continue
        p = os.path.join(logdir, "estimator_innovation_test_ratios.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        if "timestamp" not in t.column_names:
            continue

        n_total += 1
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        if len(ts) < 2:
            continue
        # exclude warm-up
        if warmup_s > 0:
            cutoff = ts[0] + warmup_s * 1e6
            mask = ts >= cutoff
        else:
            mask = np.ones(len(ts), dtype=bool)
        if mask.sum() < 5:
            continue

        any_over = False
        for ratio in RATIOS:
            if ratio not in t.column_names:
                continue
            arr = t[ratio].to_numpy(zero_copy_only=False).astype(np.float64)
            arr = arr[mask]
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            if (finite > PX4_THRESHOLD).mean() > flag_fraction:
                any_over = True
                break
        if any_over:
            n_over += 1
    return n_over, n_total


df = pd.read_csv(ANALYSIS_CSV)
print(f"loaded {len(df)} flights")
print()

print("=" * 78)
print("C1 warm-up exclusion test")
print("=" * 78)
print()
print(f"{'warmup_s':>10s}  {'n_flights':>10s}  {'n_over':>8s}  {'frac':>8s}")
print()

results = []
for w in WARMUP_WINDOWS:
    n_over, n_total = c1_for_window(df, w)
    frac = n_over / n_total if n_total > 0 else np.nan
    print(f"{w:>10d}  {n_total:>10d}  {n_over:>8d}  {frac:>8.4f}")
    results.append({"warmup_s": w, "n_flights": n_total, "n_over": n_over, "frac": frac})

rdf = pd.DataFrame(results)
rdf.to_csv("c1_warmup_exclusion.csv", index=False)

# figure
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rdf["warmup_s"], rdf["frac"] * 100, marker="o", color="indianred", linewidth=2)
for _, row in rdf.iterrows():
    ax.text(row["warmup_s"], row["frac"] * 100 + 0.3,
            f"{row['frac']*100:.1f}%", ha="center", fontsize=10)
ax.set_xlabel("Warm-up exclusion (seconds)")
ax.set_ylabel("Flights over-flagged (%)")
ax.set_title("C1: Effect of warm-up exclusion on the false alarm rate")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("c1_warmup_exclusion.png", dpi=150)
plt.show()
print()
print("saved c1_warmup_exclusion.png")