import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("regime_segments.csv")
print(f"total segments: {len(df)}")

counts = df.groupby("log_id").size()
print(f"flights with at least one segment: {len(counts)}")
print()
print("=== Segments per flight ===")
print(f"  mean:   {counts.mean():.2f}")
print(f"  median: {counts.median():.0f}")
print(f"  min:    {counts.min()}")
print(f"  max:    {counts.max()}")
print(f"  p25:    {counts.quantile(0.25):.0f}")
print(f"  p75:    {counts.quantile(0.75):.0f}")
print(f"  p90:    {counts.quantile(0.90):.0f}")
print(f"  p95:    {counts.quantile(0.95):.0f}")
print(f"  p99:    {counts.quantile(0.99):.0f}")
print()

print("=== Histogram ===")
bins = [0, 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
hist, _ = np.histogram(counts, bins=bins)
for i in range(len(bins)-1):
    if hist[i] == 0:
        continue
    print(f"  {bins[i]:>5d}-{bins[i+1]:<5d}: {hist[i]:>5d} flights ({100*hist[i]/len(counts):.1f}%)")

print()
print("=== Top 20 flights by segment count ===")
top = counts.sort_values(ascending=False).head(20)
for log_id, c in top.items():
    print(f"  {log_id[:20]:>20s}: {c} segments")

print()
print("=== Concentration: what fraction of segments come from the top X% of flights? ===")
sorted_counts = counts.sort_values(ascending=False)
total_segments = counts.sum()
for frac in [0.01, 0.05, 0.10, 0.20, 0.50]:
    n = int(len(sorted_counts) * frac)
    seg_from_top = sorted_counts.head(n).sum()
    print(f"  top {int(frac*100):>2d}% ({n:>4d} flights): "
          f"{seg_from_top:>5d} segments ({100*seg_from_top/total_segments:.1f}% of all)")

# figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(counts, bins=50, edgecolor="black")
axes[0].set_xlabel("Segments per flight")
axes[0].set_ylabel("Count of flights")
axes[0].set_title(f"Segments per flight (n={len(counts)}, median={counts.median():.0f}, max={counts.max()})")
axes[0].grid(alpha=0.3)

sorted_vals = np.sort(counts)
cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
axes[1].plot(sorted_vals, cdf)
axes[1].set_xlabel("Segments per flight")
axes[1].set_ylabel("Cumulative fraction of flights")
axes[1].set_title("CDF of segments per flight")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("segments_per_flight.png", dpi=150)
plt.show()
print()
print("saved segments_per_flight.png")