import numpy as np
import pandas as pd
from scipy import stats

cross = pd.read_csv("c3_cross_band.csv")
inband = pd.read_csv("c3_inband.csv")
weighted = pd.read_csv("c3_weighted.csv")

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)

print("=" * 78)
print("C3: Coverage with 95% Wilson confidence intervals")
print("=" * 78)
print()
print(f"{'ratio':>18s}  {'setting':>12s}  {'cov':>8s}  {'CI_low':>8s}  {'CI_high':>8s}  "
      f"{'nominal_inside':>15s}")

for df, label in [(cross, "cross"), (inband, "inband"), (weighted, "weighted")]:
    for _, row in df.iterrows():
        ratio = row["ratio"]
        n = int(row["n_test"])
        k = int(round(row["coverage"] * n))
        lo, hi = wilson_ci(k, n)
        nominal_in = "yes" if lo <= 0.95 <= hi else "NO"
        print(f"{ratio:>18s}  {label:>12s}  {row['coverage']:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  "
              f"{nominal_in:>15s}")

# aggregate
print()
print("=" * 78)
print("Aggregate means with CI")
print("=" * 78)
print()
for df, label in [(cross, "cross"), (inband, "inband"), (weighted, "weighted")]:
    n_total = df["n_test"].sum()
    k_total = int(round((df["coverage"] * df["n_test"]).sum()))
    mean = df["coverage"].mean()
    lo, hi = wilson_ci(k_total, n_total)
    print(f"{label:>12s}: mean={mean:.4f}  pooled_count={k_total}/{n_total}  "
          f"CI=[{lo:.4f}, {hi:.4f}]  nominal_in={'yes' if lo<=0.95<=hi else 'NO'}")