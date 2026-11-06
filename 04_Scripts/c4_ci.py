import numpy as np
import pandas as pd
from scipy import stats

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# Data from the combined 1627 run, track_b_v3 output
# Format: (method, threshold, n_detected, n_total)
data = [
    ("IsoForest", 1, 21, 264),
    ("IsoForest", 2, 13, 94),
    ("IsoForest", 3, 5, 24),
    ("IsoForest", 4, 2, 11),
    ("SVM", 1, 24, 264),
    ("SVM", 2, 14, 94),
    ("SVM", 3, 4, 24),
    ("SVM", 4, 3, 11),
    ("Autoencoder", 1, 27, 264),
    ("Autoencoder", 2, 15, 94),
    ("Autoencoder", 3, 4, 24),
    ("Autoencoder", 4, 2, 11),
]

print("=" * 92)
print("C4: Detection power with 95% Wilson confidence intervals")
print("=" * 92)
print()
print(f"{'method':>14s}  {'thr':>4s}  {'k/n':>8s}  {'rate':>8s}  {'CI_low':>8s}  "
      f"{'CI_high':>8s}  {'CI_width':>9s}")

rows = []
for method, thr, k, n in data:
    lo, hi = wilson_ci(k, n)
    width = hi - lo
    print(f"{method:>14s}  {thr:>4d}  {k:>3d}/{n:<4d}  {k/n:>8.4f}  {lo:>8.4f}  "
          f"{hi:>8.4f}  {width:>9.4f}")
    rows.append({
        "method": method, "threshold": thr, "n_detected": k, "n_total": n,
        "rate": k/n, "ci_low": lo, "ci_high": hi, "ci_width": width,
    })

df = pd.DataFrame(rows)
df.to_csv("c4_detection_power_ci.csv", index=False)

# Pairwise overlap at each threshold
print()
print("=" * 92)
print("Pairwise CI overlap at each threshold (do the CIs overlap?)")
print("=" * 92)
print()

methods = ["IsoForest", "SVM", "Autoencoder"]
for thr in [2, 3, 4]:
    print(f"--- n_indicators >= {thr} ---")
    sub = df[df["threshold"] == thr]
    print(f"  {'method':>14s}  {'rate':>8s}  {'CI':>20s}")
    for _, r in sub.iterrows():
        print(f"  {r['method']:>14s}  {r['rate']:>8.4f}  "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print()
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            r1 = sub[sub["method"] == m1].iloc[0]
            r2 = sub[sub["method"] == m2].iloc[0]
            overlap = (r1["ci_low"] <= r2["ci_high"]) and (r2["ci_low"] <= r1["ci_high"])
            mark = "OVERLAP" if overlap else "NO OVERLAP"
            print(f"    {m1:>14s} vs {m2:>14s}: {mark}")
    print()

print("=" * 92)
print("Rule: if the CIs overlap, the difference is not statistically significant")
print("at the 95% level with this sample size.")
print("=" * 92)