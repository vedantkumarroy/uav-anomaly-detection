import numpy as np
import pandas as pd

# load Group C results
c = pd.read_csv("detection_power_c.csv")

# load coverage
cov = pd.read_csv("track_b_coverage.csv")

print("=== Coverage comparison ===")
print(cov.to_string(index=False))
print()

# detection rate for each method at each indicator level
methods = ["flag_track_a", "flag_iso", "flag_svm", "flag_ae"]
labels = ["TrackA", "IsoForest", "OneClassSVM", "Autoencoder"]

print("=== Detection rate by indicator level (Group C) ===")
print()
print(f"{'level':>10s}  {'n_logs':>7s}" + "".join(f"{l:>14s}" for l in labels))
for level in range(6):
    sub = c[c["n_indicators"] == level]
    if len(sub) == 0:
        continue
    line = f"{level:>10d}  {len(sub):>7d}"
    for m in methods:
        rate = sub[m].mean() if len(sub) > 0 else np.nan
        line += f"{rate:>13.3f} "
    print(line)

# Combined (>= 2 and >= 3)
print()
print("=== Detection rate for anomalous subsets ===")
for thr in [2, 3, 4]:
    sub = c[c["n_indicators"] >= thr]
    if len(sub) == 0:
        continue
    print(f"\nn_indicators >= {thr}: {len(sub)} logs")
    for m, lab in zip(methods, labels):
        rate = sub[m].mean()
        n = int(sub[m].sum())
        print(f"  {lab:>14s}: {n} of {len(sub)}  ({100*rate:.1f}%)")

# Save a compact table for the paper
rows = []
for thr in [1, 2, 3, 4]:
    sub = c[c["n_indicators"] >= thr]
    for m, lab in zip(methods, labels):
        if len(sub) == 0:
            continue
        rows.append({
            "method": lab,
            "indicator_threshold": thr,
            "n_candidates": len(sub),
            "n_detected": int(sub[m].sum()),
            "detection_rate": float(sub[m].mean()),
        })
rdf = pd.DataFrame(rows)
rdf.to_csv("detection_power_table.csv", index=False)
print("\nsaved detection_power_table.csv")
print()
print(rdf.round(4).to_string(index=False))