import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("per_log_with_dropout.csv")
df = df[df["parse_ok"]]

r = df["rate_ratios"].dropna()
r_pos = r[r > 0]

print(f"total logs:          {len(r)}")
print(f"logs with rate>0:    {len(r_pos)}")
print(f"logs with rate==0:   {int((r == 0).sum())}")
print()

print("=== ratio rate distribution (Hz), rate>0 only ===")
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    print(f"  p{int(q*100):02d} = {r_pos.quantile(q):.2f}")

print(f"  min = {r_pos.min():.2f}")
print(f"  max = {r_pos.max():.2f}")

print()
print("=== proposed gate sweep ===")
for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 5.0]:
    passed = int((r >= thr).sum())
    print(f"  rate >= {thr:.1f} Hz: {passed} of {len(r)} keep "
          f"({100*passed/len(r):.1f}%)")

# histogram
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(0, 130, 2)
ax.hist(r, bins=bins, edgecolor="black", alpha=0.75)
ax.set_xlabel("ratio logging rate (Hz)")
ax.set_ylabel("number of logs")
ax.set_title(f"estimator_innovation_test_ratios sample rate (n={len(r)})")
ax.axvline(1.5, color="red", linestyle="--", label="proposed gate = 1.5 Hz")
ax.legend()
plt.tight_layout()
plt.savefig("ratio_rate_hist.png", dpi=120)
plt.show()
print("\nsaved ratio_rate_hist.png")