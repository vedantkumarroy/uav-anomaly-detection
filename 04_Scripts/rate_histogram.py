import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("sensor_rates.csv")
print(f"rows: {len(df)}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

configs = [
    ("rate_position", "vehicle_local_position (Hz)", axes[0, 0], 10),
    ("rate_attitude", "vehicle_attitude (Hz)", axes[0, 1], 20),
    ("rate_imu", "sensor_combined (Hz)", axes[1, 0], 50),
    ("rate_ratios", "estimator_innovation_test_ratios (Hz)", axes[1, 1], None),
]

for col, title, ax, old_thr in configs:
    vals = df[col].dropna()
    vals = vals[vals > 0]  # drop zeros
    bins = np.linspace(vals.min(), vals.max(), 30)
    ax.hist(vals, bins=bins, edgecolor="black", alpha=0.7)
    if old_thr is not None:
        ax.axvline(old_thr, color="red", linestyle="--",
                   label=f"old threshold = {old_thr} Hz")
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Sample rate (Hz)")
    ax.set_ylabel("Number of logs")

plt.tight_layout()
plt.savefig("rate_histogram.png", dpi=120)
plt.show()
print("saved rate_histogram.png")