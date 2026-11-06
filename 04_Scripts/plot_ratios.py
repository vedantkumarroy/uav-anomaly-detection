from pyulog import ULog
import matplotlib.pyplot as plt
import numpy as np

LOG = r"pilot\logs\04972883-b141-44c7-aef8-13b983daba55.ulg"

u = ULog(LOG)

# Find the ratios topic
d = next(x for x in u.data_list if x.name == "estimator_innovation_test_ratios")

# Convert timestamp to seconds
t = (d.data["timestamp"] - d.data["timestamp"][0]) / 1e6

# Pick fields to plot
fields = ["gps_hvel[0]", "gps_hvel[1]", "gps_hpos[0]", "gps_hpos[1]", "baro_vpos"]

fig, axes = plt.subplots(len(fields), 1, figsize=(10, 8), sharex=True)

for ax, f in zip(axes, fields):
    ax.plot(t, d.data[f], linewidth=0.8)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.7, label="threshold = 1.0")
    ax.set_ylabel(f)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Estimator Innovation Test Ratios")
plt.tight_layout()
plt.savefig("ratio_plot.png", dpi=120)
plt.show()