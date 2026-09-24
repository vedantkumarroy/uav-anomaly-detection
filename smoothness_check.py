from pyulog import ULog
import numpy as np
import matplotlib.pyplot as plt
import os

# pick one log we have on disk
LOG = "pilot/logs/04972883-b141-44c7-aef8-13b983daba55.ulg"

u = ULog(LOG)

def get(name):
    for d in u.data_list:
        if d.name == name:
            return d
    return None

inn = get("estimator_innovations")
rat = get("estimator_innovation_test_ratios")

if inn is None or rat is None:
    print("missing topics")
    raise SystemExit

# common channel: gps_hvel[0]
field = "gps_hvel[0]"

if field not in inn.data or field not in rat.data:
    print(f"{field} not in both topics. innovation fields:")
    print(sorted([k for k in inn.data.keys() if not k.startswith('timestamp')])[:10])
    print("ratio fields:")
    print(sorted([k for k in rat.data.keys() if not k.startswith('timestamp')])[:10])
    raise SystemExit

t_inn = (inn.data["timestamp"] - inn.data["timestamp"][0]) / 1e6
t_rat = (rat.data["timestamp"] - rat.data["timestamp"][0]) / 1e6
y_inn = np.asarray(inn.data[field], dtype=float)
y_rat = np.asarray(rat.data[field], dtype=float)

# basic stats
def stats(y, label):
    print(f"{label:12s}  n={len(y):5d}  std={np.nanstd(y):.4f}  "
          f"mean={np.nanmean(y):.4f}  max_abs={np.nanmax(np.abs(y)):.4f}")

stats(y_inn, "innovation")
stats(y_rat, "ratio")

# smoothness metric: ratio of mean abs first-difference to std
def roughness(y):
    d = np.abs(np.diff(y))
    return np.nanmean(d) / (np.nanstd(y) + 1e-12)

print(f"\nroughness (mean|Δ|/std):")
print(f"  innovation = {roughness(y_inn):.4f}")
print(f"  ratio      = {roughness(y_rat):.4f}")
print(f"  (lower = smoother)")

# plot
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t_inn, y_inn, linewidth=0.8, color="C0")
axes[0].set_ylabel("innovation")
axes[0].set_title(f"Raw innovation vs test ratio — {field} — {os.path.basename(LOG)}")
axes[0].grid(True)

axes[1].plot(t_rat, y_rat, linewidth=0.8, color="C1")
axes[1].set_ylabel("test ratio")
axes[1].set_xlabel("time (s)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("smoothness_check.png", dpi=120)
plt.show()
print("\nsaved: smoothness_check.png")