"""
Named finding: eb456400

179 seconds of sustained motor imbalance. Ratio traces stay flat.
Motor commands diverge.

Top panel: 10 ratio traces on one axis.
Bottom panel: 4 motor commands on same time axis.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

LOG_ID = "eb456400-8803-4062-b937-ea4b4320bb59"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]


def logdir_for(log_id):
    for base in ["pilot_2270", "pilot_extra"]:
        p = os.path.join(base, "logs", log_id)
        if os.path.isdir(p):
            return p
    return None


def main():
    logdir = logdir_for(LOG_ID)
    if logdir is None:
        print("log not found")
        return
    print(f"logdir: {logdir}")

    # ratios
    rat = pq.read_table(os.path.join(logdir, "estimator_innovation_test_ratios.parquet"))
    rat_ts = rat["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    rat_t = (rat_ts - rat_ts[0]) / 1e6

    # motors
    mot = pq.read_table(os.path.join(logdir, "actuator_motors.parquet"))
    mot_ts = mot["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    mot_t = (mot_ts - mot_ts[0]) / 1e6

    motor_cols = sorted([c for c in mot.column_names if c.startswith("control[")])
    motor_cols = [c for c in motor_cols if not np.isnan(mot[c].to_numpy(zero_copy_only=False).astype(np.float64)).all()]
    motor_cols = motor_cols[:4]
    print(f"motor columns: {motor_cols}")

    # figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # top: ratio traces
    ax = axes[0]
    for r in RATIOS:
        if r not in rat.column_names:
            continue
        arr = rat[r].to_numpy(zero_copy_only=False).astype(np.float64)
        ax.plot(rat_t, arr, linewidth=0.8, label=r, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="PX4 threshold")
    ax.set_ylabel("Innovation test ratio", fontsize=11)
    ax.set_title(f"eb456400: 179 s of motor imbalance, zero ratio response", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 2.0)

    # bottom: motor commands
    ax = axes[1]
    for c in motor_cols:
        arr = mot[c].to_numpy(zero_copy_only=False).astype(np.float64)
        ax.plot(mot_t, arr, linewidth=1.0, label=c)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1, label="saturation threshold 0.95")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Motor command (normalised)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig("eb456400_figure.png", dpi=150)
    plt.show()
    print("saved eb456400_figure.png")


if __name__ == "__main__":
    main()