"""
c1_heatmap_clean.py

Regenerate the C1 heatmap on clean flights only.
Same layout as c1_heatmap.py: ratios on y, thresholds on x, one panel per
firmware band.
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS_CSV = "analysis_set_with_log_scores.csv"
SATFRAC_CSV = "analysis_set_with_satfrac.csv"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

THRESHOLDS = [0.5, 1.0, 2.0, 5.0, 10.0]
WARMUP_S = 30.0
BANDS = ["v1.12-1.13", "v1.14-1.15", "v1.16+"]


def logdir_for(log_id):
    for d in ["pilot_2270", "pilot_extra"]:
        p = os.path.join(d, "logs", log_id)
        if os.path.isdir(p):
            return p
    return None


def load_analysis():
    a = pd.read_csv(ANALYSIS_CSV)
    if "sat_frac" in a.columns:
        a["_sat_frac"] = a["sat_frac"]
    elif "sat_frac_recomputed" in a.columns:
        a["_sat_frac"] = a["sat_frac_recomputed"]
    else:
        s = pd.read_csv(SATFRAC_CSV)
        col = "sat_frac_recomputed" if "sat_frac_recomputed" in s.columns else "sat_frac"
        a = a.merge(s[["log_id", col]].rename(columns={col: "_sat_frac"}),
                    on="log_id", how="left")
    return a


def compute_tripped(df):
    return (
        (df["failsafe_frac"] > 0)
        | (df["nav_state_forbidden_frac"] > 0)
        | (df["filter_fault_flags_frac"] > 0)
        | (df["landed_at_end"] == False)
        | (df["_sat_frac"] > 0.05)
    )


def main():
    a = load_analysis()
    a["_tripped"] = compute_tripped(a).values
    clean = a[~a["_tripped"]].copy()
    print(f"clean flights: {len(clean)}")

    per_flight = []
    for i, row in clean.iterrows():
        d = logdir_for(row["log_id"])
        if d is None:
            continue
        p = os.path.join(d, "estimator_innovation_test_ratios.parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        if "timestamp" not in t.column_names:
            continue
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        if len(ts) < 2:
            continue
        keep = (ts - ts[0]) / 1e6 >= WARMUP_S
        if keep.sum() < 10:
            continue
        rec = {"log_id": row["log_id"], "band": row["band"]}
        for r in RATIOS:
            if r not in t.column_names:
                continue
            arr = t[r].to_numpy(zero_copy_only=False).astype(np.float64)[keep]
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            for thr in THRESHOLDS:
                rec[f"rate_{r}_{thr}"] = float((arr > thr).mean())
        per_flight.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(clean)}")

    pdf = pd.DataFrame(per_flight)
    print(f"processed: {len(pdf)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, band in zip(axes, BANDS):
        sub = pdf[pdf["band"] == band]
        M = np.zeros((len(RATIOS), len(THRESHOLDS)))
        for ri, r in enumerate(RATIOS):
            for ti, thr in enumerate(THRESHOLDS):
                col = f"rate_{r}_{thr}"
                if col in sub.columns and len(sub) > 0:
                    frac = float((sub[col].fillna(0.0) > 0.05).mean())
                    M[ri, ti] = frac
        im = ax.imshow(M, aspect="auto", cmap="Reds", vmin=0, vmax=0.20)
        ax.set_xticks(range(len(THRESHOLDS)))
        ax.set_xticklabels([str(t) for t in THRESHOLDS])
        ax.set_yticks(range(len(RATIOS)))
        ax.set_yticklabels(RATIOS if band == BANDS[0] else [""] * len(RATIOS))
        ax.set_xlabel("PX4 threshold")
        ax.set_title(f"{band}  (n={len(sub)})")
    fig.colorbar(im, ax=axes, label="Fraction of clean flights over-flagged")
    fig.suptitle("C1: fraction of clean flights over-flagged, 30 s warm-up excluded")
    plt.tight_layout()
    plt.savefig("c1_heatmap_clean.png", dpi=150, bbox_inches="tight")
    print("saved c1_heatmap_clean.png")

    # Also CSV for the table
    rows = []
    for band in BANDS:
        sub = pdf[pdf["band"] == band]
        for r in RATIOS:
            for thr in THRESHOLDS:
                col = f"rate_{r}_{thr}"
                if col in sub.columns and len(sub) > 0:
                    frac = float((sub[col].fillna(0.0) > 0.05).mean())
                    rows.append({"band": band, "ratio": r, "threshold": thr,
                                 "frac_over": frac, "n": len(sub)})
    pd.DataFrame(rows).to_csv("c1_heatmap_clean.csv", index=False)
    print("saved c1_heatmap_clean.csv")


if __name__ == "__main__":
    main()
