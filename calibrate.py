import os
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
ANALYSIS_CSV = "analysis_set.csv"
RATIO_TOPIC = "estimator_innovation_test_ratios"

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

ALPHA = 0.05
SPLIT_SEED = 42


def score_ratio(logdir, ratio_name, percentile=95):
    """Return the p-th percentile of the ratio over the flight, or NaN."""
    p = os.path.join(logdir, RATIO_TOPIC + ".parquet")
    if not os.path.exists(p):
        return np.nan
    t = pq.read_table(p)
    if ratio_name not in t.column_names:
        return np.nan
    arr = t[ratio_name].to_numpy(zero_copy_only=False).astype(np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, percentile))


def assign_groups(df, seed):
    rng = random.Random(seed)
    df = df.copy()
    df["group"] = None
    for band in df["band"].dropna().unique():
        idx = df[df["band"] == band].index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_a = int(round(0.20 * n))
        n_b = int(round(0.40 * n))
        df.loc[idx[:n_a], "group"] = "A"
        df.loc[idx[n_a:n_a + n_b], "group"] = "B"
        df.loc[idx[n_a + n_b:], "group"] = "C"
    return df


def conformal_threshold(cal_scores, alpha):
    """Split conformal threshold: kth smallest calibration score."""
    s = np.sort(cal_scores)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.nan
    return s[k - 1]


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"analysis set: {len(df)} logs")

    # assign A/B/C per band
    df = assign_groups(df, SPLIT_SEED)
    df.to_csv("analysis_set_with_groups.csv", index=False)
    print(f"groups assigned with seed {SPLIT_SEED}")
    print()
    print("band x group:")
    print(pd.crosstab(df["band"], df["group"]))

    # compute scores per ratio
    print()
    print("computing per-flight scores ...")
    for ratio in RATIOS:
        scores = []
        for i, row in df.iterrows():
            logdir = os.path.join(LOGDIR, row["log_id"])
            scores.append(score_ratio(logdir, ratio))
            if (i + 1) % 200 == 0:
                pass
        df[f"score_{ratio}"] = scores
        valid = sum(1 for s in scores if not np.isnan(s))
        print(f"  {ratio}: {valid} of {len(df)} logs have a valid score")

    df.to_csv("analysis_set_with_scores.csv", index=False)
    print(f"\nsaved analysis_set_with_scores.csv")

    # calibrate and evaluate per ratio
    print()
    print(f"=== Split conformal at alpha = {ALPHA} ===")
    print()

    results = []
    for ratio in RATIOS:
        col = f"score_{ratio}"
        # per-band calibration
        for band in ["v1.12-1.13", "v1.14-1.15", "v1.16+"]:
            sub = df[df["band"] == band]
            cal = sub[sub["group"] == "B"][col].dropna().values
            test = sub[sub["group"] == "C"][col].dropna().values
            if len(cal) < 5 or len(test) < 5:
                continue
            threshold = conformal_threshold(cal, ALPHA)
            coverage = float(np.mean(test <= threshold))
            results.append({
                "ratio": ratio,
                "band": band,
                "n_cal": len(cal),
                "n_test": len(test),
                "threshold": threshold,
                "coverage": coverage,
                "nominal": 1 - ALPHA,
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv("coverage_per_ratio_per_band.csv", index=False)

    print("=== Coverage per ratio per band ===")
    print()
    pivot = rdf.pivot(index="ratio", columns="band", values="coverage")
    pivot["mean"] = pivot.mean(axis=1)
    print(pivot.round(4).to_string())

    print()
    print(f"nominal = {1 - ALPHA}")
    print(f"overall mean coverage = {rdf['coverage'].mean():.4f}")
    print(f"min coverage = {rdf['coverage'].min():.4f}")
    print(f"max coverage = {rdf['coverage'].max():.4f}")

    # pooled coverage: calibrate on all B, test on all C
    print()
    print("=== Pooled (all B calibrate, all C test) ===")
    pooled = []
    for ratio in RATIOS:
        col = f"score_{ratio}"
        cal = df[df["group"] == "B"][col].dropna().values
        test = df[df["group"] == "C"][col].dropna().values
        if len(cal) < 5 or len(test) < 5:
            continue
        threshold = conformal_threshold(cal, ALPHA)
        coverage = float(np.mean(test <= threshold))
        pooled.append({
            "ratio": ratio,
            "n_cal": len(cal),
            "n_test": len(test),
            "threshold": threshold,
            "coverage": coverage,
        })
    pdf = pd.DataFrame(pooled)
    pdf.to_csv("coverage_per_ratio_pooled.csv", index=False)
    print(pdf.round(4).to_string(index=False))
    print()
    print(f"pooled mean coverage = {pdf['coverage'].mean():.4f}")


if __name__ == "__main__":
    main()