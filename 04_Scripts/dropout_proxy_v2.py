import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
INDEX_CSV = "pilot_2270_index.csv"

# topics to check for gaps
TOPICS = ["vehicle_attitude", "vehicle_local_position", "sensor_combined",
          "estimator_innovation_test_ratios"]

# gap threshold: how many times the median gap counts as a dropout
GAP_MULTIPLIER = 10.0


def compute_dropout_proxy(logdir):
    """Compute dropout fraction from timestamp gaps.
    For each topic, compute fraction of time spent in gaps > GAP_MULTIPLIER * median gap.
    Take the maximum across topics.
    """
    worst_frac = 0.0
    worst_topic = None
    for topic in TOPICS:
        p = os.path.join(logdir, topic + ".parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        if "timestamp" not in t.column_names:
            continue
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        if len(ts) < 10:
            continue

        # inter-sample gaps in microseconds
        gaps = np.diff(ts)
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            continue

        median_gap = np.median(gaps)
        if median_gap <= 0:
            continue

        threshold = GAP_MULTIPLIER * median_gap
        large_gaps = gaps > threshold
        if not large_gaps.any():
            continue

        # fraction of total duration spent in large gaps
        total_span = ts[-1] - ts[0]
        large_gap_time = gaps[large_gaps].sum()
        frac = large_gap_time / total_span

        if frac > worst_frac:
            worst_frac = frac
            worst_topic = topic

    return worst_frac, worst_topic


def main():
    df = pd.read_csv(INDEX_CSV)
    print(f"rows: {len(df)}")

    dropouts = []
    topics = []

    for i, row in df.iterrows():
        logdir = os.path.join(LOGDIR, row["log_id"])
        if not os.path.isdir(logdir):
            dropouts.append(np.nan)
            topics.append(None)
            continue
        try:
            frac, topic = compute_dropout_proxy(logdir)
        except Exception:
            frac, topic = np.nan, None
        dropouts.append(frac)
        topics.append(topic)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)}")

    df["dropout_frac_v2"] = dropouts
    df["dropout_worst_topic"] = topics

    df.to_csv(INDEX_CSV, index=False)
    print(f"\nupdated {INDEX_CSV}")

    vals = df["dropout_frac_v2"].dropna()
    print()
    print(f"dropout_frac_v2:")
    print(f"  n:        {len(vals)}")
    print(f"  median:   {vals.median():.4f}")
    print(f"  mean:     {vals.mean():.4f}")
    print(f"  max:      {vals.max():.4f}")
    print(f"  > 0.01:   {int((vals > 0.01).sum())}")
    print(f"  > 0.02:   {int((vals > 0.02).sum())}")
    print(f"  > 0.05:   {int((vals > 0.05).sum())}")
    print()
    print("worst topics distribution:")
    print(df["dropout_worst_topic"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()