import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_2270/logs"
INDEX_CSV = "pilot_2270_index.csv"


def estimate_dropout_fraction(logdir, topic, expected_rate_hz):
    """Estimate dropout fraction from gaps in the timestamp series.
    Compares actual sample count to expected count based on duration and rate.
    """
    p = os.path.join(logdir, topic + ".parquet")
    if not os.path.exists(p):
        return np.nan

    t = pq.read_table(p)
    if "timestamp" not in t.column_names:
        return np.nan

    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if len(ts) < 2:
        return np.nan

    span_s = (ts[-1] - ts[0]) / 1e6
    if span_s <= 0:
        return np.nan

    actual_count = len(ts)
    expected_count = expected_rate_hz * span_s
    if expected_count <= 0:
        return np.nan

    # dropout fraction as fraction of expected samples missing
    missing = max(0.0, expected_count - actual_count)
    return missing / expected_count


df = pd.read_csv(INDEX_CSV)
print(f"rows: {len(df)}")

dropouts = []

for i, row in df.iterrows():
    logdir = os.path.join(LOGDIR, row["log_id"])
    if not os.path.isdir(logdir):
        dropouts.append(np.nan)
        continue

    # use the median rate to estimate expected samples
    rate = row.get("rate_ratios_hz", np.nan)
    if pd.isna(rate) or rate <= 0:
        dropouts.append(np.nan)
        continue

    frac = estimate_dropout_fraction(logdir, "estimator_innovation_test_ratios", rate)
    dropouts.append(frac)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(df)}")

df["dropout_frac_proxy"] = dropouts
df.to_csv(INDEX_CSV, index=False)
print(f"\nupdated {INDEX_CSV}")

vals = df["dropout_frac_proxy"].dropna()
print()
print(f"dropout_frac_proxy: n={len(vals)}, median={vals.median():.4f}, max={vals.max():.4f}")
print(f"  count > 0.02: {int((vals > 0.02).sum())}")