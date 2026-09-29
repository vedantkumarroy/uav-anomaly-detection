import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR = "pilot_500_v2/logs"
INDEX_CSV = "pilot_500_v2_index.csv"

MOTOR_HIGH = 0.95
MOTOR_LOW = 0.05
OUTPUT_HIGH = 1950
OUTPUT_LOW = 1050


def to_float_array(col):
    """Convert a PyArrow column to a float64 numpy array with NaN for nulls."""
    try:
        arr = col.to_numpy(zero_copy_only=False)
    except Exception:
        return None
    try:
        arr = arr.astype(np.float64, copy=False)
    except Exception:
        return None
    return arr


def active_columns(table, prefix):
    """Return column names starting with prefix that are ever non-zero."""
    cols = []
    for name in table.column_names:
        if not name.startswith(prefix):
            continue
        arr = to_float_array(table[name])
        if arr is None or len(arr) == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        if np.max(np.abs(finite)) == 0:
            continue
        cols.append(name)
    return cols


def compute_saturation(logdir):
    motors_path = os.path.join(logdir, "actuator_motors.parquet")
    outputs_path = os.path.join(logdir, "actuator_outputs.parquet")

    if os.path.exists(motors_path):
        t = pq.read_table(motors_path)
        cols = active_columns(t, "control[")
        if not cols:
            return np.nan
        arrays = [to_float_array(t[c]) for c in cols]
        arrays = [a for a in arrays if a is not None]
        if not arrays:
            return np.nan
        arr = np.column_stack(arrays)
        hi, lo = MOTOR_HIGH, MOTOR_LOW
    elif os.path.exists(outputs_path):
        t = pq.read_table(outputs_path)
        cols = active_columns(t, "output[")
        if not cols:
            return np.nan
        arrays = [to_float_array(t[c]) for c in cols]
        arrays = [a for a in arrays if a is not None]
        if not arrays:
            return np.nan
        arr = np.column_stack(arrays)
        hi, lo = OUTPUT_HIGH, OUTPUT_LOW
    else:
        return np.nan

    if arr.size == 0:
        return np.nan

    saturated = np.any((arr >= hi) | (arr <= lo), axis=1)
    # drop rows where every column is NaN
    valid_rows = ~np.isnan(arr).all(axis=1)
    saturated = saturated[valid_rows]
    if len(saturated) == 0:
        return np.nan
    return float(np.mean(saturated))


def main():
    df = pd.read_csv(INDEX_CSV)
    print(f"rows: {len(df)}")

    new_vals = []
    for i, row in df.iterrows():
        log_id = row["log_id"]
        logdir = os.path.join(LOGDIR, log_id)
        if not os.path.isdir(logdir):
            new_vals.append(np.nan)
            continue
        try:
            v = compute_saturation(logdir)
        except Exception as e:
            v = np.nan
        new_vals.append(v)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")

    df["actuator_saturation_frac"] = new_vals
    df["landed_at_end"] = df["landed_at_end"].astype("boolean")

    df.to_csv(INDEX_CSV, index=False)
    print(f"\nupdated {INDEX_CSV}")

    vals = df["actuator_saturation_frac"].dropna()
    print()
    print(f"actuator_saturation_frac:")
    print(f"  count > 0: {int((vals > 0).sum())} of {len(vals)}")
    print(f"  median:    {vals.median():.4f}")
    print(f"  mean:      {vals.mean():.4f}")
    print(f"  max:       {vals.max():.4f}")

    landed = df["landed_at_end"].dropna().astype(bool)
    print()
    print(f"landed_at_end:")
    print(f"  True:  {int(landed.sum())} of {len(landed)}")
    print(f"  False: {int((~landed).sum())} of {len(landed)}")


if __name__ == "__main__":
    main()