import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOGDIR_MAIN = "pilot_2270/logs"
LOGDIR_EXTRA = "pilot_extra/logs"
INDEX_MAIN = "pilot_2270_index.csv"
INDEX_EXTRA = "pilot_extra_index.csv"

GAP_MULTIPLIER = 10.0
TOPICS_FOR_DROPOUT = ["vehicle_attitude", "vehicle_local_position",
                      "sensor_combined", "estimator_innovation_test_ratios"]

RATE_POSITION_MIN = 5.0
RATE_ATTITUDE_MIN = 10.0
RATE_RATIOS_MIN = 1.5
DURATION_MIN = 60.0
AIRBORNE_MIN = 45.0
DROPOUT_MAX = 0.02


def compute_dropout_v2(logdir):
    worst_frac = 0.0
    for topic in TOPICS_FOR_DROPOUT:
        p = os.path.join(logdir, topic + ".parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        if "timestamp" not in t.column_names:
            continue
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        if len(ts) < 10:
            continue
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
        total_span = ts[-1] - ts[0]
        frac = gaps[large_gaps].sum() / total_span
        if frac > worst_frac:
            worst_frac = frac
    return worst_frac


def compute_rate(logdir, topic):
    p = os.path.join(logdir, topic + ".parquet")
    if not os.path.exists(p):
        return 0.0
    t = pq.read_table(p)
    if "timestamp" not in t.column_names:
        return 0.0
    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if len(ts) < 2:
        return 0.0
    span = (ts[-1] - ts[0]) / 1e6
    if span <= 0:
        return 0.0
    return (len(ts) - 1) / span


def compute_duration(logdir):
    for topic in ["vehicle_status", "vehicle_attitude", "vehicle_local_position", "sensor_combined"]:
        p = os.path.join(logdir, topic + ".parquet")
        if not os.path.exists(p):
            continue
        t = pq.read_table(p)
        if "timestamp" not in t.column_names:
            continue
        ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        if len(ts) >= 2:
            return (ts[-1] - ts[0]) / 1e6
    return 0.0


def compute_airborne(logdir):
    p = os.path.join(logdir, "vehicle_land_detected.parquet")
    if not os.path.exists(p):
        return 0.0
    t = pq.read_table(p)
    if "landed" not in t.column_names or "timestamp" not in t.column_names:
        return 0.0
    ts = t["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    landed = t["landed"].to_numpy(zero_copy_only=False).astype(bool)
    if len(ts) < 2:
        return 0.0
    air = 0.0
    for i in range(len(ts) - 1):
        if not landed[i]:
            air += (ts[i + 1] - ts[i]) / 1e6
    return air


def enrich(df, logdir):
    dropouts, rpos, ratt, rrat, dur, air = [], [], [], [], [], []
    for i, row in df.iterrows():
        d = os.path.join(logdir, row["log_id"])
        if not os.path.isdir(d):
            for lst in [dropouts, rpos, ratt, rrat, dur, air]:
                lst.append(np.nan)
            continue
        try:
            dropouts.append(compute_dropout_v2(d))
        except Exception:
            dropouts.append(np.nan)
        rpos.append(compute_rate(d, "vehicle_local_position"))
        ratt.append(compute_rate(d, "vehicle_attitude"))
        rrat.append(compute_rate(d, "estimator_innovation_test_ratios"))
        dur.append(compute_duration(d))
        air.append(compute_airborne(d))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}")
    df = df.copy()
    df["dropout_frac_v2"] = dropouts
    df["rate_position_hz"] = rpos
    df["rate_attitude_hz"] = ratt
    df["rate_ratios_hz"] = rrat
    df["duration_s"] = dur
    df["airborne_s"] = air
    return df


print("loading main index ...")
main = pd.read_csv(INDEX_MAIN)
print(f"main: {len(main)}")
print("computing gates for main ...")
main = enrich(main, LOGDIR_MAIN)
main.to_csv("pilot_2270_index_enriched.csv", index=False)

print("loading extra index ...")
extra = pd.read_csv(INDEX_EXTRA)
print(f"extra: {len(extra)}")
print("computing gates for extra ...")
extra = enrich(extra, LOGDIR_EXTRA)
extra.to_csv("pilot_extra_index_enriched.csv", index=False)

# combine
main["source"] = "main"
extra["source"] = "extra"
combined = pd.concat([main, extra], ignore_index=True)
print(f"\ncombined: {len(combined)}")

# apply gates
ok = combined["status"] == "ok"
has_band = combined["band"].notna()
rate_ok = ((combined["rate_position_hz"] >= RATE_POSITION_MIN) &
           (combined["rate_attitude_hz"] >= RATE_ATTITUDE_MIN) &
           (combined["rate_ratios_hz"] >= RATE_RATIOS_MIN))
dur_ok = combined["duration_s"] >= DURATION_MIN
air_ok = combined["airborne_s"] >= AIRBORNE_MIN
drop_ok = combined["dropout_frac_v2"] <= DROPOUT_MAX

print()
print("=== Combined funnel ===")
n = len(combined)
print(f"sampled (after 4001 filter)          {n}")
print(f"downloaded and parsed                {int(ok.sum())}")
print(f"has valid firmware band              {int((ok & has_band).sum())}")
print(f"+ rate gates                         {int((ok & has_band & rate_ok).sum())}")
print(f"+ duration >= 60s                    {int((ok & has_band & rate_ok & dur_ok).sum())}")
print(f"+ airborne >= 45s                    {int((ok & has_band & rate_ok & dur_ok & air_ok).sum())}")
gate1 = (ok & has_band & rate_ok & dur_ok & air_ok & drop_ok)
print(f"+ dropout <= 0.02                    {int(gate1.sum())}")
print(f"Gate 1 survivors                     {int(gate1.sum())}")

surv = combined[gate1].copy()
print()
print(f"=== Composition of {len(surv)} survivors ===")
print(f"is_quadrotor_final == True           {int(surv['is_quadrotor_final'].sum())}")
analysis = surv[surv["is_quadrotor_final"] == True].copy()
print(f"analysis set (quad + gate1)          {len(analysis)}")

print()
print("=== Band breakdown of analysis set ===")
print(analysis["band"].value_counts())

combined.to_csv("combined_index.csv", index=False)
analysis.to_csv("combined_analysis_set.csv", index=False)
print()
print("saved combined_index.csv")
print("saved combined_analysis_set.csv")