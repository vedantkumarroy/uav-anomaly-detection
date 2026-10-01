import pandas as pd
import numpy as np

df = pd.read_csv("pilot_2270_index.csv")

# define gates
RATE_POSITION_MIN = 5.0
RATE_ATTITUDE_MIN = 10.0
RATE_RATIOS_MIN = 1.5
DURATION_MIN = 60.0
AIRBORNE_MIN = 45.0
DROPOUT_MAX = 0.02

n_total = len(df)
print(f"=== Stage-by-stage funnel (2,270 run) ===\n")

# Stage 1
n_sampled = n_total
print(f"sampled (after 4001 metadata filter)      {n_sampled}")

# Stage 2 - downloaded (status ok or better)
n_downloaded = int((df["status"] == "ok").sum())
print(f"downloaded and parsed                     {n_downloaded}")

# Stage 3 - has ratios topic (implied by band)
has_band = df["band"].notna()
n_band = int(has_band.sum())
print(f"has valid firmware band                   {n_band}")

# From here on, restrict to clean PX4 with band
d = df[has_band].copy()

# Stage 4 - rate gates
rate_ok = (
    (d["rate_position_hz"] >= RATE_POSITION_MIN) &
    (d["rate_attitude_hz"] >= RATE_ATTITUDE_MIN) &
    (d["rate_ratios_hz"] >= RATE_RATIOS_MIN)
)
n_rate = int(rate_ok.sum())
print(f"+ rate gates (pos {RATE_POSITION_MIN}, att {RATE_ATTITUDE_MIN}, rat {RATE_RATIOS_MIN})   {n_rate}")

# Stage 5 - duration
dur_ok = d["duration_s"] >= DURATION_MIN
n_dur = int((rate_ok & dur_ok).sum())
print(f"+ duration >= {int(DURATION_MIN)}s                    {n_dur}")

# Stage 6 - airborne
air_ok = d["airborne_s"] >= AIRBORNE_MIN
n_air = int((rate_ok & dur_ok & air_ok).sum())
print(f"+ airborne >= {int(AIRBORNE_MIN)}s                    {n_air}")

# Stage 7 - dropout
drop_ok = d["dropout_frac_proxy"] <= DROPOUT_MAX
n_drop = int((rate_ok & dur_ok & air_ok & drop_ok).sum())
print(f"+ dropout <= {DROPOUT_MAX}                   {n_drop}")

# Stage 8 - clean PX4 is already included
print(f"                                   ------")
print(f"Gate 1 survivors                           {n_drop}")

# Now break down for downstream analysis
surv = d[rate_ok & dur_ok & air_ok & drop_ok].copy()

print()
print(f"=== Composition of {n_drop} survivors ===")
print()
print(f"is_quadrotor = True                       {int(surv['is_quadrotor'].sum())}")
print(f"is_quadrotor = False or NaN               {int((~surv['is_quadrotor'].astype(bool)).sum())}")

non_ground = (
    (surv["skip_reason"] != "ground_only_actuator_motors") &
    (surv["skip_reason"] != "ground_only_actuator_outputs") &
    (surv["skip_reason"] != "ground_only_none")
)
print(f"not ground-only                           {int(non_ground.sum())}")

# Combined analysis set
analysis = surv[(surv["is_quadrotor"] == True) & non_ground].copy()
n_analysis = len(analysis)
print(f"analysis set (quad AND not ground-only)   {n_analysis}")

print()
print("=== Band breakdown of analysis set ===")
print(analysis["band"].value_counts())

# A/B/C sizes for the analysis set
print()
print("=== Required A/B/C sizes for 20/40/40 split of analysis set ===")
for band in ["v1.12-1.13", "v1.14-1.15", "v1.16+"]:
    n_band = int((analysis["band"] == band).sum())
    n_a = int(round(0.20 * n_band))
    n_b = int(round(0.40 * n_band))
    n_c = n_band - n_a - n_b
    print(f"  {band}: total={n_band}, A={n_a}, B={n_b}, C={n_c}")

print()
print(f"  TOTAL: {n_analysis}")

# save analysis set
analysis.to_csv("analysis_set.csv", index=False)
print(f"\nsaved analysis_set.csv ({n_analysis} rows)")


print()
print("=== Band breakdown of analysis set ===")
print(analysis["band"].value_counts())

# A/B/C sizes for the analysis set
print()
print("=== Required A/B/C sizes for 20/40/40 split of analysis set ===")
for band in ["v1.12-1.13", "v1.14-1.15", "v1.16+"]:
    n_band = int((analysis["band"] == band).sum())
    n_a = int(round(0.20 * n_band))
    n_b = int(round(0.40 * n_band))
    n_c = n_band - n_a - n_b
    print(f"  {band}: total={n_band}, A={n_a}, B={n_b}, C={n_c}")

print()
print(f"  TOTAL: {n_analysis}")