import numpy as np
import pandas as pd

RATIOS = [
    "baro_vpos",
    "gps_hpos[0]", "gps_hpos[1]",
    "gps_hvel[0]", "gps_hvel[1]",
    "gps_vpos", "gps_vvel",
    "heading",
    "mag_field[0]",
    "hagl",
]

df = pd.read_csv("analysis_set_with_scores.csv")
print(f"loaded: {len(df)}")
print(f"groups: {df['group'].value_counts().to_dict()}")
print()

# compute clip value on Group B only
cal_mask = df["group"] == "B"
print("applying log(1+x) with clip computed on Group B only")

for r in RATIOS:
    c = f"score_{r}"
    if c not in df.columns:
        continue

    vals = df[c].values.astype(np.float64)
    vals_log = np.log1p(np.maximum(vals, 0))

    # compute clip from Group B only
    cal_vals = vals_log[cal_mask.values]
    finite = cal_vals[np.isfinite(cal_vals)]
    if len(finite) == 0:
        continue
    clip_hi = np.percentile(finite, 99.9)

    # apply to ALL rows (so the clip affects everything equally)
    vals_log_clipped = np.minimum(vals_log, clip_hi)
    df[f"score_log_{r}"] = vals_log_clipped

df.to_csv("analysis_set_with_log_scores_v2.csv", index=False)
print("saved analysis_set_with_log_scores_v2.csv")
print()
print("New ranges (Group B only clip):")
for r in RATIOS:
    c = f"score_log_{r}"
    if c not in df.columns:
        continue
    vals = df[c].dropna()
    print(f"  {r}: min={vals.min():.4f} max={vals.max():.4f} median={vals.median():.4f}")