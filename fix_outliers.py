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
print(f"loaded: {len(df)} logs")

# count extreme values per ratio
print()
print("=== extreme score counts (per ratio) ===")
for r in RATIOS:
    c = f"score_{r}"
    if c not in df.columns:
        continue
    vals = df[c].dropna()
    n_extreme_1e3 = int((vals > 1e3).sum())
    n_extreme_1e6 = int((vals > 1e6).sum())
    n_extreme_1e9 = int((vals > 1e9).sum())
    print(f"  {r}: >1e3={n_extreme_1e3}, >1e6={n_extreme_1e6}, >1e9={n_extreme_1e9}")

# log-transform and clip
print()
print("applying log(1+x) transform and clipping at 99.9th percentile")
for r in RATIOS:
    c = f"score_{r}"
    if c not in df.columns:
        continue
    vals = df[c].values.astype(np.float64)
    # log transform
    vals_log = np.log1p(np.maximum(vals, 0))
    # clip at 99.9th percentile (per-ratio, computed over all logs)
    finite = vals_log[np.isfinite(vals_log)]
    if len(finite) == 0:
        continue
    clip_hi = np.percentile(finite, 99.9)
    vals_log = np.minimum(vals_log, clip_hi)
    df[f"score_log_{r}"] = vals_log

# save
df.to_csv("analysis_set_with_log_scores.csv", index=False)
print()
print("saved analysis_set_with_log_scores.csv")

# check ranges
print()
print("=== new score ranges ===")
for r in RATIOS:
    c = f"score_log_{r}"
    if c not in df.columns:
        continue
    vals = df[c].dropna()
    print(f"  {r}: min={vals.min():.4f} max={vals.max():.4f} median={vals.median():.4f}")