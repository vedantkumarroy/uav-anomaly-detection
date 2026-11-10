"""SUPERSEDED. Pilot-era table printer, not part of the published pipeline.

This script reads pilot_2270_index.csv (2,270 logs), analysis_set.csv (the
1,566-flight partial set) and track_b_coverage.csv (an abandoned scoring
track), and hardcodes a Gate 1 count of 1901. Every one of those figures
disagrees with the paper, which reports 2,770 streamed, 2,168 Gate 1
survivors and a 1,627-flight analysis set.

It is not in run_all.py. Kept for provenance only. For the published
tables see final_summary.py and 01_Docs/RESULTS.md.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("TABLE 1: Dataset composition")
print("=" * 60)

df = pd.read_csv("pilot_2270_index.csv")
print(f"Total sampled (after 4001 metadata filter):  {len(df)}")
print(f"Downloaded and parsed:                       {int((df['status']=='ok').sum())}")
print(f"Valid PX4 with band:                         {int(df['band'].notna().sum())}")
print(f"Gate 1 survivors:                            1901")
print(f"Analysis set (quad, not ground-only):        {len(pd.read_csv('analysis_set.csv'))}")
print()

print("=" * 60)
print("TABLE 2: A/B/C split per band")
print("=" * 60)
groups = pd.read_csv("analysis_set_with_groups.csv")
print(pd.crosstab(groups["band"], groups["group"]))
print()

print("=" * 60)
print("TABLE 3: Coverage per method (nominal = 0.95)")
print("=" * 60)
cov = pd.read_csv("track_b_coverage.csv")
print(cov.round(4).to_string(index=False))
print()

print("=" * 60)
print("TABLE 4: Detection power on high-anomaly logs (Group C)")
print("=" * 60)
dp = pd.read_csv("detection_power_table.csv")
pivot = dp.pivot(index="indicator_threshold", columns="method", values="detection_rate")
pivot = pivot.round(4)
print(pivot)
print()
print("Counts:")
pivot_n = dp.pivot(index="indicator_threshold", columns="method", values="n_detected")
print(pivot_n)

# Save combined table for paper
print()
print("=" * 60)
print("Saving combined results ...")
print("=" * 60)

cov_out = cov.copy()
cov_out.to_csv("paper_table_coverage.csv", index=False)
dp.to_csv("paper_table_detection_power.csv", index=False)
groups_out = pd.crosstab(groups["band"], groups["group"]).reset_index()
groups_out.to_csv("paper_table_split.csv", index=False)
print("saved paper_table_coverage.csv")
print("saved paper_table_detection_power.csv")
print("saved paper_table_split.csv")