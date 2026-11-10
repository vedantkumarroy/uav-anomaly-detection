# Calibrated Anomaly Detection on Open UAV Flight Logs

Vedant Kumar Roy (IIT Madras)
Supervisor: Prof. Tavish Pattanayak, BITS Pilani Dubai Campus
Period: September to December 2026

## What this is

PX4 autopilots use fixed thresholds on estimator innovation test ratios to
decide whether a sensor measurement is trustworthy. This project measures the
false alarm rate of those thresholds on real flights and evaluates split
conformal prediction as a calibrated alternative.

Four claims:

- **C1.** The eight innovation ratios PX4 gates at 5σ over-flag on clean
  flights by 2.8 to 4.3 orders of magnitude against their own per-sample null.
  The magnetometer and heading ratios, gated at 3.0σ and 2.6σ by design, fire
  within roughly one order of their design rate. They produce the most raw
  flags but sit closest to specification, so the miscalibration is in the core
  navigation gates, not in the two sensors the autopilot already distrusts.
- **C2.** Conformal coverage is regime-dependent. Descent over-covers by 2.0
  percentage points against nominal.
- **C3.** Cross-band firmware shift breaks the guarantee over the corpus
  population. Underpowered over the nominal subpopulation.
- **C4.** EKF innovation ratios cannot see actuator faults by construction.
  Detection power is inversely related to fault severity. Log `eb456400` is
  direct evidence.

## Repository layout

| Folder | Contents |
|---|---|
| `01_Docs/` | `RESULTS.md` consolidated per-claim results; `DECISIONS.md` every gate and analysis choice with its reason; supporting notes |
| `02_Figures/` | 29 figures. The 8 used in the paper are listed below; the rest are supporting and diagnostic |
| `03_Data_CSVs/` | 83 derived tables. Documented in full below |
| `04_Scripts/` | Streaming, screening, feature extraction, calibration, evaluation, `fetch_by_id.py` and `run_all.py` |
| root | `README.md`, `LICENSE`, `CITATION.cff` and `.zenodo.json` (citation and deposit metadata) |

## Corpus

Streamed from the public PX4 Flight Review repository (`logs.px4.io`).
Restricted to real quadrotor flights, firmware v1.12+, the modern uORB topic
schema, 2023 onward. Screening funnel and gate rationale in `DECISIONS.md`.

- 2,770 logs streamed
- 2,168 Gate 1 survivors
- 1,627 analysis set
- Split 20/40/40 into Group A / B / C within each firmware band

Group A trains the fitted detectors. Group B is the conformal calibration set.
Group C is held out for evaluation. No flight is used for both training and
calibration.

## Data files

### Regenerating the corpus

This repository does **not** contain the Parquet corpus. The pipeline converts
each raw `.ulg` log to Parquet (26.9 GB for the full corpus) and discards the
raw log after conversion. Neither is redistributed here: the source logs belong
to the PX4 community and remain available from the public Flight Review
repository.

What is released instead is the **log-ID manifest** plus the code.

| File | Rows | Contents |
|---|---|---|
| `combined_index.csv` | 2,770 | **The manifest.** Every log streamed, with `log_id`, firmware version, decoded band, parse status, screening-gate outcomes and the five runtime indicators. This is the file to start from. |
| `combined_analysis_set.csv` | 1,627 | The analysis set: the subset of the manifest that cleared every screening gate. |

Any `log_id` resolves to `https://logs.px4.io/plot_app?log=<log_id>` for
inspection in the browser, or `https://logs.px4.io/download?log=<log_id>` for
the raw file.

**Rebuild it.** `04_Scripts/fetch_by_id.py` reads the manifest and downloads
precisely those log IDs:

```
python 04_Scripts/fetch_by_id.py --dry-run           # plan only, no network
python 04_Scripts/fetch_by_id.py                     # all 2,770 logs
python 04_Scripts/fetch_by_id.py --analysis-set-only # just the 1,627 analysed
```

It is safe to interrupt and re-run: a log already converted is skipped, so a
failed run resumes where it stopped. Each log is downloaded, converted to
zstd-compressed Parquet per uORB topic, and the raw `.ulg` deleted. A per-log
outcome report is written to `fetch_by_id_report.csv`. Expect roughly 27 GB and
several hours on a home connection. Keep `--workers` low; this is community
infrastructure.

**Why a fetch-by-ID script is necessary.** The original sampling used
`pilot_2270.py` at seed 10 and `pilot_extra.py` at seed 11, recorded in
`01_Docs/pilot_2270_seed.txt` and `01_Docs/pilot_extra_seed.txt`. Re-running
those does **not** reproduce this corpus: they sample from the live
`logs.px4.io` listing, which grows as flights are uploaded, so the same seed
selects a different set on a later date. The seeds document what was done; only
the manifest reconstructs it.

### Canonical tables

Use these. Everything else is either a sensitivity check or superseded.

| File | Rows | Contents |
|---|---|---|
| `analysis_set_with_satfrac.csv` | 1,627 | **Canonical per-flight feature table.** Superset of every other `analysis_set_with_*` file: metadata, band, group assignment, the five runtime indicators, the ten log-transformed ratio scores, and actuator saturation fraction. |
| `regime_segments.csv` | 11,330 | Hover / translation / descent segments with start, end, parent `log_id` and per-segment ratio scores. |
| `version_bands_table.csv` | 1,634 | Firmware version decoded from the packed 32-bit field, with band assignment. |
| `classify_outputs_quads.csv` | 426 | Quadrotor confirmation from actual actuator-output count, rather than from the `SYS_AUTOSTART` metadata alone. |

### Clean and dirty flights

A flight is **clean** when it trips no runtime indicator. Four of the five
must be exactly zero; the fifth, `sat_frac`, carries a tolerance and trips
only above 0.05, because brief actuator saturation is ordinary control
effort rather than evidence of a fault. This yields **957 clean** and 670
dirty flights out of 1,627. Requiring `sat_frac == 0` instead would leave
743. The 0.05 tolerance is applied consistently throughout and is recorded
in `DECISIONS.md`.

### C1 — gate false alarm rate

| File | Rows | Contents |
|---|---|---|
| `c1_per_flight.csv` | 1,627 | Per-flight, per-ratio fraction of samples exceeding 1.0. Restricted to the 957 clean flights, the column means are the observed rates in the paper's per-ratio table. |
| `c1_clean.csv` | 3 | Flight-level over-flag rate with Wilson intervals for all / clean / dirty. Source of the 18.93 / 15.57 / 23.73 percent figures. |
| `c1_clean_no_mag_heading.csv` | 2 | The same with the magnetometer and heading ratios dropped. Source of 5.54 percent. |
| `c1_summary_clean.csv` | 10 | **Per-ratio C1 result as the paper reports it.** Observed per-sample exceedance rate on the 957 clean flights against each ratio's own null, with the gate parameter, gate size, bootstrap confidence interval, ratio to null and orders above null. Written by `04_Scripts/c1_summary_clean.py`. |
| `c1_summary.csv` | 10 | Per-ratio summary over all 1,627 flights. Note the population: this is **not** the clean-flight restriction the paper reports. |
| `c1_per_band.csv` | 30 | Per-ratio rates split by firmware band. |
| `c1_heatmap_clean.csv` | 150 | Band × ratio × threshold exceedance grid on clean flights. |
| `c1_warmup_exclusion.csv` | 5 | Sensitivity to the warm-up window length. |
| `c1_warmup_excluded.csv` | 1,627 | Per-flight rates with the 30 s warm-up removed. |
| `c1_exclude_extremes.csv` | 4 | Sensitivity to excluding the 111 initialization-artefact flights. |

### C2 — regime-dependent coverage

| File | Rows | Contents |
|---|---|---|
| `c2_cell2.csv` | 3 | Per-regime coverage under cell 1 and cell 2 with bootstrap intervals and both sample sizes. Primary C2 result. |
| `c2_clean_ci.csv` | 3 | Contaminated versus clean calibration, per regime. |
| `c2_regime_summary.csv` | 3 | Pooled per-regime coverage with calibration and test counts. |
| `c2_cluster_robust.csv` | 3 | Flight-block bootstrap intervals against the naive segment-level intervals, showing the design effect. |
| `c2_alpha_sweep.csv` | 12 | Coverage against nominal α per regime. |
| `regime_c2_coverage.csv` | 30 | Per-regime, per-ratio thresholds and coverage. |

### C3 — cross-band distribution shift

| File | Rows | Contents |
|---|---|---|
| `c3_cell2.csv` | 2 | Cross-band coverage under cell 1 and cell 2. Primary C3 result. |
| `c3_cross_band.csv` | 10 | Per-ratio coverage calibrating on v1.12–v1.15 and testing on v1.16+. |
| `c3_inband.csv` | 10 | Within-band control for the same ratios. |
| `c3_weighted.csv` | 10 | Likelihood-ratio weighted conformal variant. |
| `c3_clean_ci.csv` | 2 | Clean-calibration version with intervals. |
| `c3_segments_cluster.csv` | 2 | Segment-level clustered intervals. |
| `c3_segments_final.csv` | 3 | Interval methods compared. |
| `c3_alpha_sweep.csv` | 4 | Cross-band versus weighted across α. |

### C4 — detection power and the actuator blind spot

| File | Rows | Contents |
|---|---|---|
| `c4_cells_table.csv` | 3 | Detection power per fitted detector under contaminated and clean calibration, with Wilson intervals. Primary C4 result. |
| `c4_handlabel_final.csv` | 50 | The hand-labelled top-50 flights by saturation: motor means, spread, saturation run length, and the genuine / borderline / not-genuine verdict. Source of the 11 confirmed anomalies and the 1-of-11 recovery. |
| `c4_handlabel_worksheet.csv` | 50 | The pre-verdict worksheet the labelling was done from. |
| `matched_coverage_curve.csv` | 120 | Detection power against false alarm rate for all four detectors. |
| `matched_coverage_false_alarm.csv` | 4 | Each detector at its matched operating point. |
| `matched_coverage_detection_power.csv` | 4 | Power at that matched point, with intervals. |
| `c4_detection_power_ci.csv` | 12 | Detection counts and intervals by method and threshold. |
| `c4_satfrac_ranking.csv` | 12 | Detection rate by saturation bucket. |
| `per_ratio_detection_power.csv` | 10 | Power attributable to each individual ratio. |
| `coverage_power_curve.csv` | 120 | Joint coverage and power across α. |
| `detection_power_c.csv` | 627 | Group C flights with detector scores and flags. |
| `case_study_top15.csv` | 15 | The 15 most saturated flights, including `eb456400`. |
| `extreme_times.csv` | 34,861 | Timestamps of every ratio excursion above 1e10, used to confirm these are EKF initialization artefacts. |

### Calibration-set contamination

| File | Rows | Contents |
|---|---|---|
| `contamination_control.csv` | 10 | Per-ratio coverage under the full calibration set, the clean subset, and 200 equal-sized random subsets. The control that separates contamination from sample size. |
| `contamination_check.csv` | 10 | Full versus clean calibration without the random-subset control. |
| `clean_b_comparison.csv` | 14 | Every claim recomputed on the clean Group B calibration set. |
| `coverage_per_ratio_pooled.csv` | 10 | Per-ratio conformal thresholds and coverage, pooled. |
| `coverage_per_ratio_per_band.csv` | 30 | The same split by firmware band. |

### Superseded — do not use

These predate the final analysis and are kept only for provenance. They
disagree with the paper, and in two cases describe a different corpus.

| File | Why |
|---|---|
| `analysis_set.csv` | 1,566 rows. The analysis set from the first 2,270-log batch only, written by `funnel_2270.py` before the extra 500 logs were merged. The final set is `combined_analysis_set.csv` at 1,627. Anyone reaching for the obvious filename gets the partial corpus. |
| `cooccurrence_table.csv` | Built from the 1,566-row partial set above. |
| `analysis_set_with_groups.csv` | Intermediate enrichment stage. |
| `analysis_set_with_scores.csv` | Intermediate enrichment stage. |
| `analysis_set_with_log_scores.csv` | Intermediate enrichment stage. |
| `analysis_set_with_log_scores_v2.csv` | Intermediate enrichment stage. |
| `analysis_set_with_log_scores_leaky.csv` | Deliberate negative control: the 99.9th-percentile clip computed on all flights including the test set. Retained to show the leakage was found and corrected, never for analysis. |
| `paper_table_split.csv` | Old 1,578-flight split (21/42/43, 192/384/385, 100/200/199). Superseded by the band table in `combined_analysis_set.csv`. |
| `paper_table_coverage.csv` | Older calibration sizes. |
| `paper_table_detection_power.csv` | Older indicator-threshold detection analysis. |
| `detection_power_table.csv` | Same, superseded by `c4_cells_table.csv`. |
| `c1_heatmap_v1_12-1_13.csv` | Per-band heatmaps superseded by `c1_heatmap_clean.csv`. |
| `c1_heatmap_v1_14-1_15.csv` | As above. |
| `c1_heatmap_v1_16+.csv` | As above. |
| `track_b_coverage.csv` | Abandoned alternative scoring track. |
| `track_b_v2_coverage.csv` | As above. |
| `track_b_v3_coverage.csv` | As above. |
| `band_matrix_v2.csv` | Field-presence matrices from the schema-diff exercise. |
| `band_matrix_v2_2000.csv` | As above. |
| `band_matrix_v2_2000_seed2.csv` | As above. |

### Pilot runs — historical

From the staged pilot runs that set the screening thresholds. Documented in
`DECISIONS.md`; none feed the published results.

`pilot_2270_index.csv`, `pilot_2270_index_enriched.csv`,
`pilot_2270_topic_bytes.csv`, `pilot_500_results.csv`,
`pilot_500_crosstab.csv`, `pilot_500_v2_index.csv`,
`pilot_500_v2_excluded.csv`, `pilot_500_v3_index.csv`,
`pilot_500_v3_topic_bytes.csv`, `pilot_extra_index.csv`,
`pilot_extra_index_enriched.csv`, `per_log_200_new.csv`,
`per_log_with_dropout.csv`, `sensor_rates.csv`, `version_sensor_table.csv`

## Figures

Used in the paper:

| File | Shows |
|---|---|
| `funnel_figure.png` | Screening funnel, one bar per gate |
| `split_figure.png` | Corpus composition by firmware band and group |
| `segments_per_flight.png` | Regime segment count distribution |
| `c2_cells_figure.png` | C2 regime coverage, cell 1 and cell 2 |
| `c3_cells_figure.png` | C3 cross-band coverage, cell 1 and cell 2 |
| `c4_cells_figure.png` | C4 detection power, contaminated vs clean calibration |
| `matched_coverage_curve.png` | C4 power against false alarm rate, all detectors |
| `eb456400_figure.png` | Ten ratio traces flat at zero against four divergent motor commands |

The remaining 21 figures in `02_Figures/` are supporting and diagnostic
(sensitivity sweeps, histograms, clustered-interval comparisons, per-band
heatmaps). They are not referenced by the paper.

## Requirements

Python 3.10+.

```
pip install -r 01_Docs/requirements.txt
```

## Reproduce

```
pip install -r 01_Docs/requirements.txt
python 04_Scripts/fetch_by_id.py        # rebuild the Parquet corpus, ~27 GB
python 04_Scripts/run_all.py            # regenerate every table and figure
```

`run_all.py` runs 34 steps in dependency order and regenerates every table and
figure the paper uses, plus the sensitivity checks. Package versions are pinned
in `01_Docs/requirements.txt` to prevent environment drift. Every threshold and
analysis choice is recorded with its reason in `01_Docs/DECISIONS.md`.

The repository holds 104 scripts in all. The 34 in `run_all.py` are the
published pipeline; the rest are pilot-era and exploratory work kept for
provenance. `prepare_tables.py` in particular is pilot-era and prints figures
that disagree with the paper; it is not part of the pipeline.

## Named finding: eb456400

Log `eb456400` shows 179 seconds of sustained motor imbalance. Two motors
collapse toward zero while the controller drives the remaining two upward to
compensate, until one saturates and the flight ends. All ten innovation test
ratios stay flat at zero throughout. The EKF is satisfied because the vehicle
is still flying the commanded trajectory. The controller is fighting.

Estimator health and control health are two different quantities, and no
threshold on this signal family, fixed or conformal, can bridge them.

## License

Code: MIT, see `LICENSE`.
Data: derived from public PX4 logs, released under CC-BY 4.0. The original
logs remain the property of their uploaders and are not redistributed here.
