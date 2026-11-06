# Calibrated Anomaly Detection on Open UAV Flight Logs

Vedant Kumar Roy
Supervisor: Prof. Tavish Pattanayak, BITS Pilani Dubai Campus
Period: September to December 2026

## What this is

PX4 autopilots use fixed thresholds on estimator innovation test ratios to
decide whether a sensor measurement is trustworthy. This project measures
the false alarm rate of those thresholds on real flights and evaluates
split conformal prediction as a calibrated alternative.

Four claims:

- C1. The PX4 EKF innovation gates over-flag on healthy flights at
  approximately four orders of magnitude above the implied rate. Driven by
  magnetometer and heading.
- C2. Conformal coverage is regime-dependent. Descent over-covers by 2.0
  percentage points.
- C3. Cross-band shift breaks the guarantee over the corpus population.
  Underpowered over the nominal subpopulation.
- C4. EKF innovation ratios cannot see actuator faults by construction.
  Log eb456400 is direct evidence.

## Data

Streamed from the public PX4 Flight Review repository (logs.px4.io).
Restricted to real quadrotor flights, firmware v1.12+, modern uORB topic
schema, 2023 onward. Screening funnel and gates documented in
`DECISIONS.md` and `RESULTS.md`.

- 2,770 logs streamed
- 2,168 Gate 1 survivors
- 1,627 analysis set
- Split 20/40/40 into Group A / B / C within each firmware band

Group A is used to train fitted detectors. Group B is the conformal
calibration set. Group C is held out for evaluation.

## Repository layout

- `RESULTS.md`. Consolidated results and per-claim tables
- `DECISIONS.md`. Every gate and analysis choice with its reason
- `PAPER_OUTLINE.md`. Paper section structure
- `combined_index.csv`. Master index
- `combined_analysis_set.csv`. 1,627 analysis logs
- `analysis_set_with_log_scores.csv`. Per-flight ratio scores
- `analysis_set_with_satfrac.csv`. Per-flight actuator saturation
- `regime_segments.csv`. 11,330 hover / translation / descent segments
- `c4_handlabel_final.csv`. Hand-labelled top 50 sat_frac flights
- `run_all.py`. One-command reproducibility

Key scripts:

- `pilot_2270.py`, `pilot_extra.py`. Streaming
- `px4_screen.py`. Streaming and filtering
- `c1_clean.py`, `c1_clean_no_mag_heading.py`, `c1_heatmap_clean.py`. C1
- `regime_c2.py`, `c2_figure_cells.py`. C2
- `c3_shift.py`, `c3_figure_cells.py`. C3
- `matched_coverage_table.py`, `c4_cells_figure.py`, `eb456400_figure.py`. C4
- `cell2_clean.py`. Cell 1 and cell 2 comparison
- `clean_b_ci.py`. Flight-block bootstrap CIs

## Requirements

Python 3.10+. Install:

```
pip install -r requirements.txt
```

## Reproduce

Full pipeline runs end to end. Streams logs, applies gates, computes
features, calibrates, evaluates all four claims. Requires network access
and approximately 20 GB of disk for the Parquet corpus.

## Figure list

- `c1_heatmap_clean.png`. C1 clean-flight heatmap across firmware bands
- `c1_clean_no_mag_heading.png`. C1 with mag and heading removed
- `c2_cells_figure.png`. C2 cell 1 and cell 2 regime coverage
- `c3_cells_figure.png`. C3 cell 1 and cell 2 cross-band coverage
- `c4_cells_figure.png`. C4 detection power, contaminated vs clean
- `matched_coverage_curve.png`. C4 across false alarm rate range
- `eb456400_figure.png`. 10 ratio traces (flat) and 4 motor commands
  (divergent) on one time axis

## Named finding: eb456400

Log eb456400 shows 179 seconds of sustained motor imbalance. Two motors
pinned at maximum, two near zero. All ten innovation test ratios stay flat
at zero. The EKF is satisfied. The controller is fighting. Estimator
health and control health are two different quantities.

## License

Code: MIT. See `LICENSE`.
Data: derived from public PX4 logs, released under CC-BY 4.0.
