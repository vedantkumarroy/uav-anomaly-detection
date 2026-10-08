# Calibrated Anomaly Detection on Open UAV Flight Logs

Consolidated results. Updated October 8, 2026.

## Project summary

UAV autopilots (PX4) use fixed thresholds on estimator innovation test
ratios to decide whether sensor measurements are trustworthy. This project
measures the false alarm rate of those thresholds on real flights, and
evaluates conformal prediction as a calibrated alternative.

Four claims:

- **C1**: PX4 EKF thresholds over-flag healthy flights
- **C2**: Coverage guarantees are marginal, not conditional
- **C3**: Cross-band calibration fails; weighted conformal repairs partially
- **C4**: Conformal calibration does not destroy detection power; but
  ratio-based detection is blind to motor imbalance

## Dataset

Streamed from `logs.px4.io`. Filtered to real quadrotor flights.

| Stage | Count |
|---|---|
| Sampled after SYS_AUTOSTART == 4001 metadata filter | 2,770 |
| Downloaded and parsed | 2,770 |
| Valid PX4 firmware band | 2,679 |
| + Rate gates (position 5 Hz, attitude 10 Hz, ratios 1.5 Hz) | 2,524 |
| + Duration >= 60 s | 2,510 |
| + Airborne >= 45 s | 2,314 |
| + Dropout <= 0.02 | 2,168 |
| Gate 1 survivors | 2,168 |
| Analysis set (quads, not ground-only) | **1,627** |

Split at 20/40/40 within each firmware band:

| Band | Train A | Calibrate B | Test C |
|---|---|---|---|
| v1.12-1.13 | 51 | 102 | 103 |
| v1.14-1.15 | 180 | 360 | 360 |
| v1.16+ | 94 | 188 | 189 |
| **Total** | **325** | **650** | **652** |

## Pipeline

1. Stream `.ulg` from logs.px4.io
2. Parse with pyulog
3. Record 5 runtime indicators
4. Write each uORB topic as Parquet with zstd (50% compression)
5. Discard the `.ulg`

Per flight: extract 10 universal innovation test ratios. Per-flight score for
each ratio: 95th percentile over the flight. Applied log(1+x) transform and
clipped at 99.9th percentile (computed on Group B only).

Conformal setup:
- Split conformal at alpha = 0.05
- Calibrate on Group B, test on Group C

---

## C1: PX4 EKF threshold false alarm rate

### Headline (with 30-second warm-up excluded)

| | Flights over-flagged | Rate |
|---|---|---|
| Naive (all samples) | 344/1627 | 21.1% |
| **Warm-up excluded** | **308/1627** | **18.9%** |

The warm-up artifact contributes 2.2 pp. Headline is 18.9%. The 21.1% goes
in a sensitivity note.

### Threshold sensitivity

| Threshold | chi-square | Tail prob | Flights over-flagged |
|---|---|---|---|
| 1.0 | 25 | 3.73e-6 | 18.9% |
| 2.0 | 100 | 1.93e-22 | 17.0% |
| 5.0 | 625 | ~ | 11.8% |
| 10.0 | 2500 | 5.17e-55 | 8.2% |

**Correction:** earlier I wrote threshold 10.0 implied rate as exp(-50) =
1.9e-22. This was wrong. The ratio is already divided by gate². Ratio = 10
gives chi-square = 250, tail = 5.17e-55. The difference between threshold
1.0 and threshold 10.0 is 53 orders of magnitude.

### Per-ratio breakdown at threshold 1.0 (warm-up excluded)

| Ratio | Mean rate | Flights >5% over-flagged |
|---|---|---|
| mag_field[0] | 3.96% | 10.1% |
| heading | 1.94% | 7.7% |
| gps_vpos | 1.27% | 2.4% |
| hagl | 1.01% | 2.4% |
| gps_hpos[0] | 0.71% | 1.8% |
| baro_vpos | 0.98% | 1.8% |
| gps_hpos[1] | 0.55% | 1.4% |
| gps_hvel[0] | 0.44% | 1.4% |
| gps_vvel | 0.22% | 1.1% |
| gps_hvel[1] | 0.12% | 0.7% |

### Firmware degradation

| Ratio | v1.12-1.13 | v1.14-1.15 | v1.16+ |
|---|---|---|---|
| mag_field[0] | 2.9% | 11.1% | 15.3% |
| heading | 4.7% | 8.8% | 8.9% |
| gps_hpos[0] | 0.8% | 1.6% | 4.9% |
| gps_hpos[1] | 0.0% | 1.2% | 5.3% |

Monotonic trend. Magnetometer is worst. This matches the physics.

### Figures

- `c1_heatmap.png` — ratios on y, thresholds on x, one panel per band
- `c1_figure.png` — bar chart per ratio
- `c1_false_alarm_histograms.png` — distributions
- `c1_exclude_extremes_figure.png` — sensitivity to extreme cutoffs

### Scripts

- `c1_false_alarm.py`, `c1_false_alarm_v2.py`
- `c1_figure.py`, `c1_heatmap.py`, `c1_per_band.py`
- `c1_threshold_sensitivity.py`, `c1_exclude_extremes.py`

---

## C2: Regime-level coverage

Cut 1,627 flights into segments by flight regime.

| Regime | Definition | Segments |
|---|---|---|
| Hover | horizontal speed < 0.5 m/s | 5,937 |
| Translation | horizontal speed >= 0.5 m/s | 4,303 |
| Descent | vertical velocity > 0.5 m/s downward | 1,090 |
| **Total** | | **11,330** |

### Coverage per regime

| Regime | Mean coverage | 95% CI | Nominal inside? |
|---|---|---|---|
| Hover | 0.9609 | [0.9583, 0.9634] | No |
| Translation | 0.9401 | [0.9365, 0.9436] | No |
| Descent | 0.9536 | [0.9468, 0.9594] | Yes |

Hover over-covers by 1.1 pp. Translation under-covers by 1.0 pp. Marginal
guarantee holds, conditional does not.

### Segments per flight

| Statistic | Value |
|---|---|
| Flights with >= 1 segment | 1,620 |
| Mean | 6.99 |
| Median | 5 |
| Max | **87** |
| p90 | 14 |
| p95 | 20 |
| p99 | 40 |

Concentration:

| Top fraction | Flights | Segments | % of total |
|---|---|---|---|
| Top 1% | 16 | 917 | 8.1% |
| Top 5% | 81 | 2,658 | 23.5% |
| **Top 10%** | 162 | **4,011** | **35.4%** |
| Top 20% | 324 | 5,836 | 51.5% |

A small number of long flights dominate the regime analysis. The top 10%
of flights carry 35% of the segments.

### Figure

`c2_regime_figure.png` — two panels: (A) coverage with CIs, (B) per-ratio
distribution per regime

### Scripts

`regime_c2.py`, `segment_distribution.py`

---

## C3: Cross-band distribution shift

Calibrate on Group B from v1.12-1.13 and v1.14-1.15. Test on Group C from
v1.16+.

| Method | Mean coverage | 95% CI | Nominal inside? |
|---|---|---|---|
| Cross-band | 0.9212 | [0.9081, 0.9325] | No |
| In-band reference | 0.9286 | [0.9161, 0.9393] | No |
| Weighted conformal | 0.9270 | [0.9144, 0.9379] | No |

Weighting: v1.14-1.15 x2, v1.12-1.13 x0.5.

**Key finding:** the guarantee fails cross-band. But in-band also fails.
The failure is not solely cross-band shift. v1.16+ under-covers even with
same-band calibration.

Segment-level weighted conformal with cluster bootstrap in
`c3_weighted_segments.py`.

### Figure

`c3_shift_figure.png`, `c3_weighted_segments_figure.png`

### Scripts

`c3_shift.py`, `c3_weighted_segments.py`

---

## C4: Detection power and hand-labels

### Matched-coverage table

All methods calibrated at alpha = 0.05 on Group B, tested on Group C.

| Detector | False alarm rate | Detection power (top-10% sat_frac) |
|---|---|---|
| Isolation Forest | 4.91% | 5/65 (7.7%) |
| One-Class SVM | 5.06% | 3/65 (4.6%) |
| Autoencoder | 5.37% | 6/65 (9.2%) |
| **PX4 threshold** | **4.60%** | **0/65 (0.0%)** |

All four methods fire at the same rate (~5%). PX4 catches zero of the top
saturated flights.

### Hand-labels (50 flights examined)

| Label | Count |
|---|---|
| Genuine anomaly | 11 |
| Borderline | 14 |
| Not genuine | 25 |

### Detector agreement

| Detector | True positive | False positive | Precision | Recall |
|---|---|---|---|---|
| Isolation Forest | 1/11 | 3/25 | 0.25 | 0.091 |
| One-Class SVM | 0/11 | 1/25 | 0.00 | 0.000 |
| Autoencoder | 1/11 | 3/25 | 0.25 | 0.091 |
| Any detector | 1/11 | 3/25 | 0.25 | 0.091 |

Recall 9.1%. Precision 25%.

### Root cause

`eb456400` has zero ratio activity across all 10 ratios despite 179 s of
sustained motor imbalance (motors 0.03, 0.51, 0.02, 0.98). The EKF is
satisfied; the controller is fighting. Innovation test ratios measure
estimator confidence, not control effort. Motor imbalance is a control
problem, not an estimation problem.

### Figure

`c4_handlabel_figure.png` — three panels

### Scripts

`matched_coverage_table.py`, `c4_handlabel.py`, `c4_finalize.py`,
`c4_investigate.py`, `c4_figure.py`

---

## Contamination check

| Setting | Mean coverage |
|---|---|
| Full B (656 flights) | 0.9550 |
  <!-- | Clean B (379 flights) | 0.9397 -->
| **Random B (379, 200 iterations)** | **0.9553** |

Contamination effect is real, not sample size. Removing tripped flights
drops coverage by 1.5 pp. Reducing sample size drops it by 0.03 pp.

### Scripts

`contamination_check.py`, `contamination_control.py`

---

## Datasheet notes

### The 111 extreme ratio flights

Approximately 111 flights (6.8% of corpus) carry values above 10^10 in
`gps_hpos` and `gps_vpos`. Each has ~314 such samples. At 2 Hz that is
~157 seconds of sustained behavior. Not brief spikes.

The values are the squared normalized innovation:


## Additional analyses

### Matched-coverage curve

Sweep alpha from 0.01 to 0.30. Detection power vs false alarm rate for all
four detectors on the top-10% sat_frac flights.

| Detector | FA rate (alpha=0.05) | Power (k/65) |
|---|---|---|
| Isolation Forest | 4.91% | 5/65 (7.7%) |
| One-Class SVM | 5.06% | 3/65 (4.6%) |
| Autoencoder | 5.37% | 6/65 (9.2%) |
| PX4 threshold | 4.60% | 0/65 (0.0%) |

The PX4 curve sits below every other detector across the full false alarm
rate range from 0.5% to 25%. It is not a single-threshold artifact.

Figure: `matched_coverage_curve.png`.

### Alpha sensitivity of C2 and C3

Reran C2 and C3 at alpha = 0.01, 0.05, 0.10, 0.20.

C2 regime coverage:

| alpha | hover | translation | descent |
|---|---|---|---|
| 0.01 | 0.9905 | 0.9842 | 0.9892 |
| 0.05 | 0.9609 | 0.9401 | 0.9536 |
| 0.10 | 0.9077 | 0.8921 | 0.9053 |
| 0.20 | 0.8037 | 0.7987 | 0.8175 |

C3 cross-band and weighted:

| alpha | cross | weighted | gap from nominal (cross) |
|---|---|---|---|
| 0.01 | 0.9831 | 0.9815 | -0.007 |
| 0.05 | 0.9228 | 0.9270 | -0.027 |
| 0.10 | 0.8614 | 0.8714 | -0.039 |
| 0.20 | 0.7238 | 0.7508 | -0.076 |

The C2 regime gap and the C3 cross-band failure hold across all alphas.
Not artifacts of the 95% choice.

Figure: `alpha_sensitivity.png`.

### Reproducibility

`run_all.py` runs every analysis script in the correct order. One command
reproduces the complete paper. Runtime: approximately 90 minutes on a
standard laptop.

`matched_coverage_curve.py`, `alpha_sensitivity.py`, `run_all.py` are the
new scripts.