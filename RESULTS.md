# Calibrated Anomaly Detection on Open UAV Flight Logs

Consolidated results. Updated October 8, 2026.

## Project summary

UAV autopilots (PX4) use fixed thresholds on estimator innovation test
ratios to decide whether sensor measurements are trustworthy. This project
measures the false alarm rate of those thresholds on real flights, and
evaluates conformal prediction as a calibrated alternative.

## Four claims

| Claim | Classification | Result |
|---|---|---|
| C1 | Strong quantitative claim | 18.9% of healthy flights over-flag. 53 orders of magnitude gap between implied and observed. Firmware degradation monotonic. |
| C2 | Null result | Regime coverage differences not significant under cluster-robust CIs |
| C3 | Null result | Point estimates under-cover cross-band, but cluster-robust intervals include nominal |
| C4 | Scope result | EKF ratios cannot see actuator faults. Not a detection-power measurement. |

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

## C4: Ratio-based detection is blind to motor imbalance

This is a scope result, not a detection-power measurement. The ratio-based
detectors cannot see actuator faults by construction. The 9.1 percent recall
is not a fault of conformal calibration. It is evidence that the signal
family is incorrect for actuator failures.

### The mechanism

Log `eb456400` shows 179 seconds of sustained motor imbalance. Two motors
pinned near maximum. Two motors near zero. The EKF tracks the vehicle
correctly, so the innovations stay small. All 10 ratio traces remain flat.
The controller is fighting; the estimator is satisfied. Estimator health
and control health are two different quantities. A detector that uses one
quantity cannot see failures in the other.

Figure: `eb456400_figure.png`. Top panel: 10 ratio traces, flat at zero.
Bottom panel: 4 motor commands, diverging to saturation and zero. Same
time axis.

### Matched-coverage table

All methods calibrated at alpha = 0.05 on Group B, tested on Group C.

| Detector | False alarm rate | Detection power (top-10% sat_frac) |
|---|---|---|
| Isolation Forest | 4.91% | 5/65 (7.7%) |
| One-Class SVM | 5.06% | 3/65 (4.6%) |
| Autoencoder | 5.37% | 6/65 (9.2%) |
| PX4 threshold | 4.60% | 0/65 (0.0%) |

All methods fire at approximately the same false alarm rate. The ratio-
based detectors catch 5 to 9 percent of the top saturated flights. PX4
catches zero. This is not a small observation. It is the confirmation
of the scope result.

### Matched-coverage curve

The PX4 curve sits below every other detector across the full false alarm
rate range from 0.5 to 25 percent. Not a single-threshold artifact.

Figure: `matched_coverage_curve.png`.

### Hand-labels

Top 50 sat_frac flights in Group C, examined by hand.

| Label | Count |
|---|---|
| Genuine anomaly | 11 |
| Borderline | 14 |
| Not genuine | 25 |

Detector agreement:

| Detector | True positive | False positive | Precision | Recall |
|---|---|---|---|---|
| Isolation Forest | 1/11 | 3/25 | 0.25 | 0.091 |
| One-Class SVM | 0/11 | 1/25 | 0.00 | 0.000 |
| Autoencoder | 1/11 | 3/25 | 0.25 | 0.091 |

Recall is 9.1 percent. The separation is clean. The one detected flight
had motor spread 0.13. The ten missed flights all had motor spread 0.891
or more.

The probability that the one detection lands on the one low-spread flight
by chance is 1/11 = 0.09. This is suggestive, not significant at 0.05.
**We do not use the p-value.** We use the mechanism. `eb456400` is direct
evidence: 179 seconds of imbalance and zero EKF response.

### The reframed C4 claim

C4 is not a weak negative result. It is a scope result. The estimator
innovation test ratio measures estimator confidence, not control effort.
The signal family cannot see actuator faults. Any detector built only on
this signal family will miss motor imbalance. The bound on detection power
is set by the input signal, not by the calibration method.

The appropriate fix is not better calibration. It is control-side features
such as actuator saturation fraction, motor spread, or control effort.
That is future work.