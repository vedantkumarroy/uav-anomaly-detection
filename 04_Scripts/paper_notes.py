notes = '''
============================================================
PAPER NOTES: METHODS AND RESULTS
============================================================

TITLE DRAFT:
Calibrated anomaly detection on open UAV flight logs

ABSTRACT DRAFT:
Open flight-log repositories like logs.px4.io provide a large
corpus of real UAV telemetry, but are heterogeneous across
firmware versions, airframes, and logging rates. We present a
calibrated anomaly detection pipeline that combines the EKF's
own innovation test ratios with split conformal prediction to
provide distribution-free coverage guarantees on individual
flights. Using 2,270 streamed logs from the PX4 public corpus,
we filter to 1,566 clean quadrotor flights, split them 20/40/40
into training/calibration/test groups, and compare four detection
methods at matched coverage. We find that Isolation Forest
provides the best detection power at 0.95 coverage, and that
the baro_vpos innovation ratio is the strongest single signal.

METHODS:

1. Data acquisition
   - Filter dbinfo for SYS_AUTOSTART == 4001 and Quadrotor mav_type
   - Stream logs directly from logs.px4.io without permanent
     disk storage
   - Write each uORB topic as a Parquet table with zstd compression
   - Achieved 50 percent compression

2. Runtime indicators
   - failsafe_frac: fraction of vehicle_status.failsafe == True
   - nav_state_forbidden_frac: fraction of nav_state in {10,17,18}
   - filter_fault_flags_frac: fraction of estimator_status.filter_fault_flags != 0
   - landed_at_end: whether vehicle_land_detected.landed is True at end
   - sat_frac: fraction of airborne samples with max motor >= 0.95

3. Ratio scores
   - 10 universal innovation ratios from estimator_innovation_test_ratios
   - Per-flight score: 95th percentile of each ratio over the flight
   - log(1+x) transform and clip at 99.9th percentile for stability

4. Conformal calibration
   - Split conformal with alpha = 0.05
   - Calibrate on Group B, evaluate on Group C
   - Threshold = kth smallest of B, k = ceil((n+1)(1-alpha))

5. Detectors
   - Track A: conformal on max-ratio score (no fitted model)
   - Track B: Isolation Forest, One-Class SVM, shallow Autoencoder
   - All fit on A, calibrated on B, tested on C

RESULTS:

1. Coverage at alpha = 0.05:
   Track A: 0.9521
   IsoForest: 0.9506
   SVM: 0.9506
   Autoencoder: 0.9553
   All within 0.5 pp of nominal

2. Detection power at coverage 0.95, n_indicators >= 2:
   IsoForest: 0.1386
   SVM: 0.1040
   Track A: 0.0957
   Autoencoder: 0.0891

3. Best single ratios (at n>=2):
   baro_vpos: 0.1485
   gps_hpos[0]: 0.1188
   gps_hvel[1]: 0.1188

4. Weakest single ratio:
   heading: 0.0594

5. Case study: only one log in Group C tripped all 5 indicators
   (0ae539bf-9dd1-4393-a645-98ab97a865d4, v1.16+)

FINDINGS:

1. Split conformal works on real UAV telemetry. Coverage is at
   nominal for all four methods.

2. Isolation Forest provides the best detection power at matched
   coverage. This suggests that a joint model of the ratio
   distribution catches anomalies that per-ratio thresholds miss.

3. Detection power is limited (10-14 percent at n>=2). The five
   indicators are proxies for ground truth, not labels. Future
   work should establish stronger labels.

4. The baro_vpos ratio is the strongest single innovation signal.
   This may reflect the sensitivity of the EKF's vertical
   position estimate to barometric drift.

LIMITATIONS:

1. The indicators are proxies. No ground truth labels exist.

2. Only 25 logs in Group C have 3 or more indicators. Statistical
   power for detection is limited.

3. Airframes are heterogeneous. 4001 is a default parameter, not
   a hardware guarantee. 41 of 1,835 actuator_motors logs have
   non-4 motor counts.

4. Sensor_combined and vehicle_angular_velocity topics dominate
   Parquet storage (56 percent). We did not investigate whether
   these can be dropped for downstream analysis.

5. Dropout cannot be recomputed from Parquet alone. The dropout
   gate is applied at stream time or not at all.
'''

print(notes)
print("Copy this to a text file for later use.")