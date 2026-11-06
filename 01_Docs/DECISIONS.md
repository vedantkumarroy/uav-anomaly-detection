# Decisions log

Every analysis choice made in the pipeline, with the reason. Reviewers may
ask about any of these. Ordered by when they appear in the pipeline.

## Data collection

**Only real flights, no simulation.**
SITL flights do not exhibit sensor faults or estimator faults of the kind
the project measures. They would inflate the healthy rate. Excluded at the
metadata stage.

**Only SYS_AUTOSTART = 4001.**
The project claims a single platform class. 4001 is the PX4 standard
quadrotor. Hexarotor and octorotor frames share much of the estimator code
but have different actuator redundancy and different failure modes. Mixing
them would confound the actuator fault analysis in C4.

**Only logs from 2023 onward.**
The uORB topic `estimator_innovation_test_ratios` was introduced in the
1.12 firmware line. Logs before this use `estimator_status` and expose a
different ratio schema. Restricting to 2023 onward captures the modern
schema without the legacy branch.

**Only logs exposing `estimator_innovation_test_ratios`.**
Legacy logs falling back to `estimator_status` expose ratios under
different field names and different scaling. Including them would require
a second extraction path and confound the ratio definition. About 15
percent of the corpus uses the legacy schema and is dropped.

## Screening gates

**Position rate >= 5 Hz.**
EKF2 expects position updates at 5 Hz or better from the active position
source. Below this the EKF is running on stale measurements and the
innovation statistics are not representative of nominal operation.

**Attitude rate >= 10 Hz.**
Same reasoning as position. Attitude innovation is gated at 10 Hz in EKF2.

**Ratio rate >= 1.5 Hz.**
The `estimator_innovation_test_ratios` topic publishes at the slowest of
the innovation update rates. The observed median is 2 Hz. Logs publishing
below 1.5 Hz have too few ratio samples per flight for a stable 95th
percentile. The 1.5 Hz gate is 25 percent below the median to allow for
normal jitter without rejecting real flights.

**Duration >= 60 s.**
Short flights do not produce enough regime segments to support the regime
analysis in C2. Sixty seconds is short enough to keep brief but valid
flights, long enough to give a median of several segments per flight.

**Airborne >= 45 s.**
Ground-only logs would contribute zero regime segments and zero actuator
activity. The 45-second threshold is chosen so that takeoff, hover and
landing all have a chance to appear.

**Dropout fraction <= 0.02.**
The dropout proxy is the fraction of inter-sample gaps larger than ten
times the median interval, computed per topic and taking the maximum
across `vehicle_attitude`, `vehicle_local_position`, `sensor_combined`,
and `estimator_innovation_test_ratios`. A value above 2 percent indicates
logging that is unreliable enough that per-flight statistics are not
comparable to the rest of the corpus.

## Per-flight features

**95th percentile over the flight, not maximum.**
The maximum is sensitive to single-sample spikes and to the extreme-value
flights documented in the datasheet. The 95th percentile captures
sustained deviation while ignoring isolated outliers.

**log(1 + x) transform.**
The ratio distribution has a heavy right tail. The log transform
compresses it and stabilizes the fitted detectors. Per-ratio coverage at
a quantile is invariant under the transform.

**99.9th percentile clip computed on Group B only.**
Clip on all data would let Group A and Group C influence preprocessing.
The clip value is estimated from Group B and applied unchanged to A and C.
Fix applied in `fix_outliers_v2.py`.

## Group split

**20 / 40 / 40 within each firmware band.**
Within-band splitting prevents a firmware version from appearing only in
one group. Group A is used only to fit detectors that require training.
Group B is the conformal calibration set. Group C is held out.

**Group B is the calibration set.**
Split conformal requires a calibration set separate from any model
training. Group B plays this role for all claims.

## Claim definitions

**C1 warm-up exclusion: first 30 seconds.**
The EKF origin is not set at t = 0 and the first GPS fix may not have
arrived. Ratios in the first 30 seconds reflect initialization, not
nominal operation. The 30-second window is chosen so that the EKF is
settled in all bands while excluding only the initialization transient.
The full-sample number is reported alongside.

**C2 minimum segment duration: 5 seconds.**
Below this duration the 95th percentile of the ratio series within a
segment is dominated by sampling noise. Five seconds gives roughly ten
ratio samples at the observed 2 Hz publish rate.

**C3 calibration bands: v1.12 to v1.13 and v1.14 to v1.15.**
Test band v1.16+. This split isolates the largest documented schema and
estimator changes, which occur at the 1.14 and 1.16 boundaries.

**C4 top-10 percent by sat_frac.**
`sat_frac` is the fraction of the flight with any actuator at saturation.
The top ten percent of Group C by this measure are the flights most
likely to carry a genuine actuator fault. Ten percent gives 65 flights,
enough to measure detection power with a Wilson interval.

## Contamination definitions

**Tripped flight = any of five indicators.**
`failsafe_frac > 0`, `nav_state_forbidden_frac > 0`,
`filter_fault_flags_frac > 0`, `landed_at_end == False`,
`sat_frac > 0.05`. Any one of these is a signal that something abnormal
happened during the flight. The 0.05 threshold on sat_frac matches the
"five percent of samples" rule used elsewhere.

**Cell definitions.**
- Cell 1: dirty B, dirty C. Valid. Coverage over the corpus population.
- Cell 2: clean B, clean C. Valid. Coverage over the nominal population.
- Cell 3: clean B, dirty C. Invalid. Breaks exchangeability. Dropped.

**Cell 2 is the paper experiment.**
The claim is that 95 percent of flights showing no failure indicator fall
below the threshold. That claim requires clean calibration and clean test.

## Resampling

**Flight as the bootstrap block.**
Segments within a flight are correlated. The top 10 percent of flights
carry 35.4 percent of segments. Resampling at segment level understates
CI width by roughly a factor of 2 to 3. Flight is the natural independent
unit.

**2000 bootstrap iterations.**
Enough for stable 2.5 and 97.5 percentile estimates. Runs in under a
minute per claim.
