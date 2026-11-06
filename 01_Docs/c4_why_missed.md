\# Why ratio-based detectors miss motor imbalance



\## The comparison



Three flights from Group C, examined at the ratio level:



1\. `696590b3` — detected by IsoForest and AE. Global throttle saturation, motors balanced.

2\. `c1506e1f` — missed by all. Severe imbalance, two motors near 0.

3\. `eb456400` — missed by all. Severe imbalance, two motors near 0.



Fraction of ratio samples exceeding the PX4 threshold 1.0:



| Ratio | 696590b3 (caught) | c1506e1f (missed) | eb456400 (missed) |

|---|---|---|---|

| baro\_vpos | 0.000 | 0.000 | 0.000 |

| gps\_hpos\[0] | 0.004 | 0.000 | 0.000 |

| gps\_hpos\[1] | 0.003 | 0.000 | 0.000 |

| gps\_hvel\[0] | 0.045 | 0.000 | 0.000 |

| gps\_hvel\[1] | 0.035 | 0.000 | 0.000 |

| gps\_vpos | 0.000 | 0.000 | 0.000 |

| gps\_vvel | 0.010 | 0.000 | 0.000 |

| heading | 0.185 | 0.145 | 0.000 |

| \*\*mag\_field\[0]\*\* | \*\*0.416\*\* | \*\*0.000\*\* | \*\*0.000\*\* |

| hagl | 0.000 | 0.000 | 0.000 |



The caught flight has substantial activity in `mag\_field\[0]`, `heading`,

`gps\_hvel\[0,1]`, and `gps\_vvel`. The two missed flights have almost nothing.

`eb456400` is essentially silent across all 10 ratios despite 179 seconds of

sustained saturation with motors 0.03, 0.51, 0.02, 0.98.



\## The interpretation



The innovation test ratio measures how far sensor data deviates from the

EKF's prediction.



\- When the vehicle experiences \*\*global throttle saturation\*\* (all motors

&#x20; pinned at max, spread low), the EKF is itself confused. Innovations rise.

&#x20; Ratios cross threshold. The detector fires.



\- When the vehicle experiences \*\*motor imbalance\*\* (one or two motors pinned

&#x20; at max, the rest nearly off), the EKF predicts the vehicle's motion

&#x20; correctly. The controller compensates for the imbalance by commanding

&#x20; higher throttle to the strong motors. The vehicle still flies the intended

&#x20; path. Innovations stay small. Ratios stay flat. The detector does not fire.



\*\*Motor imbalance is a control problem, not an estimation problem. The

innovation test ratio is an estimator-internal signal. It cannot see

control-side stress.\*\*



\## Implication for the paper



The C4 result is not "conformal calibration removes detection power." It is:



> The ratio-based detection signal is blind to motor imbalance by

> construction. Conformal calibration preserves whatever signal the ratios

> carry. The bound on detection power is set by the input signal, not the

> calibration method.



\## The unified story across claims



\- \*\*C1:\*\* PX4 ratios over-fire on healthy flights (false alarm problem)

\- \*\*C4:\*\* PX4 ratios miss real anomalies (missed detection problem)

\- \*\*Root cause for both:\*\* innovation test ratios measure estimator

&#x20; confidence, not actuator health



This is a coherent narrative for the paper. The contribution is not just

"conformal is better than fixed thresholds." It is:



> \*\*The industry-standard anomaly detection signal in PX4 is measuring the

> wrong thing. It is neither correctly calibrated (C1) nor sensitive to

> physical failure modes (C4).\*\*



\## What an alternative detector would need



To catch motor imbalance, the input signal must be control-side:

\- Actuator saturation fraction (our `sat\_frac`)

\- Motor spread (our `spread\_p95`)

\- Control effort relative to available authority



None of these are in the innovation ratio set. Adding them as features would

give a detector a chance. That is a natural follow-up for the paper.

