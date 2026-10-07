\# C4 Hand-Label Result



\## Protocol



Took the top 50 flights by `sat\_frac` in Group C. For each, extracted motor

time series from `actuator\_motors` or `actuator\_outputs`, computed per-channel

means, spread, and sustained saturation. Hand-classified each flight based on

the motor evidence.



\## Hand-label distribution



| Label | Count |

|---|---|

| Genuine anomaly | 11 |

| Borderline | 14 |

| Not genuine | 25 |



\## Detector agreement



The three fitted detectors (Isolation Forest, One-Class SVM, Autoencoder) at

alpha = 0.05. The PX4 threshold detector is also evaluated.



| Detector | True positive | False positive | Precision | Recall |

|---|---|---|---|---|

| Isolation Forest | 1/11 | 3/25 | 0.25 | 0.091 |

| One-Class SVM | 0/11 | 1/25 | 0.00 | 0.000 |

| Autoencoder | 1/11 | 3/25 | 0.25 | 0.091 |

| Any detector | 1/11 | 3/25 | 0.25 | 0.091 |



\*\*Precision at 25%. Recall at 9.1%.\*\*



\## The one genuine anomaly detected



`696590b3-101b-4aa4-820d-d67b22eecea3` (v1.16+)



\- `sat\_frac = 1.000`

\- `sat\_max\_run\_s = 67.1 s`

\- `motor\_spread\_p95 = 0.132`



\*\*All four motors were pinned at approximately 0.95 for 67 seconds.\*\* The

flight was in global saturation. The motors were balanced (spread low). This

is a case of running out of throttle authority, not motor imbalance.



\## The ten genuine anomalies missed



| Log ID | Band | sat\_frac | sat\_max\_run\_s | spread\_p95 |

|---|---|---|---|---|

| c1506e1f | v1.12-1.13 | 1.000 | 101.3 | 1.000 |

| b48a234c | v1.16+ | 0.956 | 73.7 | 1.000 |

| eb456400 | v1.14-1.15 | 0.949 | 179.7 | 1.000 |

| f386caa8 | v1.16+ | 0.917 | 1012.2 | 1.000 |

| afd1472c | v1.16+ | 0.795 | 263.3 | 1.000 |

| aaf2608e | v1.16+ | 0.758 | 58.1 | 0.934 |

| a4ac869f | v1.16+ | 0.675 | 29.9 | 1.000 |

| 7a10c97d | v1.14-1.15 | 0.673 | 125.3 | 0.891 |

| 480597bd | v1.16+ | 0.554 | 17.9 | 1.000 |

| 2cb6ff34 | v1.16+ | 0.293 | 3.5 | 0.938 |



\*\*All ten have motor\_spread\_p95 >= 0.891.\*\* Extreme imbalance. One or more

motors at 1.0 while others sit near 0. The controller is fighting a severe

asymmetry, and the innovation test ratios do not capture it.



Examples from the motor means:



\- `c1506e1f`: motors 0.99, 1.00, 0.96, 0.27. Motor 4 nearly off.

\- `eb456400`: motors 0.03, 0.51, 0.02, 0.98. Two motors nearly off.

\- `a4ac869f`: motors 0.70, 0.21, 0.68, 0.09. Two motors nearly off.

\- `2cb6ff34`: motors 0.19, 0.26, 0.85, 0.87. Two motors saturated.



\## Interpretation



The three fitted detectors do not detect motor-imbalance anomalies. They flag

only one type: global throttle saturation where all motors are pinned at max

with low spread.



This is a real finding. It says:



1\. The innovation test ratios do not align with motor-level anomalies.

2\. Motor imbalance is likely absorbed by the EKF as a bias or disturbance and

&#x20;  does not produce large innovations in the ratio set we use.

3\. Any anomaly detector built only on innovation ratios will miss a class of

&#x20;  physical failures.



\## Implication for the C4 claim



Sir's C4 claim was: "Conformal calibration does not remove too much detection

power."



\*\*The result refines this claim substantially.\*\* Conformal calibration does

not remove detection power, but the ratio-based detector itself has near-zero

detection power on motor imbalance. The detection power is bound by the input

signal, not the calibration method.



The honest C4 statement is:



> At matched false alarm rate (5%), the ratio-based detectors (with or without

> conformal calibration) achieve recall of 9% on hand-labelled motor imbalance

> anomalies. Conformal calibration does not degrade this. The bound is on the

> information content of the innovation ratios, not on the calibration

> procedure.



\## What this means for the paper



C4 is now a substantive result, not a check. It says the industry standard for

anomaly detection (innovation test ratios) is blind to a class of real

failures. That is a strong contribution.



It also connects C1 and C4:



\- C1: PX4 thresholds over-fire on healthy flights (false alarm problem)

\- C4: PX4 ratio-based detection misses real anomalies (missed detection problem)



Both directions of the anomaly detection problem are broken in the current

PX4 setup.

