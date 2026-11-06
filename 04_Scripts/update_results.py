"""Update RESULTS.md: replace C4 section, insert new sections before Journal shortlist."""
import re
from pathlib import Path

path = Path("RESULTS.md")
text = path.read_text(encoding="utf-8")
path.with_suffix(".md.bak").write_text(text, encoding="utf-8")

NEW_C4 = """## C4: Detection power and hand-labels

### Matched-coverage table

All four detectors calibrated at alpha = 0.05 on Group B, tested on Group C.

| Detector | False alarm rate | Detection power (top-10% sat_frac) |
|---|---|---|
| Isolation Forest | 4.91% | 5/65 (7.7%) |
| One-Class SVM | 5.06% | 3/65 (4.6%) |
| Autoencoder | 5.37% | 6/65 (9.2%) |
| PX4 threshold | 4.60% | 0/65 (0.0%) |

All four methods fire at ~5%. The three fitted detectors catch 5 to 9 percent
of the top saturated flights. **PX4 catches zero.**

### Hand-labels

50 flights examined by hand. Classified based on motor means, spread and
sustained saturation.

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

### Named finding: eb456400

Log `eb456400` shows 179 seconds of sustained motor imbalance. Two motors
pinned at maximum. Two motors near zero. The four motor commands diverge.

All ten innovation test ratios stay flat at zero for the entire 179 seconds.
They never approach the 1.0 threshold.

The EKF is satisfied. The controller is fighting. **Estimator health and
control health are two different quantities. A detector that uses one
quantity cannot see failures in the other.**

### Interpretation

C4 is not a weak negative result. It is a scope result. The 9.1% recall is
not a fault of conformal calibration. Conformal calibration preserves
whatever signal the ratios carry. The bound is on the signal family: the
innovation test ratios cannot see actuator faults because they measure
estimator confidence, not control effort.

The separation between the detected and the missed flights is clean:

- The one detected flight had motor spread 0.13 (balanced motors, global
  saturation)
- The ten missed flights all had motor spread >= 0.891 (extreme imbalance)

### p-value note

One of eleven detections landing on the one low-spread flight by chance
gives p = 0.09. This is suggestive, not significant. Do not report the
p-value. The mechanism is the evidence. 179 seconds of imbalance with zero
EKF response is direct.

### Figure

`eb456400_figure.png` — 10 ratio traces (flat) and 4 motor commands
(divergent) on one time axis. The clearest single argument in the paper.

### Scripts

- `matched_coverage_table.py`, `matched_coverage_curve.py`
- `c4_handlabel.py`, `c4_finalize.py`, `c4_investigate.py`, `c4_figure.py`
- `eb456400_figure.py`
"""

NEW_SECTIONS = """## Four-claim summary

Sir's classification after the corrections round:

| Claim | Classification | Evidence |
|---|---|---|
| C1 | Strong quantitative | 53 orders of magnitude between nominal and observed false alarm rate |
| C2 | Null | Point estimates suggest a regime gap, cluster-robust CIs include nominal |
| C3 | Null | Cross-band point estimate under-covers, CIs include nominal |
| C4 | Scope result | EKF ratios cannot see actuator faults; eb456400 is direct evidence |

**One strong claim. One mechanism result. Two null results.**

The C4 reframe is what makes this acceptable. Before the reframe, C4 looked
like a failed experiment. With the reframe, C4 tells us something the paper
needs to say: the ratio signal family is structurally wrong for actuator
failures.

## Calibration contamination

Every headline number uses contaminated calibration. All 650 Group B flights
were used. No filter for indicator-tripping flights.

| Setting | Mean coverage | Comment |
|---|---|---|
| Contaminated (650 flights) | 0.9550 | Used in all headline results |
| Clean (374 flights) | 0.9397 | Reported for transparency |
| Random control (374, 200 iterations) | 0.9553 | Proves the drop is contamination, not sample size |

Contaminated flights have high scores. They sit in the upper tail. Keeping
them pushes the 95th percentile up. A higher threshold flags fewer test
flights. Coverage then looks better than it is.

**Contamination makes the detector less sensitive than reported.** This is
a small copy of the paper's thesis inside our own pipeline.

Both versions will be reported in the paper.

"""

lines = text.split("\n")
c4_idx = None
for i, line in enumerate(lines):
    if re.match(r"^##+ C4\b", line):
        c4_idx = i
        break
if c4_idx is None:
    print("ERROR: C4 header not found in RESULTS.md")
    raise SystemExit(1)

end_idx = None
for i in range(c4_idx + 1, len(lines)):
    if re.match(r"^## [^#]", lines[i]):
        end_idx = i
        break
if end_idx is None:
    print("ERROR: next top-level section after C4 not found")
    raise SystemExit(1)

new_lines = lines[:c4_idx] + NEW_C4.rstrip().split("\n") + [""] + lines[end_idx:]
text = "\n".join(new_lines)

js = text.find("## Journal shortlist")
if js == -1:
    print("ERROR: '## Journal shortlist' not found")
    raise SystemExit(1)
text = text[:js] + NEW_SECTIONS + text[js:]

path.write_text(text, encoding="utf-8")
print("OK: RESULTS.md updated.")
print("Backup: RESULTS.md.bak")
