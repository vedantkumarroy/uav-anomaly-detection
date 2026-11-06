"""Append calibration contamination section and per-claim table to RESULTS.md."""
from pathlib import Path

path = Path("RESULTS.md")
text = path.read_text(encoding="utf-8")

ADDITION = """
---

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

Both versions are reported in the paper.

### Per-claim comparison: contaminated vs clean calibration

Every claim below is reported with contaminated B (all 650 flights) and with
clean B (374 flights, indicator-tripping removed).

| Claim | Contaminated | Clean | Note |
|---|---|---|---|
| C1 false alarm rate | 18.9% | 18.9% | C1 does not use B. Identical. |
| C2 hover coverage | 0.9609 | 0.9533 | Both above nominal |
| C2 translation coverage | 0.9401 | 0.9187 | Clean under-covers by 3.1 pp |
| C2 descent coverage | 0.9536 | 0.9215 | Clean under-covers by 2.9 pp |
| C3 cross-band coverage | 0.9228 | 0.9079 | Clean under-covers by 4.2 pp |
| C4 IsoForest power | 5/65 (7.7%) | 7/65 (10.8%) | Clean raises sensitivity |
| C4 SVM power | 4/65 (6.2%) | 6/65 (9.2%) | Clean raises sensitivity |
| C4 AE power | 6/65 (9.2%) | 7/65 (10.8%) | Clean raises sensitivity |

Direction is consistent across all claims. Contaminated B raises the
conformal threshold. A higher threshold means fewer test flights flagged.
Coverage goes up, sensitivity goes down. Clean B reverses both. This is
the contamination thesis on every claim at once.

Clean calibration still leaves C4 detection power at 10.8 percent or below.
The scope result does not depend on the contamination choice.
"""

if "## Calibration contamination" in text:
    print("SKIP: section already present")
else:
    text = text.rstrip() + "\n" + ADDITION
    path.write_text(text, encoding="utf-8")
    print("OK: contamination section and per-claim table appended.")
