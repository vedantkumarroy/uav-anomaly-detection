"""Replace segment-level C2/C3 tables with flight-block versions."""
from pathlib import Path

path = Path("RESULTS.md")
text = path.read_text(encoding="utf-8")

OLD_C2 = """| Regime | Mean coverage | 95% CI | Nominal inside? |
|---|---|---|---|
| Hover | 0.9609 | [0.9583, 0.9634] | No |
| Translation | 0.9401 | [0.9365, 0.9436] | No |
| Descent | 0.9536 | [0.9468, 0.9594] | Yes |

Hover over-covers by 1.1 pp. Translation under-covers by 1.0 pp. Marginal
guarantee holds, conditional does not."""

NEW_C2 = """Flight-block bootstrap. Unit of resampling is the flight, not the segment.
Segments within a flight are correlated (top 10% of flights carry 35% of
segments). The earlier segment-level bootstrap understated CI width.

**Contaminated calibration (all 650 B flights):**

| Regime | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Hover | 0.9570 | [0.9501, 0.9629] | No (by 0.0001) |
| Translation | 0.9441 | [0.9322, 0.9554] | Yes |
| Descent | 0.9583 | [0.9447, 0.9713] | Yes |

**Clean calibration (374 B flights, indicator-tripping removed):**

| Regime | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Hover | 0.9495 | [0.9420, 0.9561] | Yes |
| Translation | 0.9259 | [0.9124, 0.9389] | No |
| Descent | 0.9307 | [0.9123, 0.9484] | No |

Under contaminated calibration, translation and descent both include
nominal. This matches the earlier null classification. Under clean
calibration, translation and descent exclude nominal by 1.1 and 1.9 pp.
The classification is calibration-dependent."""

OLD_C3 = """| Method | Mean coverage | 95% CI | Nominal inside? |
|---|---|---|---|
| Cross-band | 0.9212 | [0.9081, 0.9325] | No |
| In-band reference | 0.9286 | [0.9161, 0.9393] | No |
| Weighted conformal | 0.9270 | [0.9144, 0.9379] | No |"""

NEW_C3 = """Flight-block bootstrap. Unit of resampling is the flight.

| Method | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Cross-band contaminated | 0.9228 | [0.9016, 0.9423] | No |
| Cross-band clean | 0.9079 | [0.8836, 0.9307] | No |

Both settings exclude nominal. The clean upper bound sits 1.9 pp below
0.95. Cross-band under-coverage is real under both calibrations.

Secondary rows (segment-level, not re-run under flight-block):

| Method | Mean coverage | 95% CI |
|---|---|---|
| In-band reference | 0.9286 | [0.9161, 0.9393] |
| Weighted conformal | 0.9270 | [0.9144, 0.9379] |"""

if OLD_C2 in text:
    text = text.replace(OLD_C2, NEW_C2)
    print("C2 table replaced.")
else:
    print("WARN: old C2 table not found. Skipped.")

if OLD_C3 in text:
    text = text.replace(OLD_C3, NEW_C3)
    print("C3 table replaced.")
else:
    print("WARN: old C3 table not found. Skipped.")

note_anchor = "| C4 | Scope result | EKF ratios cannot see actuator faults. Not a detection-power measurement. |"
note = note_anchor + """

Note: the C2 and C3 classifications above use the earlier segment-level
CIs. Under flight-block bootstrap, C2 translation and descent exclude
nominal under clean calibration, and C3 excludes nominal under both
calibrations. See the C2 and C3 sections."""

if "Note: the C2 and C3 classifications above" not in text and note_anchor in text:
    text = text.replace(note_anchor, note, 1)
    print("Four-claim note added.")
else:
    print("WARN: four-claim note skipped (either already present or anchor missing).")

path.write_text(text, encoding="utf-8")
print("done.")
