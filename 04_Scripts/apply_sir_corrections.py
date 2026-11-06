"""
Apply sir's six corrections to RESULTS.md.
"""
from pathlib import Path

path = Path("RESULTS.md")
text = path.read_text(encoding="utf-8")
path.with_suffix(".md.bak5").write_text(text, encoding="utf-8")

# 1. Replace the top four-claim table
old_table = """| Claim | Classification | Result |
|---|---|---|
| C1 | Strong quantitative claim | 18.9% of healthy flights over-flag. 53 orders of magnitude gap between implied and observed. Firmware degradation monotonic. |
| C2 | Null result | Regime coverage differences not significant under cluster-robust CIs |
| C3 | Null result | Point estimates under-cover cross-band, but cluster-robust intervals include nominal |
| C4 | Scope result | EKF ratios cannot see actuator faults. Not a detection-power measurement. |"""

new_table = """| Claim | Classification | Result |
|---|---|---|
| C1 | Strong quantitative claim | Clean flights over-flag at 15.57%. Driven by magnetometer and heading. Without those two ratios the rate is 5.54%. Implied rate is 3.73e-6. Gap is four orders of magnitude. |
| C2 | Regime-dependent | Descent over-covers by 2.0 points, significant. Hover and translation sit at nominal. |
| C3 | Split result | Significant over the corpus population (cell 1, nominal outside CI). Underpowered over the nominal population (cell 2, nominal inside CI, 107 test flights). |
| C4 | Scope result | EKF ratios cannot see actuator faults. Not a detection-power measurement. |

Four results. No null results."""

if old_table in text:
    text = text.replace(old_table, new_table)
    print("Top four-claim table: replaced.")
else:
    print("WARN: top four-claim table not found.")

# 2. Remove the outdated note after the table
old_note = """
Note: the C2 and C3 classifications above are the final ones. An earlier
draft of this file reported a reclassification based on cleaning only the
calibration set. That comparison was invalid. It broke exchangeability
between calibration and test. See the Calibration contamination section.
Cells 1 and 2 are the valid comparisons and both leave C2 and C3 as null
or over-covering."""

if old_note in text:
    text = text.replace(old_note, "")
    print("Old note: removed.")
else:
    print("Old note: not present.")

# 3. Append the CI re-examination and the new findings
addition = """

---

## Confidence interval re-examination

Sir's correction on how to read a confidence interval. The rule is: does
the target value (0.95 for coverage claims) lie inside the interval, not
whether the point estimate lies inside its own interval. Every CI
re-examined under the correct rule.

### Cell 1 (dirty B, dirty C). Corpus population.

| Claim | Point | 95% CI | Nominal 0.95 | Verdict |
|---|---|---|---|---|
| C2 hover | 0.9570 | [0.9501, 0.9629] | Outside, by 0.0001 | Marginal. Not a real exclusion. |
| C2 translation | 0.9441 | [0.9322, 0.9554] | Inside | At nominal |
| C2 descent | 0.9583 | [0.9447, 0.9713] | Inside | At nominal |
| C3 cross-band | 0.9228 | [0.9016, 0.9423] | Outside, by 0.77 pts | Significant under-coverage |

### Cell 2 (clean B, clean C). Nominal population.

| Claim | Point | 95% CI | Nominal 0.95 | Verdict |
|---|---|---|---|---|
| C2 hover | 0.9522 | [0.9433, 0.9606] | Inside | At nominal |
| C2 translation | 0.9501 | [0.9373, 0.9620] | Inside | At nominal |
| C2 descent | 0.9696 | [0.9553, 0.9821] | Outside, by 0.53 pts | Significant over-coverage |
| C3 cross-band | 0.9355 | [0.9093, 0.9579] | Inside | Underpowered. 107 test flights. |

The C2 hover cell 1 exclusion is by 0.0001. It is at the precision limit.
It is reported as marginal, not as a finding.

## What descent over-coverage means

Over-coverage gives fewer alarms than nominal. Descent is the phase where
a failure is most difficult to recover. In that phase the detector fires
less often than the nominal rate. So descent has less protection than the
nominal guarantee implies.

The physics agrees. Descent has low dynamics. Innovations stay small.
Scores sit below the pooled calibration distribution. The threshold is
slightly too high for this regime.

Two caveats. The descent and translation cell 2 intervals overlap between
0.9553 and 0.9620. The regime-to-regime difference is not established.
Only descent against nominal is established. The exclusion is 0.53 points.
Significant. Not large.

## C1 without magnetometer and heading

The C1 per-ratio table shows the effect is concentrated.

| Ratio | Clean flights over-flagged | Rate |
|---|---|---|
| mag_field[0] | 75/957 | 7.84% |
| heading | 63/957 | 6.58% |
| hagl | 18/957 | 1.88% |
| gps_vpos | 13/957 | 1.36% |
| baro_vpos | 12/957 | 1.25% |
| gps_hpos[0] | 9/957 | 0.94% |
| gps_hpos[1] | 8/957 | 0.84% |
| gps_hvel[0] | 8/957 | 0.84% |
| gps_hvel[1] | 3/957 | 0.31% |
| gps_vvel | 3/957 | 0.31% |

Union over-flag rate on clean flights:

| Subset | Over-flagged | Rate | 95% Wilson CI |
|---|---|---|---|
| All 10 ratios | 149/957 | 15.57% | [13.41, 18.00] |
| Magnetometer and heading removed | 53/957 | 5.54% | [4.26, 7.17] |

Removing the top two ratios drops the union rate to 5.54 percent. The
remaining eight ratios still over-flag. At gate 1.0 the implied rejection
probability is 3.73e-6. The gap is still approximately four orders of
magnitude even without mag and heading.

The finding: the magnetometer and heading gates carry most of the C1
effect. They should be recalibrated first. The remaining eight ratios
over-flag at a much lower but still non-nominal rate, so gate calibration
is a problem across the board.

## Clean versus dirty: independent validation of the indicator rule

| Subset | Over-flagged | Rate | 95% Wilson CI |
|---|---|---|---|
| Clean flights | 149/957 | 15.57% | [13.41, 18.00] |
| Dirty flights | 159/670 | 23.73% | [20.67, 27.10] |

The two intervals do not overlap. Flights that trip an indicator over-flag
significantly more than clean flights. This is independent evidence that
the five-indicator rule identifies something the EKF also finds. The rule
is not arbitrary.

This goes in the Data section as validation of the clean / dirty split.
"""

if "## Confidence interval re-examination" not in text:
    text = text.rstrip() + "\n" + addition
    print("New sections: appended.")
else:
    print("New sections: already present.")

path.write_text(text, encoding="utf-8")
print()
print("RESULTS.md updated. Backup: RESULTS.md.bak4")
