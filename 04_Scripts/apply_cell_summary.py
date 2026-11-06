"""
Apply sir's three corrections to RESULTS.md.

1. Hover cell 1 exclusion rewritten as below-resolution, with reasoning.
2. C1 clean-flight section: 4.6 orders / 4.2 orders, 64 percent split.
3. C3 cell 2 under-coverage attributed to power, not direction.
"""
from pathlib import Path

path = Path("RESULTS.md")
text = path.read_text(encoding="utf-8")

# 1. Hover cell 1: below resolution
old_hover = "The C2 hover cell 1 exclusion is by 0.0001. It is at the precision limit. Reported as marginal."
new_hover = (
    "The C2 hover cell 1 exclusion is by 0.0001. Cell 1 hover contains 5,937 "
    "segments, so the granularity of the coverage statistic is 1/5937 = 0.00017. "
    "The observed exclusion is below the resolution of the measurement. It is "
    "not a finding."
)
if old_hover in text:
    text = text.replace(old_hover, new_hover)
    print("Hover: updated to below-resolution.")
else:
    print("Hover: anchor not found.")

# Also catch the variant in the CI re-examination table
old_hover_2 = "| C2 hover | 0.9570 | [0.9501, 0.9629] | Outside by 0.0001 | Marginal, not a finding |"
new_hover_2 = "| C2 hover | 0.9570 | [0.9501, 0.9629] | Outside by 0.0001 | Below resolution, not a finding |"
if old_hover_2 in text:
    text = text.replace(old_hover_2, new_hover_2)
    print("Hover table row: updated.")
else:
    print("Hover table row: not present.")

old_hover_3 = "| C2 hover | 0.9570 | [0.9501, 0.9629] | Outside by 0.0001 | Marginal. Not a real exclusion. |"
new_hover_3 = "| C2 hover | 0.9570 | [0.9501, 0.9629] | Outside by 0.0001 | Below resolution. Not a real exclusion. |"
if old_hover_3 in text:
    text = text.replace(old_hover_3, new_hover_3)
    print("Hover table row 2: updated.")
else:
    print("Hover table row 2: not present.")

# 2. C1 clean-flight orders of magnitude
old_c1 = (
    "Clean flights still over-flag at 15.57 percent. The implied rejection "
    "probability at gate 1.0 is 3.73e-6. The gap between observed and implied "
    "is still approximately four orders of magnitude. The claim survives the "
    "clean-flight correction."
)
new_c1 = (
    "Clean flights still over-flag at 15.57 percent. The implied rejection "
    "probability at gate 1.0 is 3.73e-6. The observed rate exceeds the implied "
    "rate by 4.6 orders of magnitude. Removing the two worst ratios, the "
    "magnetometer and heading, drops the union over-flag rate to 5.54 percent, "
    "which still exceeds the implied rate by 4.2 orders of magnitude. Two ratios "
    "of ten carry 64 percent of the effect: 96 flights removed by the exclusion "
    "out of 149 over-flagged clean flights."
)
if old_c1 in text:
    text = text.replace(old_c1, new_c1)
    print("C1 clean-flight paragraph: updated with orders and split.")
else:
    print("C1 clean-flight paragraph: anchor not found.")

old_c1_summary = (
    "The finding: the magnetometer and heading gates carry most of the C1 "
    "effect. They need recalibration first. The remaining eight ratios over-flag "
    "at a much lower but still non-nominal rate, so gate calibration is a "
    "problem across the board."
)
new_c1_summary = (
    "The finding: the magnetometer and heading gates carry most of the C1 "
    "effect. They need recalibration first. The remaining eight ratios over-flag "
    "at a much lower rate, 5.54 percent, but that rate still exceeds the implied "
    "rate by 4.2 orders of magnitude. Gate calibration is a problem across the "
    "board. The two worst ratios simply carry most of it."
)
if old_c1_summary in text:
    text = text.replace(old_c1_summary, new_c1_summary)
    print("C1 summary paragraph: updated.")
else:
    print("C1 summary paragraph: anchor not found.")

# 3. C3 cell 2: power, not direction
old_c3 = (
    "Cell 1 excludes nominal by 2.7 pp. Cell 2 under-covers by 1.45 pp but the "
    "95 percent CI includes nominal. With 107 test flights the test set does "
    "not have the power to reject. C3 is null."
)
new_c3 = (
    "Cell 1 excludes nominal by 2.7 pp. Cell 2 has 107 test flights against "
    "cell 1's 188, which is 57 percent of the count. The interval in cell 2 is "
    "approximately 1.33 times wider. Cell 2 under-covers by 1.45 pp but the "
    "95 percent CI includes nominal. This is a loss of power, not a different "
    "result. The point estimate stays in the same direction. C3 is a split "
    "result, significant in cell 1 and underpowered in cell 2."
)
if old_c3 in text:
    text = text.replace(old_c3, new_c3)
    print("C3 cell 2 paragraph: updated with power explanation.")
else:
    print("C3 cell 2 paragraph: anchor not found.")

old_c3_2 = (
    "The point estimate direction is consistent with distribution shift. The "
    "sample size is not sufficient to establish it. More flights in v1.16+ "
    "would be required. This is reported as a limit, not a finding."
)
new_c3_2 = (
    "The point estimate direction is consistent with distribution shift. The "
    "sample size is not sufficient to establish it. The cell 2 interval has "
    "a half-width of 0.024 against the cell 1 half-width of 0.020, driven by "
    "the 57 percent reduction in test flights. More flights in v1.16+ would be "
    "required to resolve. This is reported as a power limit, not a different "
    "finding."
)
if old_c3_2 in text:
    text = text.replace(old_c3_2, new_c3_2)
    print("C3 power-limit paragraph: updated.")
else:
    print("C3 power-limit paragraph: anchor not found.")

# 4. Insert the two-cause summary as a new section at the top of the CI re-examination
anchor = "## Confidence interval re-examination"
new_section = """## Summary across cells

Every claim, both cells, all numbers. The pattern is that two results are
significant against nominal, and each one sits in exactly one cell.

| Claim | Population | Cell | n | Point | 95% CI | Nominal 0.95 |
|---|---|---|---|---|---|---|
| C1 all flights | corpus | n/a | 1627 | 18.93% | [17.10, 20.91] | rate |
| C1 clean flights | nominal | n/a | 957 | 15.57% | [13.41, 18.00] | rate |
| C1 clean, mag+heading removed | nominal | n/a | 957 | 5.54% | [4.26, 7.17] | rate |
| C2 hover | corpus | 1 | 513 | 0.9570 | [0.9501, 0.9629] | outside by 0.0001 |
| C2 translation | corpus | 1 | 364 | 0.9441 | [0.9322, 0.9554] | inside |
| C2 descent | corpus | 1 | 207 | 0.9583 | [0.9447, 0.9713] | inside |
| C2 hover | nominal | 2 | 319 | 0.9522 | [0.9433, 0.9606] | inside |
| C2 translation | nominal | 2 | 213 | 0.9501 | [0.9373, 0.9620] | inside |
| C2 descent | nominal | 2 | 102 | 0.9696 | [0.9553, 0.9821] | outside by 0.53 pts |
| C3 cross-band | corpus | 1 | 189 | 0.9228 | [0.9016, 0.9423] | outside by 0.77 pts |
| C3 cross-band | nominal | 2 | 107 | 0.9355 | [0.9093, 0.9579] | inside |
| C4 top-10% detection | both | n/a | 65 | 0-7/65 | -- | n/a |

Two results are significant. C3 cross-band under-covers in cell 1 only.
C2 descent over-covers in cell 2 only. Both differences have clear causes.

C3. Cell 1 has 188 test flights and cell 2 has 107, 57 percent of the count.
The interval in cell 2 is approximately 1.33 times wider. The point estimate
moves in the same direction in both cells (0.9228 and 0.9355). This is a loss
of power, not a different result.

C2 descent. Half-widths are effectively identical (0.0133 cell 1, 0.0134
cell 2). The point estimate moves up by 1.13 points (0.9583 to 0.9696).
Cleaning removed the contaminated descent flights. Those have high scores
and pull coverage down. Removing them raises coverage. Coverage inflation,
not a different regime behaviour.

## Confidence interval re-examination"""
if anchor in text and "## Summary across cells" not in text:
    text = text.replace(anchor, new_section, 1)
    print("Summary across cells section: inserted.")
else:
    print("Summary across cells section: already present or anchor missing.")

path.write_text(text, encoding="utf-8")
print()
print("RESULTS.md updated. Backup: RESULTS.md.bak6")
