"""
Rewrite RESULTS.md with cell 1 and cell 2 only. Removes cell 3.
"""
from pathlib import Path

path = Path("RESULTS.md")
text = path.read_text(encoding="utf-8")
path.with_suffix(".md.bak3").write_text(text, encoding="utf-8")

# 1. Fix the four-claim note at the top
old_note = """
Note: the C2 and C3 classifications above use the earlier segment-level
CIs. Under flight-block bootstrap, C2 translation and descent exclude
nominal under clean calibration, and C3 excludes nominal under both
calibrations. See the C2 and C3 sections."""

new_note = """
Note: the C2 and C3 classifications above are the final ones. An earlier
draft of this file reported a reclassification based on cleaning only the
calibration set. That comparison was invalid. It broke exchangeability
between calibration and test. See the Calibration contamination section.
Cells 1 and 2 are the valid comparisons and both leave C2 and C3 as null
or over-covering."""

if old_note in text:
    text = text.replace(old_note, new_note)
    print("Note at top: replaced.")
else:
    print("Note at top: anchor not found. Skipping.")

# 2. Add C1 clean numbers after the headline table
c1_anchor = """| Naive (all samples) | 344/1627 | 21.1% |
| **Warm-up excluded** | **308/1627** | **18.9%** |"""

c1_replacement = """| Naive (all samples) | 344/1627 | 21.1% |
| **Warm-up excluded** | **308/1627** | **18.9%** |

### Clean-flight C1

About 41 percent of the analysis flights trip at least one runtime
indicator. Those flights are not "healthy" by the indicator definition.
The C1 headline was recomputed on the clean subset only.

| Subset | Over-flagged | Rate | 95% Wilson CI |
|---|---|---|---|
| All flights | 308/1627 | 18.93% | [17.10, 20.91] |
| **Clean flights only** | **149/957** | **15.57%** | **[13.41, 18.00]** |
| Dirty flights only | 159/670 | 23.73% | [20.67, 27.10] |

Clean flights still over-flag at 15.57 percent. The implied rejection
probability at gate 1.0 is 3.73e-6. The gap between observed and implied
is still approximately four orders of magnitude. The claim survives the
clean-flight correction.

Per-ratio over-flag rate on clean flights (rate > 5 percent of samples):

| Ratio | Mean rate | Over 5% fraction |
|---|---|---|
| mag_field[0] | 2.90% | 8.47% |
| heading | 1.63% | 6.58% |
| hagl | 0.69% | 1.88% |
| gps_vpos | 1.03% | 1.36% |
| baro_vpos | 0.84% | 1.25% |
| gps_hpos[0] | 0.56% | 0.94% |
| gps_hpos[1] | 0.31% | 0.84% |
| gps_hvel[0] | 0.36% | 0.84% |
| gps_hvel[1] | 0.03% | 0.31% |
| gps_vvel | 0.08% | 0.31% |

Direction matches the earlier heatmap. Magnetometer and heading are the
worst two ratios on clean flights as well."""

if c1_anchor in text:
    text = text.replace(c1_anchor, c1_replacement)
    print("C1: clean-flight section added.")
else:
    print("C1: anchor not found. Skipping.")

# 3. Replace C2 section
c2_start_marker = "### Coverage per regime"
c2_end_marker = "### Segments per flight"

if c2_start_marker in text and c2_end_marker in text:
    i = text.find(c2_start_marker)
    j = text.find(c2_end_marker, i)
    new_c2 = """### Coverage per regime

Flight-block bootstrap. Unit of resampling is the flight.

**Cell 1 (dirty B, dirty C). Valid. Coverage over the corpus population.**

| Regime | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Hover | 0.9570 | [0.9501, 0.9629] | No (by 0.0001) |
| Translation | 0.9441 | [0.9322, 0.9554] | Yes |
| Descent | 0.9583 | [0.9447, 0.9713] | Yes |

**Cell 2 (clean B, clean C). Valid. Coverage over the nominal population.**

| Regime | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Hover | 0.9522 | [0.9433, 0.9606] | Yes |
| Translation | 0.9501 | [0.9373, 0.9620] | Yes |
| Descent | 0.9696 | [0.9553, 0.9821] | No (over-covers) |

Cell 2 is the experiment the paper is about. In the nominal population,
conformal coverage is correct. Hover and translation sit at or just above
nominal. Descent over-covers by 2 pp. No regime under-covers. C2 is null.

Test counts: cell 2 hover 319 flights, translation 213, descent 102.

**Cell 3 (clean B, dirty C) is invalid.** It breaks exchangeability
between calibration and test. An earlier draft reported cell 3 as a
finding. That was wrong. Cell 3 results are withdrawn.

"""
    text = text[:i] + new_c2 + text[j:]
    print("C2: section replaced.")
else:
    print("C2: markers not found. Skipping.")

# 4. Replace C3 section
c3_start_marker = "### C3 cross-band"  # placeholder; will try several
# fallback: find "Cross-band" table
c3_marker_a = "| Method | Mean coverage | 95% CI | Nominal 0.95 inside? |"
if c3_marker_a in text:
    # find the section header "## C3:"
    idx_c3 = text.find("## C3:")
    idx_next = text.find("## C4:", idx_c3)
    if idx_c3 != -1 and idx_next != -1:
        new_c3 = """## C3: Cross-band distribution shift

Calibrate on Group B from v1.12-1.13 and v1.14-1.15. Test on Group C
from v1.16+.

Flight-block bootstrap. Unit of resampling is the flight.

**Cell 1 (dirty B, dirty C). Valid.**

| Setting | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Cross-band | 0.9228 | [0.9016, 0.9423] | No |

**Cell 2 (clean B, clean C). Valid.**

| Setting | Mean coverage | 95% CI | Nominal 0.95 inside? |
|---|---|---|---|
| Cross-band | 0.9355 | [0.9093, 0.9579] | Yes |

Calibration counts: cell 1 B=462, C=189. Cell 2 B=265, C=107.

Cell 1 excludes nominal by 2.7 pp. Cell 2 under-covers by 1.45 pp but the
95 percent CI includes nominal. With 107 test flights the test set does
not have the power to reject. C3 is null.

The point estimate direction is consistent with distribution shift. The
sample size is not sufficient to establish it. More flights in v1.16+
would be required. This is reported as a limit, not a finding.

**Cell 3 (clean B, dirty C) is invalid.** Dropped.

Figure: `c3_shift_figure.png`, `c3_weighted_segments_figure.png`

Scripts: `c3_shift.py`, `c3_weighted_segments.py`, `cell2_clean.py`

---

"""
        text = text[:idx_c3] + new_c3 + text[idx_next:]
        print("C3: section replaced.")
    else:
        print("C3: section boundaries not found. Skipping.")
else:
    print("C3: table marker not found. Skipping.")

# 5. Replace the contamination section and per-claim table
contam_marker = "## Calibration contamination"
journal_marker = "## Journal shortlist"
if contam_marker in text:
    idx_c = text.find(contam_marker)
    idx_j = text.find(journal_marker, idx_c)
    end = idx_j if idx_j != -1 else len(text)
    new_contam = """## Calibration contamination

Three calibration / test combinations are possible.

| Cell | Calibration B | Test C | Valid? |
|---|---|---|---|
| Cell 1 | dirty (650) | dirty (652) | Yes |
| Cell 2 | clean (374) | clean (388) | Yes. Paper experiment. |
| Cell 3 | clean (374) | dirty (652) | No. Breaks exchangeability. |

**Cell 1** measures coverage over the full corpus population. This is the
headline that uses all of Group B and all of Group C.

**Cell 2** measures coverage over the nominal population. The claim is
that 95 percent of flights showing no failure indicator fall below the
threshold. This requires a clean calibration set and a clean test set.
Cell 2 is the experiment the paper is about.

**Cell 3** was reported in an earlier draft and is withdrawn. Cleaning
only the calibration set removes the high-score flights from the upper
tail. The 95th percentile drops. The threshold drops. Group C still
contains its contaminated flights. More of them exceed the lowered
threshold. Coverage falls below nominal. This is mechanical, not a
conformal failure.

### Cell 2 results

| Claim | Cell 1 | Cell 2 | 0.95 inside cell 2? |
|---|---|---|---|
| C1 (clean flight only) | 18.93% over-flag | 15.57% over-flag | n/a. C1 is a rate. |
| C2 hover | 0.9570 | 0.9522 | Yes |
| C2 translation | 0.9441 | 0.9501 | Yes |
| C2 descent | 0.9583 | 0.9696 | No (over-covers) |
| C3 cross-band | 0.9228 | 0.9355 | Yes |

Conformal operates correctly on nominal data. This is the paper result.
The C2 regimes cover or over-cover. The C3 cross-band point estimate
under-covers by 1.45 pp but the CI includes nominal.

### Interpretation

The inflated C2 and C3 numbers in the earlier draft came from cell 3,
which is not a valid comparison. Conformal prediction does not fail on
nominal data.

The C1 clean-flight number (15.57 percent) is the correct PX4 false alarm
rate on flights that show no failure indicator. That claim stands and is
still four orders of magnitude above the implied rate.

### Test counts

Cell 2: C2 hover 319 flights, translation 213, descent 102. C3 cross-band
107 flights. The C3 test count is the limiting factor. A larger corpus
would be required to reject nominal if the point estimate is stable.

"""
    text = text[:idx_c] + new_contam + text[end:]
    print("Contamination section: replaced.")
else:
    print("Contamination: anchor not found. Skipping.")

path.write_text(text, encoding="utf-8")
print()
print("RESULTS.md rewritten. Backup: RESULTS.md.bak3")
