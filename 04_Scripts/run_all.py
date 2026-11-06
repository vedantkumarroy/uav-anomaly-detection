"""
Reproduce all paper results from committed data.

Usage:
    python run_all.py

Runs each analysis script in sequence. Produces all figures and CSVs
in the working directory. Requires the Parquet logs (gitignored) to be
present in pilot_2270/logs and pilot_extra/logs.
"""

import os
import sys
import time
import subprocess

# Order matters: some scripts consume outputs of earlier scripts
STEPS = [
    # C1
    ("C1: false alarm rate (all samples)", "c1_false_alarm.py"),
    ("C1: warm-up excluded", "c1_false_alarm_v2.py"),
    ("C1: per-band breakdown", "c1_per_band.py"),
    ("C1: threshold sensitivity", "c1_threshold_sensitivity.py"),
    ("C1: extreme-value sensitivity", "c1_exclude_extremes.py"),
    ("C1: heatmap figure", "c1_heatmap.py"),
    ("C1: bar chart figure", "c1_figure.py"),

    # C2
    ("C2: regime segmentation and coverage", "regime_c2.py"),
    ("C2: segment distribution", "segment_distribution.py"),
    ("C2: paper-quality figure", "c2_figure.py"),

    # C3
    ("C3: cross-band and weighted", "c3_shift.py"),
    ("C3: segment-level with block bootstrap", "c3_weighted_segments.py"),
    ("C3: CI computation", "c3_ci.py"),

    # C4
    ("C4: matched-coverage table", "matched_coverage_table.py"),
    ("C4: matched-coverage curve", "matched_coverage_curve.py"),
    ("C4: hand-label worksheet", "c4_handlabel.py"),
    ("C4: hand-label finalization", "c4_finalize.py"),
    ("C4: root-cause investigation", "c4_investigate.py"),
    ("C4: hand-label figure", "c4_figure.py"),

    # Contamination
    ("Contamination: check", "contamination_check.py"),
    ("Contamination: control", "contamination_control.py"),

    # Alpha sensitivity
    ("Alpha sensitivity of C2 and C3", "alpha_sensitivity.py"),

    # Summary
    ("Final summary tables", "final_summary.py"),
    ("Co-occurrence structure", "cooccurrence.py"),
]


def main():
    start = time.time()
    print("=" * 78)
    print("Reproducing all paper results")
    print("=" * 78)
    print()

    failed = []
    for i, (label, script) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {label}")
        print(f"          running {script} ...")
        if not os.path.exists(script):
            print(f"          WARNING: {script} not found, skipping")
            continue
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"          FAILED after {elapsed:.1f}s")
            print(f"          stdout: {result.stdout[-500:]}")
            print(f"          stderr: {result.stderr[-500:]}")
            failed.append(script)
        else:
            print(f"          done in {elapsed:.1f}s")
        print()

    total = time.time() - start
    print("=" * 78)
    print(f"Complete in {total/60:.1f} minutes")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All scripts completed successfully.")
    print("=" * 78)


if __name__ == "__main__":
    main()