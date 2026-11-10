"""
Per-ratio C1 summary on clean flights, with each ratio's own null rate.

Produces c1_summary_clean.csv, the table behind the paper's per-ratio C1 result.
c1_summary.csv covers all 1,627 flights; this restricts to the 957 clean ones,
which is the population the paper reports.

Two facts drive the null column, both read from the PX4 source rather than
assumed:

  1. estimator_innovation_test_ratios publishes one SCALAR ratio per
     measurement axis (EKF2.cpp assigns test_ratio[0], test_ratio[1]
     individually), so the null is chi-square with ONE degree of freedom.
  2. The gate size is not 5 for every ratio. The magnetometer is gated at
     3.0 sigma (EKF2_MAG_GATE) and heading at 2.6 sigma (EKF2_HDG_GATE).

A ratio exceeds 1.0 exactly when |innovation / sigma| > gate, so under a
correctly specified filter with Gaussian innovations the per-sample rejection
probability is erfc(gate / sqrt(2)).

Usage:
    python c1_summary_clean.py
"""

import csv
import os
import random
from math import erfc, sqrt, log10

RATES = "c1_warmup_excluded.csv"       # written by c1_false_alarm_v2.py, 30 s warm-up removed
INDICATORS = "analysis_set_with_satfrac.csv"
OUT = "c1_summary_clean.csv"

# ratio -> (PX4 parameter, default gate in standard deviations)
GATES = {
    "baro_vpos":    ("EKF2_BARO_GATE",  5.0),
    "gps_hpos[0]":  ("EKF2_GPS_P_GATE", 5.0),
    "gps_hpos[1]":  ("EKF2_GPS_P_GATE", 5.0),
    "gps_vpos":     ("EKF2_BARO_GATE",  5.0),
    "gps_hvel[0]":  ("EKF2_GPS_V_GATE", 5.0),
    "gps_hvel[1]":  ("EKF2_GPS_V_GATE", 5.0),
    "gps_vvel":     ("EKF2_GPS_V_GATE", 5.0),
    "hagl":         ("EKF2_RNG_GATE",   5.0),
    "mag_field[0]": ("EKF2_MAG_GATE",   3.0),
    "heading":      ("EKF2_HDG_GATE",   2.6),
}

SAT_TOLERANCE = 0.05   # see DECISIONS.md; brief saturation is ordinary control effort
N_BOOT = 2000
SEED = 0


def null_rate(gate):
    """P(ratio > 1) for a scalar normalised innovation = P(chi2_1 > gate^2)."""
    return erfc(gate / sqrt(2.0))


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def is_true(v):
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def clean_log_ids(path):
    """Four indicators exactly zero; sat_frac within tolerance."""
    out = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (as_float(r["failsafe_frac"]) == 0
                    and as_float(r["nav_state_forbidden_frac"]) == 0
                    and as_float(r["filter_fault_flags_frac"]) == 0
                    and is_true(r["landed_at_end"])
                    and as_float(r["sat_frac"]) <= SAT_TOLERANCE):
                out.add(r["log_id"])
    return out


def boot_ci(values, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap over flights for the mean per-sample rate."""
    if len(values) < 2:
        return float("nan"), float("nan")
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def main():
    for p in (RATES, INDICATORS):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}\nRun from the directory holding the derived CSVs.")

    clean = clean_log_ids(INDICATORS)
    with open(RATES, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["log_id"] in clean]
    print(f"clean flights: {len(clean)}   rate rows matched: {len(rows)}")

    rng = random.Random(SEED)
    out = []
    for ratio, (param, gate) in GATES.items():
        col = "rate_" + ratio
        if col not in (rows[0] if rows else {}):
            print(f"  skipping {ratio}: no column {col}")
            continue
        vals = [float(r[col]) for r in rows if r[col] not in ("", "nan", "None")]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        p0 = null_rate(gate)
        lo, hi = boot_ci(vals, rng)
        out.append({
            "ratio": ratio,
            "gate_param": param,
            "gate_sigma": f"{gate:.1f}",
            "null_per_sample": f"{p0:.6e}",
            "n_flights": len(vals),
            "mean_rate": f"{mean:.6f}",
            "ci_low": f"{lo:.6f}",
            "ci_high": f"{hi:.6f}",
            "ratio_to_null": f"{mean / p0:.4g}",
            "orders_above_null": f"{log10(mean / p0):.3f}",
        })

    out.sort(key=lambda r: float(r["ratio_to_null"]), reverse=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT}")

    five = [float(r["orders_above_null"]) for r in out if r["gate_sigma"] == "5.0"]
    print(f"\n5-sigma ratios ({len(five)}): {min(five):.2f} to {max(five):.2f} orders, "
          f"mean {sum(five)/len(five):.2f}")
    for r in out:
        if r["gate_sigma"] != "5.0":
            print(f"{r['ratio']:<13} gate {r['gate_sigma']}: "
                  f"{float(r['ratio_to_null']):.1f}x null "
                  f"({float(r['orders_above_null']):.2f} orders)")


if __name__ == "__main__":
    main()
