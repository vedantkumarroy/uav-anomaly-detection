import os
import glob
import numpy as np
import pandas as pd
from pyulog import ULog

LOG_DIR = "pilot2/logs"

MIN_RATE_HZ = {
    "vehicle_local_position": 5.0,
    "vehicle_attitude": 10.0,
}

MIN_DURATION_S = 60.0
MIN_AIRBORNE_S = 45.0
MAX_DROPOUT_FRAC = 0.01


def analyze(path):
    row = {"log_id": os.path.basename(path).replace(".ulg", ""), "parse_ok": False}
    try:
        u = ULog(path)
    except Exception as e:
        row["parse_error"] = f"{type(e).__name__}: {e}"[:200]
        return row
    row["parse_ok"] = True

    present = set()
    rates = {}
    for d in u.data_list:
        present.add(d.name)
        ts = d.data.get("timestamp")
        if ts is None or len(ts) < 2:
            continue
        span = (float(ts[-1]) - float(ts[0])) / 1e6
        if span <= 0:
            continue
        hz = (len(ts) - 1) / span
        rates[d.name] = max(rates.get(d.name, 0.0), hz)

    row["n_topics"] = len(present)

    # topics
    row["has_position"] = "vehicle_local_position" in present
    row["has_attitude"] = "vehicle_attitude" in present
    row["has_status"] = "vehicle_status" in present
    row["has_imu"] = "sensor_combined" in present
    row["has_actuator"] = ("actuator_motors" in present) or ("actuator_outputs" in present)
    row["has_att_sp"] = "vehicle_attitude_setpoint" in present
    row["has_rates_sp"] = "vehicle_rates_setpoint" in present
    row["has_battery"] = "battery_status" in present
    row["has_ekf_innov"] = ("estimator_innovations" in present) or ("estimator_status" in present)
    row["has_land_det"] = "vehicle_land_detected" in present

    if "estimator_innovation_test_ratios" in present:
        row["has_ekf_ratios"] = True
        row["src_ekf_ratios"] = "estimator_innovation_test_ratios"
    elif "estimator_status" in present:
        es = next((d for d in u.data_list if d.name == "estimator_status"), None)
        legacy = es is not None and any(k.endswith("test_ratio") for k in es.data)
        row["has_ekf_ratios"] = legacy
        row["src_ekf_ratios"] = "estimator_status(legacy)" if legacy else ""
    else:
        row["has_ekf_ratios"] = False
        row["src_ekf_ratios"] = ""

    # rates
    row["rate_position"] = round(rates.get("vehicle_local_position", 0.0), 2)
    row["rate_attitude"] = round(rates.get("vehicle_attitude", 0.0), 2)
    row["rate_ratios"] = round(rates.get("estimator_innovation_test_ratios", 0.0), 2)
    row["rate_position_ok"] = row["rate_position"] >= MIN_RATE_HZ["vehicle_local_position"]
    row["rate_attitude_ok"] = row["rate_attitude"] >= MIN_RATE_HZ["vehicle_attitude"]

    # duration
    try:
        dur = (u.last_timestamp - u.start_timestamp) / 1e6
    except Exception:
        dur = 0.0
    row["duration_s"] = round(dur, 1)
    row["duration_ok"] = dur >= MIN_DURATION_S

    # dropouts
    drop_ms = sum(getattr(d, "duration", 0) for d in (u.dropouts or []))
    row["dropout_frac"] = round((drop_ms / 1000.0) / dur, 6) if dur > 0 else 1.0
    row["dropout_ok"] = row["dropout_frac"] <= MAX_DROPOUT_FRAC

    # airborne
    airborne = 0.0
    ld = next((d for d in u.data_list if d.name == "vehicle_land_detected"), None)
    if ld is not None and "landed" in ld.data and len(ld.data["timestamp"]) > 1:
        ts, landed = ld.data["timestamp"], ld.data["landed"]
        for i in range(len(ts) - 1):
            if not landed[i]:
                airborne += (ts[i + 1] - ts[i]) / 1e6
    else:
        lp = next((d for d in u.data_list if d.name == "vehicle_local_position"), None)
        if lp is not None and "z" in lp.data and len(lp.data["z"]) > 1:
            z = lp.data["z"]
            if (max(z) - min(z)) > 2.0:
                airborne = row["duration_s"]
    row["airborne_s"] = round(airborne, 1)
    row["airborne_ok"] = airborne >= MIN_AIRBORNE_S

    # SYS_AUTOSTART
    params = u.initial_parameters or {}
    row["sys_autostart"] = params.get("SYS_AUTOSTART", "")

    # composite gates
    row["topics_ok"] = all([
        row["has_position"], row["has_attitude"], row["has_status"],
        row["has_imu"], row["has_actuator"], row["has_att_sp"],
        row["has_rates_sp"], row["has_battery"], row["has_ekf_ratios"],
        row["has_ekf_innov"], row["has_land_det"],
    ])
    row["rates_ok"] = row["rate_position_ok"] and row["rate_attitude_ok"]

    return row


def main():
    paths = sorted(glob.glob(os.path.join(LOG_DIR, "*.ulg")))
    print(f"reading {len(paths)} logs from {LOG_DIR}\n")

    rows = []
    for i, p in enumerate(paths, 1):
        rows.append(analyze(p))
        if i % 25 == 0:
            print(f"  parsed {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    df.to_csv("per_log_with_dropout.csv", index=False)
    print(f"\nsaved per_log_with_dropout.csv ({len(df)} rows)")

    ok = df[df["parse_ok"]]

    # funnel
    print("\n=== Funnel ===")
    n = len(ok)
    print(f"parsed                       {n}")

    def count(cond):
        return int(cond.sum())

    print(f"  has position               {count(ok['has_position'])}")
    print(f"  has attitude               {count(ok['has_attitude'])}")
    print(f"  has status                 {count(ok['has_status'])}")
    print(f"  has imu                    {count(ok['has_imu'])}")
    print(f"  has actuator               {count(ok['has_actuator'])}")
    print(f"  has att_sp                 {count(ok['has_att_sp'])}")
    print(f"  has rates_sp               {count(ok['has_rates_sp'])}")
    print(f"  has battery                {count(ok['has_battery'])}")
    print(f"  has ekf_ratios             {count(ok['has_ekf_ratios'])}")
    print(f"  has ekf_innov              {count(ok['has_ekf_innov'])}")
    print(f"  has land_det               {count(ok['has_land_det'])}")

    n_topics = count(ok["topics_ok"])
    print(f"all required topics          {n_topics}")

    n_rates = count(ok["topics_ok"] & ok["rates_ok"])
    print(f"  + rate gates               {n_rates}")

    n_dur = count(ok["topics_ok"] & ok["rates_ok"] & ok["duration_ok"])
    print(f"  + duration >= {int(MIN_DURATION_S)}s        {n_dur}")

    n_air = count(ok["topics_ok"] & ok["rates_ok"] & ok["duration_ok"] & ok["airborne_ok"])
    print(f"  + airborne >= {int(MIN_AIRBORNE_S)}s        {n_air}")

    n_drop = count(
        ok["topics_ok"] & ok["rates_ok"] & ok["duration_ok"] & ok["airborne_ok"] & ok["dropout_ok"]
    )
    print(f"  + dropout <= {MAX_DROPOUT_FRAC}        {n_drop}")

    print(f"SURVIVES ALL GATES           {n_drop}   ({100*n_drop/n:.1f}% of parsed)")

    # dropout is the difference between airborne and final
    diff = n_air - n_drop
    print(f"\n=== Dropout impact ===")
    print(f"airborne survivors:          {n_air}")
    print(f"after dropout gate:          {n_drop}")
    print(f"dropped by dropout gate:     {diff}")

    # dropout_frac distribution
    d = ok["dropout_frac"].dropna()
    print(f"\n=== dropout_frac distribution (n={len(d)}) ===")
    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"  p{int(q*100):02d} = {d.quantile(q):.6f}")
    print(f"  max = {d.max():.6f}")
    print(f"  count <= 0.01: {int((d <= 0.01).sum())}")
    print(f"  count >  0.01: {int((d >  0.01).sum())}")

    # sys_autostart count among survivors
    surv = ok[
        ok["topics_ok"] & ok["rates_ok"] & ok["duration_ok"] & ok["airborne_ok"] & ok["dropout_ok"]
    ]
    print(f"\n=== SYS_AUTOSTART among {len(surv)} survivors ===")
    print(surv["sys_autostart"].value_counts().to_string())

    n4001 = int((surv["sys_autostart"].astype(str) == "4001").sum())
    print(f"\nSurvivors with SYS_AUTOSTART == 4001: {n4001} of {len(surv)}")


if __name__ == "__main__":
    main()