from pyulog import ULog
import pandas as pd
import numpy as np
import os
import glob

LOG_DIR = "pilot/logs"

def analyze_log(path):
    try:
        u = ULog(path)
    except Exception:
        return None

    info = u.msg_info_dict or {}
    ver_sw = str(info.get("ver_sw_release") or info.get("ver_sw") or "")

    # collect topics
    topics = set(d.name for d in u.data_list)

    # find the ratios topic
    ratio_topic = None
    for d in u.data_list:
        if d.name == "estimator_innovation_test_ratios":
            ratio_topic = d
            break

    varying_ratios = []
    total_ratios = 0
    if ratio_topic is not None:
        for field, arr in ratio_topic.data.items():
            if field in ("timestamp", "timestamp_sample"):
                continue
            total_ratios += 1
            arr = np.asarray(arr, dtype=float)
            if len(arr) < 2:
                continue
            # variation check: any non-NaN std > 0
            if np.nanstd(arr) > 0:
                varying_ratios.append(field)

    return {
        "log_id": os.path.basename(path).replace(".ulg", ""),
        "ver_sw": ver_sw,
        "n_topics": len(topics),
        "n_ratios_total": total_ratios,
        "n_ratios_varying": len(varying_ratios),
        "varying_ratios": sorted(varying_ratios),
    }

# run over all downloaded logs
rows = []
for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.ulg"))):
    r = analyze_log(path)
    if r:
        rows.append(r)

df = pd.DataFrame(rows)
df = df[df["n_ratios_total"] > 0]  # drop legacy logs with no ratios topic

print("\n=== Per-log summary (logs with ratios topic only) ===")
print(df[["log_id","ver_sw","n_topics","n_ratios_total","n_ratios_varying"]].to_string())

# group by exact ver_sw
print("\n=== Varying ratios per exact firmware version ===")
for ver in sorted(df["ver_sw"].unique()):
    sub = df[df["ver_sw"] == ver]
    print(f"\nver_sw = {ver}  ({len(sub)} logs)")
    union = set()
    for _, r in sub.iterrows():
        union.update(r["varying_ratios"])
    print(f"  Union of varying ratios: {len(union)}")
    print(f"  {sorted(union)}")

# intersection across all versions
sets = [set().union(*df[df["ver_sw"]==v]["varying_ratios"].tolist())
        for v in sorted(df["ver_sw"].unique())]

common = sets[0].intersection(*sets[1:]) if sets else set()
print("\n=== Cross-version intersection ===")
print(f"Vary in ALL versions: {len(common)}")
print(f"  {sorted(common)}")

# matrix: rows = ratio fields, cols = ver_sw, cell = varies?
all_fields = set()
for s in sets:
    all_fields.update(s)
all_fields = sorted(all_fields)
versions = sorted(df["ver_sw"].unique())

print("\n=== Vary matrix (X = varies, . = constant or absent) ===")
header = "field".ljust(22) + "".join(v.rjust(12) for v in versions)
print(header)
for f in all_fields:
    row = f.ljust(22)
    for v in versions:
        s = set().union(*df[df["ver_sw"]==v]["varying_ratios"].tolist())
        row += ("X" if f in s else ".").rjust(12)
    print(row)

# save
df_out = df[["log_id","ver_sw","n_topics","n_ratios_total","n_ratios_varying"]]
df_out.to_csv("version_sensor_table.csv", index=False)
print("\nSaved: version_sensor_table.csv")