import glob
import os
from pyulog import ULog

# inspect all logs in pilot/logs and pilot2/logs
paths = sorted(glob.glob("pilot/logs/*.ulg")) + sorted(glob.glob("pilot2/logs/*.ulg"))
print(f"inspecting {len(paths)} logs\n")

# collect parameter names containing PWM
pwm_params = {}
for path in paths[:50]:
    try:
        u = ULog(path)
    except Exception:
        continue
    params = u.initial_parameters or {}
    for k, v in params.items():
        if "PWM" in k.upper():
            pwm_params.setdefault(k, []).append(v)
    # also check for any topic with "output" or "motor"
    topic_names = [d.name for d in u.data_list]
    if "actuator_outputs" in topic_names or "actuator_motors" in topic_names:
        pass

print("=== PWM parameters found ===")
for k in sorted(pwm_params):
    vals = pwm_params[k]
    unique = sorted(set(round(float(v), 2) if isinstance(v, (int, float)) else str(v) for v in vals))
    print(f"  {k}: n={len(vals)}, unique={unique[:5]}")

print()
print("=== Check actuator_outputs structure on one log ===")
for path in paths:
    try:
        u = ULog(path)
    except Exception:
        continue
    for d in u.data_list:
        if d.name == "actuator_outputs":
            print(f"  log: {os.path.basename(path)[:20]}")
            print(f"  columns: {sorted(d.data.keys())}")
            break
    break

print()
print("=== Check actuator_motors structure on one log ===")
for path in paths:
    try:
        u = ULog(path)
    except Exception:
        continue
    for d in u.data_list:
        if d.name == "actuator_motors":
            print(f"  log: {os.path.basename(path)[:20]}")
            print(f"  columns: {sorted(d.data.keys())}")
            break
    break