from pyulog import ULog
import glob

found = False
for p in sorted(glob.glob("pilot2/logs/*.ulg")):
    try:
        u = ULog(p)
    except Exception:
        continue
    for d in u.data_list:
        if d.name == "estimator_innovation_test_ratios":
            if "is_baro_fluctuation" in d.data.keys():
                print("FOUND in:", p)
                print("All fields:", sorted(d.data.keys()))
                found = True
                break
    if found:
        break

if not found:
    print("NOT FOUND in pilot2 logs")
    print("Searching for any field containing 'baro' or 'fluct'...")
    for p in sorted(glob.glob("pilot2/logs/*.ulg"))[:20]:
        try:
            u = ULog(p)
        except Exception:
            continue
        for d in u.data_list:
            if d.name == "estimator_innovation_test_ratios":
                hits = [k for k in d.data.keys() if "baro" in k.lower() or "fluct" in k.lower()]
                if hits:
                    print("  ", p.split("\\")[-1][:20], "->", hits)