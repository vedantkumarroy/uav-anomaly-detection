from pyulog import ULog

# Pick one of your downloaded logs
LOG = r"pilot\logs\c72b04bf-b840-4401-9402-262614aab07e.ulg"

u = ULog(LOG)

print("=== ALL TOPICS IN THIS LOG ===")
for d in sorted(u.data_list, key=lambda x: x.name):
    print(f"{d.name:45s} {len(d.data['timestamp']):>8d} msgs")

print("\n=== ESTIMATOR INNOVATION RATIOS ===")
for d in u.data_list:
    if d.name == "estimator_innovation_test_ratios":
        print("Fields:", sorted(d.data.keys()))
        print("Rows:", len(d.data["timestamp"]))
        break
else:
    print("Not present — checking legacy estimator_status...")
    for d in u.data_list:
        if d.name == "estimator_status":
            print("Fields:", sorted(d.data.keys()))
            print("Rows:", len(d.data["timestamp"]))
            break