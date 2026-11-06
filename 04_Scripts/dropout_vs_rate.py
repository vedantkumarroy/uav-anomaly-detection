import pandas as pd

df = pd.read_csv("per_log_with_dropout.csv")
df = df[df["parse_ok"]]

df["rate_bucket"] = df["rate_ratios"].apply(
    lambda r: "high (>10 Hz)" if r > 10 else ("low (<=10 Hz)" if r > 0 else "no ratios")
)

print("=== dropout_frac vs rate bucket ===")
for bucket in ["low (<=10 Hz)", "high (>10 Hz)", "no ratios"]:
    sub = df[df["rate_bucket"] == bucket]
    if len(sub) == 0:
        continue
    d = sub["dropout_frac"]
    print(f"\n{bucket}: n={len(sub)}")
    print(f"  median dropout_frac: {d.median():.4f}")
    print(f"  mean dropout_frac:   {d.mean():.4f}")
    print(f"  count > 0.01:        {int((d > 0.01).sum())}")
    print(f"  count > 0.02:        {int((d > 0.02).sum())}")
    print(f"  max:                 {d.max():.4f}")

print("\n=== Cross-tab: dropout_ok_002 vs rate bucket ===")
df["dropout_ok_002"] = df["dropout_frac"] <= 0.02
print(pd.crosstab(df["rate_bucket"], df["dropout_ok_002"]))