import pandas as pd

df = pd.read_csv("pilot_500_v2_index.csv")
before = len(df)

# add an audit column for why each log is dropped
df["excluded_reason"] = ""

for i, row in df.iterrows():
    if pd.isna(row["band"]):
        ver = row["ver_sw"]
        if pd.isna(ver):
            df.at[i, "excluded_reason"] = "missing_ver_sw"
        else:
            try:
                v = int(float(ver))
                major = (v >> 24) & 0xFF
                if major != 1:
                    df.at[i, "excluded_reason"] = f"not_px4_v{major}"
                else:
                    df.at[i, "excluded_reason"] = "unknown_band"
            except Exception:
                df.at[i, "excluded_reason"] = "unparseable_ver_sw"

# keep only rows with a valid band
df_clean = df[df["band"].notna()].copy()
after = len(df_clean)

print(f"before: {before}")
print(f"after:  {after}")
print(f"dropped: {before - after}")
print()
print("excluded reasons:")
print(df[df["band"].isna()]["excluded_reason"].value_counts())
print()
print("group counts after drop:")
print(df_clean["group"].value_counts())
print()
print("band x group after drop:")
print(pd.crosstab(df_clean["band"], df_clean["group"]))

df_clean.to_csv("pilot_500_v2_index.csv", index=False)
print()
print("saved pilot_500_v2_index.csv")

# also save the excluded rows for reference
df[df["band"].isna()].to_csv("pilot_500_v2_excluded.csv", index=False)
print("saved pilot_500_v2_excluded.csv")