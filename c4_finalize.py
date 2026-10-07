"""
Finalize C4 hand-labels.

Apply labels to the worksheet, compute agreement with detector flags,
produce a summary table.
"""

import pandas as pd
import numpy as np

# Hand labels from manual examination of motor means
hand_labels = {
    "696590b3-101b-4aa4-820d-d67b22eecea3": "genuine",
    "c1506e1f-9ee3-4640-b073-1483605b9e32": "genuine",
    "b48a234c-03d3-4dae-9f9d-cf335744cad3": "genuine",
    "eb456400-8803-4062-b937-ea4b4320bb59": "genuine",
    "f386caa8-c372-40e0-b1c5-4687f8d614b7": "genuine",
    "cdfac74e-bc29-4ca0-ae91-fa20c4973386": "not_genuine",
    "afd1472c-f54f-403d-8dd8-402d8ec951ce": "genuine",
    "376cfb8d-d741-4115-801e-2d60ee5e53c9": "borderline",
    "aaf2608e-2a6e-4e1d-b887-45cd26723511": "genuine",
    "a4ac869f-6f57-4ac3-91ec-63b1997596b8": "genuine",
    "7a10c97d-79c5-4186-b2cd-2fc728ee62ff": "genuine",
    "df14c679-3e75-46f5-a495-7ed0285cd9ff": "borderline",
    "c8b3aec5-645b-41e2-96a3-b38a4b6f6095": "borderline",
    "332fb065-a9ff-498c-81f0-ee7818ef6895": "not_genuine",
    "9e082b2f-6292-4e70-84cf-e520db2b4144": "borderline",
    "bb754bf2-b9bc-4137-8629-ff36b91e737c": "not_genuine",
    "480597bd-2f52-4bfd-a6c4-9b8ec7ea5f12": "genuine",
    "b542b8e7-2953-4c05-a5f6-12841b7bb488": "borderline",
    "46969280-f01c-4d74-9520-e2de5dc6fcfe": "not_genuine",
    "897fc073-14b4-4366-8ab6-c98d06d91c1f": "not_genuine",
    "1d33794c-79e8-4200-be54-c6fa4bcacdf5": "not_genuine",
    "e5b15dd2-ceec-4e8c-a852-72580d8e1884": "not_genuine",
    "78d77df7-6edd-42c6-81d1-1fd45fed372c": "not_genuine",
    "558d60f8-8a59-405b-835c-3ab1f032ce24": "not_genuine",
    "3a0a0ea1-ad59-4140-9651-c893c21971c1": "borderline",
    "def4855c-0135-4bfc-9c9b-b0802a50d926": "borderline",
    "44b6a14d-9a49-4046-9a60-e161e645d4f4": "not_genuine",
    "d3299236-fbda-4af8-99cd-f59debbc12a2": "borderline",
    "e6582484-f30a-4828-9d86-cf39ccaf3af8": "not_genuine",
    "4d9f0dc8-288f-4fe8-8fd8-be37ff06278b": "not_genuine",
    "4287b232-f4a8-417d-aa49-b21eb88081fb": "not_genuine",
    "efe75f5d-87ba-4204-b10a-139c0da4c698": "not_genuine",
    "b58e2067-08eb-4ee0-a119-e138823b8cd1": "not_genuine",
    "2cb6ff34-eef4-4466-9b2d-f13f0db891c4": "genuine",
    "66705865-6555-4077-852a-8d75706e6937": "borderline",
    "dd152b77-5609-44e0-8e6a-90f703990876": "not_genuine",
    "0ae539bf-9dd1-4393-a645-98ab97a865d4": "not_genuine",
    "d19ecb79-8949-44a4-8921-88159a8c5404": "not_genuine",
    "697f6dce-243c-4b83-937f-653e8f4f218d": "not_genuine",
    "ef996020-534c-4fbe-9c66-9070bbfa3f10": "not_genuine",
    "996bdb97-9e64-4e4e-aff1-68205889b246": "not_genuine",
    "c5aa149c-84f7-40d3-984b-9660fca0ff40": "borderline",
    "1609a2f5-7ef4-46b2-ad6a-384aa9dc256d": "not_genuine",
    "e1a04338-574c-4c12-b115-cadd9467dd6e": "not_genuine",
    "36c55341-13f7-40d9-8736-5a53eea5c970": "borderline",
    "6019c4a4-97ed-4741-9ea4-990e9b992717": "borderline",
    "71e2b054-2464-4455-8ed2-03d2ae475e88": "not_genuine",
    "0bbdcab9-0dc0-40e2-900c-7b619b591181": "borderline",
    "6cf28ab8-22ca-4e27-926c-122372ec9799": "borderline",
    "8a54b5fe-5299-4e84-b668-6df1a9a93441": "not_genuine",
}

df = pd.read_csv("c4_handlabel_worksheet.csv")
df["hand_label"] = df["log_id"].map(hand_labels)
df.to_csv("c4_handlabel_final.csv", index=False)

print(f"labelled {df['hand_label'].notna().sum()} of {len(df)} flights")
print()
print("=== Hand-label distribution ===")
print(df["hand_label"].value_counts().to_string())
print()

# agreement analysis
print("=" * 70)
print("Detector agreement with hand labels")
print("=" * 70)
print()

# For each detector, consider "detector flagged" as positive prediction.
# Consider "genuine" as positive truth.
genuine = df[df["hand_label"] == "genuine"]
not_genuine = df[df["hand_label"] == "not_genuine"]
borderline = df[df["hand_label"] == "borderline"]

for method in ["flag_iso", "flag_svm", "flag_ae"]:
    print(f"--- {method} ---")
    n_gen = len(genuine)
    n_ng = len(not_genuine)
    tp = int(genuine[method].sum())  # flagged as anomaly, correctly
    fp = int(not_genuine[method].sum())  # flagged, but actually normal
    tn = n_ng - fp
    fn = n_gen - tp
    print(f"  true positive:   {tp}/{n_gen}  (genuine flights flagged)")
    print(f"  false positive:  {fp}/{n_ng}  (normal flights flagged)")
    print(f"  true negative:   {tn}/{n_ng}")
    print(f"  false negative:  {fn}/{n_gen}  (missed genuine)")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    print(f"  precision: {precision:.3f}")
    print(f"  recall:    {recall:.3f}")
    print()

# Any detector combined
df["flag_any"] = df[["flag_iso", "flag_svm", "flag_ae"]].any(axis=1)
print("--- Any detector ---")
tp = int(genuine["flag_any"].sum())
fp = int(not_genuine["flag_any"].sum())
tn = len(not_genuine) - fp
fn = len(genuine) - tp
print(f"  true positive:   {tp}/{len(genuine)}")
print(f"  false positive:  {fp}/{len(not_genuine)}")
print(f"  precision: {tp/(tp+fp) if (tp+fp)>0 else float('nan'):.3f}")
print(f"  recall:    {tp/(tp+fn) if (tp+fn)>0 else float('nan'):.3f}")
print()

# Borderline treatment
print("--- Borderline flights (flagged) ---")
print(f"  n_borderline: {len(borderline)}")
print(f"  flagged by any detector: {int(borderline['flag_any'].sum())} ({100*borderline['flag_any'].mean():.1f}%)")
print()

# Sanity check: which genuine flights were missed?
print("=== Genuine flights missed by all detectors ===")
missed = genuine[~genuine["flag_any"]]
for _, row in missed.iterrows():
    print(f"  {row['log_id'][:20]:>20s}  sat_frac={row['sat_frac']:.3f}  "
          f"sat_max_run_s={row['sat_max_run_s']:.1f}  spread={row['spread_p95']:.3f}")

print()
print("saved c4_handlabel_final.csv")