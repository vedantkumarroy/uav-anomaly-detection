import os
import tempfile
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from pyulog import ULog

LOG_ID = "04972883-b141-44c7-aef8-13b983daba55"
URL = f"https://logs.px4.io/download?log={LOG_ID}"

print("streaming log...")
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg", dir="pilot")
with requests.get(URL, stream=True) as r:
    r.raise_for_status()
    n = 0
    with open(tmp.name, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
            n += len(chunk)
print(f"  downloaded {n/1e6:.1f} MB")

print("parsing with pyulog...")
u = ULog(tmp.name)

outdir = "parquet_test"
os.makedirs(outdir, exist_ok=True)

total_snappy = 0
total_zstd = 0

for d in u.data_list:
    arrays = {k: pa.array(v) for k, v in d.data.items()}
    table = pa.table(arrays)

    path_snappy = os.path.join(outdir, f"{d.name}_snappy.parquet")
    pq.write_table(table, path_snappy, compression="snappy")
    total_snappy += os.path.getsize(path_snappy)

    path_zstd = os.path.join(outdir, f"{d.name}_zstd.parquet")
    pq.write_table(table, path_zstd, compression="zstd")
    total_zstd += os.path.getsize(path_zstd)

print(f"  wrote {len(u.data_list)} topics")
print()
print(f"original ulg size:      {n/1e6:.1f} MB")
print(f"snappy parquet total:   {total_snappy/1e6:.1f} MB  ({100*total_snappy/n:.1f}% of original)")
print(f"zstd parquet total:     {total_zstd/1e6:.1f} MB  ({100*total_zstd/n:.1f}% of original)")

os.remove(tmp.name)
print()
print("done")