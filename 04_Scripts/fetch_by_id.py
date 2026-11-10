"""
Rebuild the Parquet corpus by fetching exact log IDs from the manifest.

The pilot scripts (pilot_2270.py, pilot_extra.py) sampled logs at random from
the live logs.px4.io listing. That listing grows, so re-running them with the
recorded seeds selects a different set of logs and does not reproduce this
corpus. This script closes that gap: it reads the committed manifest and
downloads precisely the logs the analysis used.

Usage:
    python fetch_by_id.py                        # all 2,770 streamed logs
    python fetch_by_id.py --analysis-set-only    # just the 1,627 the analysis used
    python fetch_by_id.py --dry-run              # plan only, no network needed
    python fetch_by_id.py --manifest some_other_index.csv

Output layout, matching what run_all.py expects:
    pilot_2270/logs/<log_id>/<topic>.parquet     (manifest source == "main")
    pilot_extra/logs/<log_id>/<topic>.parquet    (manifest source == "extra")

Safe to interrupt and re-run: a log whose directory already holds the
estimator ratio topic is skipped. Expect roughly 27 GB and several hours on a
home connection. Raw .ulg files are deleted after conversion.
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE = "https://logs.px4.io"
DOWNLOAD_PATH = "/download?log={log_id}"
USER_AGENT = "px4-corpus-fetch/1.0"

# manifest "source" -> output root, matching pilot_2270.py / pilot_extra.py
SOURCE_DIRS = {"main": "pilot_2270", "extra": "pilot_extra"}

DEFAULT_MANIFEST = "03_Data_CSVs/combined_index.csv"          # all 2,770 streamed
DEFAULT_ANALYSIS_SET = "03_Data_CSVs/combined_analysis_set.csv"  # the 1,627 analysed

# presence of this topic marks a log as already converted
SENTINEL_TOPIC = "estimator_innovation_test_ratios.parquet"

RETRIES = 3
BACKOFF_S = 5


def log_dir(source, log_id):
    root = SOURCE_DIRS.get(source)
    if root is None:
        return None
    return os.path.join(root, "logs", log_id)


def already_done(source, log_id):
    d = log_dir(source, log_id)
    return bool(d) and os.path.exists(os.path.join(d, SENTINEL_TOPIC))


def authoritative_sources():
    """log_id -> source from the full index, which records where each log came from.

    combined_analysis_set.csv carries its own source column, but it disagrees with
    the index for a handful of logs. The index is the record of what was actually
    downloaded where, so it wins when available. The analysis scripts look in both
    roots regardless, so this only keeps the rebuilt layout identical to the
    original.
    """
    if not os.path.exists(DEFAULT_MANIFEST):
        return {}
    try:
        with open(DEFAULT_MANIFEST, newline="", encoding="utf-8-sig") as f:
            return {(r.get("log_id") or "").strip(): (r.get("source") or "").strip()
                    for r in csv.DictReader(f)}
    except OSError:
        return {}


def read_manifest(path):
    if not os.path.exists(path):
        sys.exit(f"manifest not found: {path}\nPass --manifest with the path to combined_index.csv")
    index_src = authoritative_sources() if os.path.abspath(path) != os.path.abspath(DEFAULT_MANIFEST) else {}
    out, skipped, relocated = [], 0, 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        for col in ("log_id", "source"):
            if col not in (rd.fieldnames or []):
                sys.exit(f"manifest is missing the '{col}' column; got {rd.fieldnames}")
        for r in rd:
            lid, src = (r.get("log_id") or "").strip(), (r.get("source") or "").strip()
            if not lid or src not in SOURCE_DIRS:
                skipped += 1
                continue
            true_src = index_src.get(lid)
            if true_src in SOURCE_DIRS and true_src != src:
                src = true_src
                relocated += 1
            out.append({"log_id": lid, "source": src})
    if relocated:
        print(f"note: {relocated} logs routed per {DEFAULT_MANIFEST}, which "
              f"disagrees with this file's source column")
    if skipped:
        print(f"note: skipped {skipped} manifest rows with no log_id or an unknown source")
    return out


def stream_to_temp(log_id, session, outdir):
    """Download one .ulg to a temp file inside outdir. Returns (path, error)."""
    import requests
    url = BASE + DOWNLOAD_PATH.format(log_id=log_id)
    os.makedirs(outdir, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        tmp_name = None
        try:
            r = session.get(url, timeout=180, stream=True)
            if r.status_code != 200:
                if r.status_code in (404, 410):
                    return None, f"http_{r.status_code}"
                raise requests.RequestException(f"http {r.status_code}")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ulg",
                                              dir=outdir, prefix="tmp_")
            tmp.close()
            tmp_name = tmp.name
            n = 0
            with open(tmp_name, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n += len(chunk)
            if n < 1024:
                os.remove(tmp_name)
                return None, "too_small"
            return tmp_name, None
        except Exception as e:
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            if attempt == RETRIES:
                return None, f"download_failed: {type(e).__name__}"
            time.sleep(BACKOFF_S * attempt)
    return None, "download_failed"


def write_parquet(u, dest):
    """Write every uORB topic as zstd Parquet. Mirrors pilot_2270.write_parquet_and_count."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(dest, exist_ok=True)
    total, n_topics = 0, 0
    for d in u.data_list:
        try:
            table = pa.table({k: pa.array(v) for k, v in d.data.items()})
            path = os.path.join(dest, f"{d.name}.parquet")
            pq.write_table(table, path, compression="zstd")
            total += os.path.getsize(path)
            n_topics += 1
        except Exception:
            continue
    return total, n_topics


def process(rec, session):
    log_id, source = rec["log_id"], rec["source"]
    dest = log_dir(source, log_id)
    out = {"log_id": log_id, "source": source, "status": "unknown",
           "n_topics": 0, "parquet_bytes": 0, "elapsed_s": 0.0}
    t0 = time.time()

    path, err = stream_to_temp(log_id, session, SOURCE_DIRS[source])
    if path is None:
        out["status"] = err or "download_failed"
        out["elapsed_s"] = round(time.time() - t0, 1)
        return out

    try:
        from pyulog import ULog
        try:
            u = ULog(path)
        except Exception:
            out["status"] = "parse_failed"
            return out
        total, n_topics = write_parquet(u, dest)
        out["parquet_bytes"], out["n_topics"] = total, n_topics
        if not os.path.exists(os.path.join(dest, SENTINEL_TOPIC)):
            # converted, but the topic this study needs is absent
            out["status"] = "ok_no_ratio_topic"
        else:
            out["status"] = "ok"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None,
                    help="CSV with log_id and source columns "
                         "(default: 03_Data_CSVs/combined_index.csv, all 2,770 logs)")
    ap.add_argument("--analysis-set-only", action="store_true",
                    help="fetch only the 1,627 flights the analysis used, by reading "
                         "03_Data_CSVs/combined_analysis_set.csv instead")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent downloads; keep low, this is community "
                         "infrastructure (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N logs (0 = no limit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched and exit without network access")
    ap.add_argument("--report", default="fetch_by_id_report.csv",
                    help="per-log outcome CSV (default: %(default)s)")
    args = ap.parse_args()

    if args.manifest is None:
        args.manifest = (DEFAULT_ANALYSIS_SET if args.analysis_set_only
                         else DEFAULT_MANIFEST)
    elif args.analysis_set_only:
        print("note: --manifest given explicitly, so --analysis-set-only is ignored")

    records = read_manifest(args.manifest)
    done = [r for r in records if already_done(r["source"], r["log_id"])]
    todo = [r for r in records if not already_done(r["source"], r["log_id"])]
    if args.limit:
        todo = todo[:args.limit]

    by_src = {}
    for r in todo:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1

    print(f"manifest:        {args.manifest}")
    print(f"selected:        {len(records)} logs")
    print(f"already present: {len(done)}")
    print(f"to fetch:        {len(todo)}"
          + (f"  ({', '.join(f'{k}={v}' for k, v in sorted(by_src.items()))})" if by_src else ""))
    for src, root in sorted(SOURCE_DIRS.items()):
        print(f"  source '{src}' -> {os.path.join(root, 'logs', '<log_id>', '<topic>.parquet')}")

    if args.dry_run:
        print("\ndry run, nothing fetched")
        return
    if not todo:
        print("\nnothing to do")
        return

    import requests
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process, r, session): r["log_id"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as e:
                res = {"log_id": futs[fut], "source": "", "status": f"error: {type(e).__name__}",
                       "n_topics": 0, "parquet_bytes": 0, "elapsed_s": 0.0}
            results.append(res)
            if i % 25 == 0 or i == len(todo):
                gb = sum(r["parquet_bytes"] for r in results) / 1e9
                rate = i / max(1e-9, time.time() - t0) * 60
                print(f"  {i}/{len(todo)}  {gb:.1f} GB  {rate:.0f} logs/min", flush=True)

    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["log_id", "source", "status",
                                          "n_topics", "parquet_bytes", "elapsed_s"])
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: r["log_id"]))

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nwrote {args.report}")
    for k in sorted(counts):
        print(f"  {k:<22} {counts[k]}")
    total_gb = sum(r['parquet_bytes'] for r in results) / 1e9
    print(f"  {'parquet total':<22} {total_gb:.1f} GB")
    failed = sum(v for k, v in counts.items() if not k.startswith("ok"))
    if failed:
        print(f"\n{failed} logs did not convert. Re-run to retry only those; "
              f"successful logs are skipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
