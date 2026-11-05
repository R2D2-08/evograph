#!/usr/bin/env python3
"""Download a balanced set of CAPEv2 JSON *reports only* from a public dataset.

No executable samples are downloaded. By default this selects 60 malware and
60 benign raw CAPEv2 reports from Thcrull/win-exe-malware-analysis.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = "Thcrull/win-exe-malware-analysis"
REVISION = "main"
BENIGN_SOURCES = ("Benign-NET", "PortableApps", "PortableAppsInstallers", "SysinternalsSuite")


def list_report_files(max_size_mb: float):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install huggingface_hub") from exc

    api = HfApi()
    entries = list(api.list_repo_tree(REPO_ID, repo_type="dataset", revision=REVISION,
                                      recursive=True, expand=True))
    records = []
    for entry in entries:
        path = getattr(entry, "path", "")
        if not path.endswith(".json"):
            continue
        # Strictly include raw report folders; exclude filtered reports and all binary folders.
        is_malware = path.startswith("malware_reports/reports/")
        benign_source = None
        for source in BENIGN_SOURCES:
            if path.startswith(f"benign_reports/{source}/reports/"):
                benign_source = source
                break
        if not (is_malware or benign_source):
            continue
        size = getattr(entry, "size", None)
        if size is not None and max_size_mb > 0 and size > max_size_mb * 1024 * 1024:
            continue
        records.append({"path": path, "size": size,
                        "label": 1 if is_malware else 0,
                        "source": "MalwareBazaar" if is_malware else benign_source})
    if not records:
        raise RuntimeError("No raw report JSONs found. Check dataset paths/revision or HF availability.")
    return records


def api_event_count(doc: dict) -> int:
    behavior = doc.get("behavior") or {}
    total = 0
    for proc in behavior.get("processes") or []:
        for call in proc.get("calls") or []:
            if isinstance(call, dict) and call.get("api"):
                total += 1
    for _pid, apis in (behavior.get("apistats") or {}).items():
        if isinstance(apis, dict):
            total += sum(int(v) for v in apis.values() if isinstance(v, (int, float)) and v > 0)
    if total == 0:
        summary = behavior.get("summary") or {}
        total += len(summary.get("resolved_apis") or [])
    return total


def report_time(doc: dict) -> str:
    """Return CAPE execution time only; never substitute a binary compile time."""
    info = doc.get("info") or {}
    for key in ("started", "ended"):
        value = info.get(key)
        if not value:
            continue
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
            stamp = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(stamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            pass
    return ""


def sample_sha256(doc: dict, repo_path: str) -> str:
    target = doc.get("target") or {}
    file_info = target.get("file") or {}
    value = file_info.get("sha256") or target.get("sha256") or ""
    if value:
        return str(value).lower()
    # Malware filenames commonly begin with SHA-256; benign filenames usually do not.
    match = re.match(r"^([0-9a-fA-F]{64})_", Path(repo_path).name)
    return match.group(1).lower() if match else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(record: dict, out: Path):
    from huggingface_hub import hf_hub_download
    local = Path(hf_hub_download(repo_id=REPO_ID, filename=record["path"],
                                 repo_type="dataset", revision=REVISION,
                                 local_dir=str(out)))
    try:
        with local.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        count = api_event_count(doc)
        if count <= 0:
            return None, "no API trace in raw report"
        report_rel = Path(record["path"])
        sample_id = report_rel.stem
        return {
            "sample_id": sample_id,
            "report_path": report_rel.as_posix(),
            "label": record["label"],
            "observed_at": report_time(doc),
            "sha256": sample_sha256(doc, record["path"]),
            "report_sha256": sha256_file(local),
            "source": record["source"],
            "repo_path": record["path"],
            "api_event_count": count,
            "timestamp_semantics": "CAPE execution time" if report_time(doc) else "unavailable",
        }, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, f"invalid/unreadable JSON: {exc}"


def collect(records, label, target, out, rng, seen_paths, seen_hashes, benign_source=None):
    pool = [r for r in records if r["label"] == label and
            (benign_source is None or r["source"] == benign_source)]
    rng.shuffle(pool)
    rows, failures = [], Counter()
    for record in pool:
        if len(rows) >= target:
            break
        if record["path"] in seen_paths:
            continue
        seen_paths.add(record["path"])
        row, error = download_one(record, out)
        if error:
            failures[error] += 1
            (out / record["path"]).unlink(missing_ok=True)
            continue
        sha = row["sha256"]
        if sha and sha in seen_hashes:
            failures["duplicate sample SHA-256"] += 1
            (out / record["path"]).unlink(missing_ok=True)
            continue
        if sha:
            seen_hashes.add(sha)
        rows.append(row)
        print(f"[{len(rows):>3}/{target}] label={label} source={row['source']} {Path(row['repo_path']).name}")
    return rows, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="real120", help="Output directory")
    ap.add_argument("--benign", type=int, default=60)
    ap.add_argument("--malware", type=int, default=60)
    ap.add_argument("--max-size-mb", type=float, default=30,
                    help="Skip individual reports larger than this; 0 disables the cap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.benign < 1 or args.malware < 1:
        ap.error("--benign and --malware must both be positive")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Listing report-only files from {REPO_ID} (this does not fetch binaries)...")
    records = list_report_files(args.max_size_mb)
    counts = Counter((r["label"], r["source"]) for r in records)
    print("Eligible raw report counts:", dict(counts))

    rng = random.Random(args.seed)
    seen_paths, seen_hashes = set(), set()
    benign_rows = []
    # Source-diverse benign selection: aim for an equal contribution from each source.
    base, rem = divmod(args.benign, len(BENIGN_SOURCES))
    quotas = {s: base + (1 if i < rem else 0) for i, s in enumerate(BENIGN_SOURCES)}
    failures = Counter()
    for source, quota in quotas.items():
        if quota <= 0:
            continue
        rows, bad = collect(records, 0, quota, out, rng, seen_paths, seen_hashes, source)
        benign_rows.extend(rows); failures.update(bad)
    # Fill any shortfall from all remaining benign sources.
    if len(benign_rows) < args.benign:
        rows, bad = collect(records, 0, args.benign - len(benign_rows), out, rng,
                            seen_paths, seen_hashes, None)
        benign_rows.extend(rows); failures.update(bad)

    malware_rows, bad = collect(records, 1, args.malware, out, rng, seen_paths, seen_hashes)
    failures.update(bad)
    rows = benign_rows + malware_rows

    fields = ["sample_id", "report_path", "label", "observed_at", "sha256",
              "report_sha256", "source", "repo_path", "api_event_count", "timestamp_semantics"]
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    manifest = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "dataset_license_as_listed": "MIT",
        "reports_only": True,
        "no_executable_samples_downloaded": True,
        "requested": {"benign": args.benign, "malware": args.malware},
        "downloaded": {"benign": len(benign_rows), "malware": len(malware_rows), "total": len(rows)},
        "benign_source_counts": dict(Counter(r["source"] for r in benign_rows)),
        "timestamp_note": "observed_at is CAPE execution time when present, not first-seen time; blank means unavailable.",
        "max_report_size_mb": args.max_size_mb,
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nManifest:")
    print(json.dumps(manifest, indent=2))
    if failures:
        print("Skipped candidates:", dict(failures))
    if len(benign_rows) != args.benign or len(malware_rows) != args.malware:
        print("\nERROR: requested counts not reached. Try --max-size-mb 50 or 0.", file=sys.stderr)
        raise SystemExit(2)
    print(f"\nCreated {out / 'metadata.csv'} with {len(rows)} report-only records.")
    print("No original EXE/DLL samples were downloaded or executed.")


if __name__ == "__main__":
    main()
