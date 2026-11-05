#!/usr/bin/env python3
"""Audit a downloaded CAPE-Evolve metadata.csv before training."""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime


def count_apis(doc):
    behavior = doc.get("behavior") or {}
    n = 0
    for proc in behavior.get("processes") or []:
        n += sum(1 for call in (proc.get("calls") or [])
                 if isinstance(call, dict) and call.get("api"))
    if n == 0:
        for values in (behavior.get("apistats") or {}).values():
            if isinstance(values, dict):
                n += sum(1 for v in values.values() if isinstance(v, (int, float)) and v > 0)
    if n == 0:
        n = len((behavior.get("summary") or {}).get("resolved_apis") or [])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="real120/metadata.csv")
    args = ap.parse_args()
    meta = Path(args.metadata).resolve()
    rows = list(csv.DictReader(meta.open(encoding="utf-8", newline="")))
    labels = Counter(r["label"] for r in rows)
    sources = Counter(r.get("source", "unspecified") for r in rows)
    missing_time = sum(not r.get("observed_at", "").strip() for r in rows)
    missing_files = 0
    bad_json = 0
    zero_api = 0
    months = defaultdict(Counter)
    for r in rows:
        report = Path(r["report_path"])
        if not report.is_absolute():
            report = meta.parent / report
        if not report.exists():
            missing_files += 1
            continue
        try:
            doc = json.loads(report.read_text(encoding="utf-8"))
            if count_apis(doc) <= 0:
                zero_api += 1
        except Exception:
            bad_json += 1
        stamp = r.get("observed_at", "").strip()
        if stamp:
            try:
                dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                months[dt.strftime("%Y-%m")][r["label"]] += 1
            except ValueError:
                pass
    print(f"Total manifest rows: {len(rows)}")
    print(f"Labels (0=benign, 1=malware): {dict(labels)}")
    print(f"Sources: {dict(sources)}")
    print(f"Missing CAPE execution timestamps: {missing_time}/{len(rows)}")
    print(f"Missing report files: {missing_files}; invalid JSON: {bad_json}; reports with no API trace: {zero_api}")
    if months:
        print("Month-by-label distribution (CAPE execution month, not first-seen month):")
        for month in sorted(months):
            print(f"  {month}: {dict(months[month])}")
    else:
        print("No usable CAPE execution timestamps.")
    print("\nRecommendation:")
    if missing_files or bad_json or zero_api:
        print("Resolve missing/invalid/no-API reports before training; run prepare and inspect skipped.csv too.")
    if missing_time or len(months) < 3 or any(len(c) < 2 for c in months.values()):
        print("Use --split-mode stratified for this prototype. Do NOT call it a concept-drift evaluation.")
        print("For a temporal drift claim, collect older/newer benign AND malware observations with genuine acquisition timestamps.")
    else:
        print("Temporal coverage exists, but inspect class counts and split boundaries before attempting --split-mode chronological.")

if __name__ == "__main__":
    main()
