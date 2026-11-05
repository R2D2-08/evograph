#!/usr/bin/env bash
set -euo pipefail

# Run after activating the project's virtual environment.
python download_real120.py --out real120 --benign 60 --malware 60 --max-size-mb 30 --seed 42
python audit_dataset.py --metadata real120/metadata.csv

MAL_REPORT="$(find real120/malware_reports/reports -type f -name '*.json' | head -n 1)"
if [[ -z "${MAL_REPORT}" ]]; then
  echo "No malware report was downloaded. Check the download output." >&2
  exit 2
fi
python run.py inspect --report "${MAL_REPORT}"
python run.py prepare --metadata real120/metadata.csv --out prepared
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
rows=[json.loads(x) for x in Path("prepared/dataset.jsonl").read_text().splitlines() if x.strip()]
counts=Counter(r["label"] for r in rows)
print(f"Prepared usable reports: {len(rows)}; labels: {dict(counts)}")
if len(rows) < 24 or len(counts) < 2:
    raise SystemExit("Not enough parseable reports or only one class. Inspect prepared/skipped.csv and obtain more reports.")
PY
python run.py train --dataset prepared/dataset.jsonl --out results --split-mode stratified --epochs 24 --update-epochs 12 --replay 12
python run.py predict --report "${MAL_REPORT}" --model-dir results

echo "Finished. Inspect results/metrics.json, splits.csv, predictions.csv, and PNG figures."
