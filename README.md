# CAPE-Evolve: Real 120-report prototype

This version is intended for the Semester 7 project. It replaces the synthetic-only demo path with a downloader for **120 real CAPEv2 JSON reports** from a public dataset. It downloads reports only; it does **not** download or execute EXE/DLL samples.

## Dataset used

**Windows Executable Malware Analysis Dataset** — Hugging Face: `Thcrull/win-exe-malware-analysis` (dataset card lists MIT license; DOI `10.57967/hf/10720`). The dataset card describes 1,844 samples: 1,082 malware reports sourced from MalwareBazaar and 762 benign reports sourced from PortableApps, PortableApps installers, Sysinternals Suite, and Benign-NET. The reports were generated with CAPEv2 in a Windows 10 Pro detonation VM.

The repository contains both raw CAPEv2 JSON reports and reduced reports. This starter downloads the **raw JSON report folders** because the current feature extractor uses process/API traces. It never downloads the `malware_samples/` or `benign_samples/` directories.

Dataset card: https://huggingface.co/datasets/Thcrull/win-exe-malware-analysis

## 1. Install

Use Python 3.11 or 3.12 on Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the PyTorch wheel fails, install the CPU wheel using the selector at https://pytorch.org/get-started/locally/ and then rerun `pip install -r requirements.txt`.

## 2. Download exactly 120 report JSONs

```bash
python download_real120.py \
  --out real120 \
  --benign 60 \
  --malware 60 \
  --max-size-mb 30 \
  --seed 42
```

Or run the entire download → audit → prepare → train → predict sequence with:

```bash
./run_real120.sh
```

The downloader selects 60 malware reports and 60 benign reports. It tries to spread the benign reports across the four source groups. It only requests files in:

- `malware_reports/reports/*.json`
- `benign_reports/<source>/reports/*.json`

It excludes `filtered_reports/` and every executable sample folder. The output contains the original report directory structure, `metadata.csv`, and `manifest.json`. The manifest records source and report hashes.

**Storage:** allow a few GB free. Raw CAPE reports can be large. The default per-report size cap is 30 MB. If the script cannot find 60 usable reports in each class, retry with `--max-size-mb 50`; use `--max-size-mb 0` only if you have sufficient storage.

## 3. Audit before training

```bash
python audit_dataset.py --metadata real120/metadata.csv
```

Check that it reports 60 benign and 60 malware. It also reports sources, missing CAPE execution timestamps, and month-by-month class counts.

The dataset's labels come from the dataset authors' source/provenance categories. They are useful for a prototype, but they are not an independently adjudicated ground-truth audit. Keep that limitation in your report.

## 4. Parse and prepare the reports

Inspect one malware report:

```bash
MAL_REPORT=$(find real120/malware_reports/reports -name '*.json' | head -n 1)
python run.py inspect --report "$MAL_REPORT"
```

Prepare the dataset:

```bash
python run.py prepare \
  --metadata real120/metadata.csv \
  --out prepared
```

Expected output is approximately 120 parsed reports. Check:

- `prepared/dataset.jsonl`
- `prepared/skipped.csv`

If some reports have no API traces, they are skipped rather than assigned fake features. If too many are skipped, inspect the report schema and the reasons in `skipped.csv`.

## 5. Train the baseline, graph model, and one incremental update

Start with the stratified split:

```bash
python run.py train \
  --dataset prepared/dataset.jsonl \
  --out results \
  --split-mode stratified \
  --epochs 24 \
  --update-epochs 12 \
  --replay 12
```

For 120 usable samples, the intended split is approximately:

- 60 initial training
- 18 validation
- 18 update batch
- 24 held-out test

Models:

1. Random Forest on API-frequency features.
2. Two-layer message-passing GNN on process → API → capability graphs.
3. One incremental GNN update using the update batch plus a small replay memory from initial training.

The held-out test set is not used for training or threshold selection. Outputs are written to `results/`.

## 6. When can you use a chronological split?

Only use:

```bash
python run.py train --dataset prepared/dataset.jsonl --out results_time --split-mode chronological
```

if the audit shows genuine CAPE execution timestamps across a useful time range and both labels occur throughout the time windows. `observed_at` is the CAPE execution time (`info.started`/`info.ended`) when present. It is **not** the malware's first-seen date. The program rejects missing timestamps, tied time boundaries, or one-class windows.

If you use `--split-mode stratified`, call the update a **stratified incremental-learning demonstration**. Do not claim that it proves concept-drift resistance or temporal generalization. The dataset is a compact prototype corpus, not a longitudinal malware stream.

## 7. Results and presentation

The training command produces:

- `results/metrics.json` — precision, recall, F1, balanced accuracy, confusion matrices, sample counts, split mode.
- `results/splits.csv` — exact sample-to-split assignments.
- `results/predictions.csv` — held-out probabilities from RF, GNN before update, and GNN after update.
- `results/sample_graph.png` — one generated behavior graph.
- `results/f1_comparison.png` — measured model comparison.
- `results/confusion_matrix.png` — updated graph model confusion matrix.
- `results/random_forest.joblib`, `graph_before.pt`, `graph_after.pt` — model artifacts.

Run one inference:

```bash
python run.py predict --report "$MAL_REPORT" --model-dir results
```

## 8. What you can claim

For a stratified 120-report experiment, a defensible claim is:

> We built a CAPEv2 report-processing pipeline, represented process/API behavior as simplified graphs, compared a Random Forest baseline with a graph classifier, and demonstrated one incremental update with replay on a held-out test set.

Do **not** claim:

- a fully autonomous/self-evolving production detector;
- measured concept-drift resistance from a random split;
- robust generalization to unseen malware families;
- a large-scale longitudinal dataset;
- conference-level validation from 120 reports alone.

The graph is deliberately simplified: process → API → ten capability categories. It does not yet encode full temporal event order, file/registry/network object identities, or rich API arguments.

## 9. Safety and provenance

This workflow processes JSON reports only. It does not need the underlying malware binaries. Do not download or execute the dataset's binaries for this project. If you later collect additional samples in your own CAPE installation, keep them in a disposable isolated guest, disable host-shared folders, restrict network egress, restore snapshots between runs, and retain hashes and acquisition provenance.

## Dataset citation

Thcrull. *Windows Executable Malware Analysis Dataset*. Hugging Face Datasets, DOI: `10.57967/hf/10720`. Cite the dataset card and its associated thesis/research project when you use the data. Confirm the dataset license and attribution requirements before redistribution.
