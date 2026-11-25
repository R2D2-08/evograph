# EvoGraph Multimodal Pipeline

The foundational static/dynamic pipeline is implemented additively in `run_multimodal.py` and `multimodal/`. It does not replace the original `run.py` API-frequency baseline, simplified graph model, or incremental replay demonstration documented in [README.md](README.md).

This guide describes the current implementation, not every proposal in [overall_architecture.md](../overall_architecture.md) and [planned_features.md](../planned_features.md). The default classifier is **hybrid**; the CLI trains `baseline` and `hybrid` by default.

> **Current local120 limitation:** The local corpus has no EMBER static vectors; `real120/metadata.csv` contains no static paths or explicit family/application groups. Dynamic-only training works without EMBER; baseline, graph and hybrid runs require `--allow-missing-static` until real static features are supplied. Static-only training is rejected when all static features are absent, even with that flag. These incomplete-modality runs are not full multimodal results, and family-disjoint evaluation is unavailable until every record is annotated.

## Installation and Safety

Use **Python 3.11 or 3.12**, preferably in a virtual environment, from `evograph/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` includes PyTorch and PyTorch Geometric. The training CLI currently runs on CPU. Official [Elastic EMBER](https://github.com/elastic/ember) is optional when consuming precomputed sidecars. Do **not** install an unrelated package merely named `ember` from PyPI or downgrade the model environment's dependencies to accommodate the legacy extractor.

### Dedicated Static Runtime

The [locked extraction environment](environments/ember/README.md) uses official EMBER source pinned to a commit, Python 3.11.14, NumPy 1.23.5, scikit-learn 1.1.3, and LIEF 0.12.3. With `uv` installed, provision or verify it from `evograph/`:

```bash
python3 scripts/setup_ember.py
python3 scripts/setup_ember.py --verify-only
.venv-ember/bin/python run_multimodal.py extract-static \
  --binary /path/to/sample.exe --out /path/to/static.json
```

Use the model environment with the optional subprocess bridge for metadata `binary_path` or prediction `--binary`:

```bash
export EVOGRAPH_EMBER_PYTHON="$(pwd)/.venv-ember/bin/python"
python3 run_multimodal.py prepare --metadata /path/to/metadata.csv \
  --out prepared_multimodal/dataset.jsonl
```

The bridge launches only the dedicated Python extractor, validates its JSON and input hash, and times out after 120 seconds. Extracted sidecars include dependency/source provenance, preserved as `static_extractor` in prepared records. Upstream warnings go to stderr rather than corrupting JSON output. The environment was verified on benign bundled PE32 and PE32+ launchers with real headers, sections, imports, and deterministic 2,381-dimensional output; these fixtures were never executed.

LIEF 0.12.3 differs from canonical EMBER2018's 0.9.0. Keep this extraction profile consistent for future training and inference; published pretrained EMBER model compatibility is not established. A virtual environment is dependency isolation, **not a security sandbox**. Use appropriate unprivileged containment for untrusted native parsing. Full static/dynamic pairing and training-grade validation remain outstanding.

This tool downloads no binaries and executes none. `extract-static`, metadata `binary_path`, and prediction `--binary` only read a **user-supplied** file for hashing and static parsing. They do not run CAPE or detonate a sample. Keep untrusted files isolated and retain the original CAPE JSON, original EMBER dictionaries, and acquisition provenance. A prediction is a research output, not a production safety verdict.

## Inputs and Identity

Preparation requires CSV columns `sample_id`, `report_path`, and `label` (`0`/`benign` or `1`/`malware`). Optional columns include `sha256`, `observed_at`, `static_path`, `binary_path`, `family`, and `group`. Report/static/binary paths are relative to the **CSV directory**, unless absolute. Supply `static_path` **or** `binary_path` per record, never both; both may be blank for missing-static experiments.

Schematic metadata, not a sufficient training dataset:

```csv
sample_id,report_path,label,sha256,observed_at,static_path,binary_path,family,group
sample_a,reports/sample_a.json,malware,<sample SHA-256>,2026-01-02T10:00:00Z,static/sample_a.json,,example_family,malware:example_family
sample_b,reports/sample_b.json,benign,<sample SHA-256>,2026-01-03T10:00:00Z,static/sample_b.json,,,app:example_application
```

`sha256` identifies the executable, not the report JSON. Preparation hashes report bytes separately as `report_sha256`, checks available metadata/report/static identity claims, and hashes binary bytes when extracting. Conflicting identities or labels fail; identical sample/label duplicates and blank labels are recorded as skips in the manifest. If the report lacks its target hash, a metadata claim alone cannot cryptographically establish report-to-binary correspondence.

A static sidecar contains exactly **2,381 finite numbers** and the **complete original EMBER raw-feature object**, not a replacement containing only import names. Template below uses placeholders, not loadable feature values:

```json
{
  "schema_version": "evograph-embfv-v1",
  "feature_version": 2,
  "sha256": "<64-hex executable SHA-256>",
  "features": "<replace with a JSON array of exactly 2381 finite numbers>",
  "raw": {
    "imports": {"KERNEL32.dll": ["CreateFileW"]},
    "<remaining original EMBER fields>": "<retain their original values>"
  }
}
```

Replace the entire illustrative `raw` object with the extractor's unmodified dictionary. Its `sha256`, if present, must agree. EMBER feature version 2 is required; older 2,351-dimensional vectors are not padded. `schema_version` and `feature_version` may be absent on existing sidecars, but are checked when present. Raw import names create static import evidence, never synthetic executed calls.

For PE payloads identified in CAPE, `--payload-static-dir payload_static_dir` loads optional sidecars named **`payload_static_dir/<sha>.json`**, checked against each payload's SHA-256. This directory is relative to the command's working directory, not the CSV. Preparation and prediction both support it; neither downloads or extracts missing payload binaries.

## Frozen Representation

Schema **`evograph-embfv-v1`** freezes static **2,381** + dynamic **2,048** = **4,429** dimensions. Static occupies full-vector indices `0-2380`; dynamic occupies `2381-4428`. All ranges below are zero-based and inclusive; add 2,381 to a dynamic offset to locate it in the full vector.

| Block | Dimensions | Dynamic Indices | Current Encoding |
| --- | ---: | --- | --- |
| D1 Statistics | 128 | 0-127 | Eight groups of 16 observed numerical features and availability flags |
| D2 API identity | 512 | 128-639 | Stable signed BLAKE2b hashing of normalized executed API frequencies |
| D3 API sequences | 256 | 640-895 | Per-process/thread BiGRU, event attention, then stream attention |
| D4 Files | 256 | 896-1151 | 128-dimensional path CharCNN, mean/max pooling |
| D5 Processes/commands | 256 | 1152-1407 | Name CharCNN64 and command token BiGRU64, separate mean/max pooling |
| D6 Registry | 128 | 1408-1535 | Character-encoded path components, hierarchical BiLSTM64, mean/max |
| D7 Network | 128 | 1536-1663 | Identifier CharCNN64 with literal-IP masking, mean/max |
| D8 Other resources | 128 | 1664-1791 | Shared CharCNN64 plus mutex/service/module type embeddings, mean/max |
| D9 Arguments | 128 | 1792-1919 | Typed semantics projected to 64 per retained event, mean/max |
| D10 Payloads | 128 | 1920-2047 | EMBER2381 projected to 64, mean/max plus availability embeddings |

Preparation writes **raw, versioned JSONL**, not pretrained embeddings: normalized CAPE evidence, unscaled D1 statistics, optional static vector/raw dictionary, availability, original metadata, report path/hash, and source pointers. Original reports are not rewritten or embedded wholesale; keep those files for evidence omitted by normalization or caps.

D3-D10 contain trainable modules fitted jointly with the classifier on **training data only**. D1 extraction and D2 hashing are not learned. Split before fitting: the scaler applies `log1p` and training-population standardization to non-mask D1 values, leaves availability/missingness/truncation flags unchanged, and standardizes static vectors using present training sample vectors only. Payload vectors use those same static moments; no training static vectors means identity static scaling. Validation/test/prediction reuse the saved scaler without refitting.

Missing static features remain `null` with explicit availability rather than pretending to be a measured zero-valued PE. Models use learned availability embeddings; payloads have per-vector availability and report-level coverage. D1 and graph telemetry distinguish unavailable sections from observed empty sections. Numerical zero placeholders must be interpreted with these masks.

File encoders mask local Windows profile usernames without changing raw evidence. Network encoders replace literal IPv4/IPv6 hosts and their endpoint ports with coarse markers, while retaining domain names and URL paths. Both baseline and graph branches use these normalizations to reduce identity memorization.

## Graph and Classifiers

The graph has exactly six node types, with one binary node per report:

| Node Type | Initial Encoding Before Common Hidden Projection |
| --- | --- |
| `binary` | Static MLP `2381 -> 512 -> 256`, availability embedding, and projected coverage/truncation flags |
| `process` | Name64 + projected command32 + 32 local statistics = 128 |
| `api` | API64 + category8 + imported/resolved flags2, projected to 128 |
| `window` | Sequence BiGRU/attention256 projected to 128 |
| `resource` | File/registry/network/mutex/service/module string encoders, type and local evidence statistics, combined at 128 |
| `payload` | Mask-aware EMBER projection64 projected to 128 |

Typed forward relations include binary `launches` process / `imports` API / `produces` payload; process `spawns` process / `calls` API / `executes` window / `loads` or `accesses` resource / `drops` payload; window `contains_api` API / `next_window` window / `reads`, `writes`, `deletes`, `modifies`, `connects_to`, or `accesses` resource. Each has an exact `rev_` relation, with independently learned parameters.

Windows preserve source order within each process/thread, including a separate unidentified-thread stream per process. They do not impose a global timestamp order. Imported/resolved APIs are not counted as executions; count-only `apistats` fallback does not fabricate sequences. Resource links use explicit names/paths/pretty values: numeric API handles are **not** resolved into guessed resource bindings. Summary/network inventories do not invent process ownership or event edges.

All types project to common hidden size **256** by default. Three **relation-mean R-GCN-style layers** use PyG `HeteroConv(aggr="sum")` over relation-specific `SAGEConv(aggr="mean", root_weight=False, bias=False)` modules, with no SAGE normalization. This is not an untyped GraphSAGE stack or a call to `RGCNConv`. The exact relation normalization and layer update are:

$$
m_{v,r}^{(l)} = \frac{1}{\max(1, |N_r(v)|)}
    \sum_{u \in N_r(v)} W_r^{(l)} h_u^{(l)}
$$

$$
h_v^{(l+1)} = \operatorname{LayerNorm}_{t(v)}^{(l)}\left(
 h_v^{(l)} + \operatorname{Dropout}\left(\operatorname{GELU}\left(
 W_{0,t(v)}^{(l)} h_v^{(l)} + b_{0,t(v)}^{(l)} + \sum_r m_{v,r}^{(l)}
 \right)\right)\right).
$$

Here `N_r(v)` is the incoming neighbor set for relation `r`; empty sums contribute zero. Neighbor means are normalized **per relation**, then summed, not averaged across relation types or divided by total graph degree. The self transform, bias and LayerNorm are node-type-specific; the outer addition is residual. Dropout defaults to 0.2.

After three layers, learned attention pools each non-binary type separately and projects it to 128 dimensions; the binary representation is used directly. At default hidden size, concatenation is `256 + 5*128 = 896`, projected to a **512-dimensional graph embedding**. Empty type pools are masked; payload coverage is retained.

| Mode | Classifier Input |
| --- | --- |
| `baseline` | Static + all dynamic blocks, 4,429 dimensions, with jointly trained encoders |
| `static` | Static-only ablation, 2,381 dimensions |
| `dynamic` | Dynamic-only ablation, 2,048 dimensions |
| `graph` | Static binary node + behavioral graph, pooled to 512; **no global D1 branch** |
| `hybrid` | Graph512 + D1 projected `128 -> 64`, giving 576 dimensions; default classifier |

Graph-only still includes **coverage/truncation telemetry** on the binary node, static/payload availability, and local process/resource statistics. It excludes global D1 values, not missingness evidence. Graph/hybrid do not compute or append the entire 2,048-dimensional baseline branch. The default hybrid head is `576 -> 256 -> 64 -> 1`, followed by sigmoid for malware probability.

## Commands

Prepare the existing report-only corpus and compare the default baseline/hybrid modes as an explicitly incomplete-modality experiment:

```bash
python3 run_multimodal.py prepare --metadata real120/metadata.csv \
  --out prepared_multimodal/dataset.jsonl
python3 run_multimodal.py train --dataset prepared_multimodal/dataset.jsonl \
  --out results_multimodal --allow-missing-static
```

Use `--models dynamic` for dynamic-only training without that override. `--models baseline hybrid graph static dynamic` selects all supported modes, but `static` needs actual static features. Once metadata supplies real sidecars or user-supplied binaries, prepare again and omit `--allow-missing-static` for complete-modality experiments. Add `--payload-static-dir payload_static_dir` to preparation/prediction when available.

Inspect report coverage, validation warnings, statistics, graph counts and truncation limits:

```bash
python3 run_multimodal.py inspect --report /path/to/report.json
```

Predict using the checkpoint's saved extraction settings, scaler and validation-selected threshold:

```bash
python3 run_multimodal.py predict --report /path/to/report.json \
  --checkpoint results_multimodal/hybrid.pt
```

That report-only prediction requires a dynamic checkpoint or one trained with missing static explicitly allowed. Otherwise add `--static /path/to/static.json` or `--binary /path/to/user-supplied/sample.exe`, not both.

With official EMBER available, read a user-supplied binary and write its validated sidecar:

```bash
python3 run_multimodal.py extract-static \
  --binary /path/to/user-supplied/sample.exe --out static/sample.json
```

Export fixed vectors **only after fitting a `baseline` or `dynamic` checkpoint**; graph/hybrid/static checkpoints are rejected rather than exporting random encoder outputs:

```bash
python3 run_multimodal.py export-features \
  --dataset prepared_multimodal/dataset.jsonl \
  --checkpoint results_multimodal/baseline.pt --out results_multimodal/features.npz
```

The NPZ contains standardized `static_features` (`N x 2381`), learned `dynamic_features` (`N x 2048`), concatenated `features` (`N x 4429`), `static_available`, sample IDs, SHA-256s, labels, schema version and block names. Missing static rows export zeros **with their availability mask**, not extracted zero-valued features. D1 coverage flags remain inside the dynamic vector. The fitted scaler remains in the checkpoint; vectors are checkpoint-dependent, not immutable raw features.

Export requires the dataset's `extraction.max_events` to match the checkpoint. Reprepare with the recorded limit rather than mixing extraction configurations. Prediction rejects unreadable/corrupt input files; a valid JSON object with genuinely missing telemetry is still representable.

## Evaluation and Outputs

The new pipeline uses **60/20/20 train/validation/test**, approximately **72/24/24** for 120 usable records. All modes in one invocation use the **same partitions** and train their own parameters. This is separate from the original `run.py` **50/15/15/20** initial-train/validation/update/test split; there is no new-pipeline update partition.

- `--split-mode stratified` is the default. It does not establish temporal or unseen-family generalization.
- `--split-mode family` requires explicit `group` or fallback `family` on **every record**, including benign application groups. No families/groups are inferred from filenames, sources, labels, or CAPE verdicts. Enough disjoint groups must support both labels in every partition; group sizes can make proportions approximate. Current local120 metadata lacks these annotations, so this evaluation is unavailable until they are supplied.
- `--split-mode chronological` requires genuine `observed_at` datetimes on every record. The new preparer does not infer them from CAPE automatically. Missing times, tied boundaries and **one-class time windows** are rejected. CAPE execution time is not first-seen time.

The scaler and encoder gradients use training data only. Validation loss selects the checkpoint; validation F1 selects the threshold. Test data are reserved for final evaluation. Load only trusted checkpoints, even though loading uses `weights_only=True`.

- `prepared_multimodal/dataset.jsonl`: versioned raw records; `dataset.manifest.json`: extraction settings, feature names, counts and skipped rows.
- `results_multimodal/<mode>.pt`: fitted weights, model/extraction configuration, scaler, threshold, schema, dataset hash and available preparation provenance.
- `results_multimodal/metrics.json`: split sizes, missing-static count, loss histories, test metrics, confusion matrices and research limitations.
- `results_multimodal/splits.csv`: exact sample assignments; `predictions.csv`: held-out probabilities for each requested mode.

## Limits and Deferred Work

The completed foundation covers validated preparation, a verified isolated EMBER runtime, D1-D10 encoders, six-type graph construction, three relation-aware layers, type attention, ablations, shared splits, checkpoints, prediction and fitted-vector export. It does **not** establish research performance. A concise [IEEE Transactions-style methodology note](paper/evograph_methodology.tex) and [compiled PDF](paper/evograph_methodology.pdf) describe the design without empirical performance claims.

- Default limits are `--max-events 20000` retained source rows per report, `--window-size 32`, and `--max-windows 256`. Zero event/window budgets retain none, not unlimited data. Repeats remain compressed, capped at 1,000,000 occurrences per source row. The window budget keeps a prefix in process/first-seen-thread order and may bias retained evidence.
- Event retention is not a scan limit: full API totals and resource inventories can include rows beyond it, while sequences and event-level features use retained rows. The window cap further limits graph windows, not baseline streams or process/resource inventories. Coverage/truncation flags report these distinctions.
- Reports are loaded and scanned in memory; process/API/resource/payload node counts are not globally capped. Strings and token/hierarchy lengths are bounded by encoders, but graph caps do not guarantee bounded RAM or training cost. Reduce batch size/window budgets and audit resource usage; large-scale scalability has not been established.
- HGT, RGAT and GINE experiments, the Dambra-style high-dimensional baseline/comparisons, large-scale benchmarks, and incremental learning for this **new pipeline** are deferred. The original `run.py` replay demonstration remains available and unchanged.
- A training-grade labeled dataset, complete static/dynamic pairing with real EMBER integration, and family/time-disjoint research evaluation are not yet validated. Audit labels, provenance, duplication, sandbox/source confounding and missing telemetry before reporting results. No superiority, robustness, concept-drift resistance or production-detection claim follows from this implementation or local120 smoke runs.

## Verification

Run the original and new unit/integration tests from `evograph/` in the installed environment:

```bash
python3 -m unittest discover -v
```

Tests use small synthetic fixtures for extraction, identity validation, scaling, graph relations, encoder gradients, model ablations, checkpoint prediction and feature export. They do not execute sample binaries or access the network. Missing optional test dependencies are reported as skips; install the requirements to exercise all numerical and graph tests.
