# Official EMBER Extraction Environment

This dedicated environment uses Elastic's official EMBER 0.1.0 source at commit
`d97a0b523de02f3fe5ea6089d080abacab6ee931` and produces EMBER v2's 2,381 features
plus the original raw metadata. The source archive and all runtime packages are
SHA-256-locked in `requirements.lock`; `requirements.in` pins every dependency,
including NumPy 1.23.5, scikit-learn 1.1.3, and LIEF 0.12.3
(`lief.__version__ == "0.12.3-39115d10"`). The isolated EMBER source build uses
the setuptools and wheel versions in `build-constraints.txt`.

## Setup and Verification

Validated on Linux x86_64 with CPython **3.11.14** and uv **0.12.21**. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) separately, then run
from the project root using your ordinary Python (including Python 3.14):

```bash
python3 scripts/setup_ember.py
python3 scripts/setup_ember.py --verify-only
.venv-ember/bin/python scripts/verify_ember.py
```

Setup creates `.venv-ember` with exactly Python 3.11.14 if absent, syncs the
hash-locked packages with isolated-build constraints, then verifies extraction.
It never deletes or recreates an existing environment. A wrong Python version,
non-venv directory, or broken environment is an error; choose a new dedicated
path with `--venv /path/to/new/ember-venv` instead. Relative `--venv` paths are
relative to the project root, regardless of the current directory. Sync removes
packages outside this lock, so do not point it at your model environment.
`--verify-only` performs no downloads, installation, or syncing and needs no uv.

LightGBM needs the system OpenMP runtime. If importing it fails with
`libgomp.so.1` missing on Debian/Ubuntu, have the system administrator install
`libgomp1` separately. These scripts never invoke sudo or install OS packages.

The verifier checks Python, every installed version, the official EMBER
`direct_url.json` source URL/commit (and its archive hash when recorded), and
LIEF's runtime version. It extracts setuptools' bundled benign `cli-32.exe`
(PE32) and `cli-64.exe` (PE32+) twice each. It checks 2,381 finite features,
input/raw/sidecar SHA-256 identity, nonzero virtual size, COFF machine and PE
magic, parsed sections and imports, normalized byte histograms, and deterministic
repeated extraction. The binaries are **never executed** and no sample is
downloaded. A JSON report goes to stdout; diagnostics go to stderr. Verification
unsets `EVOGRAPH_EMBER_PYTHON` internally to test its own interpreter, not a bridge.

## Usage

Extract directly with the dedicated interpreter, without changing the packages
in your separate modern model environment:

```bash
.venv-ember/bin/python run_multimodal.py extract-static \
  --binary /path/to/sample.exe --out /path/to/static.json
```

The optional subprocess bridge lets the modern model environment request
extraction with this interpreter. Set an **absolute** path, then run the usual
CLI with your model environment's Python:

```bash
export EVOGRAPH_EMBER_PYTHON="$(pwd)/.venv-ember/bin/python"
python run_multimodal.py extract-static \
  --binary /path/to/sample.exe --out /path/to/static.json
```

Alternatively, generate JSON sidecars directly and consume their precomputed
features and raw metadata from the model environment. No CAPE client or dynamic
execution is installed or configured here.

## Rebuild the Lock

After an intentional pin change, regenerate rather than hand-editing hashes:

```bash
uv pip compile --python .venv-ember/bin/python --generate-hashes \
  --build-constraints environments/ember/build-constraints.txt \
  -o environments/ember/requirements.lock environments/ember/requirements.in
python3 scripts/setup_ember.py
```

The official source archive's SHA-256 is
`8eb5e9dbf9982c9453cf75e7f3d323756c4572d3e804b26cc1dad2c46acc8eed`.
Runtime hashes are enforced by `uv pip sync --require-hashes`; build constraints
pin isolated build-tool versions but are not a separate hash lock of build tools.

## Compatibility and Safety

This is official EMBER source with a verified compatible dependency stack, not a
replacement vectorizer or synthetic zero-padding. However, LIEF 0.12.3 differs
from the original EMBER 2018 extractor's LIEF 0.9.0. The official warning is
expected: **do not claim bit-exact EMBER 2018 compatibility or use published
pretrained EMBER models as if that compatibility had been established**. Keep
the extraction stack consistent for model training and inference, and retain
raw metadata and environment provenance alongside derived data.

These are old dependencies, including a native PE parser. A virtual environment
isolates Python packages; **it is not a security sandbox**. For untrusted PEs,
run unprivileged in a suitably hardened container or isolated VM with read-only
inputs and no network egress. Such containment is recommended, not implemented
by these setup scripts. Static extraction reads bytes and never launches inputs.
