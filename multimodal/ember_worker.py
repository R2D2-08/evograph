"""Standalone JSON worker for the dedicated, legacy-compatible EMBER runtime.

Run as a script with the extraction interpreter; no model dependencies or
project imports are required. Input bytes are parsed, never executed.
"""

import contextlib
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys


def extract_locally(binary_path):
    """Use the official extractor unchanged and retain its runtime provenance."""
    with contextlib.redirect_stdout(sys.stderr):
        from ember import PEFeatureExtractor

        content = Path(binary_path).read_bytes()
        if not content:
            raise ValueError("Cannot extract EMBER features from an empty file")
        extractor = PEFeatureExtractor(feature_version=2)
        raw = extractor.raw_features(content)
        features = extractor.process_raw_features(raw)
    if hasattr(features, "tolist"):
        features = features.tolist()
    versions = {}
    for package in ("ember", "numpy", "scikit-learn", "lief"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    source = {}
    try:
        direct_url = metadata.distribution("ember").read_text("direct_url.json")
        source = json.loads(direct_url) if direct_url else {}
    except metadata.PackageNotFoundError:
        pass
    url = source.get("url", "")
    match = re.search(r"/([0-9a-f]{40})\.zip(?:#.*)?$", url)
    revision = source.get("vcs_info", {}).get("commit_id") or (match[1] if match else None)
    lief_runtime = getattr(sys.modules.get("lief"), "__version__", versions["lief"])
    return {
        "schema_version": "evograph-embfv-v1", "feature_version": 2,
        "sha256": hashlib.sha256(content).hexdigest(), "features": features, "raw": raw,
        "extractor": {"implementation": "ember.PEFeatureExtractor", "source_revision": revision,
                      "source": source, "packages": versions, "lief_runtime": lief_runtime,
                      "python": platform.python_version(),
                      "reference_lief": "0.9.0"},
    }


def main():
    if len(sys.argv) != 2:
        print("usage: ember_worker.py BINARY_PATH", file=sys.stderr)
        return 2
    try:
        document = extract_locally(sys.argv[1])
        print(json.dumps(document, allow_nan=False))
    except Exception as exc:
        print(f"EMBER extraction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
