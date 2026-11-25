#!/usr/bin/env python3
"""Verify pinned official EMBER and extract bundled benign PEs without executing them."""

import argparse
import contextlib
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urldefrag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = (3, 11, 14)
LIEF_VERSION = "0.12.3-39115d10"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify():
    check(
        sys.version_info[:3] == PYTHON_VERSION and sys.implementation.name == "cpython",
        f"Run with the dedicated CPython 3.11.14 interpreter, not {sys.version.split()[0]}.",
    )
    check(sys.prefix != sys.base_prefix, "Verification requires the dedicated virtual environment.")
    pins = {"ember": "0.1.0"}
    source = None
    requirements = PROJECT_ROOT / "environments/ember/requirements.in"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ember @ "):
            source = line.split(" @ ", 1)[1]
        else:
            name, version = line.split("==", 1)
            pins[name] = version
    check(source is not None, "The official EMBER source pin is missing from requirements.in.")
    source_url, fragment = urldefrag(source)
    source_hash = parse_qs(fragment).get("sha256", [""])[0]
    check(len(source_hash) == 64, "The EMBER source archive must have a SHA-256 pin.")
    installed = {}
    for name, expected in pins.items():
        installed[name] = metadata.version(name)
        check(
            installed[name] == expected,
            f"{name}=={expected} is required; found {installed[name]}. Rerun setup to sync the lock.",
        )

    provenance_text = metadata.distribution("ember").read_text("direct_url.json")
    check(provenance_text, "EMBER has no direct_url.json proving its pinned official source URL.")
    provenance = json.loads(provenance_text)
    check(
        urldefrag(provenance.get("url", ""))[0] == source_url,
        f"EMBER must come from {source_url}, not {provenance.get('url')!r}.",
    )
    archive_info = provenance.get("archive_info", {})
    check(
        archive_info.get("hashes", {}).get("sha256") in (None, source_hash)
        and archive_info.get("hash") in (None, "sha256=" + source_hash),
        "Installed EMBER source archive hash disagrees with the source pin.",
    )

    # Exercise this interpreter, never a caller's optional subprocess bridge.
    os.environ.pop("EVOGRAPH_EMBER_PYTHON", None)
    sys.path.insert(0, str(PROJECT_ROOT))
    fixtures = []
    with contextlib.redirect_stdout(sys.stderr):
        for module in ("numpy", "scipy", "sklearn", "lief", "pandas"):
            importlib.import_module(module)
        try:
            importlib.import_module("lightgbm")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"LightGBM could not load: {exc}. On Linux, a missing libgomp.so.1 "
                "requires the system OpenMP runtime (libgomp1 on Debian/Ubuntu). "
                "Ask your administrator to install it separately; setup never runs sudo."
            ) from exc
        lief = importlib.import_module("lief")
        check(lief.__version__ == LIEF_VERSION, f"Expected LIEF runtime {LIEF_VERSION}, got {lief.__version__}.")
        from multimodal.static import extract_ember

        setuptools = metadata.distribution("setuptools")
        for name, machine, magic in (
            ("cli-32.exe", "I386", "PE32"),
            ("cli-64.exe", "AMD64", "PE32_PLUS"),
        ):
            fixture = Path(setuptools.locate_file("setuptools/" + name))
            content = fixture.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            document = extract_ember(fixture)
            raw = document["raw"]
            vector = document["features"]
            check(len(vector) == 2381, f"{name}: expected 2381 EMBER v2 features.")
            check(all(math.isfinite(value) for value in vector), f"{name}: nonfinite features.")
            check(
                document["sha256"] == raw.get("sha256") == digest,
                f"{name}: sidecar/raw SHA-256 does not match the input bytes.",
            )
            general = raw["general"]
            check(general["size"] == len(content), f"{name}: incorrect raw file size.")
            check(general["vsize"] > 0, f"{name}: PE parsing failed (zero virtual size).")
            check(raw["header"]["coff"]["machine"] == machine, f"{name}: wrong or empty COFF machine.")
            check(raw["header"]["optional"]["magic"] == magic, f"{name}: wrong PE32/PE32+ magic.")
            sections = raw["section"]["sections"]
            check(len(sections) > 0, f"{name}: no parsed PE sections.")
            imports_count = sum(len(entries) for entries in raw["imports"].values())
            check(general["imports"] > 0 and imports_count > 0, f"{name}: no parsed imports.")
            histogram = raw["histogram"]
            check(
                len(histogram) == 256 and min(histogram) >= 0 and sum(histogram) == len(content),
                f"{name}: raw byte histogram does not account for the input bytes.",
            )
            histogram_sum = sum(vector[:256])
            check(
                math.isclose(histogram_sum, 1.0, rel_tol=0, abs_tol=1e-6)
                and all(math.isclose(value, count / len(content), rel_tol=1e-6, abs_tol=1e-8)
                        for value, count in zip(vector[:256], histogram)),
                f"{name}: byte histogram features are not normalized raw counts.",
            )
            repeated = extract_ember(fixture)
            check(
                vector == repeated["features"] and raw == repeated["raw"],
                f"{name}: repeated extraction is not deterministic.",
            )
            fixtures.append({
                "path": str(fixture), "sha256": digest, "machine": machine,
                "pe_type": "PE32+" if magic == "PE32_PLUS" else "PE32",
                "feature_dimension": len(vector), "finite": True,
                "virtual_size": general["vsize"], "section_count": len(sections),
                "imports_count": imports_count, "byte_histogram_sum": histogram_sum,
                "deterministic": True,
            })

    return {
        "status": "ok", "python": sys.version.split()[0], "interpreter": sys.executable,
        "packages": installed, "lief_runtime_version": lief.__version__,
        "ember_source": {
            "url": source_url, "sha256": source_hash, "installed_direct_url": provenance,
        },
        "feature_version": 2, "fixtures": fixtures, "binaries_executed": False,
        "compatibility": "LIEF 0.12.3 differs from original EMBER 2018 LIEF 0.9.0; "
                         "not bit-exact EMBER 2018 or validated for published pretrained EMBER models.",
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        report = verify()
    except Exception as exc:
        print(f"EMBER verification failed: {exc}", file=sys.stderr)
        print("Run python3 scripts/setup_ember.py; see environments/ember/README.md.", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
