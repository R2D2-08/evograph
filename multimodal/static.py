"""Read-only official EMBER v2 extraction and precomputed JSON sidecars.

Sidecars contain ``sha256``, ``features`` (2381 finite numbers), and ``raw``
(the original EMBER raw-feature object). Optional ``schema_version`` and
``feature_version`` are checked when present; new extractions include both.
An unversioned 2381-vector is accepted, never padded or treated as EMBER v1.
No file in this module is ever executed.
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from numbers import Real
from pathlib import Path

from .cape import canonical_api
from .schema import SCHEMA_VERSION, STATIC_DIM


def _sha256(value, name="sha256", required=False):
    if not required and (value is None or (isinstance(value, str) and not value.strip())):
        return ""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()):
        raise ValueError(f"{name} must be a valid 64-hex SHA-256")
    return value.strip().lower()


def _vector(value, dimension, name, nonnegative=False):
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValueError(f"{name} must contain exactly {dimension} finite numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name} must contain only finite numbers")
        try:
            number = float(item)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite numbers") from exc
        if not math.isfinite(number) or (nonnegative and number < 0):
            qualifier = "finite nonnegative" if nonnegative else "finite"
            raise ValueError(f"{name} must contain only {qualifier} numbers")
        result.append(number)
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _validate_static(document, expected_sha256=""):
    if not isinstance(document, dict):
        raise ValueError("Static sidecar must be a JSON object")
    if document.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Static sidecar schema_version does not match " + SCHEMA_VERSION)
    version = document.get("feature_version", 2)
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ValueError("Static features require official EMBER feature_version=2")
    digest = _sha256(document.get("sha256"), required=True)
    expected = _sha256(expected_sha256, "expected_sha256")
    if expected and digest != expected:
        raise ValueError("Static SHA-256 identity mismatch with expected_sha256")
    raw = document.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("Static raw must be the original EMBER object")
    if "sha256" in raw and _sha256(raw["sha256"], "raw.sha256", required=True) != digest:
        raise ValueError("Static raw.sha256 identity mismatch")
    json.dumps(raw, allow_nan=False)
    return {**document, "sha256": digest,
            "features": _vector(document.get("features"), STATIC_DIM, "EMBER v2 features")}


def extract_ember(binary_path):
    """Read-only extraction, optionally through EVOGRAPH_EMBER_PYTHON.

    The subprocess bridge keeps legacy numerical/parser dependencies out of
    the model environment. It is dependency isolation, not a security sandbox.
    """
    python = os.environ.get("EVOGRAPH_EMBER_PYTHON", "").strip()
    if python:
        python = Path(python).expanduser().absolute()
        if not python.is_file():
            raise ValueError(f"EVOGRAPH_EMBER_PYTHON does not exist: {python}. Run python3 scripts/setup_ember.py")
        binary_path = Path(binary_path).expanduser().resolve()
        try:
            result = subprocess.run(
                [str(python), "-I", str(Path(__file__).with_name("ember_worker.py")), str(binary_path)],
                capture_output=True, text=True, timeout=120, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("EMBER extraction timed out after 120 seconds") from exc
        if result.returncode:
            raise ValueError(f"EMBER worker failed (exit {result.returncode}): {result.stderr.strip()}")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        try:
            document = json.loads(result.stdout, parse_constant=_reject_constant)
        except ValueError as exc:
            raise ValueError("EMBER worker did not return valid finite JSON") from exc
        digest = hashlib.sha256(binary_path.read_bytes()).hexdigest()
        return _validate_static(document, digest)
    from .ember_worker import extract_locally

    try:
        return _validate_static(extract_locally(binary_path))
    except ImportError as exc:
        raise ImportError(
            "Official Elastic EMBER (https://github.com/elastic/ember) is required. "
            "Run python3 scripts/setup_ember.py, then use .venv-ember/bin/python for extraction "
            "or set EVOGRAPH_EMBER_PYTHON to its absolute path in the model environment. "
            "Alternatively provide a precomputed EMBER v2 JSON via static_path."
        ) from exc


def load_static(path, expected_sha256=""):
    """Validate a sidecar, including every available SHA-256 identity claim."""
    expected_sha256 = _sha256(expected_sha256, "expected_sha256")
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            document = json.load(stream, parse_constant=_reject_constant)
        return _validate_static(document, expected_sha256)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid static sidecar {path}: {exc}") from exc


def imports_from_raw(raw):
    """Return deduplicated canonical function names from EMBER's DLL mapping.

    DLL keys and ordinal integers are not API names. Imports are evidence of
    imports only, never synthetic executed calls.
    """
    imports = raw.get("imports", {}) if isinstance(raw, dict) else {}
    if not isinstance(imports, dict):
        return []
    return list(dict.fromkeys(
        name for functions in imports.values() if isinstance(functions, list)
        for function in functions if isinstance(function, str)
        for name in [canonical_api(function)] if name
    ))
