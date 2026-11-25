"""Versioned preparation, strict JSONL loading, and training-only scaling.

``prepare_dataset`` accepts a CSV path or an iterable of metadata dictionaries.
Paths are relative to the CSV directory (cwd for dictionaries). It writes JSONL
and ``output_path.with_suffix('.manifest.json')``; skips are in the manifest.
Only blank labels and identical sample/label duplicates are skipped. The CSV
and original reports are never rewritten. Metadata SHA claims without a target
hash are not cryptographic verification of report-to-binary correspondence.

Records and manifests carry SCHEMA_VERSION. Stored statistics are always RAW;
call FeatureScaler.fit on training records only, then transform each partition.
Scaler static moments use present training *sample* vectors, not payloads;
payloads use those same moments. With no training static vectors, static
scaling is the identity. Naive ISO datetimes are interpreted as UTC; missing
observation times remain empty, and date-only values are rejected.
"""

import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .cape import COVERAGE_KEYS, parse_report
from .schema import SCHEMA_VERSION, STATIC_DIM, STATS_DIM
from .static import (_reject_constant, _sha256, _vector, extract_ember,
                     imports_from_raw, load_static)


def _parse_observed_at(value):
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("observed_at must be an ISO datetime")
    value = value.strip()
    if not value:
        return None
    if not any(separator in value for separator in ("T", "t", " ")):
        raise ValueError("observed_at requires an ISO datetime, not just a date")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid ISO observed_at: {value!r}") from exc


def _label(value):
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip().lower() in ("0", "1", "benign", "malware"):
        return int(value.strip().lower() in ("1", "malware"))
    raise ValueError("label must be binary: 0/benign or 1/malware")


def _text(value, name, required=False):
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"{name} must be {'a nonempty' if required else 'a'} string")
    return value.strip()


def _validate_record(record):
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Record schema_version must be " + SCHEMA_VERSION)
    required = {"sample_id", "label", "sha256", "report_path", "report_sha256", "observed_at",
                "family", "group", "metadata", "report", "static_features", "statistics", "extraction"}
    if not required.issubset(record):
        raise ValueError("Record missing fields: " + ", ".join(sorted(required - record.keys())))
    record = copy.deepcopy(record)
    extraction = record["extraction"]
    if (not isinstance(extraction, dict) or type(extraction.get("max_events")) is not int
            or extraction["max_events"] < 0):
        raise ValueError("extraction.max_events must be a nonnegative integer; reprepare the dataset")
    if (type(record.get("feature_version", 2)) is not int
            or record.get("feature_version", 2) != 2):
        raise ValueError("Record static features require EMBER feature_version=2")
    record["sample_id"] = _text(record["sample_id"], "sample_id", required=True)
    if type(record["label"]) is not int or record["label"] not in (0, 1):
        raise ValueError("Record label must be a binary integer 0 or 1")
    record["sha256"] = _sha256(record["sha256"])
    record["report_sha256"] = _sha256(record["report_sha256"], "report_sha256", required=True)
    if not isinstance(record["report_path"], str) or not Path(record["report_path"]).is_absolute():
        raise ValueError("report_path must be absolute")
    stamp = _parse_observed_at(record["observed_at"])
    record["observed_at"] = stamp.isoformat() if stamp else ""
    for key in ("family", "group"):
        record[key] = _text(record[key], key)
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("metadata must preserve the original metadata object")
    metadata_hash = _sha256(metadata.get("sha256"), "metadata.sha256")
    if metadata_hash and metadata_hash != record["sha256"]:
        raise ValueError("metadata.sha256 identity mismatch")
    if "label" in metadata and _label(metadata["label"]) != record["label"]:
        raise ValueError("metadata.label disagrees with record label")
    for key in ("family", "group"):
        if _text(metadata.get(key), "metadata." + key) != record[key]:
            raise ValueError(key + " must come from explicit user metadata")
    report = record["report"]
    if not isinstance(report, dict):
        raise ValueError("report must be normalized CAPE IR")
    if {"label", "family", "group", "metadata", "signatures", "verdict", "malscore"} & report.keys():
        raise ValueError("report IR must not contain labels, verdicts, or family metadata")
    report_hash = _sha256(report.get("sha256"), "report.sha256")
    if report_hash and report_hash != record["sha256"]:
        raise ValueError("report.sha256 identity mismatch")
    report["sha256"] = report_hash
    for key in ("processes", "resources", "payloads"):
        if not isinstance(report.get(key), list) or any(not isinstance(item, dict) for item in report[key]):
            raise ValueError(f"report.{key} must be a list of objects")
    for key in ("imports", "resolved_apis", "validation_warnings"):
        if not isinstance(report.get(key), list) or any(not isinstance(item, str) for item in report[key]):
            raise ValueError(f"report.{key} must be a list of strings")
    coverage = report.get("coverage")
    if not isinstance(coverage, dict) or any(type(coverage.get(key)) is not bool for key in COVERAGE_KEYS):
        raise ValueError("report.coverage must contain the schema's boolean availability flags")
    for process in report["processes"]:
        if (not isinstance(process.get("events"), list)
                or any(not isinstance(event, dict) for event in process["events"])
                or not isinstance(process.get("api_counts"), dict)):
            raise ValueError("IR processes require events objects and an api_counts mapping")
    if record["static_features"] is not None:
        record["static_features"] = _vector(record["static_features"], STATIC_DIM, "static_features")
        if not record["sha256"]:
            raise ValueError("Present static_features require a sample sha256")
    if "static_available" in record and (type(record["static_available"]) is not bool
                                        or record["static_available"] != (record["static_features"] is not None)):
        raise ValueError("static_available disagrees with static_features")
    raw = record.get("static_raw")
    if raw is not None:
        if not isinstance(raw, dict) or record["static_features"] is None:
            raise ValueError("static_raw requires an object and present static_features")
        if "sha256" in raw and _sha256(raw["sha256"], "static_raw.sha256", True) != record["sha256"]:
            raise ValueError("static_raw.sha256 identity mismatch")
    for payload in report["payloads"]:
        if payload.get("static_features") is not None:
            payload["static_features"] = _vector(payload["static_features"], STATIC_DIM, "payload static_features")
    record["statistics"] = _vector(record["statistics"], STATS_DIM, "statistics", nonnegative=True)
    json.dumps(record, allow_nan=False)
    return record


def prepare_dataset(metadata, output_path, max_events=20000, payload_static_dir=None):
    """Prepare validated raw records; malformed labeled samples raise, not skip."""
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise ValueError("max_events must be a nonnegative integer")
    required = {"sample_id", "report_path", "label"}
    base = Path.cwd()
    if isinstance(metadata, (str, Path)):
        source = Path(metadata).resolve()
        base = source.parent
        with source.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Metadata CSV requires sample_id, report_path, label")
            rows = list(reader)
        source_name = str(source)
    else:
        rows = list(metadata)
        source_name = ""
    records, skipped, by_id, by_hash = [], [], {}, {}
    for row_number, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError("Metadata rows require sample_id, report_path, label")
            if row["label"] is None or (isinstance(row["label"], str) and not row["label"].strip()):
                skipped.append({"row": row_number, "sample_id": row.get("sample_id"), "reason": "blank_label"})
                continue
            sample_id = _text(row["sample_id"], "sample_id", required=True)
            label = _label(row["label"])
            family = _text(row.get("family"), "family")
            group = _text(row.get("group"), "group")
            json.dumps(row, allow_nan=False)
            stamp = _parse_observed_at(row.get("observed_at"))
            report_path = (base / _text(row["report_path"], "report_path", True)).resolve()
            static_path = _text(row.get("static_path"), "static_path")
            binary_path = _text(row.get("binary_path"), "binary_path")
            if static_path and binary_path:
                raise ValueError("Specify static_path OR binary_path, not both (ambiguous static source)")
            report_bytes = report_path.read_bytes()
            document = json.loads(report_bytes.decode("utf-8-sig"), parse_constant=_reject_constant)
            if not isinstance(document, dict):
                raise ValueError("Raw report must be a JSON object")
            json.dumps(document, allow_nan=False)
            report = parse_report(document, max_events=max_events)
            metadata_hash = _sha256(row.get("sha256"), "metadata.sha256")
            report_hash = _sha256(report["sha256"], "report.sha256")
            if metadata_hash and report_hash and metadata_hash != report_hash:
                raise ValueError("Metadata/report SHA-256 identity mismatch")
            expected_hash = report_hash or metadata_hash
            static = None
            if static_path:
                static = load_static(base / static_path, expected_hash)
            elif binary_path:
                static = extract_ember(base / binary_path)
                if expected_hash and static["sha256"] != expected_hash:
                    raise ValueError("Binary/report SHA-256 identity mismatch")
            sample_hash = expected_hash or (static["sha256"] if static else "")
            digest = hashlib.sha256(report_bytes).hexdigest()
            identity = sample_hash or digest
            if identity in by_hash and by_hash[identity]["label"] != label:
                raise ValueError(f"Conflicting labels for duplicate SHA-256: {identity}")
            if sample_id in by_id and by_id[sample_id] != (identity, label):
                raise ValueError(f"sample_id collision: {sample_id}")
            by_id[sample_id] = (identity, label)
            if identity in by_hash:
                skipped.append({"row": row_number, "sample_id": sample_id, "reason": "duplicate_sample",
                                "duplicate_of": by_hash[identity]["sample_id"],
                                "sha256": sample_hash, "report_sha256": digest})
                continue
            if static:
                report["imports"] = list(dict.fromkeys(report["imports"] + imports_from_raw(static["raw"])))
                report["provenance"]["imports"].append("static_raw/imports")
            if payload_static_dir:
                for payload in report["payloads"]:
                    digest_key = _sha256(payload.get("sha256"), "payload.sha256")
                    if not digest_key or not payload.get("is_pe"):
                        continue
                    sidecar = Path(payload_static_dir) / (digest_key + ".json")
                    if sidecar.is_file():
                        payload_static = load_static(sidecar, digest_key)
                        payload["static_features"] = payload_static["features"]
                        payload["static_raw"] = payload_static["raw"]
                        payload["static_extractor"] = payload_static.get("extractor")
            from .statistics import behavioral_statistics

            record = _validate_record({
                "schema_version": SCHEMA_VERSION, "feature_version": 2,
                "sample_id": sample_id, "label": label,
                "sha256": sample_hash, "report_sha256": digest, "report_path": str(report_path),
                "observed_at": stamp.isoformat() if stamp else "",
                "family": family, "group": group,
                "metadata": copy.deepcopy(row), "report": report,
                "static_features": static["features"] if static else None,
                "static_raw": static["raw"] if static else None, "static_available": static is not None,
                "static_extractor": static.get("extractor") if static else None,
                "extraction": {"max_events": max_events},
                "statistics": behavioral_statistics(report).tolist(),
            })
            records.append(record)
            by_hash[identity] = record
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            raise ValueError(f"Metadata row {row_number} ({row.get('sample_id', '') if isinstance(row, dict) else ''}): {exc}") from exc
    from .statistics import STATS_NAMES

    output = Path(output_path)
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.resolve() == output.resolve():
        raise ValueError("output_path must differ from its .manifest.json sidecar")
    manifest = {"schema_version": SCHEMA_VERSION, "feature_version": 2,
                "static_dim": STATIC_DIM, "statistics_dim": STATS_DIM,
                "statistics_names": list(STATS_NAMES), "statistics_transform": "raw",
                "metadata_path": source_name, "max_events": max_events,
                "record_count": len(records), "skipped_count": len(skipped), "skipped": skipped,
                "label_counts": {str(label): sum(item["label"] == label for item in records) for label in (0, 1)}}
    serialized = "".join(json.dumps(record, allow_nan=False) + "\n" for record in records)
    manifest_json = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    manifest_path.write_text(manifest_json, encoding="utf-8")
    return records


def load_dataset(path):
    """Load raw versioned JSONL, rejecting invalid records and all duplicates.

    Raw report hashes are preserved, not reverified against external paths on
    load. Preparation is the boundary that hashes the report bytes.
    """
    records, ids, hashes = [], set(), set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = _validate_record(json.loads(line, parse_constant=_reject_constant))
                identity = record["sha256"] or record["report_sha256"]
                if record["sample_id"] in ids:
                    raise ValueError("Duplicate sample_id: " + record["sample_id"])
                if identity in hashes:
                    raise ValueError("Duplicate sample/report SHA-256: " + identity)
                ids.add(record["sample_id"])
                hashes.add(identity)
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid dataset {path}, line {line_number}: {exc}") from exc
    return records


def _statistics_mask():
    from .statistics import STATS_NAMES

    return [name.startswith(("available_", "missing_", "mask_")) or name.endswith(("_mask", "_truncated"))
            for name in STATS_NAMES]


class FeatureScaler:
    """Log1p raw statistics, standardize using training population moments.

    Availability/missingness/mask/truncation flags remain unchanged. Statistics
    whose names merely contain 'missing' (event counts) are still scaled.
    Metadata, labels, hashes, raw EMBER, and all other IR fields are untouched.
    """

    def __init__(self):
        self._state = None

    def fit(self, records):
        import numpy as np
        from .statistics import STATS_NAMES

        records = list(records)
        if not records:
            raise ValueError("FeatureScaler.fit requires nonempty training records")
        static = [_vector(record["static_features"], STATIC_DIM, "static_features")
                  for record in records if record.get("static_features") is not None]
        statistics = np.asarray([_vector(record.get("statistics"), STATS_DIM, "statistics", True)
                                 for record in records], dtype=np.float64)
        mask = np.asarray(_statistics_mask())
        if not np.isin(statistics[:, mask], (0, 1)).all():
            raise ValueError("Statistics availability/missing/truncation masks must be binary")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                logged = np.log1p(statistics)
                stats_mean, stats_std = logged.mean(axis=0), logged.std(axis=0)
                stats_mean[mask], stats_std[mask] = 0.0, 1.0
                static_array = np.asarray(static, dtype=np.float64)
                static_mean = static_array.mean(axis=0) if static else np.zeros(STATIC_DIM)
                static_std = static_array.std(axis=0) if static else np.ones(STATIC_DIM)
                stats_std[stats_std == 0], static_std[static_std == 0] = 1.0, 1.0
        except FloatingPointError as exc:
            raise ValueError("Nonfinite scaler training moments") from exc
        self._state = {"schema_version": SCHEMA_VERSION, "feature_version": 2,
                       "statistics_names": list(STATS_NAMES), "statistics_transform": "log1p",
                       "training_count": len(records), "static_count": len(static),
                       "static_mean": static_mean.tolist(), "static_std": static_std.tolist(),
                       "statistics_mean": stats_mean.tolist(), "statistics_std": stats_std.tolist()}
        return self

    def transform(self, record):
        import numpy as np

        state = self.state_dict()
        result = copy.deepcopy(record)
        values = np.asarray(_vector(record.get("statistics"), STATS_DIM, "statistics", True))
        mask = np.asarray(_statistics_mask())
        if not np.isin(values[mask], (0, 1)).all():
            raise ValueError("Statistics availability/missing/truncation masks must be binary")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                logged = np.log1p(values)
                logged[mask] = values[mask]
                result["statistics"] = ((logged - state["statistics_mean"]) / state["statistics_std"]).tolist()
                targets = [result] + result.get("report", {}).get("payloads", [])
                for target in targets:
                    if target.get("static_features") is not None:
                        features = np.asarray(_vector(target["static_features"], STATIC_DIM, "static_features"))
                        target["static_features"] = ((features - state["static_mean"]) / state["static_std"]).tolist()
        except FloatingPointError as exc:
            raise ValueError("Nonfinite scaled features") from exc
        return result

    def state_dict(self):
        """Return detached JSON-friendly primitives, safe for tensor-only loaders."""
        if self._state is None:
            raise ValueError("FeatureScaler must be fitted before use")
        return copy.deepcopy(self._state)

    @classmethod
    def from_state_dict(cls, state):
        from .statistics import STATS_NAMES

        if (not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION
                or type(state.get("feature_version")) is not int or state["feature_version"] != 2
                or state.get("statistics_names") != list(STATS_NAMES)
                or state.get("statistics_transform") != "log1p"):
            raise ValueError("Incompatible FeatureScaler state schema/version")
        state = copy.deepcopy(state)
        for name, dimension in (("static", STATIC_DIM), ("statistics", STATS_DIM)):
            for suffix in ("mean", "std"):
                key = name + "_" + suffix
                state[key] = _vector(state.get(key), dimension, key)
                if suffix == "std" and any(value <= 0 for value in state[key]):
                    raise ValueError(key + " must be positive")
        for key in ("training_count", "static_count"):
            if type(state.get(key)) is not int or state[key] < (1 if key == "training_count" else 0):
                raise ValueError("Invalid scaler " + key)
        if state["static_count"] > state["training_count"]:
            raise ValueError("static_count exceeds training_count")
        for index, flag in enumerate(_statistics_mask()):
            if flag and (state["statistics_mean"][index] != 0 or state["statistics_std"][index] != 1):
                raise ValueError("Scaler state must preserve unscaled statistics masks")
        json.dumps(state, allow_nan=False)
        scaler = cls()
        scaler._state = state
        return scaler
