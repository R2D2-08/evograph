"""Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_multimodal_data -v.

Only synthetic JSON and inert byte strings are used. Official EMBER is mocked;
no malware is executed/downloaded. NumPy/sklearn tests skip when unavailable.
"""

import copy
import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from multimodal.cape import parse_report
from multimodal.data import FeatureScaler, load_dataset, prepare_dataset
from multimodal.schema import SCHEMA_VERSION, STATIC_DIM, STATS_DIM
from multimodal.splits import make_splits
from multimodal.static import extract_ember, imports_from_raw, load_static

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_SKLEARN = HAS_NUMPY and importlib.util.find_spec("sklearn") is not None


def digest(index):
    return hashlib.sha256(str(index).encode()).hexdigest()


def record(index=0, label=0):
    return {"schema_version": SCHEMA_VERSION, "sample_id": f"sample-{index}",
            "label": label, "sha256": digest(index), "report_sha256": digest(f"report-{index}"),
            "report_path": f"/synthetic/report-{index}.json", "observed_at": "", "family": "", "group": "",
            "metadata": {}, "report": parse_report({}), "static_features": None, "statistics": [0.] * STATS_DIM,
            "extraction": {"max_events": 20000}}


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir="/tmp/opencode")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_json(self, name, document):
        path = self.root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def write_dataset(self, records):
        path = self.root / "dataset.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return path


class TestStatic(TemporaryFiles):
    def sidecar(self):
        return {"sha256": digest(0), "features": [0.] * STATIC_DIM,
                "raw": {"sha256": digest(0), "imports": {"KERNEL32.dll": ["WriteFile"]}}}

    def test_sidecar_identity_raw_and_optional_version(self):
        document = self.sidecar()
        document.update(schema_version=SCHEMA_VERSION, feature_version=2)
        result = load_static(self.write_json("static.json", document), digest(0).upper())
        self.assertEqual(result, document)
        self.assertEqual(len(result["features"]), 2381)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_static(self.root / "static.json", digest(1))

    def test_rejects_invalid_sha_version_shape_and_nonfinite(self):
        changes = [{"sha256": ""}, {"sha256": "a" * 63}, {"sha256": "z" * 64},
                   {"features": [0.] * 2351}, {"features": [0.] * 2380},
                   {"features": [[0.]] * STATIC_DIM}, {"features": ["0"] * STATIC_DIM},
                   {"features": [True] * STATIC_DIM}, {"features": [float("nan")] * STATIC_DIM},
                   {"features": [float("inf")] * STATIC_DIM}, {"feature_version": 1},
                   {"feature_version": 2.0}, {"schema_version": "unrelated-v1"},
                   {"raw": []}, {"raw": {"sha256": digest(1)}}, {"raw": {"bad": float("nan")}}]
        for change in changes:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    load_static(self.write_json("invalid.json", {**self.sidecar(), **change}))

    def test_official_extractor_v2_reads_bytes_and_retains_raw(self):
        content = b"inert test fixture, not a PE"
        path = self.root / "fixture.bin"
        path.write_bytes(content)
        raw = {"sha256": hashlib.sha256(content).hexdigest(), "imports": {"x.dll": ["X"]}}
        calls = []

        class Extractor:
            def __init__(self, feature_version):
                calls.append(feature_version)

            def raw_features(self, value):
                calls.append(value)
                return raw

            def process_raw_features(self, value):
                calls.append(value)
                return [0.] * STATIC_DIM

        with patch.dict("sys.modules", {"ember": types.SimpleNamespace(PEFeatureExtractor=Extractor)}):
            result = extract_ember(path)
        self.assertEqual(calls, [2, content, raw])
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertIs(result["raw"], raw)
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["feature_version"], 2)

    def test_extractor_output_is_validated(self):
        path = self.root / "fixture.bin"
        path.write_bytes(b"inert")
        for features, raw in (([0.] * 2351, {}), ([float("nan")] * STATIC_DIM, {}),
                              ([0.] * STATIC_DIM, {"sha256": digest(7)})):
            extractor = types.SimpleNamespace(raw_features=lambda _: raw,
                                              process_raw_features=lambda _: features)
            with patch.dict("sys.modules", {"ember": types.SimpleNamespace(PEFeatureExtractor=lambda **_: extractor)}):
                with self.assertRaises(ValueError):
                    extract_ember(path)

    def test_ember_optional_import_error_is_actionable(self):
        with patch.dict("sys.modules", {"ember": None}):
            load_static(self.write_json("static.json", self.sidecar()))
            with self.assertRaisesRegex(ImportError, "elastic/ember.*static_path"):
                extract_ember(self.root / "unused.bin")

    def test_imports_are_functions_not_dlls_or_executed_calls(self):
        raw = {"imports": {"KERNEL32.dll": ["KERNEL32.dll!WriteFile", "WriteFile", "CreateFileW", 123],
                           "ntdll.dll": ["ntdll.NtOpenFile"], "ignored.dll": "not a list"}}
        original = copy.deepcopy(raw)
        self.assertEqual(imports_from_raw(raw), ["writefile", "createfilew", "ntopenfile"])
        self.assertEqual(raw, original)
        self.assertEqual(imports_from_raw({"imports": []}), [])


@unittest.skipUnless(HAS_NUMPY, "dataset preparation requires NumPy for behavioral_statistics")
class TestPreparation(TemporaryFiles):
    def metadata(self, rows):
        path = self.root / "metadata.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, index=0, **updates):
        self.write_json(f"report-{index}.json", {"target": {"file": {"sha256": digest(index)}}})
        return {"sample_id": str(index), "report_path": f"report-{index}.json", "label": "benign", **updates}

    def prepare(self, rows):
        return prepare_dataset(self.metadata(rows), self.root / "prepared.jsonl")

    def test_provenance_missing_modalities_raw_statistics_and_manifest(self):
        row = self.row(observed_at="2026-01-02T03:00:00+03:00", acquisition="local", label="malware")
        report_path = self.write_json("report-0.json", {"label": 0, "family": "untrusted",
            "signatures": [{"name": "untrusted"}], "static": {"pe": {}},
            "behavior": {"processes": [{"pid": 1, "calls": [{"api": "ReadFile", "repeated": 2}]}]}})
        original = report_path.read_bytes()
        result = self.prepare([row])[0]
        from multimodal.statistics import STATS_NAMES

        self.assertIsNone(result["static_features"])
        self.assertIsNone(result["static_raw"])
        self.assertFalse(result["static_available"])
        self.assertEqual(result["metadata"], row)
        self.assertEqual(result["label"], 1)
        self.assertEqual(result["family"], "")
        self.assertEqual(result["group"], "")
        self.assertEqual(result["observed_at"], "2026-01-02T00:00:00+00:00")
        self.assertEqual(result["report_path"], str(report_path.resolve()))
        self.assertEqual(result["report_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(result["sha256"], "")
        self.assertEqual(result["statistics"][STATS_NAMES.index("api_occurrence_count")], 3)
        for key in ("label", "family", "signatures", "metadata"):
            self.assertNotIn(key, result["report"])
        self.assertEqual(report_path.read_bytes(), original)
        self.assertEqual(load_dataset(self.root / "prepared.jsonl"), [result])
        manifest = json.loads((self.root / "prepared.manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["statistics_names"], list(STATS_NAMES))
        self.assertEqual(manifest["statistics_transform"], "raw")

    def test_sidecar_paths_identity_and_imports(self):
        row = self.row(static_path="static.json", sha256=digest(0), family="explicit-app", group="explicit-group")
        raw = {"sha256": digest(0), "imports": {"KERNEL32.dll": ["WriteFile"]}}
        self.write_json("static.json", {"sha256": digest(0), "features": [2.] * STATIC_DIM, "raw": raw})
        result = self.prepare([row])[0]
        self.assertEqual(result["static_features"], [2.] * STATIC_DIM)
        self.assertEqual(result["static_raw"], raw)
        self.assertEqual(result["report"]["imports"], ["writefile"])
        self.assertEqual(result["report"]["processes"], [])
        self.assertEqual((result["family"], result["group"]), ("explicit-app", "explicit-group"))
        for changes in ({"sha256": digest(1)}, {"binary_path": "sample.bin"}):
            with self.assertRaisesRegex(ValueError, "identity mismatch|ambiguous"):
                self.prepare([{**row, **changes}])
        self.write_json("static.json", {"sha256": digest(1), "features": [2.] * STATIC_DIM, "raw": {}})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.prepare([row])

    def test_binary_source_identity_is_checked(self):
        row = self.row(binary_path="inert.bin")
        static = {"sha256": digest(0), "features": [1.] * STATIC_DIM, "raw": {}}
        with patch("multimodal.data.extract_ember", return_value=static) as extract:
            self.assertEqual(self.prepare([row])[0]["sha256"], digest(0))
            extract.assert_called_once_with(self.root / "inert.bin")
        with patch("multimodal.data.extract_ember", return_value={**static, "sha256": digest(1)}):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.prepare([row])

    def test_missing_target_hash_accepts_metadata_claim_without_rewriting_report(self):
        row = self.row(sha256=digest(0))
        self.write_json("report-0.json", {})
        result = self.prepare([row])[0]
        self.assertEqual(result["sha256"], digest(0))
        self.assertEqual(result["report"]["sha256"], "")

    def test_blank_and_duplicate_skips_are_manifested(self):
        row = self.row()
        duplicate = {**row, "sample_id": "alias"}
        blank = {"sample_id": "unlabeled", "report_path": "nonexistent.json", "label": " "}
        self.assertEqual(len(self.prepare([row, duplicate, blank])), 1)
        manifest = json.loads((self.root / "prepared.manifest.json").read_text())
        self.assertEqual([item["reason"] for item in manifest["skipped"]], ["duplicate_sample", "blank_label"])
        self.assertEqual(manifest["skipped_count"], 2)
        self.assertEqual(manifest["record_count"], 1)

    def test_report_hash_fallback_deduplicates_and_conflicts_raise(self):
        row = self.row()
        self.write_json("report-0.json", {})
        self.assertEqual(len(self.prepare([row, {**row, "sample_id": "alias"}])), 1)
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            self.prepare([row, {**row, "sample_id": "alias", "label": 1}])
        other = self.row(1, sample_id=row["sample_id"])
        with self.assertRaisesRegex(ValueError, "sample_id collision"):
            self.prepare([row, other])

    def test_invalid_labeled_samples_are_not_silently_skipped(self):
        for document in ([], None, 42):
            row = self.row()
            self.write_json("report-0.json", document)
            with self.assertRaisesRegex(ValueError, "row 1.*JSON object"):
                self.prepare([row])
        row = self.row()
        (self.root / "report-0.json").write_text("{bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "row 1"):
            self.prepare([row])
        for changes in ({"label": "unknown"}, {"sample_id": ""}, {"observed_at": "not a timestamp"},
                        {"report_path": "missing.json"}, {"static_path": "missing-static.json"},
                        {"sha256": "not-a-sha"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare([{**self.row(), **changes}])


class TestLoading(TemporaryFiles):
    def test_load_normalizes_timestamps_and_keeps_missing_static(self):
        item = record()
        item["observed_at"] = "2026-02-01T02:00:00+02:00"
        result = load_dataset(self.write_dataset([item]))[0]
        self.assertEqual(result["observed_at"], "2026-02-01T00:00:00+00:00")
        self.assertIsNone(result["static_features"])
        self.assertEqual(item["observed_at"], "2026-02-01T02:00:00+02:00")
        item["observed_at"] = "20260201T000000Z"
        self.assertEqual(load_dataset(self.write_dataset([item]))[0]["observed_at"], result["observed_at"])

    def test_every_record_schema_and_required_shapes_are_validated(self):
        changes = [{"schema_version": "old"}, {"schema_version": None}, {"label": True},
                   {"feature_version": 1}, {"feature_version": True},
                   {"label": "1"}, {"label": 2}, {"static_features": [0.] * 2351},
                   {"static_features": [[0.]] * STATIC_DIM}, {"statistics": [0.] * 127},
                   {"statistics": [-1.] * STATS_DIM}, {"statistics": [float("nan")] * STATS_DIM},
                   {"static_features": [float("inf")] * STATIC_DIM}, {"sha256": "invalid"},
                   {"report_sha256": ""}, {"report_path": "relative.json"},
                   {"observed_at": "2026-01-01"}, {"report": {}}, {"metadata": []},
                   {"static_available": True}]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaisesRegex(ValueError, "line 2"):
                load_dataset(self.write_dataset([record(1, 1), {**record(), **change}]))
        item = record()
        del item["statistics"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            load_dataset(self.write_dataset([item]))

    def test_loaded_identity_mismatches_and_payload_shape(self):
        for mutate in (lambda item: item["metadata"].update(sha256=digest(8)),
                       lambda item: item["metadata"].update(label=1),
                       lambda item: item.update(family="inferred-not-explicit"),
                       lambda item: item["report"].update(label=0),
                       lambda item: item["report"].update(sha256=digest(8)),
                       lambda item: item.update(static_features=[0.] * STATIC_DIM,
                                                static_raw={"sha256": digest(8)}),
                       lambda item: item["report"].update(payloads=[{"static_features": [0.] * 2380}])):
            item = record()
            mutate(item)
            with self.assertRaises(ValueError):
                load_dataset(self.write_dataset([item]))

    def test_load_cannot_bypass_duplicate_checks(self):
        for second in (record(), {**record(1), "sha256": digest(0)},
                       {**record(1), "sample_id": "sample-0"}, {**record(0, 1), "sample_id": "alias"}):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_dataset(self.write_dataset([record(), second]))
        first, second = record(), record(1)
        first["sha256"] = second["sha256"] = ""
        second["report_sha256"] = first["report_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_dataset(self.write_dataset([first, second]))


@unittest.skipUnless(HAS_NUMPY, "FeatureScaler requires NumPy")
class TestScaling(unittest.TestCase):
    def sample(self, static, count, available=0):
        from multimodal.statistics import STATS_NAMES

        item = record()
        item["static_features"] = [static] * STATIC_DIM if static is not None else None
        item["statistics"][0] = count
        item["statistics"][STATS_NAMES.index("available_calls")] = available
        item["statistics"][STATS_NAMES.index("missing_calls")] = 1 - available
        item["statistics"][STATS_NAMES.index("trace_truncated")] = available
        item["statistics"][STATS_NAMES.index("events_missing_thread_id_count")] = count
        return item

    def test_train_only_log1p_missing_static_masks_and_payload_stats(self):
        from multimodal.statistics import STATS_NAMES

        train = [self.sample(2, 0), self.sample(4, 3, 1), self.sample(None, 0)]
        before = copy.deepcopy(train)
        scaler = FeatureScaler().fit(train)
        state = scaler.state_dict()
        self.assertEqual(state["static_mean"], [3.] * STATIC_DIM)
        self.assertEqual(state["static_std"], [1.] * STATIC_DIM)
        self.assertEqual(state["static_count"], 2)
        self.assertEqual(state["training_count"], 3)
        self.assertAlmostEqual(state["statistics_mean"][0], math.log1p(3) / 3)
        heldout = self.sample(100, 1000, 1)
        heldout["report"]["payloads"] = [{"static_features": [5.] * STATIC_DIM}, {}]
        transformed = scaler.transform(heldout)
        self.assertEqual(transformed["static_features"], [97.] * STATIC_DIM)
        self.assertEqual(transformed["report"]["payloads"][0]["static_features"], [2.] * STATIC_DIM)
        self.assertNotIn("static_features", transformed["report"]["payloads"][1])
        for name in ("available_calls", "missing_calls", "trace_truncated"):
            index = STATS_NAMES.index(name)
            self.assertEqual(transformed["statistics"][index], heldout["statistics"][index])
        count_index = STATS_NAMES.index("events_missing_thread_id_count")
        self.assertAlmostEqual(transformed["statistics"][count_index], transformed["statistics"][0])
        self.assertIsNone(scaler.transform(train[2])["static_features"])
        self.assertEqual(scaler.state_dict(), state)
        self.assertEqual(train, before)
        self.assertEqual(heldout["static_features"], [100.] * STATIC_DIM)
        self.assertEqual(transformed["metadata"], heldout["metadata"])
        self.assertEqual(transformed["sha256"], heldout["sha256"])

    def test_state_is_json_primitives_roundtrips_and_is_detached(self):
        scaler = FeatureScaler().fit([self.sample(2, 0), self.sample(4, 3)])
        state = json.loads(json.dumps(scaler.state_dict(), allow_nan=False))
        restored = FeatureScaler.from_state_dict(state)
        sample = self.sample(None, 7)
        self.assertEqual(restored.transform(sample), scaler.transform(sample))
        state["static_mean"][0] = 999
        self.assertEqual(restored.state_dict()["static_mean"][0], 3)
        for change in ({"schema_version": "old"}, {"static_std": [0.] * STATIC_DIM},
                       {"static_mean": [float("nan")] * STATIC_DIM}, {"statistics_std": [1.] * 127},
                       {"statistics_transform": "raw"}, {"static_count": 999}):
            with self.assertRaises(ValueError):
                FeatureScaler.from_state_dict({**scaler.state_dict(), **change})

    def test_all_missing_static_identity_and_zero_variance(self):
        scaler = FeatureScaler().fit([self.sample(None, 3)])
        self.assertEqual(scaler.state_dict()["static_count"], 0)
        self.assertEqual(scaler.transform(self.sample(7, 3))["static_features"], [7.] * STATIC_DIM)
        self.assertEqual(scaler.transform(self.sample(None, 3))["statistics"][0], 0)
        self.assertIsNone(scaler.transform(self.sample(None, 3))["static_features"])

    def test_invalid_values_and_unfitted_use_fail(self):
        with self.assertRaises(ValueError):
            FeatureScaler().state_dict()
        with self.assertRaises(ValueError):
            FeatureScaler().fit([])
        scaler = FeatureScaler().fit([self.sample(None, 1)])
        for key, value in (("statistics", [-1.] * STATS_DIM),
                           ("statistics", [float("nan")] * STATS_DIM),
                           ("static_features", [float("inf")] * STATIC_DIM),
                           ("static_features", [0.] * 2351)):
            item = {**self.sample(None, 1), key: value}
            with self.assertRaises(ValueError):
                FeatureScaler().fit([item])
            with self.assertRaises(ValueError):
                scaler.transform(item)


class TestSplits(unittest.TestCase):
    def samples(self, count=30):
        return [{**record(index, index % 2), "observed_at": f"2026-01-{index + 1:02d}T00:00:00Z"}
                for index in range(count)]

    def assert_partition(self, records, splits):
        self.assertEqual(set(splits), {"train", "validation", "test"})
        indices = [index for partition in splits.values() for index in partition]
        self.assertEqual(sorted(indices), list(range(len(records))))
        for partition in splits.values():
            self.assertEqual({records[index]["label"] for index in partition}, {0, 1})

    def test_chronological_original_indices_utc_not_string_sort(self):
        records = self.samples(10)
        records[0]["observed_at"] = "2026-01-01T23:00:00-02:00"
        records[1]["observed_at"] = "2026-01-02T02:00:00+03:00"
        records.reverse()
        before = copy.deepcopy(records)
        splits = make_splits(records, "chronological")
        self.assert_partition(records, splits)
        self.assertEqual([len(splits[key]) for key in ("train", "validation", "test")], [6, 2, 2])
        self.assertEqual(splits["train"][:2], [8, 9])
        self.assertEqual(records, before)

    def test_chronological_rejects_missing_dates_ties_and_single_class_windows(self):
        for stamp in ("", "invalid", "2026-01-01"):
            records = self.samples(10)
            records[0]["observed_at"] = stamp
            with self.assertRaises(ValueError):
                make_splits(records, "chronological")
        for left, right in ((5, 6), (7, 8)):
            records = self.samples(10)
            records[left]["observed_at"] = records[right]["observed_at"]
            with self.assertRaisesRegex(ValueError, "boundary"):
                make_splits(records, "chronological")
        records = self.samples(10)
        records[5]["observed_at"] = "2026-01-07T01:00:00+01:00"
        with self.assertRaisesRegex(ValueError, "boundary"):
            make_splits(records, "chronological")
        records = self.samples(10)
        for item in records:
            item["label"] = int(int(item["sample_id"].split("-")[1]) >= 5)
        with self.assertRaisesRegex(ValueError, "both labels"):
            make_splits(records, "chronological")

    def test_duplicate_id_hash_and_missing_labels_rejected_in_every_mode(self):
        for mode in ("stratified", "chronological", "family"):
            for key, value in (("sha256", digest(0)), ("sample_id", "sample-0"), ("label", None), ("label", True)):
                records = self.samples()
                records[1][key] = value
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    make_splits(records, mode)
        records = self.samples()
        for item in records[:2]:
            item["sha256"] = ""
            item["report_sha256"] = digest("duplicate-report")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_splits(records, "chronological")

    def test_small_feasible_chronological_partition(self):
        records = self.samples(6)
        splits = make_splits(records, "chronological")
        self.assert_partition(records, splits)
        self.assertEqual([len(partition) for partition in splits.values()], [2, 2, 2])

    def test_family_requires_explicit_groups_even_for_benign_samples(self):
        records = self.samples(6)
        for item in records:
            item["group"] = "explicit-" + item["sample_id"]
        records[0]["group"] = ""
        records[0]["report"]["family"] = "must-not-be-inferred"
        with self.assertRaisesRegex(ValueError, "including benign"):
            make_splits(records, "family")

    @unittest.skipUnless(HAS_SKLEARN, "stratified/family splits require scikit-learn")
    def test_stratified_deterministic_disjoint_and_no_rigid_minimum(self):
        for count in (6, 10, 30):
            records = self.samples(count)
            splits = make_splits(records, seed=12)
            self.assert_partition(records, splits)
            self.assertEqual(splits, make_splits(records, seed=12))
        self.assertEqual([len(partition) for partition in splits.values()], [18, 6, 6])
        for count in (10, 12, 20):
            records = self.samples(count)
            for index, item in enumerate(records):
                item["label"] = int(index >= 3)
            for seed in (0, 42, 99):
                self.assert_partition(records, make_splits(records, seed=seed))

    @unittest.skipUnless(HAS_SKLEARN, "family splits require scikit-learn")
    def test_family_group_disjoint_explicit_benign_and_deterministic(self):
        records = self.samples()
        for index, item in enumerate(records):
            item["family"] = f"explicit-application-{index // 2}"
        splits = make_splits(records, "family", seed=7)
        self.assert_partition(records, splits)
        self.assertEqual(splits, make_splits(records, "family", seed=7))
        groups = [{records[index]["family"] for index in partition} for partition in splits.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        records[0]["family"] = ""
        with self.assertRaisesRegex(ValueError, "including benign"):
            make_splits(records, "family")
        records[0]["group"] = "explicit-benign-group"
        self.assert_partition(records, make_splits(records, "family"))

    @unittest.skipUnless(HAS_SKLEARN, "family splits require scikit-learn")
    def test_small_family_splits_allow_required_rounding_for_both_labels(self):
        records = self.samples(6)
        for item in records:
            item["group"] = "explicit-" + item["sample_id"]
        self.assert_partition(records, make_splits(records, "family"))
        for index, item in enumerate(records):
            item["group"] = f"mixed-{index // 2}"
        self.assert_partition(records, make_splits(records, "family"))

    @unittest.skipUnless(HAS_SKLEARN, "family splits require scikit-learn")
    def test_family_infeasible_labels_raise_instead_of_leaking_groups(self):
        records = self.samples()
        for item in records:
            item["group"] = "only-benign" if item["label"] == 0 else f"malware-{item['sample_id']}"
        with self.assertRaisesRegex(ValueError, "label-feasible"):
            make_splits(records, "family")


if __name__ == "__main__":
    unittest.main()
