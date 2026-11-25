"""Small CPU integration tests; no binary execution, network or real dataset use."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_multimodal import main, read_report
from multimodal.static import load_static

HAS_DEPS = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "torch", "torch_geometric", "sklearn"))
if HAS_DEPS:
    import numpy as np
    import torch
    from multimodal.data import load_dataset, prepare_dataset
    from multimodal.training import export_features, load_checkpoint, probabilities, run_experiment, select_threshold


class TestInputBoundary(unittest.TestCase):
    def test_missing_corrupt_and_nonobject_reports_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(OSError):
                read_report(path, 32)
            for text in ("broken", "[]", "NaN", '{"info": NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_report(path, 32)
            path.write_text("{}")
            self.assertFalse(read_report(path, 32)["coverage"]["calls"])

    def test_static_expected_hash_validated_before_open(self):
        with patch("pathlib.Path.open") as opened:
            with self.assertRaisesRegex(ValueError, "expected_sha256"):
                load_static("unused.json", "../../outside")
            opened.assert_not_called()


@unittest.skipUnless(HAS_DEPS, "requires numeric and graph dependencies")
class TestMultimodalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "dataset.jsonl"
        metadata = []
        for index in range(20):
            digest = hashlib.sha256(str(index).encode()).hexdigest()
            path = cls.root / f"report_{index}.json"
            path.write_text(json.dumps({
                "target": {"file": {"sha256": digest}},
                "behavior": {"processes": [{"process_id": 1, "process_name": "sample.exe", "calls": [
                    {"api": "WriteFile" if index % 2 else "ReadFile", "thread_id": 1,
                     "timestamp": index, "arguments": [{"name": "FileName", "value": "C:\\Temp\\test.txt"}]},
                    {"api": "CloseHandle", "thread_id": 1, "timestamp": index + 1},
                ]}], "summary": {"files": []}}, "network": {},
            }))
            metadata.append({"sample_id": f"sample_{index}", "report_path": str(path),
                             "label": index % 2, "sha256": digest})
        cls.records = prepare_dataset(metadata, cls.dataset, max_events=2)
        cls.args = SimpleNamespace(
            dataset=str(cls.dataset), out=str(cls.root / "results"), models=["baseline", "hybrid"],
            split_mode="stratified", seed=42, epochs=1, patience=1, batch_size=4,
            hidden=16, dropout=0., lr=.001, window_size=1, max_windows=2, threads=1,
            allow_missing_static=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            cls.results = run_experiment(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_shared_splits_and_train_only_scaler(self):
        self.assertEqual(self.results["split_sizes"], {"train": 12, "validation": 4, "test": 4})
        for mode in ("baseline", "hybrid"):
            model, scaler, checkpoint = load_checkpoint(self.root / "results" / (mode + ".pt"))
            self.assertEqual(scaler.state_dict()["training_count"], 12)
            self.assertEqual(checkpoint["max_events"], 2)
            self.assertEqual(len(probabilities(model, [scaler.transform(self.records[0])])), 1)
            self.assertTrue(np.isfinite(self.results["models"][mode]["history"][0]["train_loss"]))

    def test_prediction_matches_prepared_inference(self):
        for mode in ("baseline", "hybrid"):
            checkpoint = self.root / "results" / (mode + ".pt")
            model, scaler, _ = load_checkpoint(checkpoint)
            expected = probabilities(model, [scaler.transform(self.records[0])])[0]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["predict", "--report", self.records[0]["report_path"], "--checkpoint", str(checkpoint)])
            actual = json.loads(output.getvalue())
            self.assertAlmostEqual(expected, actual["malware_probability"], places=6)
            self.assertFalse(actual["static_available"])
            if mode == "hybrid":
                self.assertFalse(actual["graph_limits"]["windows_truncated"])

    def test_export_uses_fitted_encoders_and_exact_dimensions(self):
        checkpoint = self.root / "results" / "baseline.pt"
        output = self.root / "features.npz"
        self.assertEqual(export_features(checkpoint, self.dataset, output), 20)
        model, scaler, _ = load_checkpoint(checkpoint)
        sample = scaler.transform(self.records[0])
        with np.load(output, allow_pickle=False) as features, torch.inference_mode():
            self.assertEqual(features["features"].shape, (20, 4429))
            self.assertEqual(features["static_features"].shape, (20, 2381))
            self.assertEqual(features["dynamic_features"].shape, (20, 2048))
            self.assertFalse(features["static_available"].any())
            np.testing.assert_allclose(features["dynamic_features"][0],
                                       model.dynamic(sample["report"], sample["statistics"]).numpy())
        with self.assertRaisesRegex(ValueError, "baseline or dynamic"):
            export_features(self.root / "results" / "hybrid.pt", self.dataset, output)

    def test_export_rejects_incompatible_extraction(self):
        incompatible = self.root / "incompatible.jsonl"
        records = copy.deepcopy(self.records)
        records[0]["extraction"]["max_events"] = 1
        incompatible.write_text("".join(json.dumps(record) + "\n" for record in records))
        with self.assertRaisesRegex(ValueError, "max_events"):
            export_features(self.root / "results" / "baseline.pt", incompatible, self.root / "bad.npz")

    def test_missing_extraction_configuration_is_not_guessed(self):
        path = self.root / "unversioned_extraction.jsonl"
        records = copy.deepcopy(self.records)
        del records[0]["extraction"]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        with self.assertRaisesRegex(ValueError, "extraction"):
            load_dataset(path)

    def test_missing_static_requires_explicit_opt_in(self):
        args = copy.copy(self.args)
        args.allow_missing_static = False
        with self.assertRaisesRegex(ValueError, "missing"):
            run_experiment(args)
        args.models = ["static"]
        with self.assertRaisesRegex(ValueError, "static-only"):
            run_experiment(args)

    def test_payload_path_cannot_escape_sidecar_directory(self):
        report = self.root / "bad_payload.json"
        report.write_text(json.dumps({"dropped": [{"type": "PE32 executable", "sha256": "../../outside"}]}))
        with patch("multimodal.static.load_static") as opened, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["predict", "--report", str(report), "--checkpoint", str(self.root / "results" / "baseline.pt"),
                      "--payload-static-dir", str(self.root)])
            opened.assert_not_called()

    def test_payload_sidecars_are_attached_during_preparation(self):
        digest = "a" * 64
        report = self.root / "payload_report.json"
        report.write_text(json.dumps({"dropped": [{"type": "PE32 executable", "sha256": digest}]}))
        (self.root / (digest + ".json")).write_text(json.dumps({
            "sha256": digest, "features": [1.] * 2381, "raw": {"imports": {}}}))
        records = prepare_dataset([{"sample_id": "payload", "report_path": str(report), "label": 0}],
                                  self.root / "payload_dataset.jsonl", payload_static_dir=self.root)
        self.assertEqual(records[0]["report"]["payloads"][0]["static_features"], [1.] * 2381)
        self.assertEqual(len(load_dataset(self.root / "payload_dataset.jsonl")), 1)

    def test_threshold_prefers_neutral_tie(self):
        self.assertEqual(select_threshold([0, 1], np.array([.1, .9])), .5)


if __name__ == "__main__":
    unittest.main()
