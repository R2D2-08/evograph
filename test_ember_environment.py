"""Extraction bridge tests and optional real, read-only PE parsing checks."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multimodal.static import extract_ember


ROOT = Path(__file__).resolve().parent
EMBER_PYTHON = ROOT / ".venv-ember" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class TestEmberBridge(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.binary = root / "sample with spaces.exe"
        self.binary.write_bytes(b"inert mock fixture")
        self.python = root / "python with spaces"
        self.python.touch()
        self.digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.document = {"sha256": self.digest, "features": [0.] * 2381,
                         "raw": {"sha256": self.digest, "imports": {}},
                         "extractor": {"source_revision": "test"}}
        self.environment = patch.dict(os.environ, {"EVOGRAPH_EMBER_PYTHON": str(self.python)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def result(self, **changes):
        return SimpleNamespace(**({"returncode": 0, "stdout": json.dumps(self.document), "stderr": ""} | changes))

    def test_bridge_uses_isolated_python_no_shell_and_checks_hash(self):
        with patch("multimodal.static.subprocess.run", return_value=self.result()) as run:
            document = extract_ember(self.binary)
        self.assertEqual(document["extractor"], self.document["extractor"])
        self.assertEqual(document["sha256"], self.digest)
        args, kwargs = run.call_args
        self.assertEqual(args[0][:2], [str(self.python), "-I"])
        self.assertEqual(args[0][-1], str(self.binary))
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(kwargs["timeout"], 120)

    def test_worker_failures_and_invalid_json_are_not_features(self):
        for result, message in (
            (self.result(returncode=-11, stderr="parser failure"), "exit -11"),
            (self.result(stdout="not JSON"), "valid finite JSON"),
            (self.result(stdout="NaN"), "valid finite JSON"),
            (self.result(stdout=json.dumps({**self.document, "sha256": "a" * 64})), "identity mismatch"),
            (self.result(stdout=json.dumps({**self.document, "features": [0.] * 2351})), "2381"),
        ):
            with self.subTest(message=message), patch("multimodal.static.subprocess.run", return_value=result):
                with self.assertRaisesRegex(ValueError, message):
                    extract_ember(self.binary)

    def test_timeout_is_actionable(self):
        with patch("multimodal.static.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 120)):
            with self.assertRaisesRegex(ValueError, "timed out"):
                extract_ember(self.binary)

    def test_missing_interpreter_is_actionable(self):
        with patch.dict(os.environ, {"EVOGRAPH_EMBER_PYTHON": str(self.python) + "-missing"}):
            with self.assertRaisesRegex(ValueError, "setup_ember.py"):
                extract_ember(self.binary)

    def test_worker_diagnostics_stay_off_stdout(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("multimodal.static.subprocess.run", return_value=self.result(stderr="diagnostic\n")):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                extract_ember(self.binary)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "diagnostic\n")


@unittest.skipUnless(EMBER_PYTHON.is_file(), "run python3 scripts/setup_ember.py for real EMBER checks")
class TestRealEmberEnvironment(unittest.TestCase):
    def fixture(self, name):
        result = subprocess.run(
            [str(EMBER_PYTHON), "-I", "-c", "from importlib.metadata import distribution; "
             f"print(distribution('setuptools').locate_file('setuptools/{name}'))"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return Path(result.stdout.strip())

    def test_real_pe32_and_pe32plus_features_and_provenance(self):
        with patch.dict(os.environ, {"EVOGRAPH_EMBER_PYTHON": str(EMBER_PYTHON)}):
            for name, machine in (("cli-32.exe", "I386"), ("cli-64.exe", "AMD64")):
                path = self.fixture(name)
                with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                    document = extract_ember(path)
                    self.assertEqual(len(document["features"]), 2381)
                    self.assertEqual(document["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertEqual(document["raw"]["header"]["coff"]["machine"], machine)
                    self.assertGreater(document["raw"]["general"]["vsize"], 0)
                    self.assertEqual(document["extractor"]["source_revision"],
                                     "d97a0b523de02f3fe5ea6089d080abacab6ee931")
                    self.assertEqual(document["extractor"]["packages"]["numpy"], "1.23.5")
                    self.assertEqual(document["extractor"]["packages"]["scikit-learn"], "1.1.3")
                    self.assertEqual(document["extractor"]["lief_runtime"], "0.12.3-39115d10")
                    self.assertAlmostEqual(sum(document["features"][:256]), 1., places=6)

    def test_standalone_worker_stdout_is_one_json_document(self):
        result = subprocess.run(
            [str(EMBER_PYTHON), "-I", str(ROOT / "multimodal/ember_worker.py"), str(self.fixture("cli-64.exe"))],
            check=True, capture_output=True, text=True, timeout=120,
        )
        document = json.loads(result.stdout)
        self.assertEqual(len(document["features"]), 2381)
        self.assertNotIn("WARNING", result.stdout)
        self.assertIn("WARNING", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "preparation requires NumPy in the caller")
    def test_binary_report_pair_preserves_real_features_and_provenance(self):
        from multimodal.data import load_dataset, prepare_dataset

        binary = self.fixture("cli-64.exe")
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"target": {"file": {"sha256": digest}}, "behavior": {}}))
            output = Path(directory) / "dataset.jsonl"
            with patch.dict(os.environ, {"EVOGRAPH_EMBER_PYTHON": str(EMBER_PYTHON)}):
                with contextlib.redirect_stderr(io.StringIO()):
                    prepare_dataset([{"sample_id": "benign-launcher", "label": "benign",
                                      "report_path": str(report), "binary_path": str(binary)}], output)
            record = load_dataset(output)[0]
            self.assertTrue(record["static_available"])
            self.assertEqual(record["sha256"], digest)
            self.assertEqual(record["static_raw"]["sha256"], digest)
            self.assertEqual(len(record["static_features"]), 2381)
            self.assertEqual(record["static_extractor"]["packages"]["numpy"], "1.23.5")
            self.assertEqual(record["static_extractor"]["source_revision"],
                             "d97a0b523de02f3fe5ea6089d080abacab6ee931")


if __name__ == "__main__":
    unittest.main()
