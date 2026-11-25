"""Run from evograph: python3 -m unittest test_multimodal_models -v.

Only small in-memory reports are used; no PE files, downloads, or sandbox runs.
Neural/graph tests are explicitly skipped when torch or PyG is unavailable.
"""

import copy
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from multimodal.cape import COVERAGE_KEYS, parse_report
from multimodal.schema import BLOCK_SLICES, FORWARD_RELATIONS, NODE_TYPES, RELATIONS

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "torch_geometric"))
if HAS_DEPS:
    import torch
    from torch_geometric.data import HeteroData

    from multimodal.encoders import (CharCNN, DynamicEncoder, NetworkEncoder,
                                     TypedArgumentEncoder, api_frequencies, normalize_network_identifier,
                                     normalize_user_path, stable_bucket)
    from multimodal.graph import BINARY_TELEMETRY_NAMES, PROCESS_STAT_NAMES, RESOURCE_STAT_NAMES, build_graph, process_statistics
    from multimodal.models import EvoGraphClassifier, RelationalLayer


def rich_report():
    report = parse_report({
        "target": {"file": {"pe": {"imports": [{"name": "KERNEL32.dll!WriteFile"}]}}},
        "behavior": {"processes": [
            {"process_id": 7, "process_name": "sample.exe", "command_line": '"sample.exe" --run', "calls": [
                {"api": "WriteFile", "thread_id": 1, "timestamp": 10, "category": "file", "status": True,
                 "arguments": [{"name": "FileName", "value": "C:\\Temp\\child.exe"},
                               {"name": "NumberOfBytesToWrite", "value": 4096}]},
                {"api": "connect", "thread_id": 2, "timestamp": 2, "category": "network",
                 "arguments": [{"name": "HostName", "value": "update.example.test"}]},
                {"api": "RegSetValueExW", "thread_id": 1, "timestamp": 9, "category": "registry", "status": False,
                 "arguments": [{"name": "KeyName", "value": "HKCU\\Software\\Example\\Run"},
                               {"name": "Access", "value": "0x20", "pretty_value": "KEY_WRITE"}]},
                {"api": "CreateMutexW", "thread_id": 2, "timestamp": 3,
                 "arguments": [{"name": "MutexName", "value": "Local\\Example"}]},
                {"api": "LoadLibraryW", "thread_id": 1, "timestamp": 11,
                 "arguments": [{"name": "LibraryName", "value": "example.dll"}]},
            ]},
            {"process_id": 8, "parent_id": 7, "process_name": "child.exe", "command_line": "child.exe /q", "calls": [
                {"api": "ReadFile", "thread_id": 1, "timestamp": 1,
                 "arguments": [{"name": "FileName", "value": "C:\\Temp\\child.exe"}]}]},
        ], "summary": {"files": ["C:\\summary-only.txt"], "resolved_apis": ["VirtualAllocEx"]}},
        "dropped": [{"name": "child.exe", "process_id": 7, "type": "PE32 executable", "size": 512},
                    {"name": "unknown.bin"}],
    })
    report["payloads"][0]["static_features"] = [0.25] * 2381
    return report


def sample(report=None, static=True):
    return {"report": rich_report() if report is None else report,
            "static_features": [0.125] * 2381 if static else None,
            "statistics": [index / 128 for index in range(128)]}


@unittest.skipUnless(HAS_DEPS, "requires torch and torch_geometric")
class TestGraphConstruction(unittest.TestCase):
    def test_schema_empty_graph_and_reverse_relations(self):
        for report in (parse_report({}), rich_report()):
            graph = build_graph(report, window_size=1)
            self.assertIsInstance(graph, HeteroData)
            self.assertEqual(tuple(graph.node_types), NODE_TYPES)
            self.assertEqual(set(graph.edge_types), set(RELATIONS))
            self.assertEqual(graph["binary"].num_nodes, 1)
            for src, relation, dst in FORWARD_RELATIONS:
                edges = graph[src, relation, dst].edge_index
                self.assertEqual(edges.dtype, torch.long)
                self.assertEqual(tuple(edges.shape[:1]), (2,))
                self.assertTrue(torch.equal(edges.flip(0), graph[dst, "rev_" + relation, src].edge_index))
                if edges.numel():
                    self.assertLess(int(edges[0].max()), graph[src].num_nodes)
                    self.assertLess(int(edges[1].max()), graph[dst].num_nodes)
        self.assertEqual(build_graph({})["window"].num_nodes, 0)
        self.assertEqual(tuple(build_graph({})["process"].statistics.shape), (0, 32))
        self.assertEqual(tuple(build_graph({})["resource"].statistics.shape), (0, 12))

    def test_sequences_keep_stream_source_order_and_no_cross_edges(self):
        graph = build_graph(rich_report(), window_size=1)
        windows = graph["window"].raw
        self.assertEqual([window["events"][0]["api"] for window in windows],
                         ["writefile", "regsetvalueexw", "loadlibraryw", "connect", "createmutexw", "readfile"])
        self.assertEqual([window["events"][0]["timestamp"] for window in windows[:3]], [10, 9, 11])
        for left, right in graph["window", "next_window", "window"].edge_index.t().tolist():
            self.assertEqual((windows[left]["process_index"], windows[left]["thread_id"]),
                             (windows[right]["process_index"], windows[right]["thread_id"]))
            self.assertEqual(windows[left]["ordinal"] + 1, windows[right]["ordinal"])
        unknown = parse_report({"behavior": {"processes": [{"pid": 1, "calls": [{"api": "A"}, {"api": "B"}]},
                                                           {"pid": 2, "calls": [{"api": "C"}]}]}})
        graph = build_graph(unknown, window_size=1)
        self.assertEqual(graph["window", "next_window", "window"].edge_index.t().tolist(), [[0, 1]])

    def test_global_budget_and_determinism(self):
        report = rich_report()
        before = copy.deepcopy(report)
        graph = build_graph(report, window_size=1, max_windows=2)
        again = build_graph(report, window_size=1, max_windows=2)
        self.assertEqual(graph["window"].num_nodes, 2)
        self.assertEqual(graph.graph_metadata["total_windows"], 6)
        self.assertEqual(graph.graph_metadata["dropped_windows"], 4)
        self.assertEqual(graph.graph_metadata["dropped_events"], 4)
        self.assertTrue(graph.graph_metadata["windows_truncated"])
        self.assertEqual(graph["window"].raw, again["window"].raw)
        self.assertEqual(report, before)
        for relation in RELATIONS:
            self.assertTrue(torch.equal(graph[relation].edge_index, again[relation].edge_index))
        graph = build_graph(report, max_windows=0)
        self.assertEqual(graph["window"].num_nodes, 0)
        self.assertTrue(graph.graph_metadata["windows_truncated"])
        self.assertGreater(graph["process", "accesses", "resource"].edge_index.size(1), 0)
        for kind in ("process", "resource"):
            torch.testing.assert_close(graph[kind].statistics, again[kind].statistics)

    def test_import_runtime_alignment_without_resolved_calls(self):
        report = rich_report()
        graph = build_graph(report)
        names = [api["name"] for api in graph["api"].raw]
        self.assertEqual(names.count("writefile"), 1)
        index = names.index("writefile")
        self.assertIn([0, index], graph["binary", "imports", "api"].edge_index.t().tolist())
        self.assertIn([0, index], graph["process", "calls", "api"].edge_index.t().tolist())
        resolved = names.index("virtualallocex")
        self.assertNotIn(resolved, graph["process", "calls", "api"].edge_index[1].tolist())
        self.assertNotIn(resolved, graph["window", "contains_api", "api"].edge_index[1].tolist())
        report = parse_report({"behavior": {"apistats": {"7": {"KERNEL32.dll!WriteFile": 20}},
                                            "summary": {"resolved_apis": ["ReadFile"]}}})
        graph = build_graph(report)
        self.assertEqual(graph["window"].num_nodes, 0)
        self.assertEqual(graph["process", "calls", "api"].edge_index.size(1), 1)
        self.assertEqual(graph["api"].raw[graph["process", "calls", "api"].edge_index[1, 0]]["name"], "writefile")

    def test_resource_edges_require_events_and_payloads_require_pids(self):
        graph = build_graph(rich_report(), window_size=1)
        resources = graph["resource"].raw
        summary = next(index for index, item in enumerate(resources) if item["value"] == "C:\\summary-only.txt")
        for src, relation, dst in FORWARD_RELATIONS:
            if dst == "resource":
                self.assertNotIn(summary, graph[src, relation, dst].edge_index[1].tolist())
        writes = [(graph["window"].raw[window]["events"][0]["api"], resources[resource]["kind"], resources[resource]["value"])
                  for window, resource in graph["window", "writes", "resource"].edge_index.t().tolist()]
        self.assertCountEqual(writes, [("writefile", "file", "C:\\Temp\\child.exe"),
                                      ("regsetvalueexw", "registry", "HKCU\\Software\\Example\\Run"),
                                      ("createmutexw", "mutex", "Local\\Example")])
        self.assertEqual(graph["process", "loads", "resource"].edge_index.size(1), 1)
        self.assertEqual(graph["binary", "launches", "process"].edge_index.t().tolist(), [[0, 0]])
        self.assertEqual(graph["process", "spawns", "process"].edge_index.t().tolist(), [[0, 1]])
        self.assertEqual(graph["process", "drops", "payload"].edge_index.t().tolist(), [[0, 0]])
        self.assertEqual(graph["binary", "produces", "payload"].edge_index.size(1), 2)
        self.assertEqual(graph["payload"].static_available.tolist(), [True, False])
        # Resource inventory remains after parser truncation, but no invented event ownership.
        report = parse_report({"behavior": {"processes": [{"pid": 7, "calls": [
            {"api": "WriteFile", "arguments": [{"name": "FileName", "value": "x.txt"}]}]}]}}, max_events=0)
        graph = build_graph(report)
        self.assertEqual(graph["resource"].num_nodes, 1)
        self.assertEqual(graph["process", "accesses", "resource"].edge_index.size(1), 0)
        self.assertTrue(graph.graph_metadata["trace_truncated"])

    def test_bad_limits(self):
        for kwargs in ({"window_size": 0}, {"window_size": True}, {"max_windows": -1}, {"max_windows": 1.5}):
            with self.assertRaises(ValueError):
                build_graph({}, **kwargs)

    def test_binary_telemetry_is_whitelisted_and_matches_raw_metadata(self):
        report = rich_report()
        report["trace_truncated"] = True
        report["label"], report["verdict"], report["family"] = 1, "malicious", "example"
        report["coverage"]["label"] = True
        graph = build_graph(report, window_size=1, max_windows=1)
        expected = {"available_" + key: bool(report["coverage"][key]) for key in COVERAGE_KEYS}
        expected.update(trace_truncated=True, windows_truncated=True)
        self.assertEqual(graph["binary"].raw[0]["telemetry"], expected)
        self.assertEqual(set(expected), set(BINARY_TELEMETRY_NAMES))
        self.assertEqual(tuple(graph["binary"].telemetry.shape), (1, len(BINARY_TELEMETRY_NAMES)))
        self.assertEqual(graph["binary"].telemetry.dtype, torch.float32)
        torch.testing.assert_close(graph["binary"].telemetry,
                                   torch.tensor([[float(expected[name]) for name in BINARY_TELEMETRY_NAMES]]))
        self.assertTrue(graph.graph_metadata["trace_truncated"])
        self.assertTrue(graph.graph_metadata["windows_truncated"])

    def test_process_statistics_are_local_and_clock_safe(self):
        self.assertEqual(len(PROCESS_STAT_NAMES), 32)
        report = rich_report()
        stats = process_statistics(report["processes"])
        first = dict(zip(PROCESS_STAT_NAMES, stats[0].tolist()))
        self.assertAlmostEqual(first["api_occurrences"], math.log1p(5), places=6)
        self.assertAlmostEqual(first["identified_threads"], math.log1p(2), places=6)
        self.assertAlmostEqual(first["observed_children"], math.log1p(1), places=6)
        self.assertAlmostEqual(first["observed_stream_span_seconds"], math.log1p(2), places=6)
        self.assertAlmostEqual(first["file_resources"], math.log1p(1), places=6)
        self.assertAlmostEqual(first["success_fraction"], .2, places=6)
        self.assertAlmostEqual(first["failure_fraction"], .2, places=6)
        self.assertAlmostEqual(first["unknown_status_fraction"], .6, places=6)
        self.assertEqual(first["observed_parent"], 0)
        self.assertEqual(float(stats[1, PROCESS_STAT_NAMES.index("observed_parent")]), 1)
        shifted = copy.deepcopy(report["processes"])
        for event in shifted[0]["events"]:
            if event["thread_id"] == "2":
                event["timestamp"] += 10000
        torch.testing.assert_close(stats, process_statistics(shifted))
        shifted[0]["events"][-1]["timestamp_domain"] = "datetime"
        self.assertAlmostEqual(float(process_statistics(shifted)[0, 7]), math.log1p(1), places=6)
        unrelated = parse_report({"behavior": {"processes": [{"pid": 999, "calls": [{"api": "A"}]}]}})["processes"]
        torch.testing.assert_close(stats, process_statistics(report["processes"] + unrelated)[:2])
        renumbered = copy.deepcopy(report["processes"])
        renumbered[0]["pid"], renumbered[1]["pid"], renumbered[1]["ppid"] = "70", "80", "70"
        torch.testing.assert_close(stats, process_statistics(renumbered))

    def test_process_statistics_preserve_counts_truncation_and_missingness(self):
        report = parse_report({"behavior": {"processes": [{"pid": 7, "calls": [
            {"api": "A", "repeated": 9, "status": True}, {"api": "B", "status": False}]}]}}, max_events=1)
        stats = dict(zip(PROCESS_STAT_NAMES, build_graph(report)["process"].statistics[0].tolist()))
        self.assertAlmostEqual(stats["api_occurrences"], math.log1p(11), places=6)
        self.assertAlmostEqual(stats["trace_rows"], math.log1p(2), places=6)
        self.assertAlmostEqual(stats["retained_rows"], math.log1p(1), places=6)
        self.assertAlmostEqual(stats["retained_occurrences"], math.log1p(10), places=6)
        self.assertEqual(stats["trace_retained_fraction"], .5)
        self.assertEqual(stats["trace_truncated"], 1)
        self.assertEqual(stats["success_fraction"], 1)
        self.assertEqual(stats["timestamps_available"], 0)
        counts_only = parse_report({"behavior": {"apistats": {"7": {"A": 10**400}}}})
        stats = build_graph(counts_only)["process"].statistics[0]
        self.assertTrue(torch.isfinite(stats).all())
        self.assertGreater(float(stats[0]), 900)
        self.assertEqual(float(stats[PROCESS_STAT_NAMES.index("counts_from_apistats")]), 1)
        self.assertEqual(float(stats[PROCESS_STAT_NAMES.index("retained_rows")]), 0)
        missing = process_statistics([{}])[0]
        empty = process_statistics(parse_report({"behavior": {"processes": [{"calls": []}]}})["processes"])[0]
        self.assertEqual(float(missing[PROCESS_STAT_NAMES.index("calls_available")]), 0)
        self.assertEqual(float(empty[PROCESS_STAT_NAMES.index("calls_available")]), 1)

    def test_resource_statistics_count_evidence_without_fabricated_ownership(self):
        self.assertEqual(len(RESOURCE_STAT_NAMES), 12)
        report = rich_report()
        # Duplicate references/operations in one event still describe one row.
        event = report["processes"][0]["events"][0]
        event["repeat"] = 4
        event["resources"].append(dict(event["resources"][0]))
        graph = build_graph(report, max_windows=0)
        by_value = {resource["value"]: dict(zip(RESOURCE_STAT_NAMES, stats.tolist()))
                    for resource, stats in zip(graph["resource"].raw, graph["resource"].statistics)}
        file = by_value["C:\\Temp\\child.exe"]
        self.assertEqual(file["reads"], 1)
        self.assertEqual(file["writes"], 1)
        self.assertAlmostEqual(file["event_rows"], math.log1p(2), places=6)
        self.assertAlmostEqual(file["event_occurrences"], math.log1p(5), places=6)
        self.assertAlmostEqual(file["attributed_processes"], math.log1p(2), places=6)
        self.assertEqual(file["event_evidence"], 1)
        summary = by_value["C:\\summary-only.txt"]
        self.assertEqual(summary["accesses"], 1)
        self.assertEqual(summary["summary_evidence"], 1)
        for name in ("event_rows", "event_occurrences", "attributed_processes", "event_evidence"):
            self.assertEqual(summary[name], 0, name)


@unittest.skipUnless(HAS_DEPS, "requires torch and torch_geometric")
class TestTrainableModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_all_modes_shapes_finite_and_batch_matches_single(self):
        samples = [sample(), sample(parse_report({}), static=False), sample(parse_report({
            "behavior": {"apistats": {"42": {"ReadFile": 20}}}}))]
        for mode in EvoGraphClassifier.MODES:
            with self.subTest(mode=mode):
                model = EvoGraphClassifier(mode=mode, hidden=32, dropout=0, window_size=2).eval()
                with torch.no_grad():
                    batched = model(samples)
                    singles = torch.cat([model([item]) for item in samples])
                self.assertEqual(tuple(batched.shape), (3,))
                self.assertTrue(torch.isfinite(batched).all())
                torch.testing.assert_close(batched, singles, atol=2e-5, rtol=1e-4)
                self.assertEqual(tuple(model([]).shape), (0,))
                if mode in ("graph", "hybrid"):
                    self.assertFalse(hasattr(model, "dynamic"))
        model = EvoGraphClassifier().eval()
        self.assertEqual(model.head[0].in_features, 576)
        self.assertEqual(model.head[0].out_features, 256)
        self.assertEqual(tuple(model([sample(parse_report({}), static=False)]).shape), (1,))

    def test_dynamic_dimensions_statistics_and_empty_blocks(self):
        encoder = DynamicEncoder().eval()
        stats = torch.arange(128, dtype=torch.float32) / 128
        for report in (rich_report(), parse_report({})):
            vector = encoder(report, stats)
            self.assertEqual(tuple(vector.shape), (2048,))
            self.assertTrue(torch.isfinite(vector).all())
            torch.testing.assert_close(vector[BLOCK_SLICES["statistics"]], stats)
        vector = encoder(parse_report({}), stats)
        for name in ("api_identity", "api_sequences", "files", "processes", "registry", "network", "other_resources", "arguments"):
            self.assertEqual(int(torch.count_nonzero(vector[BLOCK_SLICES[name]])), 0, name)
        available = parse_report({"dropped": []})
        self.assertFalse(torch.equal(encoder(available, stats)[BLOCK_SLICES["payloads"]], vector[BLOCK_SLICES["payloads"]]))

    def test_gradients_reach_all_learned_encoders(self):
        for mode in ("baseline", "dynamic", "hybrid", "graph"):
            with self.subTest(mode=mode):
                model = EvoGraphClassifier(mode=mode, hidden=32, dropout=0, window_size=2)
                logits = model([sample(), sample(parse_report({}), static=False)])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.]))
                loss.backward()
                enc = model.encoders if mode in ("graph", "hybrid") else model.dynamic.encoders
                parameters = {"file CNN": enc.files.convs[0].weight,
                              "process CNN": enc.process_names.embedding.weight,
                              "command GRU": enc.commands.gru.weight_ih_l0,
                              "registry LSTM": enc.registry.lstm.weight_ih_l0,
                              "network CNN": enc.network.convs[0].weight,
                              "other resource type": enc.other_types.weight,
                              "argument projection": enc.arguments.projection[0].weight,
                              "API embedding": enc.sequences.api_embedding.weight,
                              "sequence GRU": enc.sequences.gru.weight_ih_l0,
                              "payload projection": enc.payloads.projection.weight,
                              "payload missingness": enc.payloads.availability.weight}
                if mode in ("graph", "hybrid"):
                    parameters["static encoder"] = model.graph.static.mlp[0].weight
                    parameters["static missingness"] = model.graph.static.availability.weight
                    parameters["binary telemetry"] = model.graph.binary_telemetry.weight
                    parameters["graph command projection"] = model.graph.command_projection.weight
                    parameters["resource statistics"] = model.graph.resource_statistics.weight
                for name, parameter in parameters.items():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
                if mode in ("graph", "hybrid"):
                    self.assertGreater(float(model.graph.initial_projection["process"].weight.grad[:, 96:].abs().sum()), 0)
                for layer in getattr(getattr(model, "graph", None), "layers", []):
                    self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                        for parameter in layer.relations.parameters()))

    def test_ablation_inputs_and_no_dynamic_forward_for_graph(self):
        with patch.object(DynamicEncoder, "forward", side_effect=AssertionError("baseline computed")):
            for mode in ("graph", "hybrid"):
                self.assertTrue(torch.isfinite(EvoGraphClassifier(mode=mode, hidden=16)([sample()])).all())
        graph_sample = {"report": parse_report({}), "static_features": None}
        self.assertEqual(tuple(EvoGraphClassifier(mode="graph", hidden=16)([graph_sample]).shape), (1,))
        self.assertEqual(tuple(EvoGraphClassifier(mode="static", hidden=16)([{"static_features": None}]).shape), (1,))
        dynamic_sample = {"report": parse_report({}), "statistics": [0.] * 128, "static_features": "not used"}
        self.assertEqual(tuple(EvoGraphClassifier(mode="dynamic", hidden=16)([dynamic_sample]).shape), (1,))

    def test_missing_static_and_payload_are_not_zero_vectors(self):
        model = EvoGraphClassifier(hidden=16, dropout=0).eval()
        values = torch.zeros((2, 2381))
        encoded = model.graph.static(values, torch.tensor([False, True]))
        self.assertFalse(torch.equal(encoded[0], encoded[1]))
        payloads = model.encoders.payloads([{}, {"static_features": [0.] * 2381}])
        self.assertFalse(torch.equal(payloads[0], payloads[1]))
        missing = sample(parse_report({}), static=False)
        empty = sample(parse_report({"dropped": []}), static=False)
        self.assertFalse(torch.equal(model([missing]), model([empty])))

    def test_graph_only_distinguishes_missing_from_empty_without_label_leakage(self):
        model = EvoGraphClassifier(mode="graph", hidden=16, dropout=0).eval()
        reports = [parse_report({}), parse_report({"behavior": {"processes": [], "summary": {
            "files": [], "keys": [], "mutexes": []}}, "network": {}})]
        graphs = [build_graph(report) for report in reports]
        for kind in NODE_TYPES[1:]:
            self.assertEqual(graphs[0][kind].num_nodes, 0)
            self.assertEqual(graphs[1][kind].num_nodes, 0)
        self.assertEqual(graphs[0]["binary"].raw[0]["payloads_available"], graphs[1]["binary"].raw[0]["payloads_available"])
        nodes = [model.graph.encode_nodes(graph, torch.zeros(2381), torch.tensor(False), model.encoders)["binary"]
                 for graph in graphs]
        self.assertFalse(torch.equal(*nodes))
        self.assertEqual(model.graph.static.mlp[0].in_features, 2381)
        self.assertEqual(model.graph.initial_projection["binary"].in_features, 256)
        samples = [{"report": report, "static_features": None} for report in reports]  # No D1 input.
        logits = model(samples)
        self.assertFalse(torch.equal(logits[0], logits[1]))
        torch.testing.assert_close(logits, torch.cat([model([item]) for item in samples]), atol=2e-5, rtol=1e-4)
        labeled = copy.deepcopy(samples)
        for item in labeled:
            item.update(label=1, family="example", verdict="malicious")
            item["report"].update(label=1, family="example", verdict="malicious", signatures=["example"])
            item["report"]["coverage"]["label"] = True
        torch.testing.assert_close(logits, model(labeled))

    def test_graph_only_encodes_trace_and_window_truncation_independently(self):
        report = parse_report({"behavior": {"processes": [{"pid": 7, "calls": [
            {"api": "A"}, {"api": "B"}, {"api": "C"}]}]}}, max_events=2)
        model = EvoGraphClassifier(mode="graph", hidden=16, dropout=0, window_size=1, max_windows=1).eval()
        graph = build_graph(report, window_size=1, max_windows=1)
        self.assertTrue(graph.graph_metadata["trace_truncated"])
        self.assertTrue(graph.graph_metadata["windows_truncated"])
        item = {"report": report, "static_features": None}  # No D1 input.
        original = model([item])
        for flag in ("trace_truncated", "windows_truncated"):
            # Hold topology, static features and all other telemetry fixed to
            # prove the binary flag itself reaches the graph-only logit.
            ablated = copy.deepcopy(graph)
            column = BINARY_TELEMETRY_NAMES.index(flag)
            ablated["binary"].telemetry[0, column] = 0
            ablated["binary"].raw[0]["telemetry"][flag] = False
            with patch("multimodal.models.build_graph", return_value=ablated):
                self.assertFalse(torch.equal(original, model([item])), flag)
        original.sum().backward()
        gradient = model.graph.binary_telemetry.weight.grad
        for flag in ("trace_truncated", "windows_truncated"):
            self.assertGreater(float(gradient[:, BINARY_TELEMETRY_NAMES.index(flag)].abs().sum()), 0, flag)

    def test_char_padding_and_sequence_padding_do_not_change_embeddings(self):
        cnn = CharCNN().eval()
        torch.testing.assert_close(cnn(["a"])[0], cnn(["a", "a much longer string"])[0])
        encoder = DynamicEncoder().encoders
        a, b = {"api": "readfile", "thread_id": "1"}, {"api": "writefile", "thread_id": "1"}
        alone = encoder.sequences([[a]], encoder.arguments)[0]
        padded = encoder.sequences([[a], [a, b, a]], encoder.arguments)[0]
        torch.testing.assert_close(alone, padded, atol=1e-6, rtol=1e-5)
        forward = encoder.sequences([[a, b]], encoder.arguments)
        reverse = encoder.sequences([[b, a]], encoder.arguments)
        self.assertFalse(torch.allclose(forward, reverse))

    def test_graph_process_layout_does_not_change_baseline_d5(self):
        model = EvoGraphClassifier(hidden=16, dropout=0).eval()
        report = rich_report()
        graph = build_graph(report)
        raw = model.encoders.process_features(report["processes"])
        self.assertEqual(tuple(raw.shape), (2, 128))  # name64 + command64 for D5.
        inputs = []
        hook = model.graph.initial_projection["process"].register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        try:
            model.graph.encode_nodes(graph, torch.zeros(2381), torch.tensor(False), model.encoders)
        finally:
            hook.remove()
        self.assertEqual(tuple(inputs[0].shape), (2, 128))
        torch.testing.assert_close(inputs[0][:, :64], raw[:, :64])
        torch.testing.assert_close(inputs[0][:, 64:96], model.graph.command_projection(raw[:, 64:]))
        torch.testing.assert_close(inputs[0][:, 96:], graph["process"].statistics)
        dynamic = DynamicEncoder(model.encoders)(report, torch.zeros(128))
        torch.testing.assert_close(dynamic[BLOCK_SLICES["processes"]],
                                   torch.cat((raw[:, :64].mean(0), raw[:, :64].max(0).values,
                                              raw[:, 64:].mean(0), raw[:, 64:].max(0).values)))

    def test_file_user_normalization_preserves_raw_identity_and_filename(self):
        left = "C:\\Users\\Alice\\AppData\\Local\\Temp\\a82f91.exe"
        right = "C:\\Users\\LongerBob\\AppData\\Local\\Temp\\a82f91.exe"
        expected = "C:\\Users\\<USER>\\AppData\\Local\\Temp\\a82f91.exe"
        self.assertEqual(normalize_user_path(left), expected)
        self.assertEqual(normalize_user_path(right), expected)
        self.assertEqual(normalize_user_path("D:/Documents and Settings/Jane Doe/Temp/a.exe"),
                         "D:/Documents and Settings/<USER>/Temp/a.exe")
        for path in ("C:\\Users\\Public\\a.exe", "C:\\Users\\Default\\a.exe", "C:\\Program Files\\a.exe",
                     "C:\\Users\\desktop.ini", "C:\\Users\\Alice", "C:\\Users\\..\\Windows\\a.exe",
                     "Users\\Alice\\a.exe", "\\\\server\\Users\\Alice\\a.exe"):
            self.assertEqual(normalize_user_path(path), path)
        self.assertNotEqual(normalize_user_path(left), normalize_user_path(left.replace("a82f91", "b77d12")))
        self.assertNotEqual(normalize_user_path(left), normalize_user_path(left.replace(".exe", ".dll")))
        enc = DynamicEncoder().eval()
        torch.testing.assert_close(enc.encoders.files([left]), enc.encoders.files([right]))
        reports = [parse_report({"behavior": {"summary": {"files": [path]}}}) for path in (left, right)]
        vectors = [enc(report, torch.zeros(128))[BLOCK_SLICES["files"]] for report in reports]
        torch.testing.assert_close(*vectors)
        model = EvoGraphClassifier(hidden=16, dropout=0).eval()
        nodes = [model.graph.encode_nodes(build_graph(report), torch.zeros(2381), torch.tensor(False),
                                         model.encoders)["resource"] for report in reports]
        torch.testing.assert_close(*nodes)
        report = parse_report({"behavior": {"summary": {"files": [left, right]}}})
        graph = build_graph(report)
        self.assertEqual([resource["value"] for resource in graph["resource"].raw], [left, right])
        self.assertEqual([resource["value"] for resource in report["resources"]], [left, right])

    def test_network_masks_literal_ip_families_endpoints_and_url_hosts(self):
        encoder = NetworkEncoder().eval()
        cases = [
            ("192.0.2.1", "198.51.100.20", "<IPV4>"),
            ("2001:db8::1", "::ffff:192.0.2.5", "<IPV6>"),
            ("192.0.2.1:80", "198.51.100.20:8080", "<IPV4>:<PORT>"),
            ("[2001:db8::1]:443", "[fe80::2%eth0]:8443", "[<IPV6>]:<PORT>"),
            ("http://192.0.2.1:80/a%2Fb?q=1#frag", "http://198.51.100.20:8080/a%2Fb?q=1#frag",
             "http://<IPV4>:<PORT>/a%2Fb?q=1#frag"),
            ("https://[2001:db8::1]/a?q=1", "https://[fe80::2%25eth0]/a?q=1", "https://[<IPV6>]/a?q=1"),
            ("//192.0.2.1/path?", "//198.51.100.20/path?", "//<IPV4>/path?"),
            ("192.0.2.1:80/path", "198.51.100.20:8080/path", "<IPV4>:<PORT>/path"),
            ("http://user:pass@192.0.2.1/a", "http://user:pass@198.51.100.20/a", "http://user:pass@<IPV4>/a"),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left):
                self.assertEqual(normalize_network_identifier(left), expected)
                self.assertEqual(normalize_network_identifier(right), expected)
                torch.testing.assert_close(encoder([left]), encoder([right]))
        self.assertFalse(torch.allclose(encoder(["192.0.2.1"]), encoder(["2001:db8::1"])))

    def test_network_preserves_hostnames_and_url_context(self):
        identifiers = ["update.example.test", "update.example.test:443", "192.0.2.1.example.test",
                       "https://User:Pass@Example.test:8443/a%2Fb?q=192.0.2.1#frag",
                       "/download/192.0.2.1/file.exe", "http://[invalid/path", ""]
        for identifier in identifiers:
            self.assertEqual(normalize_network_identifier(identifier), identifier)
        encoder = NetworkEncoder().eval()
        torch.testing.assert_close(encoder(identifiers), CharCNN.forward(encoder, identifiers))
        self.assertNotEqual(normalize_network_identifier("https://192.0.2.1/a"),
                            normalize_network_identifier("https://192.0.2.1/b"))

    def test_network_masking_is_shared_without_mutating_report_or_graph_identity(self):
        addresses = ["192.0.2.1", "198.51.100.20"]
        reports = [parse_report({"network": {"hosts": [address]}}) for address in addresses]
        before = copy.deepcopy(reports)
        model = EvoGraphClassifier(mode="graph", hidden=16, dropout=0).eval()
        dynamic = DynamicEncoder(model.encoders).eval()
        vectors = [dynamic(report, torch.zeros(128))[BLOCK_SLICES["network"]] for report in reports]
        torch.testing.assert_close(*vectors)
        nodes = [model.graph.encode_nodes(build_graph(report), torch.zeros(2381), torch.tensor(False),
                                         model.encoders)["resource"] for report in reports]
        torch.testing.assert_close(*nodes)
        logits = model([{"report": report, "static_features": None} for report in reports])
        torch.testing.assert_close(logits[0], logits[1])
        self.assertEqual(reports, before)
        graph = build_graph(parse_report({"network": {"hosts": addresses}}))
        self.assertEqual(graph["resource"].num_nodes, 2)
        self.assertEqual([resource["value"] for resource in graph["resource"].raw], addresses)

    def test_timing_category_status_repeat_affect_sequence(self):
        encoder = DynamicEncoder().encoders
        events = [{"api": "a", "timestamp": 10., "timestamp_domain": "numeric"},
                  {"api": "b", "timestamp": 9., "timestamp_domain": "numeric"}]
        reference = encoder.sequences([events], encoder.arguments)
        for update in ({"timestamp": 12.}, {"status": False}, {"category": "registry"}, {"repeat": 12}):
            changed = copy.deepcopy(events)
            changed[1].update(update)
            self.assertFalse(torch.allclose(reference, encoder.sequences([changed], encoder.arguments)))
        shifted = copy.deepcopy(events)
        for event in shifted:
            event["timestamp"] += 10000
        torch.testing.assert_close(reference, encoder.sequences([shifted], encoder.arguments))

    def test_argument_semantics_ignore_arbitrary_handles_text_and_pids(self):
        def semantics(name, value, **extra):
            return TypedArgumentEncoder.semantics({"arguments": [{"name": name, "value": value, **extra}]})
        for name, left, right in (("ProcessHandle", "0x123", "0x999"), ("Buffer", "hello", "world"),
                                  ("ProcessId", 7, 123456), ("Filename", "a.txt", "b.txt")):
            self.assertEqual(semantics(name, left), semantics(name, right))
        self.assertNotEqual(semantics("Protection", "0x20"), semantics("Protection", "0x40"))
        self.assertNotEqual(semantics("Size", 1), semantics("Size", 4096))
        self.assertNotEqual(semantics("Access", "GENERIC_READ"), semantics("Access", "GENERIC_WRITE"))

    def test_relation_normalization_is_per_relation_not_global_degree(self):
        layer = RelationalLayer(1, 0)
        with torch.no_grad():
            for conv in layer.relations.convs.values():
                conv.lin_l.weight.fill_(1)
        x = {kind: torch.empty((0, 1)) for kind in NODE_TYPES}
        x.update(binary=torch.tensor([[2.]]), process=torch.tensor([[3.], [9.]]), api=torch.tensor([[0.]]))
        edges = {("binary", "imports", "api"): torch.tensor([[0], [0]]),
                 ("process", "calls", "api"): torch.tensor([[0, 1], [0, 0]])}
        result = layer.relations(x, edges)["api"]
        torch.testing.assert_close(result, torch.tensor([[8.]]))  # 2 + mean(3, 9).

    def test_hashes_are_stable_signed_canonical_and_normalized(self):
        name = "writefile"
        digest = hashlib.blake2b(name.encode(), digest_size=16).digest()
        self.assertEqual(stable_bucket(name), (int.from_bytes(digest[:8], "little") % 512, 1 if digest[8] & 1 else -1))
        output = []
        for seed in ("1", "200"):
            output.append(subprocess.check_output([sys.executable, "-c",
                          "from multimodal.encoders import stable_bucket; print(stable_bucket('writefile'))"],
                          env={**os.environ, "PYTHONHASHSEED": seed}, text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(*output)
        report = parse_report({"behavior": {"apistats": {"7": {"KERNEL32.dll!WriteFile": 3, "ReadFile": 1}}}})
        vector = api_frequencies(report)
        expected = torch.zeros(512)
        for api, frequency in (("writefile", .75), ("readfile", .25)):
            bucket, sign = stable_bucket(api)
            expected[bucket] += sign * frequency
        torch.testing.assert_close(vector, expected)
        scaled = copy.deepcopy(report)
        scaled["processes"][0]["api_counts"] = {name: count * 10 for name, count in report["processes"][0]["api_counts"].items()}
        torch.testing.assert_close(vector, api_frequencies(scaled))
        self.assertEqual(int(torch.count_nonzero(api_frequencies(parse_report({"behavior": {
            "summary": {"resolved_apis": ["A"]}}})))), 0)

    def test_invalid_input_dimensions_and_constructor(self):
        for kwargs in ({"mode": "unknown"}, {"hidden": 0}, {"dropout": 1}, {"window_size": 0}, {"max_windows": -1}):
            with self.assertRaises(ValueError):
                EvoGraphClassifier(**kwargs)
        for vector in ([0.] * 127, [float("nan")] * 128):
            with self.assertRaises(ValueError):
                DynamicEncoder()(parse_report({}), vector)
        with self.assertRaises(ValueError):
            EvoGraphClassifier(mode="static")([{"static_features": [0.] * 2380}])


if __name__ == "__main__":
    unittest.main()
