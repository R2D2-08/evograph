"""Focused, tiny CAPE fixtures. Run: python3 -m unittest test_multimodal_extraction."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from multimodal.cape import COVERAGE_KEYS, MAX_REPEAT, canonical_api, parse_report
from multimodal.schema import STATS_DIM

try:
    import numpy as np
except ModuleNotFoundError:
    np = None
else:
    from multimodal.statistics import STATS_NAMES, behavioral_statistics


def calls_report(calls, **process_fields):
    return {"behavior": {"processes": [{"process_id": 7, "calls": calls, **process_fields}]}}


class TestCapeNormalization(unittest.TestCase):
    def test_repeat_counts_and_global_cap(self):
        raw = {"behavior": {"processes": [
            {"process_id": 1, "calls": [{"api": "KERNEL32.dll!WriteFile", "repeated": 2},
                                         {"api": "ReadFile", "repeated": 1}]},
            {"process_id": 2, "calls": [{"api": "CloseHandle"}]},
        ]}}
        report = parse_report(raw, max_events=1)
        first, second = report["processes"]
        self.assertEqual(first["events"][0]["repeat"], 3)
        self.assertEqual(first["events"][0]["api"], "writefile")
        self.assertEqual(first["api_counts"], {"writefile": 3, "readfile": 2})
        self.assertEqual(second["api_counts"], {"closehandle": 1})
        self.assertEqual(second["events"], [])
        self.assertEqual(first["trace_event_count"], 2)
        self.assertTrue(report["trace_truncated"])
        self.assertTrue(second["trace_truncated"])
        self.assertFalse(parse_report(raw, max_events=3)["trace_truncated"])

    def test_repeat_validation_and_sensible_cap(self):
        raw = calls_report([{"api": "A", "repeated": item}
                            for item in (-5, "2", None, float("inf"), 10**400)])
        report = parse_report(raw)
        self.assertEqual([event["repeat"] for event in report["processes"][0]["events"]],
                         [1, 3, 1, 1, MAX_REPEAT])
        json.dumps(report, allow_nan=False)

    def test_source_order_and_thread_ids_are_preserved(self):
        raw = calls_report([
            {"api": "B", "thread_id": 2, "timestamp": 10},
            {"api": "A", "thread_id": 1, "timestamp": 1},
            {"api": "D", "thread_id": 2, "timestamp": 11},
            {"api": "C", "thread_id": 1, "timestamp": 2},
        ])
        events = parse_report(raw)["processes"][0]["events"]
        self.assertEqual([event["api"] for event in events], ["b", "a", "d", "c"])
        self.assertEqual([event["timestamp"] for event in events if event["thread_id"] == "2"], [10, 11])
        self.assertEqual([event["source"] for event in events],
                         [f"/behavior/processes/0/calls/{index}" for index in range(4)])

    def test_comma_milliseconds_numeric_and_invalid_timestamps(self):
        raw = calls_report([
            {"api": "A", "timestamp": "2026-01-01 01:02:03,125"},
            {"api": "B", "timestamp": "2026-01-01 01:02:03,375"},
            {"api": "C", "timestamp": "2,500"},
            {"api": "D", "timestamp": "nonsense"},
        ], first_seen="2026-01-01 01:02:03,000")
        process = parse_report(raw)["processes"][0]
        events = process["events"]
        self.assertAlmostEqual(events[1]["timestamp"] - events[0]["timestamp"], 0.25)
        self.assertAlmostEqual(events[0]["timestamp"] - process["first_seen"], 0.125)
        self.assertEqual(events[2]["timestamp"], 2.5)
        self.assertEqual(events[2]["timestamp_domain"], "numeric")
        self.assertIsNone(events[3]["timestamp"])

    def test_apistats_fallback_is_per_process_not_a_sequence(self):
        raw = {"behavior": {"processes": [
            {"process_id": 1, "calls": [{"api": "ReadFile"}]},
            {"process_id": 2, "calls": []},
        ], "apistats": {"1": {"ReadFile": 99}, "2": {"ws2_32.dll.connect": 3},
                        "3": {"send": 2, "bad": -1}},
            "summary": {"resolved_apis": ["kernel32.dll.WriteFile", "ntdll!NtOpenFile"]}}}
        report = parse_report(raw)
        first, second, third = report["processes"]
        self.assertEqual(first["api_counts"], {"readfile": 1})
        self.assertEqual(first["api_count_source"], "calls")
        self.assertEqual(first["apistats"], {"readfile": 99})
        self.assertEqual(second["api_counts"], {"connect": 3})
        self.assertEqual(third["api_counts"], {"send": 2})
        self.assertEqual(second["events"], [])
        self.assertEqual(third["events"], [])
        self.assertEqual(third["pid"], "3")
        self.assertEqual(second["api_count_source"], "apistats")
        self.assertEqual(report["resolved_apis"], ["writefile", "ntopenfile"])

    def test_integer_apistats_do_not_round_through_float(self):
        count = 2**53 + 1
        report = parse_report({"behavior": {"apistats": {"7": {"A": count}}}})
        self.assertEqual(report["processes"][0]["api_counts"]["a"], count)

    def test_imports_and_resolved_apis_never_fabricate_processes(self):
        raw = {"target": {"file": {"sha256": "ABC", "pe": {"imports": {
            "kernel32": {"dll": "KERNEL32.dll", "imports": [{"name": "CreateFileW"}, {"ordinal": 1}]}
        }}}}, "static": {"pe_imports": [{"dll": "ntdll.dll", "imports": [{"name": "NtOpenFile"}]}]},
            "behavior": {"summary": {"resolved_apis": ["kernel32.dll.WriteFile"]}}}
        report = parse_report(raw)
        self.assertEqual(report["sha256"], "abc")
        self.assertEqual(report["imports"], ["createfilew", "ntopenfile"])
        self.assertEqual(report["resolved_apis"], ["writefile"])
        self.assertEqual(report["processes"], [])
        self.assertFalse(report["coverage"]["calls"])
        self.assertEqual(canonical_api("KERNEL32.DLL!CreateFileW"), "createfilew")

    def test_tree_metadata_merges_without_creating_calls(self):
        raw = calls_report([{"api": "A"}])
        raw["behavior"]["processtree"] = [{
            "pid": 7, "parent_id": 3, "name": "app.exe",
            "environ": {"CommandLine": '"app.exe" --flag'},
            "children": [{"pid": 8, "name": "child.exe", "command_line": "child.exe /q"}],
        }]
        report = parse_report(raw)
        self.assertEqual(len(report["processes"]), 2)
        parent, child = report["processes"]
        self.assertEqual(parent["command_line"], '"app.exe" --flag')
        self.assertEqual(parent["ppid"], "3")
        self.assertEqual(child["command_line"], "child.exe /q")
        self.assertEqual(child["ppid"], "7")
        self.assertEqual(child["events"], [])
        self.assertEqual(child["api_counts"], {})

    def test_empty_and_absent_sections_remain_distinct(self):
        absent = parse_report({})
        empty = parse_report({"behavior": {"processes": [], "apistats": {}, "summary": {
            "files": [], "keys": [], "mutexes": []}}, "network": {}, "CAPE": {"payloads": []}})
        self.assertEqual(set(absent["coverage"]), set(COVERAGE_KEYS))
        self.assertTrue(all(value is False for value in absent["coverage"].values()))
        for key in ("behavior", "processes", "apistats", "summary", "files", "registry",
                    "other_resources", "network", "payloads"):
            self.assertTrue(empty["coverage"][key], key)
        self.assertEqual(empty["processes"], [])
        self.assertEqual(empty["resources"], [])
        with_arguments = parse_report(calls_report([{"api": "A", "arguments": []}]))
        self.assertTrue(with_arguments["coverage"]["arguments"])
        self.assertTrue(with_arguments["coverage"]["calls"])
        self.assertFalse(parse_report(calls_report([{"api": "A"}]))["coverage"]["arguments"])

    def test_argument_resources_keep_raw_values_and_do_not_resolve_handles(self):
        raw = calls_report([
            {"api": "WriteFile", "arguments": [
                {"name": "FileName", "value": "short.txt", "pretty_value": "C:\\work\\long.txt"},
                {"name": "FileHandle", "value": "0x123"},
                {"name": "Buffer", "value": {"data": [1, 2, 3], "text": "C:\\not-a-resource.txt"}},
            ]},
            {"api": "RegSetValueExW", "arguments": [{"name": "KeyName", "value": "HKEY_CURRENT_USER\\Software\\X"}]},
            {"api": "connect", "arguments": [{"name": "sockaddr", "value": {"ip": "192.0.2.1", "port": 80}}]},
            {"api": "CreateMutexW", "arguments": [{"name": "MutexName", "value": "Local\\X"}]},
            {"api": "CreateServiceW", "arguments": [{"name": "ServiceName", "value": "Example"}]},
            {"api": "LoadLibraryW", "arguments": [{"name": "FileName", "value": "kernel32.dll"}]},
        ])
        report = parse_report(raw)
        events = report["processes"][0]["events"]
        self.assertEqual(events[0]["arguments"][0]["value"], "short.txt")
        self.assertEqual(events[0]["arguments"][2]["value"], raw["behavior"]["processes"][0]["calls"][0]["arguments"][2]["value"])
        self.assertEqual(events[0]["resources"], [{"kind": "file", "value": "C:\\work\\long.txt", "operation": "writes"}])
        self.assertEqual([event["resources"][0]["kind"] for event in events],
                         ["file", "registry", "network", "mutex", "service", "module"])
        self.assertEqual(events[2]["resources"][0]["operation"], "connects_to")
        self.assertEqual(report["resources"][0]["sources"], ["/behavior/processes/0/calls/0"])
        handles = parse_report(calls_report([{"api": "ReadFile", "arguments": [{"name": "FileHandle", "value": "0x123"}]}]))
        self.assertEqual(handles["resources"], [])

    def test_create_file_is_not_automatically_a_write(self):
        report = parse_report(calls_report([{"api": "CreateFileW", "arguments": [{"name": "FileName", "value": "x.txt"}]}]))
        self.assertEqual(report["resources"][0]["operations"], ["accesses"])

    def test_cape_fullname_and_prefixed_argument_names(self):
        report = parse_report(calls_report([
            {"api": "RegQueryValueExW", "arguments": [{"name": "FullName", "value": "HKCU\\X\\Value"}]},
            {"api": "DeleteFileW", "arguments": [{"name": "lpFileName", "value": "x.txt"}]},
            {"api": "ReadFile", "arguments": [{"name": "FileHandle", "value": "0x123", "pretty_value": "C:\\explicit.txt"}]},
        ]))
        self.assertEqual([(item["kind"], item["operations"]) for item in report["resources"]],
                         [("registry", ["reads"]), ("file", ["deletes"]), ("file", ["reads"])])

    def test_resource_source_cap_keeps_summary_provenance(self):
        raw = calls_report([{"api": "ReadFile", "arguments": [{"name": "FileName", "value": "x.txt"}]} for _ in range(40)])
        raw["behavior"]["summary"] = {"read_files": ["x.txt"]}
        report = parse_report(raw, max_events=0)
        self.assertTrue(report["resources"][0]["sources_truncated"])
        self.assertIn("/behavior/summary/read_files", report["resources"][0]["sources"])

    def test_summaries_and_network_have_provenance_not_fake_attribution(self):
        report = parse_report({"behavior": {"summary": {"files": ["x"], "write_files": ["x"],
                                                       "read_keys": ["HKCU\\X"], "created_services": ["s"]}},
                               "network": {"tcp": [{"dst": "192.0.2.1"}], "dns": [{"request": "example.invalid"}]}})
        self.assertEqual(report["processes"], [])
        self.assertEqual(report["resources"][0]["operations"], ["accesses", "writes"])
        self.assertEqual(report["resources"][0]["sources"], ["/behavior/summary/files", "/behavior/summary/write_files"])
        self.assertEqual(report["network_counts"]["tcp"], 1)
        self.assertEqual(report["network_counts"]["dns"], 1)

    def test_payload_names_paths_pids_and_pe_detection(self):
        report = parse_report({"dropped": [{"name": ["a.exe", "b.exe"], "path": ["/a", "/b"],
                                              "sha256": "ABC", "process_id": [7, "8"], "size": "42", "type": "PE32+ executable"}],
                               "CAPE": {"payloads": [{"path": "C:\\x.dll", "pid": 9, "pe": {}, "size": 5},
                                                        {"name": "not-pe.exe", "size": -1}]}})
        first, second, third = report["payloads"]
        self.assertEqual(first["name"], "a.exe")
        self.assertEqual(first["names"], ["a.exe", "b.exe"])
        self.assertEqual(first["paths"], ["/a", "/b"])
        self.assertEqual(first["pid"], "7")
        self.assertEqual(first["pids"], ["7", "8"])
        self.assertEqual(first["size"], 42)
        self.assertTrue(first["is_pe"])
        self.assertTrue(second["is_pe"])
        self.assertEqual(second["name"], "C:\\x.dll")
        self.assertFalse(third["is_pe"])
        self.assertFalse(third["size_available"])

    def test_zero_retention_still_scans_counts_resources_and_coverage(self):
        report = parse_report(calls_report([{"api": "DeleteFileW", "timestamp": 1, "repeated": 2,
                                             "arguments": [{"name": "FileName", "value": "x.txt"}]}]), max_events=0)
        self.assertEqual(report["processes"][0]["events"], [])
        self.assertEqual(report["processes"][0]["api_counts"], {"deletefilew": 3})
        self.assertEqual(report["resources"][0]["operations"], ["deletes"])
        self.assertTrue(report["coverage"]["timestamps"])
        self.assertTrue(report["coverage"]["arguments"])
        self.assertTrue(report["trace_truncated"])

    def test_input_validation_is_json_serializable_and_nonmutating(self):
        raw = calls_report([None, {}, {"api": [], "arguments": 4}, {"api": "A", "arguments": [{"name": "x", "value": float("nan")}]}])
        report = parse_report(raw)
        self.assertEqual(len(report["processes"][0]["events"]), 1)
        self.assertIsNone(report["processes"][0]["events"][0]["arguments"][0]["value"])
        json.dumps(report, allow_nan=False)
        for value in (None, [], 3, {"behavior": [], "network": [], "dropped": {}},
                      {"behavior": {"processes": [None, "bad"], "apistats": {"x": []}}}):
            json.dumps(parse_report(value), allow_nan=False)
        valid = calls_report([{"api": "A", "arguments": [{"name": "x", "value": {"a": [1]}}]}])
        original = copy.deepcopy(valid)
        parsed = parse_report(valid)
        parsed["processes"][0]["events"][0]["arguments"][0]["value"]["a"].append(2)
        self.assertEqual(valid, original)
        for limit in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                parse_report({}, max_events=limit)

    def test_path_input_leaves_original_report_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            original = json.dumps(calls_report([{"api": "A"}]))
            path.write_text(original, encoding="utf-8")
            report = parse_report(path)
            self.assertEqual(report["processes"][0]["api_counts"], {"a": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertNotIn("raw_report", report)
            self.assertTrue(parse_report(Path(directory) / "missing.json")["validation_warnings"])


@unittest.skipIf(np is None, "NumPy is not installed; extraction-only tests still run")
class TestBehavioralStatistics(unittest.TestCase):
    def statistics(self, report):
        vector = behavioral_statistics(report)
        self.assertEqual(vector.shape, (128,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((vector >= 0).all())
        return dict(zip(STATS_NAMES, vector))

    def test_documented_feature_dimensions_and_empty_missingness(self):
        self.assertIsInstance(STATS_NAMES, tuple)
        self.assertEqual(len(STATS_NAMES), STATS_DIM)
        self.assertEqual(len(set(STATS_NAMES)), 128)
        stats = self.statistics(parse_report({}))
        self.assertEqual(stats["api_occurrence_count"], 0)
        for key in COVERAGE_KEYS:
            self.assertEqual(stats["available_" + key], 0)
            self.assertEqual(stats["missing_" + key], 1)
        empty = self.statistics(parse_report({"behavior": {"processes": [], "summary": {"files": []}}, "network": {}}))
        self.assertEqual(empty["available_behavior"], 1)
        self.assertEqual(empty["available_processes"], 1)
        self.assertEqual(empty["available_files"], 1)
        self.assertEqual(empty["file_resource_count"], 0)

    def test_repeat_and_truncation_features_are_raw_counts(self):
        report = parse_report(calls_report([{"api": "A", "repeated": 2, "status": True},
                                             {"api": "B", "status": False}]), max_events=1)
        stats = self.statistics(report)
        self.assertEqual(stats["api_occurrence_count"], 4)
        self.assertEqual(stats["trace_event_count"], 2)
        self.assertEqual(stats["retained_event_count"], 1)
        self.assertEqual(stats["retained_occurrence_count"], 3)
        self.assertEqual(stats["successful_occurrence_count"], 3)
        self.assertEqual(stats["failed_occurrence_count"], 0)
        self.assertEqual(stats["trace_truncated"], 1)

    def test_apistats_and_resolutions_do_not_create_sequences(self):
        stats = self.statistics(parse_report({"behavior": {"apistats": {"7": {"A": 3, "B": 2}},
                                                             "summary": {"resolved_apis": ["C"]}}}))
        self.assertEqual(stats["api_occurrence_count"], 5)
        self.assertEqual(stats["unique_executed_api_count"], 2)
        self.assertEqual(stats["retained_event_count"], 0)
        self.assertEqual(stats["observed_thread_count"], 0)
        self.assertEqual(stats["event_span_seconds_max"], 0)
        self.assertEqual(stats["apistats_fallback_process_count"], 1)
        self.assertEqual(stats["available_apistats"], 1)
        self.assertEqual(stats["missing_calls"], 1)

    def test_guest_timing_never_uses_host_runtime_or_other_threads(self):
        raw = calls_report([
            {"api": "A", "thread_id": 1, "timestamp": 1},
            {"api": "B", "thread_id": 2, "timestamp": 10000},
            {"api": "C", "thread_id": 1, "timestamp": 3},
            {"api": "D", "thread_id": 2, "timestamp": 10002},
        ], first_seen="2026-01-01 00:00:00,000")
        raw["info"] = {"duration": 60, "started": "2026-01-01 00:00:00,000"}
        stats = self.statistics(parse_report(raw))
        self.assertEqual(stats["report_duration_seconds"], 60)
        self.assertEqual(stats["event_span_seconds_max"], 2)
        self.assertEqual(stats["interarrival_seconds_mean"], 2)
        self.assertEqual(stats["observed_thread_count"], 2)
        self.assertEqual(stats["thread_span_seconds_max"], 2)

    def test_clock_domains_backwards_and_equal_times_are_not_sorted(self):
        raw = calls_report([
            {"api": "A", "thread_id": 1, "timestamp": 3},
            {"api": "B", "thread_id": 1, "timestamp": "2026-01-01 00:00:00,000"},
            {"api": "C", "thread_id": 1, "timestamp": 1},
            {"api": "D", "thread_id": 1, "timestamp": 1},
            {"api": "E", "thread_id": 1, "timestamp": 2},
        ])
        stats = self.statistics(parse_report(raw))
        self.assertEqual(stats["event_span_seconds_max"], 2)
        self.assertEqual(stats["interarrival_seconds_max"], 1)
        self.assertEqual(stats["thread_backwards_timestamp_pair_count"], 1)
        self.assertEqual(stats["thread_equal_timestamp_pair_count"], 1)

    def test_label_signature_family_verdict_metadata_cannot_change_features(self):
        raw = calls_report([{"api": "A", "timestamp": 1}])
        original = copy.deepcopy(raw)
        baseline = behavioral_statistics(parse_report(raw))
        raw.update({"label": 1, "family": "untrusted", "verdict": "malware", "malscore": 10,
                    "signatures": [{"name": "untrusted", "severity": 99}], "statistics": {"processing": [{"time": 999}]}})
        np.testing.assert_array_equal(baseline, behavioral_statistics(parse_report(raw)))
        self.assertEqual(original, calls_report([{"api": "A", "timestamp": 1}]))

    def test_resource_and_payload_counts(self):
        raw = calls_report([{"api": "WriteFile", "repeated": 2,
                             "arguments": [{"name": "FileName", "value": "x.txt"}]}])
        raw["behavior"]["summary"] = {"read_files": ["x.txt", "y.txt"], "keys": ["HKCU\\X"]}
        raw["dropped"] = [{"name": "x", "sha256": "a", "size": 42, "type": "PE32 executable", "pid": 7}]
        stats = self.statistics(parse_report(raw))
        self.assertEqual(stats["file_resource_count"], 2)
        self.assertEqual(stats["file_write_resource_count"], 1)
        self.assertEqual(stats["file_read_resource_count"], 2)
        self.assertEqual(stats["file_occurrence_count"], 3)
        self.assertEqual(stats["registry_resource_count"], 1)
        self.assertEqual(stats["payload_total_bytes"], 42)
        self.assertEqual(stats["pe_payload_count"], 1)
        self.assertEqual(stats["payload_with_process_count"], 1)

    def test_extreme_counts_remain_finite_float32(self):
        stats = self.statistics(parse_report({"behavior": {"apistats": {"1": {"A": 1e100, "B": 1e100}}}}))
        self.assertEqual(stats["api_occurrence_count"], np.finfo(np.float32).max)

    def test_partial_ir_with_invalid_container_types_is_safe(self):
        for report in (None, {}, {"processes": None, "resources": 2, "payloads": "bad", "coverage": []},
                       {"processes": [{"api_counts": [], "events": None}]}):
            self.statistics(report)


if __name__ == "__main__":
    unittest.main()
