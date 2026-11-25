"""Frozen, documented 128-dimensional raw behavioral statistics.

The eight consecutive groups each contain 16 features. Counts are observed
counts, not estimates of unobserved activity. ``api_counts`` and trace row
totals include scanned-but-unretained events; all other event/thread features
use retained events only. Resource inventories include unattributed summary
evidence. API imports/resolutions are deliberately not executed-call counts.
No labels, signatures, families, verdicts, scaling, or logarithms are used.

Timing uses source-order neighbors within one process/thread/clock domain.
Backward differences are excluded, not sorted or made positive. Unknown
thread IDs contribute to execution timing, but not identified-thread counts.
Report duration is a separate host-report measurement, never subtracted from
guest timestamps. Missing numerical observations are zero; availability and
missingness flags preserve their distinction from measured empty sections.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

from .schema import STATS_DIM


STATS_NAMES = (
    # Execution: 0:16.
    "api_occurrence_count",              # Full per-process executed/fallback API totals.
    "unique_executed_api_count",         # Union of positive-count executed APIs.
    "trace_event_count",                # Full valid source rows, before retention.
    "retained_event_count",             # Retained compressed source rows.
    "retained_occurrence_count",        # Sum of retained repeat multiplicities.
    "successful_occurrence_count",      # Retained occurrences with status True.
    "failed_occurrence_count",          # Retained occurrences with status False.
    "unknown_status_occurrence_count",  # Retained occurrences with unknown status.
    "repeated_event_count",             # Retained rows with repeat > 1.
    "max_event_repeat",                 # Largest retained row multiplicity.
    "api_frequency_max",                # Largest global API occurrence count.
    "api_frequency_mean",               # Mean count over observed executed APIs.
    "event_span_seconds_max",           # Longest within-thread/clock observed span.
    "interarrival_seconds_mean",        # Mean nonnegative within-thread/clock gap.
    "interarrival_seconds_max",         # Largest such gap; repeats are not expanded.
    "report_duration_seconds",          # Explicit duration or host end minus host start.
    # Process: 16:32.
    "process_count",                    # Processes explicitly represented by evidence.
    "process_with_events_count",        # Processes with valid full-trace rows.
    "process_with_apistats_count",      # Processes with an available apistats section.
    "process_with_command_line_count",  # Processes with nonempty observed command lines.
    "process_with_parent_count",        # Nonempty, nonzero explicit parent IDs.
    "observed_parent_edge_count",       # Parent ID references another observed process.
    "root_process_count",               # Processes without an observed parent edge.
    "unique_process_name_count",        # Distinct nonempty names, case folded.
    "process_api_occurrences_mean",     # Mean full API total over observed processes.
    "process_api_occurrences_max",      # Largest per-process full API total.
    "process_unique_apis_mean",         # Mean positive-count APIs per process.
    "process_unique_apis_max",          # Largest positive-count API vocabulary.
    "process_retained_events_mean",     # Mean retained rows per observed process.
    "process_retained_events_max",      # Largest retained process trace.
    "available_processes",              # Process list/tree or per-PID apistats present.
    "missing_processes",                # Complement of available_processes.
    # Thread: 32:48.
    "observed_thread_count",            # Distinct (process, nonempty thread ID) pairs.
    "process_with_thread_ids_count",    # Processes with an identified retained thread.
    "events_with_thread_id_count",      # Retained rows with explicit thread ID.
    "events_missing_thread_id_count",   # Retained rows without an explicit thread ID.
    "thread_retained_events_mean",      # Mean rows over identified threads.
    "thread_retained_events_max",       # Largest identified-thread retained trace.
    "thread_occurrences_mean",          # Mean retained occurrences per identified thread.
    "thread_occurrences_max",           # Largest retained identified-thread total.
    "thread_unique_apis_mean",          # Mean API vocabulary per identified thread.
    "thread_unique_apis_max",           # Largest identified-thread API vocabulary.
    "thread_span_seconds_mean",         # Mean identified-thread/clock observed span.
    "thread_span_seconds_max",          # Largest identified-thread/clock span.
    "thread_backwards_timestamp_pair_count",  # Negative adjacent known-thread/clock gaps.
    "thread_equal_timestamp_pair_count",      # Zero adjacent known-thread/clock gaps.
    "available_timestamps",             # At least one valid source-event timestamp.
    "missing_timestamps",               # Complement of available_timestamps.
    # File: 48:64. Operation counts count distinct resources, not events.
    "file_resource_count",              # Distinct observed file identifiers.
    "file_read_resource_count",         # Files explicitly read.
    "file_write_resource_count",        # Files explicitly written/created.
    "file_delete_resource_count",       # Files explicitly deleted.
    "file_modify_resource_count",       # Files explicitly modified/renamed.
    "file_access_resource_count",       # Files accessed without stronger operation.
    "file_event_count",                 # Retained rows with explicit file resources.
    "file_occurrence_count",            # Repeat-weighted file rows, once per row.
    "file_api_count",                   # APIs on retained rows with file resources.
    "file_resource_reference_count",    # File references on retained rows.
    "file_path_character_count",        # Total characters in distinct file identifiers.
    "file_path_length_max",             # Longest observed file identifier.
    "file_extension_count",             # Distinct nonempty final file extensions.
    "file_summary_resource_count",      # Files with behavior-summary provenance.
    "available_files",                  # File evidence or well-formed empty file section.
    "missing_files",                    # Complement of available_files.
    # Registry: 64:80.
    "registry_resource_count",          # Distinct observed registry identifiers.
    "registry_read_resource_count",     # Registry identifiers explicitly read.
    "registry_write_resource_count",    # Registry identifiers explicitly written/created.
    "registry_delete_resource_count",   # Registry identifiers explicitly deleted.
    "registry_modify_resource_count",   # Registry identifiers explicitly modified.
    "registry_access_resource_count",   # Registry accesses of unspecified operation.
    "registry_event_count",             # Retained rows with explicit registry resources.
    "registry_occurrence_count",        # Repeat-weighted registry rows, once per row.
    "registry_api_count",               # APIs on registry-resource retained rows.
    "registry_resource_reference_count",  # Registry references on retained rows.
    "registry_path_character_count",    # Total characters in registry identifiers.
    "registry_path_length_max",         # Longest observed registry identifier.
    "registry_root_count",              # Distinct explicit registry root strings.
    "registry_summary_resource_count",  # Registry identifiers with summary provenance.
    "available_registry",               # Registry evidence or well-formed empty section.
    "missing_registry",                 # Complement of available_registry.
    # Network: 80:96.
    "network_resource_count",           # Distinct endpoints/hosts/URLs, not inferred peers.
    "network_connect_resource_count",   # Network identifiers with connects_to evidence.
    "network_access_resource_count",    # Network identifiers with accesses evidence.
    "network_event_count",              # Retained rows with explicit network resources.
    "network_occurrence_count",         # Repeat-weighted network rows, once per row.
    "network_api_count",                # APIs on network-resource retained rows.
    "dns_record_count",                 # Observed top-level network DNS records.
    "http_record_count",                # Observed HTTP records (http or http_ex).
    "https_record_count",               # Observed HTTPS records (https or https_ex).
    "tcp_connection_count",             # Observed TCP records, not deduced from APIs.
    "udp_connection_count",             # Observed UDP records.
    "icmp_record_count",                # Observed ICMP records.
    "network_host_count",               # Entries in network.hosts.
    "network_domain_count",             # Entries in network.domains.
    "available_network",                # Network section or explicit network resources.
    "missing_network",                  # Complement of available_network.
    # Other resources: 96:112.
    "mutex_resource_count",             # Distinct explicit mutex identifiers.
    "service_resource_count",           # Distinct explicit service identifiers.
    "module_resource_count",            # Distinct explicitly observed module identifiers.
    "mutex_event_count",                # Retained rows referencing a mutex.
    "service_event_count",              # Retained rows referencing a service.
    "module_event_count",               # Retained rows referencing a module.
    "other_resource_occurrence_count",  # Rows with any other resource, repeat weighted once.
    "other_resource_api_count",         # API vocabulary across these retained rows.
    "payload_count",                    # Observed payload records; no hash deduplication.
    "pe_payload_count",                 # Payloads explicitly identified as PE.
    "payload_total_bytes",              # Sum of available nonnegative payload sizes.
    "payload_max_bytes",                # Largest available payload size.
    "payload_with_process_count",       # Payload records with explicitly supplied PID(s).
    "unique_payload_hash_count",        # Distinct nonempty payload SHA-256 strings.
    "available_other_resources",        # Other-resource evidence or empty summary section.
    "missing_other_resources",          # Complement of available_other_resources.
    # Coverage: 112:128. All availability flags are 0/1.
    "available_behavior",               # Well-formed behavior object, including {}.
    "missing_behavior",                 # Complement of available_behavior.
    "available_calls",                  # Any well-formed calls list, including [].
    "missing_calls",                    # Complement of available_calls.
    "available_apistats",               # Well-formed global/per-process apistats object.
    "missing_apistats",                 # Complement of available_apistats.
    "available_summary",                # Well-formed behavior.summary, including {}.
    "missing_summary",                  # Complement of available_summary.
    "available_arguments",              # Any well-formed event arguments, including [].
    "missing_arguments",                # Complement of available_arguments.
    "available_payloads",               # Any well-formed payload list, including [].
    "missing_payloads",                 # Complement of available_payloads.
    "trace_truncated",                  # At least one valid row omitted by retention cap.
    "retained_events_with_timestamp_count",     # Retained rows with valid timestamps.
    "retained_events_missing_timestamp_count",  # Retained rows with absent/invalid timestamps.
    "apistats_fallback_process_count",   # Processes using counts without a call sequence.
)
assert len(STATS_NAMES) == len(set(STATS_NAMES)) == STATS_DIM == 128


def _nonnegative(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else 0.0
    except OverflowError:
        return float("inf") if isinstance(value, int) and value > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _mean(values):
    return sum(value / len(values) for value in values) if values else 0.0


def _objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def behavioral_statistics(report):
    """Extract STATS_NAMES in order as finite, nonnegative raw numpy float32.

    Input is the normalized IR from :func:`multimodal.cape.parse_report`.
    Empty/partial IRs are accepted. Values above float32 range saturate at
    float32 max solely to enforce the finite-output contract, not as scaling.
    """
    report = report if isinstance(report, dict) else {}
    processes = _objects(report.get("processes"))
    resources = _objects(report.get("resources"))
    payloads = _objects(report.get("payloads"))
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    features = {}
    api_counts = Counter()
    process_totals, process_vocabularies, process_lengths = [], [], []
    events, thread_events, timelines = [], defaultdict(list), defaultdict(list)
    for process_index, process in enumerate(processes):
        raw_counts = process.get("api_counts") if isinstance(process.get("api_counts"), dict) else {}
        counts = {api: _nonnegative(count) for api, count in raw_counts.items()
                  if _nonnegative(count) > 0}
        api_counts.update(counts)
        process_totals.append(sum(counts.values()))
        process_vocabularies.append(len(counts))
        retained = _objects(process.get("events"))
        process_lengths.append(len(retained))
        for event in retained:
            events.append(event)
            thread = event.get("thread_id", "")
            if thread:
                thread_events[process_index, thread].append(event)
            stamp = event.get("timestamp")
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp):
                timelines[process_index, thread, event.get("timestamp_domain", "numeric")].append(stamp)

    all_spans, all_gaps, thread_spans = [], [], []
    backwards = equal = 0
    for (_, thread, _), stamps in timelines.items():
        # Subtract only timestamps from the same guest stream/clock domain.
        span = max(stamps) - min(stamps)
        if math.isfinite(span):
            all_spans.append(span)
            if thread:
                thread_spans.append(span)
        for before, after in zip(stamps, stamps[1:]):
            gap = after - before
            if not math.isfinite(gap):
                continue
            if gap >= 0:
                all_gaps.append(gap)
            if thread:
                backwards += gap < 0
                equal += gap == 0
    repeats = [_nonnegative(event.get("repeat", 1)) for event in events]
    features.update({
        "api_occurrence_count": sum(api_counts.values()),
        "unique_executed_api_count": len(api_counts),
        "trace_event_count": sum(_nonnegative(process.get("trace_event_count", length))
                                 for process, length in zip(processes, process_lengths)),
        "retained_event_count": len(events),
        "retained_occurrence_count": sum(repeats),
        "successful_occurrence_count": sum(repeat for event, repeat in zip(events, repeats) if event.get("status") is True),
        "failed_occurrence_count": sum(repeat for event, repeat in zip(events, repeats) if event.get("status") is False),
        "unknown_status_occurrence_count": sum(repeat for event, repeat in zip(events, repeats) if event.get("status") is not True and event.get("status") is not False),
        "repeated_event_count": sum(repeat > 1 for repeat in repeats),
        "max_event_repeat": max(repeats, default=0),
        "api_frequency_max": max(api_counts.values(), default=0),
        "api_frequency_mean": _mean(list(api_counts.values())),
        "event_span_seconds_max": max(all_spans, default=0),
        "interarrival_seconds_mean": _mean(all_gaps),
        "interarrival_seconds_max": max(all_gaps, default=0),
        "report_duration_seconds": _nonnegative(report.get("duration")),
    })
    pids = {process.get("pid") for process in processes if process.get("pid")}
    parent_edges = sum(bool(process.get("ppid")) and process.get("ppid") != process.get("pid") and
                       process.get("ppid") in pids and process.get("ppid") != "0" for process in processes)
    features.update({
        "process_count": len(processes),
        "process_with_events_count": sum(_nonnegative(process.get("trace_event_count", length)) > 0
                                         for process, length in zip(processes, process_lengths)),
        "process_with_apistats_count": sum(bool(process.get("apistats_available")) for process in processes),
        "process_with_command_line_count": sum(bool(process.get("command_line")) for process in processes),
        "process_with_parent_count": sum(bool(process.get("ppid")) and process.get("ppid") != "0" for process in processes),
        "observed_parent_edge_count": parent_edges,
        "root_process_count": len(processes) - parent_edges,
        "unique_process_name_count": len({process["name"].casefold() for process in processes if process.get("name")}),
        "process_api_occurrences_mean": _mean(process_totals),
        "process_api_occurrences_max": max(process_totals, default=0),
        "process_unique_apis_mean": _mean(process_vocabularies),
        "process_unique_apis_max": max(process_vocabularies, default=0),
        "process_retained_events_mean": _mean(process_lengths),
        "process_retained_events_max": max(process_lengths, default=0),
    })
    thread_lengths = [len(items) for items in thread_events.values()]
    thread_totals = [sum(_nonnegative(event.get("repeat", 1)) for event in items) for items in thread_events.values()]
    thread_vocabularies = [len({event.get("api") for event in items if event.get("api")}) for items in thread_events.values()]
    identified_events = sum(thread_lengths)
    features.update({
        "observed_thread_count": len(thread_events),
        "process_with_thread_ids_count": len({key[0] for key in thread_events}),
        "events_with_thread_id_count": identified_events,
        "events_missing_thread_id_count": len(events) - identified_events,
        "thread_retained_events_mean": _mean(thread_lengths),
        "thread_retained_events_max": max(thread_lengths, default=0),
        "thread_occurrences_mean": _mean(thread_totals),
        "thread_occurrences_max": max(thread_totals, default=0),
        "thread_unique_apis_mean": _mean(thread_vocabularies),
        "thread_unique_apis_max": max(thread_vocabularies, default=0),
        "thread_span_seconds_mean": _mean(thread_spans),
        "thread_span_seconds_max": max(thread_spans, default=0),
        "thread_backwards_timestamp_pair_count": backwards,
        "thread_equal_timestamp_pair_count": equal,
    })
    by_kind = defaultdict(list)
    for resource in resources:
        by_kind[resource.get("kind")].append(resource)
    event_kinds = [{resource.get("kind") for resource in _objects(event.get("resources"))}
                   for event in events]
    for kind in ("file", "registry", "network"):
        items = by_kind[kind]
        selected = [(event, repeat) for event, repeat, kinds in zip(events, repeats, event_kinds) if kind in kinds]
        features[kind + "_resource_count"] = len(items)
        operations = ("connects_to", "accesses") if kind == "network" else ("reads", "writes", "deletes", "modifies", "accesses")
        for operation in operations:
            stem = {"connects_to": "connect", "accesses": "access", "reads": "read", "writes": "write", "deletes": "delete", "modifies": "modify"}[operation]
            features[f"{kind}_{stem}_resource_count"] = sum(operation in item.get("operations", []) for item in items)
        features[kind + "_event_count"] = len(selected)
        features[kind + "_occurrence_count"] = sum(repeat for _, repeat in selected)
        features[kind + "_api_count"] = len({event.get("api") for event, _ in selected if event.get("api")})
        if kind == "network":
            continue
        features[kind + "_resource_reference_count"] = sum(
            resource.get("kind") == kind for event, _ in selected for resource in _objects(event.get("resources")))
        lengths = [len(item.get("value", "")) for item in items]
        features[kind + "_path_character_count"] = sum(lengths)
        features[kind + "_path_length_max"] = max(lengths, default=0)
        features[kind + "_summary_resource_count"] = sum(
            any(source.startswith("/behavior/summary/") for source in item.get("sources", [])) for item in items)
    extensions = set()
    for item in by_kind["file"]:
        basename = re.split(r"[\\/]", item.get("value", ""))[-1]
        if "." in basename and not basename.endswith("."):
            extensions.add(basename.rsplit(".", 1)[1].casefold())
    features["file_extension_count"] = len(extensions)
    features["registry_root_count"] = len({re.split(r"[\\/]", item.get("value", "").lstrip("\\/"))[0].casefold()
                                            for item in by_kind["registry"] if item.get("value")})
    network_counts = report.get("network_counts") if isinstance(report.get("network_counts"), dict) else {}
    for protocol, name in (("dns", "dns_record_count"), ("http", "http_record_count"),
                           ("https", "https_record_count"), ("tcp", "tcp_connection_count"),
                           ("udp", "udp_connection_count"), ("icmp", "icmp_record_count"),
                           ("hosts", "network_host_count"), ("domains", "network_domain_count")):
        features[name] = _nonnegative(network_counts.get(protocol))
    other_kinds = {"mutex", "service", "module"}
    for kind in other_kinds:
        features[kind + "_resource_count"] = len(by_kind[kind])
        features[kind + "_event_count"] = sum(kind in kinds for kinds in event_kinds)
    other_events = [(event, repeat) for event, repeat, kinds in zip(events, repeats, event_kinds) if kinds & other_kinds]
    sizes = [_nonnegative(payload.get("size")) for payload in payloads]
    features.update({
        "other_resource_occurrence_count": sum(repeat for _, repeat in other_events),
        "other_resource_api_count": len({event.get("api") for event, _ in other_events if event.get("api")}),
        "payload_count": len(payloads),
        "pe_payload_count": sum(payload.get("is_pe") is True for payload in payloads),
        "payload_total_bytes": sum(sizes),
        "payload_max_bytes": max(sizes, default=0),
        "payload_with_process_count": sum(bool(payload.get("pid") or payload.get("pids")) for payload in payloads),
        "unique_payload_hash_count": len({payload["sha256"] for payload in payloads if payload.get("sha256")}),
        "trace_truncated": bool(report.get("trace_truncated")),
        "retained_events_with_timestamp_count": sum(len(stamps) for stamps in timelines.values()),
        "retained_events_missing_timestamp_count": len(events) - sum(len(stamps) for stamps in timelines.values()),
        "apistats_fallback_process_count": sum(process.get("api_count_source") == "apistats" for process in processes),
    })
    for section in ("behavior", "calls", "apistats", "summary", "files", "registry", "network",
                    "other_resources", "payloads", "timestamps", "arguments", "processes"):
        available = bool(coverage.get(section))
        features["available_" + section] = available
        features["missing_" + section] = not available
    assert set(features) == set(STATS_NAMES)
    limit = float(np.finfo(np.float32).max)
    # Saturate overflow instead of emitting NaN/Inf; do not transform counts.
    values = []
    for name in STATS_NAMES:
        value = features[name]
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = limit if isinstance(value, int) and value > 0 else 0.0
        values.append(min(number, limit) if number > 0 else 0.0)
    return np.asarray(values, dtype=np.float32)
