"""Deterministic, evidence-backed CAPE IR to six-type PyG graphs.

``build_graph(report, window_size=32, max_windows=256)`` accepts parse_report's
IR, not a raw CAPE document. Each node store has ``num_nodes`` and ``raw`` (a
list of dictionaries in node-index order); every schema relation has a long
``edge_index`` of shape [2, E], including empty relations and exact reverses.
There is always one binary node. API records canonicalize imports, resolutions,
counts and traces into the same identity, without treating resolutions as calls.
Binary raw records contain ``sha256``, ``payloads_available`` and ``telemetry``.
The latter is a whitelisted mapping in BINARY_TELEMETRY_NAMES order: parser
coverage availability flags, then trace/window truncation flags. The binary
store also exposes these 0/1 values as float32 ``telemetry`` of shape [1, 14],
independent of D1 and static PE features. Labels/verdicts are never included.
Process/resource/payload records retain their IR fields. API records contain
``name``, ``imported``, ``resolved``, full executed ``count``, and sorted observed
``categories``.
Process and resource stores additionally expose float32 ``statistics`` tensors
of shapes [N, 32] and [N, 12], in PROCESS_STAT_NAMES/RESOURCE_STAT_NAMES order.
Process scaling is documented by process_statistics. Resource columns 0:6 are
observed operation flags, 6:9 are log1p retained event rows/occurrences/owning
process counts, and 9:12 are summary/event evidence and provenance-cap flags.
Summary operation flags are inventory evidence, never fabricated event edges.

Window records contain ``process_index``, ``thread_id``, ``ordinal``, ``events``
and source ``event_indices``. Source order is preserved separately per process
and thread, including an unidentified-thread stream per process. Repeats remain
compressed rows, never expanded. The global window budget retains a prefix in
process order, then first-seen thread order; it never sorts on timestamps.
``graph.graph_metadata`` reports total/retained/dropped windows and events,
``window_size``, ``max_windows``, ``windows_truncated`` and ``trace_truncated``.
It does not shadow PyG's ``graph.metadata()`` schema method. Summary-only
resources have no invented ownership.

Raw metadata is for encoders/debugging, not PyG collation: models construct
tensor-only graphs after encoding, then use Batch.from_data_list. Payload stores
also expose a boolean ``static_available`` mask. No learned features are cached.
"""

from collections import Counter
import math

import torch
from torch_geometric.data import HeteroData

from .cape import COVERAGE_KEYS, canonical_api
from .schema import FORWARD_RELATIONS, NODE_TYPES


BINARY_TELEMETRY_NAMES = tuple("available_" + key for key in COVERAGE_KEYS) + ("trace_truncated", "windows_truncated")
PROCESS_STAT_NAMES = (
    "api_occurrences", "unique_apis", "trace_rows", "retained_rows", "retained_occurrences",
    "identified_threads", "observed_children", "observed_stream_span_seconds", "timestamped_rows", "categories",
    "file_resources", "registry_resources", "network_resources", "mutex_resources", "service_resources", "module_resources",
    "success_fraction", "failure_fraction", "unknown_status_fraction", "identified_thread_fraction",
    "arguments_fraction", "largest_api_fraction", "trace_retained_fraction", "calls_available", "apistats_available",
    "counts_from_calls", "counts_from_apistats", "trace_truncated", "name_available", "command_available",
    "observed_parent", "timestamps_available",
)
RESOURCE_OPERATIONS = ("reads", "writes", "deletes", "modifies", "connects_to", "accesses")
RESOURCE_STAT_NAMES = RESOURCE_OPERATIONS + (
    "event_rows", "event_occurrences", "attributed_processes", "summary_evidence", "event_evidence", "sources_truncated",
)


def process_statistics(processes):
    """Return [N, 32] local numerical features in PROCESS_STAT_NAMES order.

    Columns 0:16 are log1p counts/observed spans; 16:23 are within-process
    fractions; 23:32 are flags. No batch/dataset scaling or raw PID values enter
    these features. Timing spans never combine threads or clock domains and
    are observations, not inferred process lifetimes. Full trace/count totals
    and retained-only features remain distinct under parser/window truncation.
    """
    pids = {process.get("pid") for process in processes if process.get("pid")}
    children = Counter(process.get("ppid") for process in processes
                       if process.get("ppid") in pids and process.get("ppid") != process.get("pid")
                       and process.get("ppid") != "0")
    rows = []
    for process in processes:
        events = process.get("events", [])
        counts = [count for count in process.get("api_counts", {}).values() if count > 0]
        total = sum(counts)
        occurrences = sum(event.get("repeat", 1) for event in events)
        threads, categories, timestamps, resources = set(), set(), {}, set()
        statuses = [0, 0, 0]
        timestamped = identified = arguments = 0
        for event in events:
            thread = event.get("thread_id", "")
            if thread:
                threads.add(thread)
                identified += 1
            if event.get("category"):
                categories.add(event["category"])
            stamp = event.get("timestamp")
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and math.isfinite(stamp):
                timestamps.setdefault((thread, event.get("timestamp_domain", "numeric")), []).append(stamp)
                timestamped += 1
            resources.update((resource.get("kind"), resource.get("value")) for resource in event.get("resources", []))
            status = event.get("status")
            statuses[0 if status is True else 1 if status is False else 2] += event.get("repeat", 1)
            arguments += bool(event.get("arguments"))
        spans = [max(stamps) - min(stamps) for stamps in timestamps.values()]
        span = max((value for value in spans if math.isfinite(value)), default=0)
        trace_rows = process.get("trace_event_count", len(events))
        values = [total, len(counts), trace_rows, len(events), occurrences, len(threads),
                  children[process.get("pid")], span, timestamped, len(categories)]
        values.extend(sum(resource_kind == kind for resource_kind, _ in resources)
                      for kind in ("file", "registry", "network", "mutex", "service", "module"))
        # math.log on an integer avoids overflowing huge-but-valid JSON counts.
        values = [math.log(1 + value) for value in values]
        values.extend(value / occurrences if occurrences else 0 for value in statuses)
        values.extend([identified / len(events) if events else 0, arguments / len(events) if events else 0,
                       max(counts, default=0) / total if total else 0, len(events) / trace_rows if trace_rows else 0])
        values.extend([bool(process.get("calls_available")), bool(process.get("apistats_available")),
                       process.get("api_count_source") == "calls", process.get("api_count_source") == "apistats",
                       bool(process.get("trace_truncated")), bool(process.get("name")), bool(process.get("command_line")),
                       process.get("ppid") in pids and process.get("ppid") != process.get("pid") and process.get("ppid") != "0",
                       bool(timestamped)])
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, len(PROCESS_STAT_NAMES))


def event_streams(report):
    """Return (process_index, thread_id, [(source_index, event), ...]) streams."""
    streams = []
    for process_index, process in enumerate(report.get("processes", [])):
        threads = {}
        for index, event in enumerate(process.get("events", [])):
            if canonical_api(event.get("api")):
                threads.setdefault(str(event.get("thread_id", "")), []).append((index, event))
        streams.extend((process_index, thread, events) for thread, events in threads.items())
    return streams


def build_graph(report, window_size=32, max_windows=256):
    """Build an unembedded HeteroData graph; see the module's exact contract."""
    for name, value, minimum in (("window_size", window_size, 1), ("max_windows", max_windows, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    report = report if isinstance(report, dict) else {}
    processes = report.get("processes", [])
    payloads = report.get("payloads", [])
    edges = {relation: set() for relation in FORWARD_RELATIONS}
    raw = {kind: [] for kind in NODE_TYPES}
    raw["process"] = list(processes)
    raw["payload"] = list(payloads)

    imports = {canonical_api(name) for name in report.get("imports", [])} - {""}
    resolved = {canonical_api(name) for name in report.get("resolved_apis", [])} - {""}
    counts, categories = Counter(), {}
    process_calls = []
    for process in processes:
        calls = Counter()
        for name, count in process.get("api_counts", {}).items():
            api = canonical_api(name)
            if api and count > 0:
                calls[api] += count
        # Trace rows remain evidence even in a partial IR without api_counts.
        observed = Counter()
        for event in process.get("events", []):
            api = canonical_api(event.get("api"))
            if api:
                observed[api] += event.get("repeat", 1)
                categories.setdefault(api, set()).add(event.get("category", ""))
        for api, count in observed.items():
            if api not in calls:
                calls[api] = count
        counts.update(calls)
        process_calls.append(calls)
    names = sorted(imports | resolved | set(counts))
    api_index = {name: index for index, name in enumerate(names)}
    raw["api"] = [{"name": name, "imported": name in imports, "resolved": name in resolved,
                   "count": counts[name], "categories": sorted(categories.get(name, set()) - {""})}
                  for name in names]
    edges["binary", "imports", "api"].update((0, api_index[name]) for name in imports)
    by_pid = {str(process["pid"]): index for index, process in enumerate(processes) if process.get("pid")}
    for index, process in enumerate(processes):
        parent = by_pid.get(str(process.get("ppid", "")))
        if parent is not None and parent != index and str(process.get("ppid")) != "0":
            edges["process", "spawns", "process"].add((parent, index))
        else:
            edges["binary", "launches", "process"].add((0, index))
        edges["process", "calls", "api"].update((index, api_index[name]) for name in process_calls[index])

    resource_index, resource_evidence = {}, []

    def resource_node(resource):
        key = resource.get("kind", ""), resource.get("value", "")
        if not all(key):
            return None
        if key not in resource_index:
            resource_index[key] = len(raw["resource"])
            raw["resource"].append(dict(resource))
            resource_evidence.append({"rows": 0, "occurrences": 0, "processes": set(),
                                      "operations": set(resource.get("operations", []))})
        return resource_index[key]

    for resource in report.get("resources", []):
        resource_node(resource)
    # Process attribution survives the window cap, but requires an actual event.
    for index, process in enumerate(processes):
        for event in process.get("events", []):
            seen = set()
            for resource in event.get("resources", []):
                target = resource_node(resource)
                if target is None:
                    continue
                evidence = resource_evidence[target]
                evidence["operations"].add(resource.get("operation", "accesses"))
                if target not in seen:
                    evidence["rows"] += 1
                    evidence["occurrences"] += event.get("repeat", 1)
                    evidence["processes"].add(index)
                    seen.add(target)
                api = canonical_api(event.get("api"))
                loads = resource.get("kind") == "module" and any(
                    part in api for part in ("loadlibrary", "ldrloaddll", "loadmodule", "loadpackagedlibrary"))
                edges["process", "loads" if loads else "accesses", "resource"].add((index, target))

    total_windows = total_events = retained_events = 0
    for process_index, thread_id, stream in event_streams(report):
        total_windows += (len(stream) + window_size - 1) // window_size
        total_events += len(stream)
        previous = None
        for start in range(0, len(stream), window_size):
            if len(raw["window"]) >= max_windows:
                break
            chunk = stream[start:start + window_size]
            index = len(raw["window"])
            events = [event for _, event in chunk]
            raw["window"].append({"process_index": process_index, "thread_id": thread_id,
                                   "ordinal": start // window_size, "events": events,
                                   "event_indices": [offset for offset, _ in chunk]})
            retained_events += len(events)
            edges["process", "executes", "window"].add((process_index, index))
            if previous is not None:
                edges["window", "next_window", "window"].add((previous, index))
            previous = index
            for event in events:
                edges["window", "contains_api", "api"].add((index, api_index[canonical_api(event["api"])]))
                for resource in event.get("resources", []):
                    target = resource_node(resource)
                    operation = resource.get("operation", "accesses")
                    relation = ("window", operation, "resource")
                    if target is not None and relation in edges:
                        edges[relation].add((index, target))

    for index, payload in enumerate(payloads):
        edges["binary", "produces", "payload"].add((0, index))
        pids = payload.get("pids", []) or [payload.get("pid", "")]
        for pid in pids:
            if str(pid) in by_pid:
                edges["process", "drops", "payload"].add((by_pid[str(pid)], index))

    telemetry = {"available_" + key: bool(report.get("coverage", {}).get(key)) for key in COVERAGE_KEYS}
    telemetry.update(trace_truncated=bool(report.get("trace_truncated")),
                     windows_truncated=total_windows > len(raw["window"]))
    raw["binary"] = [{"sha256": report.get("sha256", ""),
                      "payloads_available": telemetry["available_payloads"], "telemetry": telemetry}]
    graph = HeteroData()
    for kind in NODE_TYPES:
        graph[kind].num_nodes = len(raw[kind])
        graph[kind].raw = raw[kind]
    graph["binary"].telemetry = torch.tensor([[float(telemetry[name]) for name in BINARY_TELEMETRY_NAMES]], dtype=torch.float32)
    graph["process"].statistics = process_statistics(processes)
    graph["resource"].statistics = torch.tensor([
        [float(operation in evidence["operations"]) for operation in RESOURCE_OPERATIONS] +
        [math.log1p(evidence["rows"]), math.log1p(evidence["occurrences"]), math.log1p(len(evidence["processes"])),
         float(any(source.startswith("/behavior/summary/") for source in resource.get("sources", []))),
         float(evidence["rows"] > 0), float(bool(resource.get("sources_truncated")))]
        for resource, evidence in zip(raw["resource"], resource_evidence)
    ], dtype=torch.float32).reshape(-1, len(RESOURCE_STAT_NAMES))
    graph["payload"].static_available = torch.tensor(
        [payload.get("static_features") is not None for payload in payloads], dtype=torch.bool)
    for (src, relation, dst), pairs in edges.items():
        edge_index = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        graph[src, relation, dst].edge_index = edge_index
        graph[dst, "rev_" + relation, src].edge_index = edge_index.flip(0).contiguous()
    graph.graph_metadata = {"window_size": window_size, "max_windows": max_windows,
                            "total_windows": total_windows, "retained_windows": len(raw["window"]),
                            "dropped_windows": total_windows - len(raw["window"]),
                             "windows_truncated": telemetry["windows_truncated"],
                            "total_events": total_events, "retained_events": retained_events,
                            "dropped_events": total_events - retained_events,
                             "trace_truncated": telemetry["trace_truncated"]}
    return graph
