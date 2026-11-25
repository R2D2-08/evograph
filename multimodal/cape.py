"""Normalize CAPE JSON without executing samples or inferring missing behavior.

Source strings are JSON-pointer-like paths into the caller's original report.
The input is never modified or embedded in the IR; keep its path separately.
Resource source samples are capped at 32 (summary pointers are always kept),
with ``sources_truncated`` marking omitted provenance references.
An available, empty section differs from an absent or malformed section.
Only retained events form sequences. ``api_counts`` covers the entire trace,
or per-process apistats when that process has no valid trace events.
"""

import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone


MAX_REPEAT = 1_000_000
COVERAGE_KEYS = (
    "behavior", "calls", "apistats", "summary", "files", "registry", "network",
    "other_resources", "payloads", "timestamps", "arguments", "processes",
)
RESOURCE_KINDS = ("file", "registry", "network", "mutex", "service", "module")

# These are report fields, not signatures or verdict-derived categories.
_SUMMARY_FIELDS = {
    "files": ("file", "accesses"),
    "keys": ("registry", "accesses"),
    "regkeys": ("registry", "accesses"),
    "mutexes": ("mutex", "accesses"),
    "services": ("service", "accesses"),
    "dll_loaded": ("module", "accesses"),
    "loaded_dlls": ("module", "accesses"),
    "modules": ("module", "accesses"),
    "created_services": ("service", "writes"),
    "started_services": ("service", "modifies"),
    "deleted_services": ("service", "deletes"),
}
for _kind, _noun, _cape_noun in (("file", "files", "file"),
                                ("registry", "keys", "regkey")):
    for _verb, _summary_operation in (("read", "reads"), ("write", "writes"),
                              ("delete", "deletes"), ("open", "accesses")):
        _SUMMARY_FIELDS[f"{_verb}_{_noun}"] = (_kind, _summary_operation)
        _past = {"read": "read", "write": "written", "delete": "deleted",
                 "open": "opened"}[_verb]
        _SUMMARY_FIELDS[f"{_cape_noun}_{_past}"] = (_kind, _summary_operation)
_SUMMARY_FIELDS.update({"file_created": ("file", "writes"),
                        "file_moved": ("file", "modifies"),
                        "regkey_created": ("registry", "writes")})


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _identifier(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def canonical_api(value):
    """Lowercase an API and remove DLL qualification, retaining A/W suffixes."""
    return re.split(r"[!.]", _text(value))[-1].strip().lower()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        result = float(value.replace(",", ".") if isinstance(value, str) else value)
        return result if math.isfinite(result) else None
    except (ValueError, OverflowError):
        return None


def _timestamp(value):
    number = _number(value)
    if number is not None:
        return number, "numeric"
    text = _text(value).replace(",", ".")
    if not text:
        return None, ""
    try:
        if re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", text):
            hours, minutes, seconds = map(float, text.split(":"))
            if hours < 24 and minutes < 60 and seconds < 60:
                return hours * 3600 + minutes * 60 + seconds, "timeofday"
            return None, ""
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp(), "datetime"
    except (ValueError, OverflowError, OSError):
        return None, ""


def _json_value(value):
    # Non-JSON caller values and NaN must not leak into a serializable IR.
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return None


def _strings(value):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for key in ("name", "path", "value", "ObjectName", "FileName", "FullName", "ip", "host")
                for text in _strings(value.get(key))]
    return []


def _counts(value):
    result = Counter()
    for name, count in _mapping(value).items():
        api = canonical_api(name)
        number = count if isinstance(count, int) and not isinstance(count, bool) else _number(count)
        if api and number is not None and number >= 1:
            result[api] += int(number)
    return dict(result)


def _arguments(value):
    if isinstance(value, dict):
        value = [{"name": key, "value": item} for key, item in value.items()]
    if not isinstance(value, list):
        return []
    return [{"name": _text(item.get("name")),
             "value": _json_value(item.get("value")),
             "pretty_value": _json_value(item.get("pretty_value"))}
            for item in value if isinstance(item, dict)]


def _operation(api, kind, argument=""):
    if kind == "module":
        return "accesses"
    if kind == "network":
        return "connects_to" if "connect" in api else "accesses"
    if "delete" in api or "remove" in api or "unlink" in api:
        return "deletes"
    if kind == "file" and "copyfile" in api:
        if any(part in argument for part in ("existing", "source", "src")):
            return "reads"
        if any(part in argument for part in ("new", "destination", "dst")):
            return "writes"
        return "accesses"
    if any(part in api for part in ("rename", "movefile", "setfile", "setendoffile")):
        return "modifies"
    if kind == "service" and any(part in api for part in ("start", "control", "change")):
        return "modifies"
    if "write" in api or (kind == "registry" and "set" in api):
        return "writes"
    if any(part in api for part in ("read", "query", "getfile", "enum", "findfile")):
        return "reads"
    if (kind in ("registry", "mutex", "service") and "create" in api) or "createdirectory" in api:
        return "writes"
    # CreateFile/NtCreateFile can merely open an existing file.
    return "accesses"


def _event_resources(api, category, arguments):
    result, seen = [], set()
    context = api + " " + category
    for argument in arguments:
        name = re.sub(r"[^a-z0-9]", "", argument["name"].lower())
        name = re.sub(r"^(?:lpsz|lp|psz|sz)", "", name)
        pretty = _strings(argument["pretty_value"])
        values = pretty or _strings(argument["value"])
        kind = ""
        if name in {"keyname", "keypath", "regkey", "registrykey", "subkey", "fullkeyname"}:
            kind = "registry"
        elif name in {"mutexname", "mutantname"}:
            kind = "mutex"
        elif name in {"servicename"}:
            kind = "service"
        elif name in {"modulename", "libraryname", "dllname", "libfilename", "module", "library"}:
            kind = "module"
        elif name in {"hostname", "host", "ip", "ipaddress", "remoteip", "remoteaddress",
                      "url", "uri", "domain", "servername", "sinaddr", "sockaddr"}:
            kind = "network"
        elif name in {"filename", "filepath", "file", "oldfilename", "newfilename",
                      "existingfilename", "sourcefile", "destinationfile", "srcfilename",
                      "dstfilename", "directory", "directoryname", "dirpath"}:
            kind = "module" if "loadlibrary" in api or "ldrloaddll" in api else "file"
        elif name in {"objectname", "objectattributes", "fullname", "fullpath", "path", "name", "address"}:
            if any(part in context for part in ("registry", "regopen", "regcreate", "regset", "regquery",
                                                "regdelete", "regenum", "regclose", "regload", "regsave", "key")):
                kind = "registry"
            elif "mutex" in context or "mutant" in context:
                kind = "mutex"
            elif "service" in context:
                kind = "service"
            elif "loadlibrary" in api or "ldrloaddll" in api:
                kind = "module"
            elif "file" in context or "directory" in context:
                kind = "file"
            elif any(part in context for part in ("network", "socket", "connect", "dns", "http")):
                kind = "network"
        for value in values:
            resource_kind = kind
            # Explicit pretty paths are evidence; numeric handles are never resolved.
            if not resource_kind and pretty:
                if re.match(r"(?i)^(?:HKEY_|HKLM\\|HKCU\\|\\REGISTRY\\)", value):
                    resource_kind = "registry"
                elif re.match(r"(?i)^(?:[a-z]:\\|\\\\|\\Device\\|\\\?\?\\)", value):
                    resource_kind = "file"
            if not resource_kind or re.fullmatch(r"(?i)(?:0x[0-9a-f]+|\d+|null|none)", value):
                continue
            operation = _operation(api, resource_kind, name)
            key = resource_kind, value, operation
            if key not in seen:
                result.append({"kind": resource_kind, "value": value, "operation": operation})
                seen.add(key)
    return result


def _import_names(value):
    if isinstance(value, str):
        name = canonical_api(value)
        return [name] if name else []
    if isinstance(value, list):
        return [name for item in value for name in _import_names(item)]
    if isinstance(value, dict):
        if "imports" in value:
            return _import_names(value["imports"])
        if "name" in value:
            return _import_names(value["name"])
        return [name for item in value.values() if isinstance(item, (dict, list))
                for name in _import_names(item)]
    return []


def parse_report(path_or_dict, max_events=20000):
    """Return a JSON-serializable IR, including an empty IR for invalid reports.

    ``max_events`` is a nonnegative integer global retention limit, not a scan
    limit. Repeated occurrences are capped at MAX_REPEAT per source row. All
    valid rows contribute to API totals/resources even beyond the limit.
    Read/JSON/type failures are recorded in ``validation_warnings``. Imports,
    resolutions, and summary resources never create calls or process ownership.
    """
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise ValueError("max_events must be a nonnegative integer")
    warnings = []
    if isinstance(path_or_dict, dict):
        document = path_or_dict
    elif isinstance(path_or_dict, (str, os.PathLike)):
        try:
            with open(path_or_dict, encoding="utf-8-sig") as stream:
                document = json.load(stream)
        except (OSError, ValueError, UnicodeError):
            document = {}
            warnings.append("unreadable_or_invalid_json")
    else:
        document = {}
        warnings.append("report_must_be_an_object_or_path")
    if not isinstance(document, dict):
        document = {}
        warnings.append("report_must_be_an_object")

    coverage = {key: False for key in COVERAGE_KEYS}
    behavior = _mapping(document.get("behavior"))
    summary = _mapping(behavior.get("summary"))
    coverage["behavior"] = isinstance(document.get("behavior"), dict)
    coverage["summary"] = isinstance(behavior.get("summary"), dict)
    report = {"sha256": "", "processes": [], "resources": [], "payloads": [],
              "imports": [], "resolved_apis": [], "coverage": coverage,
              "duration": None, "network_counts": {}, "trace_truncated": False,
              "validation_warnings": warnings,
              "provenance": {"sha256": "", "duration": "", "imports": [], "resolved_apis": []}}
    target = _mapping(document.get("target"))
    target_file = _mapping(target.get("file"))
    for container, source in ((target_file, "/target/file"), (target, "/target"), (document, "")):
        if _text(container.get("sha256")):
            report["sha256"] = container["sha256"].strip().lower()
            report["provenance"]["sha256"] = source + "/sha256"
            break
    info = _mapping(document.get("info"))
    duration = _number(info.get("duration"))
    if duration is not None and duration >= 0:
        report["duration"] = duration
        report["provenance"]["duration"] = "/info/duration"
    else:
        start, start_domain = _timestamp(info.get("started"))
        end, end_domain = _timestamp(info.get("ended"))
        if start is not None and end is not None and start_domain == end_domain and end >= start:
            report["duration"] = end - start
            report["provenance"]["duration"] = "/info/started,/info/ended"

    resources = {}

    def add_resource(kind, value, operation, source):
        value = _text(value)
        if not value:
            return
        coverage[{"file": "files", "registry": "registry", "network": "network"}.get(
            kind, "other_resources")] = True
        key = kind, value
        if key not in resources:
            resources[key] = {"kind": kind, "value": value, "operations": [],
                              "sources": [], "sources_truncated": False}
        resource = resources[key]
        if operation not in resource["operations"]:
            resource["operations"].append(operation)
        if source not in resource["sources"]:
            if len(resource["sources"]) < 32 or source.startswith("/behavior/summary/"):
                resource["sources"].append(source)
            else:
                resource["sources_truncated"] = True

    by_pid = {}

    def process_for(raw, source):
        pid = _identifier(raw.get("process_id")) or _identifier(raw.get("pid"))
        process = by_pid.get(pid) if pid else None
        if process is None:
            process = {"pid": pid, "ppid": "", "name": "", "command_line": "",
                       "first_seen": None, "events": [], "api_counts": {},
                       "api_count_source": "none", "trace_event_count": 0,
                       "trace_occurrences": 0, "trace_truncated": False,
                       "calls_available": False, "apistats_available": False,
                       "apistats": {}, "sources": []}
            report["processes"].append(process)
            if pid:
                by_pid[pid] = process
        environ = _mapping(raw.get("environ"))
        for key, value in (("ppid", _identifier(raw.get("parent_id")) or _identifier(raw.get("ppid"))),
                           ("name", _text(raw.get("process_name")) or _text(raw.get("name"))),
                           ("command_line", _text(raw.get("command_line")) or
                            _text(raw.get("commandline")) or _text(environ.get("CommandLine")))):
            if not process[key] and value:
                process[key] = value
        if process["first_seen"] is None:
            process["first_seen"] = _timestamp(raw.get("first_seen"))[0]
        if source not in process["sources"]:
            process["sources"].append(source)
        return process

    raw_processes = behavior.get("processes")
    coverage["processes"] = isinstance(raw_processes, list) or isinstance(behavior.get("processtree"), list)
    retained = 0
    for index, raw in enumerate(raw_processes if isinstance(raw_processes, list) else []):
        if not isinstance(raw, dict):
            continue
        source = f"/behavior/processes/{index}"
        process = process_for(raw, source)
        if isinstance(raw.get("apistats"), dict):
            process["apistats_available"] = coverage["apistats"] = True
            process["apistats"].update(_counts(raw["apistats"]))
        calls = raw.get("calls")
        if not isinstance(calls, list):
            continue
        process["calls_available"] = coverage["calls"] = True
        for event_index, call in enumerate(calls):
            if not isinstance(call, dict) or not canonical_api(call.get("api")):
                continue
            api = canonical_api(call["api"])
            raw_repeated = call.get("repeated")
            repeated = raw_repeated if isinstance(raw_repeated, int) and not isinstance(raw_repeated, bool) else _number(raw_repeated)
            repeat = 1 + min(MAX_REPEAT - 1, max(0, int(repeated))) if repeated is not None else 1
            timestamp, domain = _timestamp(call.get("timestamp", call.get("time")))
            coverage["timestamps"] |= timestamp is not None
            coverage["arguments"] |= isinstance(call.get("arguments"), (list, dict))
            arguments = _arguments(call.get("arguments"))
            category = _text(call.get("category")).lower()
            event_source = f"{source}/calls/{event_index}"
            event_resources = _event_resources(api, category, arguments)
            for resource in event_resources:
                add_resource(resource["kind"], resource["value"], resource["operation"], event_source)
            status = call.get("status")
            if not isinstance(status, bool):
                status_text = _text(status).lower()
                status = True if status_text in ("true", "success", "1") or status == 1 else (
                    False if status_text in ("false", "failure", "failed", "0") or status == 0 else None)
            process["trace_event_count"] += 1
            process["trace_occurrences"] += repeat
            process["api_counts"][api] = process["api_counts"].get(api, 0) + repeat
            process["api_count_source"] = "calls"
            if retained < max_events:
                process["events"].append({"api": api, "thread_id": _identifier(call.get("thread_id")) or _identifier(call.get("tid")),
                                          "timestamp": timestamp, "timestamp_domain": domain,
                                          "category": category, "status": status, "repeat": repeat,
                                          "arguments": arguments, "resources": event_resources,
                                          "source": event_source})
                retained += 1
            else:
                report["trace_truncated"] = process["trace_truncated"] = True

    tree = behavior.get("processtree")
    stack = [(node, f"/behavior/processtree/{index}", "")
             for index, node in reversed(list(enumerate(tree if isinstance(tree, list) else [])))]
    while stack:
        node, source, parent_pid = stack.pop()
        if not isinstance(node, dict):
            continue
        process = process_for(node, source)
        if not process["ppid"] and parent_pid:
            process["ppid"] = parent_pid
        children = node.get("children")
        if isinstance(children, list):
            stack.extend((child, f"{source}/children/{index}", process["pid"])
                         for index, child in reversed(list(enumerate(children))))

    apistats = behavior.get("apistats")
    if isinstance(apistats, dict):
        coverage["apistats"] = True
        for pid, counts in apistats.items():
            if not isinstance(counts, dict):
                continue
            coverage["processes"] = True
            process = process_for({"pid": pid}, "/behavior/apistats/" + str(pid))
            process["apistats_available"] = True
            process["apistats"].update(_counts(counts))
    for process in report["processes"]:
        if not process["trace_event_count"] and process["apistats_available"]:
            process["api_counts"] = dict(process["apistats"])
            process["api_count_source"] = "apistats"

    for field, (kind, operation) in _SUMMARY_FIELDS.items():
        values = summary.get(field)
        if not isinstance(values, (list, str)):
            continue
        coverage[{"file": "files", "registry": "registry"}.get(kind, "other_resources")] = True
        for value in _strings(values):
            add_resource(kind, value, operation, "/behavior/summary/" + field)
    resolved = summary.get("resolved_apis")
    if isinstance(resolved, (list, str)):
        report["resolved_apis"] = list(dict.fromkeys(canonical_api(value) for value in _strings(resolved)
                                                     if canonical_api(value)))
        report["provenance"]["resolved_apis"].append("/behavior/summary/resolved_apis")

    static = _mapping(document.get("static"))
    for container, source in ((target_file, "/target/file"), (static, "/static"), (document, "")):
        pe = _mapping(container.get("pe"))
        for values, location in ((pe.get("imports"), source + "/pe/imports"),
                                 (container.get("pe_imports"), source + "/pe_imports"),
                                 (container.get("imports"), source + "/imports")):
            if isinstance(values, (list, dict)):
                report["imports"].extend(_import_names(values))
                report["provenance"]["imports"].append(location)
    report["imports"] = list(dict.fromkeys(report["imports"]))

    network = document.get("network")
    coverage["network"] |= isinstance(network, dict)
    network = _mapping(network)
    for protocol in ("dns", "http", "https", "tcp", "udp", "icmp", "hosts", "domains"):
        field = protocol if protocol in network else protocol + "_ex"
        entries = network.get(field)
        entries = entries if isinstance(entries, list) else []
        report["network_counts"][protocol] = sum(isinstance(item, (dict, str)) for item in entries)
        for index, entry in enumerate(entries):
            source = f"/network/{field}/{index}"
            if isinstance(entry, str):
                add_resource("network", entry, "accesses", source)
            elif isinstance(entry, dict):
                operation = "connects_to" if protocol in ("tcp", "udp", "http", "https") else "accesses"
                for key in ("dst", "dstip", "host", "hostname", "ip", "domain", "uri", "url", "request"):
                    for value in _strings(entry.get(key)):
                        add_resource("network", value, operation, source)

    payload_sources = [(document.get("dropped"), "/dropped"),
                       (document.get("payloads"), "/payloads")]
    for key in ("CAPE", "cape"):
        payload_sources.append((_mapping(document.get(key)).get("payloads"), f"/{key}/payloads"))
    for values, source in payload_sources:
        if not isinstance(values, list):
            continue
        coverage["payloads"] = True
        for index, payload in enumerate(values):
            if not isinstance(payload, dict):
                continue
            names, paths = _strings(payload.get("name")), _strings(payload.get("path"))
            raw_pids = payload.get("process_id", payload.get("pid", payload.get("pids")))
            pids = [_identifier(pid) for pid in (raw_pids if isinstance(raw_pids, list) else [raw_pids])]
            pids = list(dict.fromkeys(pid for pid in pids if pid))
            size = _number(payload.get("size"))
            size_available = size is not None and size >= 0
            is_pe = payload.get("is_pe")
            if not isinstance(is_pe, bool):
                is_pe = isinstance(payload.get("pe"), dict) or bool(
                    re.search(r"(?i)\bPE(?:32|64)?\b|portable executable", _text(payload.get("type"))))
            report["payloads"].append({"sha256": _text(payload.get("sha256")).lower(),
                                       "name": next(iter(names or paths), ""),
                                       "pid": next(iter(pids), ""), "size": size if size_available else 0,
                                       "is_pe": is_pe, "names": names, "paths": paths, "pids": pids,
                                       "size_available": size_available, "sources": [f"{source}/{index}"]})
    report["resources"] = list(resources.values())
    return report
