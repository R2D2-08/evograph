#!/usr/bin/env python3
"""Versioned EMBFV and heterogeneous EvoGraph experiments. Never executes binaries."""

import argparse
import json
from pathlib import Path


def read_report(path, max_events):
    """File/JSON failures are not equivalent to a valid empty telemetry report."""
    from multimodal.cape import parse_report
    from multimodal.static import _reject_constant, _sha256

    with Path(path).open(encoding="utf-8-sig") as stream:
        document = json.load(stream, parse_constant=_reject_constant)
    if not isinstance(document, dict):
        raise ValueError("CAPE report must be a JSON object")
    report = parse_report(document, max_events)
    report["sha256"] = _sha256(report["sha256"], "report.sha256")
    return report


def prepare(args):
    from multimodal.data import prepare_dataset

    records = prepare_dataset(args.metadata, args.out, args.max_events, args.payload_static_dir)
    print(json.dumps({"records": len(records), "missing_static": sum(
        record["static_features"] is None for record in records), "dataset": str(Path(args.out).resolve())}, indent=2))


def extract_static(args):
    from multimodal.static import extract_ember

    features = extract_ember(args.binary)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(features, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": features["sha256"], "dimensions": len(features["features"]), "out": str(output)}))


def inspect(args):
    from multimodal.graph import build_graph
    from multimodal.statistics import STATS_NAMES, behavioral_statistics

    report = read_report(args.report, args.max_events)
    graph = build_graph(report, args.window_size, args.max_windows)
    print(json.dumps({"sha256": report["sha256"], "coverage": report["coverage"],
                      "validation_warnings": report["validation_warnings"],
                      "nodes": {kind: graph[kind].num_nodes for kind in graph.node_types},
                      "edges": {"/".join(relation): graph[relation].edge_index.size(1)
                                for relation in graph.edge_types if graph[relation].edge_index.size(1)},
                      "limits": graph.graph_metadata,
                      "statistics": dict(zip(STATS_NAMES, behavioral_statistics(report).tolist()))}, indent=2))


def train(args):
    from multimodal.training import run_experiment

    run_experiment(args)


def predict(args):
    import torch
    from multimodal.static import _sha256, extract_ember, imports_from_raw, load_static
    from multimodal.statistics import behavioral_statistics
    from multimodal.training import load_checkpoint

    torch.set_num_threads(4)
    model, scaler, checkpoint = load_checkpoint(args.checkpoint)
    report = read_report(args.report, checkpoint["max_events"])
    static = load_static(args.static, report["sha256"]) if args.static else None
    if args.binary:
        static = extract_ember(args.binary)
        if report["sha256"] and static["sha256"] != report["sha256"]:
            raise ValueError("Binary/report SHA-256 identity mismatch")
    if static is None and model.mode != "dynamic" and not checkpoint["allow_missing_static"]:
        raise ValueError("This checkpoint requires EMBER features; supply --static or --binary")
    if static:
        report["imports"] = list(dict.fromkeys(report["imports"] + imports_from_raw(static["raw"])))
    if args.payload_static_dir:
        for payload in report["payloads"]:
            if payload.get("is_pe") and payload.get("sha256"):
                digest = _sha256(payload["sha256"], "payload.sha256", required=True)
                sidecar = Path(args.payload_static_dir) / (digest + ".json")
                if sidecar.is_file():
                    payload["static_features"] = load_static(sidecar, digest)["features"]
    sample = scaler.transform({"report": report, "static_features": static["features"] if static else None,
                               "statistics": behavioral_statistics(report).tolist()})
    with torch.inference_mode():
        probability = float(model([sample]).sigmoid().item())
    graph_limits = None
    if model.mode in ("graph", "hybrid"):
        from multimodal.graph import build_graph

        graph_limits = build_graph(report, model.window_size, model.max_windows).graph_metadata
    print(json.dumps({"malware_probability": probability, "threshold": checkpoint["threshold"],
                      "predicted_label": "malware" if probability >= checkpoint["threshold"] else "benign",
                      "static_available": static is not None, "coverage": report["coverage"],
                      "trace_truncated": report["trace_truncated"],
                      "graph_limits": graph_limits, "validation_warnings": report["validation_warnings"],
                      "warning": "Research prototype; not a production safety verdict."}, indent=2))


def export(args):
    import torch
    from multimodal.training import export_features

    torch.set_num_threads(4)
    count = export_features(args.checkpoint, args.dataset, args.out)
    print(json.dumps({"records": count, "dimensions": 4429, "out": args.out}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("extract-static", help="Read bytes using official optional EMBER v2")
    command.add_argument("--binary", required=True)
    command.add_argument("--out", required=True)
    command.set_defaults(func=extract_static)
    command = commands.add_parser("prepare", help="Preserve normalized evidence and raw numerical features")
    command.add_argument("--metadata", required=True)
    command.add_argument("--out", default="prepared_multimodal/dataset.jsonl")
    command.add_argument("--max-events", type=int, default=20000)
    command.add_argument("--payload-static-dir")
    command.set_defaults(func=prepare)
    command = commands.add_parser("inspect")
    command.add_argument("--report", required=True)
    command.add_argument("--max-events", type=int, default=20000)
    command.add_argument("--window-size", type=int, default=32)
    command.add_argument("--max-windows", type=int, default=256)
    command.set_defaults(func=inspect)
    command = commands.add_parser("train", help="Joint encoder training on shared 60/20/20 partitions")
    command.add_argument("--dataset", default="prepared_multimodal/dataset.jsonl")
    command.add_argument("--out", default="results_multimodal")
    command.add_argument("--models", nargs="+", choices=["baseline", "hybrid", "graph", "static", "dynamic"],
                         default=["baseline", "hybrid"])
    command.add_argument("--split-mode", choices=["stratified", "chronological", "family"], default="stratified")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--epochs", type=int, default=24)
    command.add_argument("--patience", type=int, default=8)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--hidden", type=int, default=256)
    command.add_argument("--dropout", type=float, default=.2)
    command.add_argument("--lr", type=float, default=.001)
    command.add_argument("--window-size", type=int, default=32)
    command.add_argument("--max-windows", type=int, default=256)
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--allow-missing-static", action="store_true")
    command.set_defaults(func=train)
    command = commands.add_parser("predict")
    command.add_argument("--report", required=True)
    command.add_argument("--checkpoint", required=True)
    static = command.add_mutually_exclusive_group()
    static.add_argument("--static", help="Precomputed EMBER v2 JSON")
    static.add_argument("--binary", help="Read-only official EMBER extraction")
    command.add_argument("--payload-static-dir")
    command.set_defaults(func=predict)
    command = commands.add_parser("export-features", help="Export 4429 features using a fitted encoder checkpoint")
    command.add_argument("--dataset", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--out", required=True)
    command.set_defaults(func=export)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
