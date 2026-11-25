"""Shared-split experiments, train-only preprocessing, and trusted checkpoints."""

import csv
import hashlib
import json
import random
import platform
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch import nn

from .data import FeatureScaler, load_dataset
from .models import EvoGraphClassifier
from .schema import BLOCK_DIMS, SCHEMA_VERSION
from .splits import make_splits


def probabilities(model, records, batch_size=8):
    model.eval()
    with torch.inference_mode():
        return np.asarray([
            value for start in range(0, len(records), batch_size)
            for value in model(records[start:start + batch_size]).sigmoid().cpu().tolist()
        ])


def select_threshold(labels, values):
    """Use validation only; prefer 0.5 when F1 scores tie."""
    candidates = np.unique(np.r_[0.5, np.arange(.05, .951, .05)])
    return float(max(candidates, key=lambda threshold: (
        f1_score(labels, values >= threshold, zero_division=0), -abs(threshold - .5))))


def evaluate(labels, values, threshold):
    predictions = values >= threshold
    return {
        "n": len(labels), "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def fit_model(model, train, validation, *, epochs=24, batch_size=8, lr=.001,
              patience=8, seed=42):
    """Jointly fit all encoders; early stop on sample-weighted validation loss."""
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, patience and lr must be positive")
    rng = random.Random(seed)
    device = next(model.parameters()).device
    positives = sum(record["label"] for record in train)
    if not 0 < positives < len(train):
        raise ValueError("Training needs both labels")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        (len(train) - positives) / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        train_loss = 0.
        for start in range(0, len(order), batch_size):
            batch = [train[index] for index in order[start:start + batch_size]]
            labels = torch.tensor([record["label"] for record in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; check numerical features")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            train_loss += loss.item() * len(batch)
        model.eval()
        validation_loss = 0.
        with torch.inference_mode():
            for start in range(0, len(validation), batch_size):
                batch = validation[start:start + batch_size]
                labels = torch.tensor([record["label"] for record in batch], dtype=torch.float32, device=device)
                validation_loss += criterion(model(batch), labels).item() * len(batch)
        validation_loss /= len(validation)
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train),
                        "validation_loss": validation_loss})
        if validation_loss < best_loss - .0001:
            best_loss, stale = validation_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def load_checkpoint(path):
    """Only load checkpoints from trusted sources; no custom pickle classes."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    model = EvoGraphClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaler = FeatureScaler.from_state_dict(checkpoint["scaler"])
    return model, scaler, checkpoint


def run_experiment(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    records = load_dataset(args.dataset)
    groups = make_splits(records, args.split_mode, args.seed)
    modes = list(dict.fromkeys(args.models))
    missing_static = sum(record["static_features"] is None for record in records)
    if "static" in modes and missing_static == len(records):
        raise ValueError("Cannot run a static-only experiment without any EMBER features")
    if missing_static and any(mode != "dynamic" for mode in modes) and not args.allow_missing_static:
        raise ValueError("EMBER features are missing. Supply static_path/binary_path in metadata, or use "
                         "--allow-missing-static for an explicitly incomplete-modality experiment.")
    limits = {record["extraction"]["max_events"] for record in records}
    if len(limits) != 1:
        raise ValueError("All records must use the same max_events extraction configuration")
    if isinstance(next(iter(limits)), bool) or not isinstance(next(iter(limits)), int) or min(limits) < 0:
        raise ValueError("Invalid max_events extraction configuration")
    scaler = FeatureScaler().fit([records[index] for index in groups["train"]])
    if "static" in modes and scaler.state_dict()["static_count"] == 0:
        raise ValueError("Static-only training partition contains no EMBER features")
    partitions = {name: [scaler.transform(records[index]) for index in indices]
                  for name, indices in groups.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "sha256", "label", "observed_at", "family", "group", "split"])
        for name, indices in groups.items():
            for index in indices:
                row = records[index]
                writer.writerow([row[key] for key in ("sample_id", "sha256", "label", "observed_at", "family", "group")] + [name])
    manifest = Path(args.dataset).with_suffix(".manifest.json")
    provenance = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    result = {
        "schema_version": SCHEMA_VERSION, "seed": args.seed, "split_mode": args.split_mode,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "sample_count": len(records), "missing_static": missing_static,
        "split_sizes": {name: len(indices) for name, indices in groups.items()},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__},
        "models": {},
        "limitations": ["Research prototype, not a production safety verdict.",
                        "Random splits do not establish temporal or unseen-family generalization.",
                        "CAPE execution time is not sample first-seen time.",
                        "Extraction and graph window limits can truncate telemetry.",
                        "Check labels and source/sandbox confounding before reporting research results."],
    }
    if missing_static:
        result["limitations"].append("Missing EMBER vectors: these are incomplete-modality results, not a full static-dynamic evaluation.")
    predictions = {record["sample_id"]: {"sample_id": record["sample_id"], "label": record["label"]}
                   for record in partitions["test"]}
    for mode in modes:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        config = {"mode": mode, "hidden": args.hidden, "dropout": args.dropout,
                  "window_size": args.window_size, "max_windows": args.max_windows}
        model = EvoGraphClassifier(**config)
        history = fit_model(model, partitions["train"], partitions["validation"], epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, patience=args.patience, seed=args.seed)
        threshold = select_threshold([record["label"] for record in partitions["validation"]],
                                     probabilities(model, partitions["validation"], args.batch_size))
        values = probabilities(model, partitions["test"], args.batch_size)
        score = evaluate([record["label"] for record in partitions["test"]], values, threshold)
        result["models"][mode] = {**score, "history": history}
        checkpoint = {
            "schema_version": SCHEMA_VERSION, "model_config": config,
            "state_dict": model.state_dict(), "scaler": scaler.state_dict(), "threshold": threshold,
            "max_events": next(iter(limits)), "allow_missing_static": args.allow_missing_static,
            "seed": args.seed, "split_mode": args.split_mode,
            "dataset_sha256": result["dataset_sha256"], "preparation": provenance,
        }
        torch.save(checkpoint, out / (mode + ".pt"))
        for record, probability in zip(partitions["test"], values):
            predictions[record["sample_id"]][mode + "_probability"] = float(probability)
        print(json.dumps({"model": mode, **score}), flush=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "label"] + [mode + "_probability" for mode in modes])
        writer.writeheader()
        writer.writerows(predictions.values())
    return result


def export_features(checkpoint_path, dataset_path, output_path):
    """Export embeddings using a fitted baseline/dynamic checkpoint, never random encoders."""
    model, scaler, checkpoint = load_checkpoint(checkpoint_path)
    if model.mode not in ("baseline", "dynamic"):
        raise ValueError("Feature export needs a fitted baseline or dynamic checkpoint")
    records = load_dataset(dataset_path)
    if any(record["extraction"]["max_events"] != checkpoint["max_events"]
           for record in records):
        raise ValueError("Extraction max_events differs from checkpoint; reprepare with the checkpoint's limit")
    if Path(output_path).suffix != ".npz":
        raise ValueError("Feature export output must end in .npz")
    static, dynamic = [], []
    model.eval()
    with torch.inference_mode():
        for record in records:
            scaled = scaler.transform(record)
            dynamic.append(model.dynamic(scaled["report"], scaled["statistics"]).cpu().numpy())
            static.append(scaled["static_features"] if scaled["static_features"] is not None else np.zeros(2381))
    if not records:
        raise ValueError("Cannot export an empty dataset")
    static_array, dynamic_array = np.asarray(static, dtype=np.float32), np.asarray(dynamic, dtype=np.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, static_features=static_array, dynamic_features=dynamic_array,
                        features=np.concatenate((static_array, dynamic_array), axis=1),
                        static_available=np.asarray([record["static_features"] is not None for record in records]),
                        sample_id=np.asarray([record["sample_id"] for record in records]),
                        sha256=np.asarray([record["sha256"] for record in records]),
                        label=np.asarray([record["label"] for record in records]),
                        schema_version=np.asarray(SCHEMA_VERSION),
                        dynamic_blocks=np.asarray(list(BLOCK_DIMS)))
    return len(records)
