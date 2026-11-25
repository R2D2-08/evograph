"""Leakage-resistant 60/20/20 index splits; no timestamps/groups are inferred."""

import math

from .data import _parse_observed_at
from .static import _sha256


def make_splits(records, mode="stratified", seed=42):
    """Return train/validation/test original indices, each with both labels.

    Rounded-up held-out sample counts have a floor of two, the mathematical
    requirement for both labels, not an arbitrary dataset minimum. Family mode
    searches at most 256 deterministic GroupShuffleSplit candidates, selecting
    the feasible candidate closest to sample-level 60/20/20 proportions.
    Chronological ties within a partition are allowed, never across boundaries.
    """
    if mode not in ("stratified", "chronological", "family"):
        raise ValueError(f"Unknown split mode: {mode}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    records = list(records)
    labels, seen_hashes, seen_ids = [], set(), set()
    for record in records:
        if not isinstance(record, dict) or type(record.get("label")) is not int or record["label"] not in (0, 1):
            raise ValueError("Every split record requires a binary integer label")
        labels.append(record["label"])
        identity = _sha256(record.get("sha256")) or _sha256(record.get("report_sha256"), "report_sha256")
        if identity:
            if identity in seen_hashes:
                raise ValueError("Duplicate sample/report SHA-256 before splitting: " + identity)
            seen_hashes.add(identity)
        sample_id = record.get("sample_id")
        if sample_id:
            if sample_id in seen_ids:
                raise ValueError("Duplicate sample_id before splitting: " + str(sample_id))
            seen_ids.add(sample_id)
    n = len(records)
    if any(labels.count(label) < 3 for label in (0, 1)):
        raise ValueError("Both labels are required in all three splits; need at least three samples per label")
    heldout = max(2, math.ceil(n * 0.2))
    if mode == "chronological":
        stamps = [_parse_observed_at(record.get("observed_at")) for record in records]
        if any(stamp is None for stamp in stamps):
            raise ValueError("Chronological split requires a genuine observed_at datetime for every record")
        order = sorted(range(n), key=lambda index: stamps[index])
        first, second = n - 2 * heldout, n - heldout
        if stamps[order[first - 1]] >= stamps[order[first]] or stamps[order[second - 1]] >= stamps[order[second]]:
            raise ValueError("Chronological split boundary has tied/overlapping UTC observation times")
        result = {"train": order[:first], "validation": order[first:second], "test": order[second:]}
    elif mode == "stratified":
        from sklearn.model_selection import train_test_split

        try:
            # Resolve sklearn's minority-class allocation ties without relaxing stratification.
            for attempt in range(32):
                development, test = train_test_split(list(range(n)), test_size=heldout,
                                                      random_state=(seed + 2 * attempt) % 2**32, stratify=labels)
                train, validation = train_test_split(development, test_size=heldout,
                                                      random_state=(seed + 2 * attempt + 1) % 2**32,
                                                      stratify=[labels[index] for index in development])
                if all({labels[index] for index in partition} == {0, 1}
                       for partition in (train, validation, test)):
                    break
        except ValueError as exc:
            raise ValueError(f"Cannot make stratified 60/20/20 splits with both labels: {exc}") from exc
        result = {"train": train, "validation": validation, "test": test}
    else:
        groups = []
        for record in records:
            group = record.get("group")
            if isinstance(group, str):
                group = group.strip()
            group = group or record.get("family")
            if not isinstance(group, str) or not group.strip():
                raise ValueError("Family split requires explicit group or family for EVERY record, including benign applications")
            groups.append(group.strip())
        group_count = len(set(groups))
        if group_count < 3:
            raise ValueError("Family split requires at least three distinct explicit groups")
        from sklearn.model_selection import GroupShuffleSplit

        best, best_score = None, float("inf")
        test_count = math.ceil(group_count * 0.2)
        validation_count = math.ceil((group_count - test_count) * 0.25)
        # Single-class groups may require two groups per held-out partition.
        sizes = list(dict.fromkeys((test_size, validation_size)
                     for test_size in (test_count, max(2, test_count))
                     for validation_size in (validation_count, max(2, validation_count))
                     if test_size + validation_size < group_count))
        for attempt in range(256):
            test_size, validation_size = sizes[attempt % len(sizes)]
            try:
                outer = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                          random_state=(seed + 2 * attempt) % 2**32)
                development, test = next(outer.split(range(n), labels, groups))
                inner = GroupShuffleSplit(n_splits=1, test_size=validation_size,
                                          random_state=(seed + 2 * attempt + 1) % 2**32)
                train_local, validation_local = next(inner.split(
                    development, [labels[index] for index in development], [groups[index] for index in development]))
            except ValueError:
                continue
            candidate = {"train": development[train_local].tolist(),
                         "validation": development[validation_local].tolist(), "test": test.tolist()}
            if any({labels[index] for index in indices} != {0, 1} for indices in candidate.values()):
                continue
            score = sum(abs(len(candidate[name]) / n - fraction)
                        for name, fraction in (("train", 0.6), ("validation", 0.2), ("test", 0.2)))
            if score < best_score:
                best, best_score = candidate, score
                if score < 1e-12:
                    break
        if best is None:
            raise ValueError("No label-feasible disjoint family split found in 256 attempts; supply more explicit groups per label")
        result = best
    for name, indices in result.items():
        if {labels[index] for index in indices} != {0, 1}:
            raise ValueError(f"{name} requires both labels; collect more samples or use a different split mode")
    return result
