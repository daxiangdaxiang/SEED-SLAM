#!/usr/bin/env python3
"""Analyze feedback-safe DBA variants with edge, partition, and frame metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Iterable, Mapping

import numpy as np


CopyKey = tuple[int, int]


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _selected_directed_edges(
    rows: Iterable[Mapping[str, object]],
    minimum_weight: float = 0.5,
    evaluated_pairs: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int, int, int]]:
    selected = set()
    for row in rows:
        receiver_robot = int(row["receiver_robot"])
        sender_robot = int(row["sender_robot"])
        if (
            evaluated_pairs is not None
            and (receiver_robot, sender_robot) not in evaluated_pairs
        ):
            continue
        if float(row.get("weight", 1.0)) < minimum_weight:
            continue
        selected.add(
            (
                receiver_robot,
                sender_robot,
                int(row["receiver_object"]),
                int(row["sender_object"]),
            )
        )
    return selected


def _prf(
    selected: set[object], truth: set[object]
) -> dict[str, float | int]:
    true_positive = len(selected & truth)
    false_positive = len(selected - truth)
    false_negative = len(truth - selected)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def directed_precision_recall_f1(
    estimate_rows: Iterable[Mapping[str, object]],
    truth_rows: Iterable[Mapping[str, object]],
    *,
    evaluated_pairs: set[tuple[int, int]] | None = None,
    minimum_weight: float = 0.5,
) -> dict[str, float | int]:
    return _prf(
        _selected_directed_edges(
            estimate_rows, minimum_weight, evaluated_pairs
        ),
        _selected_directed_edges(
            truth_rows, minimum_weight, evaluated_pairs
        ),
    )


def _relation_edges(
    rows: Iterable[Mapping[str, object]],
    minimum_weight: float = 0.5,
    evaluated_pairs: set[tuple[int, int]] | None = None,
) -> set[tuple[CopyKey, CopyKey]]:
    result = set()
    for edge in _selected_directed_edges(
        rows, minimum_weight, evaluated_pairs
    ):
        first = (edge[0], edge[2])
        second = (edge[1], edge[3])
        if first != second:
            result.add(tuple(sorted((first, second))))
    return result


def _components(
    rows: Iterable[Mapping[str, object]],
    support: Iterable[CopyKey],
    minimum_weight: float = 0.5,
    evaluated_pairs: set[tuple[int, int]] | None = None,
) -> tuple[tuple[CopyKey, ...], ...]:
    nodes = set((int(value[0]), int(value[1])) for value in support)
    edges = _relation_edges(rows, minimum_weight, evaluated_pairs)
    nodes.update(node for edge in edges for node in edge)
    parent = {node: node for node in nodes}

    def find(node: CopyKey) -> CopyKey:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    for first, second in sorted(edges):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            low, high = sorted((first_root, second_root))
            parent[high] = low
    grouped: dict[CopyKey, list[CopyKey]] = {}
    for node in sorted(nodes):
        grouped.setdefault(find(node), []).append(node)
    return tuple(
        sorted(tuple(sorted(component)) for component in grouped.values())
    )


def _component_pairs(
    components: Iterable[Iterable[CopyKey]],
) -> set[tuple[CopyKey, CopyKey]]:
    pairs = set()
    for supplied in components:
        component = tuple(supplied)
        for index, first in enumerate(component):
            for second in component[index + 1 :]:
                if first[0] != second[0]:
                    pairs.add(tuple(sorted((first, second))))
    return pairs


def all_pairs_precision_recall_f1(
    estimate_rows: Iterable[Mapping[str, object]],
    truth_rows: Iterable[Mapping[str, object]],
    support: Iterable[CopyKey],
    *,
    minimum_weight: float = 0.5,
) -> dict[str, float | int]:
    return _prf(
        _component_pairs(
            _components(estimate_rows, support, minimum_weight)
        ),
        _component_pairs(_components(truth_rows, support, minimum_weight)),
    )


def bcubed_precision_recall_f1(
    estimate_rows: Iterable[Mapping[str, object]],
    truth_rows: Iterable[Mapping[str, object]],
    support: Iterable[CopyKey],
    *,
    minimum_weight: float = 0.5,
) -> dict[str, float]:
    support = tuple(sorted(set(support)))
    estimate_components = _components(
        estimate_rows, support, minimum_weight
    )
    truth_components = _components(truth_rows, support, minimum_weight)
    estimate_by_copy = {
        copy: set(component)
        for component in estimate_components
        for copy in component
    }
    truth_by_copy = {
        copy: set(component)
        for component in truth_components
        for copy in component
    }
    if not support:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = statistics.fmean(
        len(estimate_by_copy[copy] & truth_by_copy[copy])
        / len(estimate_by_copy[copy])
        for copy in support
    )
    recall = statistics.fmean(
        len(estimate_by_copy[copy] & truth_by_copy[copy])
        / len(truth_by_copy[copy])
        for copy in support
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def duplicate_robot_component_count(
    rows: Iterable[Mapping[str, object]],
    support: Iterable[CopyKey],
    *,
    minimum_weight: float = 0.5,
) -> int:
    return sum(
        len(component) != len({copy[0] for copy in component})
        for component in _components(rows, support, minimum_weight)
    )


def _compose(
    first: tuple[float, np.ndarray],
    second: tuple[float, np.ndarray],
) -> tuple[float, np.ndarray]:
    yaw = first[0] + second[0]
    rotation = np.array(
        [
            [np.cos(first[0]), -np.sin(first[0])],
            [np.sin(first[0]), np.cos(first[0])],
        ]
    )
    return yaw, rotation @ second[1] + first[1]


def _inverse(
    transform: tuple[float, np.ndarray],
) -> tuple[float, np.ndarray]:
    yaw, translation = transform
    inverse_yaw = -yaw
    rotation = np.array(
        [
            [np.cos(inverse_yaw), -np.sin(inverse_yaw)],
            [np.sin(inverse_yaw), np.cos(inverse_yaw)],
        ]
    )
    return inverse_yaw, -(rotation @ translation)


def _wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def shared_transform_error(
    estimate: Mapping[int, tuple[float, Iterable[float]]],
    truth: Mapping[int, tuple[float, Iterable[float]]],
    *,
    root_robot: int,
) -> dict[str, float]:
    if set(estimate) != set(truth) or root_robot not in estimate:
        raise ValueError("transform maps must share support and contain the root")
    estimated = {
        int(robot): (
            float(value[0]),
            np.asarray(value[1], dtype=float).reshape(2),
        )
        for robot, value in estimate.items()
    }
    expected = {
        int(robot): (
            float(value[0]),
            np.asarray(value[1], dtype=float).reshape(2),
        )
        for robot, value in truth.items()
    }
    gauge = _compose(
        expected[root_robot], _inverse(estimated[root_robot])
    )
    translation_errors = []
    rotation_errors = []
    for robot in sorted(estimated):
        aligned = _compose(gauge, estimated[robot])
        translation_errors.append(
            float(np.linalg.norm(aligned[1] - expected[robot][1]))
        )
        rotation_errors.append(
            np.degrees(_wrap_angle(aligned[0] - expected[robot][0]))
        )
    return {
        "translation_rmse": float(
            np.sqrt(np.mean(np.square(translation_errors)))
        ),
        "rotation_rmse_deg": float(
            np.sqrt(np.mean(np.square(rotation_errors)))
        ),
    }


def _read_transforms(path: Path) -> dict[int, tuple[float, tuple[float, float]]]:
    result = {}
    for row in _read_csv(path):
        result[int(row["robot"])] = (
            float(row["yaw"]),
            (
                float(row["translation_x"]),
                float(row["translation_y"]),
            ),
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case_result(case: Mapping[str, object]) -> dict[str, object]:
    case_dir = Path(str(case["output_dir"]))
    summary = json.loads(
        (case_dir / "summary.json").read_text(encoding="utf-8")
    )
    estimate_path = Path(str(summary["final_association"]))
    if not estimate_path.is_absolute():
        working_directory_path = estimate_path.resolve()
        estimate_path = (
            working_directory_path
            if working_directory_path.is_file()
            else case_dir / estimate_path
        )
    estimate = _read_csv(estimate_path)
    truth = _read_csv(Path(str(case["ground_truth"])))
    support = tuple(tuple(map(int, value)) for value in case["support"])
    topology_rows = _read_csv(Path(str(case["topology"])))
    topology = {
        tuple(
            sorted(
                (
                    int(row.get("first_robot", row.get("robot_i", 0))),
                    int(row.get("second_robot", row.get("robot_j", 0))),
                )
            )
        )
        for row in topology_rows
    }
    directed_pairs = {
        directed
        for first, second in topology
        for directed in ((first, second), (second, first))
    }
    topology_metrics = directed_precision_recall_f1(
        estimate, truth, evaluated_pairs=directed_pairs
    )
    pair_metrics = all_pairs_precision_recall_f1(
        estimate, truth, support
    )
    bcubed = bcubed_precision_recall_f1(estimate, truth, support)
    row = {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "initializer": case["initializer"],
        "variant": case["variant"],
        "status": summary.get("status", ""),
        "topology_precision": topology_metrics["precision"],
        "topology_recall": topology_metrics["recall"],
        "topology_f1": topology_metrics["f1"],
        "all_pairs_precision": pair_metrics["precision"],
        "all_pairs_recall": pair_metrics["recall"],
        "all_pairs_f1": pair_metrics["f1"],
        "bcubed_precision": bcubed["precision"],
        "bcubed_recall": bcubed["recall"],
        "bcubed_f1": bcubed["f1"],
        "duplicate_robot_components": duplicate_robot_component_count(
            estimate, support
        ),
        "total_communication_mb": summary.get(
            "total_communication_mb", ""
        ),
        "transform_translation_rmse": "",
        "transform_rotation_rmse_deg": "",
    }
    truth_transforms = case.get("truth_transforms")
    if truth_transforms:
        estimate_transform_path = case_dir / "final_transforms.csv"
        if not estimate_transform_path.is_file():
            epoch_dirs = sorted(case_dir.glob("epoch_*/upper/transforms.csv"))
            if epoch_dirs:
                estimate_transform_path = epoch_dirs[-1]
        if estimate_transform_path.is_file():
            error = shared_transform_error(
                _read_transforms(estimate_transform_path),
                _read_transforms(Path(str(truth_transforms))),
                root_robot=int(case.get("root_robot", 0)),
            )
            row["transform_translation_rmse"] = error[
                "translation_rmse"
            ]
            row["transform_rotation_rmse_deg"] = error[
                "rotation_rmse_deg"
            ]
    return row


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["dataset"]), str(row["variant"])), []
        ).append(row)
    result = []
    metric_names = (
        "topology_f1",
        "all_pairs_f1",
        "bcubed_f1",
        "duplicate_robot_components",
        "total_communication_mb",
    )
    for (dataset, variant), values in sorted(groups.items()):
        aggregate = {
            "dataset": dataset,
            "variant": variant,
            "case_count": len(values),
        }
        for name in metric_names:
            numeric = [
                float(row[name])
                for row in values
                if row[name] not in ("", None)
            ]
            aggregate[f"mean_{name}"] = (
                statistics.fmean(numeric) if numeric else ""
            )
        result.append(aggregate)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.result_root / "manifest.json").read_text(encoding="utf-8")
    )
    cases = manifest.get("cases", ())
    rows = [_case_result(case) for case in cases]
    aggregates = _aggregate(rows)
    claims = [
        {
            "claim": "held-out network association",
            "variant": row["variant"],
            "case_id": row["case_id"],
            "topology_f1": row["topology_f1"],
            "all_pairs_f1": row["all_pairs_f1"],
            "network_feasible": (
                int(row["duplicate_robot_components"]) == 0
            ),
        }
        for row in rows
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "case_results.csv", rows)
    _write_csv(args.output_dir / "aggregate_results.csv", aggregates)
    _write_csv(args.output_dir / "claim_evidence.csv", claims)
    lines = [
        "# DBA feedback-safe ablation",
        "",
        "| Dataset | Variant | Cases | Topology F1 | All-pairs F1 | B-cubed F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['case_count']} | "
            f"{float(row['mean_topology_f1']):.4f} | "
            f"{float(row['mean_all_pairs_f1']):.4f} | "
            f"{float(row['mean_bcubed_f1']):.4f} |"
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
