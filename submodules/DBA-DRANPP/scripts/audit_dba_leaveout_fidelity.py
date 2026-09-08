#!/usr/bin/env python3
"""Audit finite held-out Lower predictions against explicit nonlinear reruns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Optional

import numpy as np


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _topology(path: Path) -> tuple[tuple[int, int], ...]:
    result = set()
    for row in _read_csv(path):
        if "first_robot" in row:
            first, second = int(row["first_robot"]), int(row["second_robot"])
        elif "robot_i" in row:
            first, second = int(row["robot_i"]), int(row["robot_j"])
        else:
            first = int(row["receiver_robot"])
            second = int(row["sender_robot"])
        if first == second:
            raise ValueError("topology edge must connect different robots")
        result.add(tuple(sorted((first, second))))
    if not result:
        raise ValueError("topology must contain at least one edge")
    return tuple(sorted(result))


def _robot_components(
    topology: Iterable[tuple[int, int]],
    removed_pair: tuple[int, int],
) -> tuple[tuple[int, ...], ...]:
    edges = tuple(topology)
    robots = {robot for edge in edges for robot in edge}
    parent = {robot: robot for robot in robots}

    def find(robot: int) -> int:
        while parent[robot] != robot:
            parent[robot] = parent[parent[robot]]
            robot = parent[robot]
        return robot

    target = tuple(sorted(removed_pair))
    for first, second in edges:
        if tuple(sorted((first, second))) == target:
            continue
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(
                first_root, second_root
            )
    components: dict[int, list[int]] = {}
    for robot in sorted(robots):
        components.setdefault(find(robot), []).append(robot)
    return tuple(tuple(values) for values in components.values())


def remove_pair_relations(
    rows: Iterable[Mapping[str, str]],
    pair: tuple[int, int],
) -> tuple[dict[str, str], ...]:
    target = tuple(sorted(pair))
    return tuple(
        dict(row)
        for row in rows
        if tuple(
            sorted(
                (
                    int(row["receiver_robot"]),
                    int(row["sender_robot"]),
                )
            )
        )
        != target
    )


def _replace_option(
    command: list[str], name: str, value: str
) -> list[str]:
    result = list(command)
    if name in result:
        index = result.index(name)
        if index + 1 >= len(result):
            raise ValueError(f"missing value after {name}")
        result[index + 1] = value
    else:
        result.extend((name, value))
    return result


def build_lower_command(
    executable: Path,
    common_args: Iterable[str],
    association: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    command = [str(Path(executable).resolve()), *map(str, common_args)]
    command = _replace_option(
        command, "--association_file", str(association.resolve())
    )
    command = _replace_option(
        command, "--output_dir", str(output_dir.resolve())
    )
    return tuple(command)


def _read_means(path: Path) -> dict[tuple[int, int], np.ndarray]:
    result = {}
    for row in _read_csv(path):
        if int(row.get("valid", 1)) == 0:
            continue
        coordinates = [
            float(row[name]) for name in ("x", "y", "z") if name in row
        ]
        result[(int(row["robot"]), int(row["object_id"]))] = np.asarray(
            coordinates, dtype=float
        )
    return result


def _rigid_align(
    estimate: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    estimate_center = np.mean(estimate, axis=0)
    reference_center = np.mean(reference, axis=0)
    centered_estimate = estimate - estimate_center
    centered_reference = reference - reference_center
    covariance = centered_estimate.T @ centered_reference
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return centered_estimate @ rotation + reference_center


def compare_means(
    estimate_path: Path,
    reference_path: Path,
    *,
    robot_components: Optional[Iterable[Iterable[int]]] = None,
) -> tuple[float, float, int]:
    estimate = _read_means(estimate_path)
    reference = _read_means(reference_path)
    keys = tuple(sorted(set(estimate) & set(reference)))
    if not keys:
        raise ValueError("mean exports have no common valid object copies")
    components = (
        (tuple(sorted({key[0] for key in keys})),)
        if robot_components is None
        else tuple(tuple(component) for component in robot_components)
    )
    displacement_groups = []
    compared_count = 0
    for component in components:
        component_robots = set(component)
        component_keys = tuple(
            key for key in keys if key[0] in component_robots
        )
        if not component_keys:
            continue
        estimate_values = np.stack(
            [estimate[key] for key in component_keys]
        )
        reference_values = np.stack(
            [reference[key] for key in component_keys]
        )
        aligned = _rigid_align(estimate_values, reference_values)
        displacement_groups.append(
            np.linalg.norm(aligned - reference_values, axis=1)
        )
        compared_count += len(component_keys)
    if not displacement_groups:
        raise ValueError("no robot component contains comparable object copies")
    displacements = np.concatenate(displacement_groups)
    return (
        float(np.mean(displacements)),
        float(np.max(displacements)),
        compared_count,
    )


def _read_covariance(path: Path) -> dict[tuple[int, int, int], float]:
    result = {}
    for row in _read_csv(path):
        if int(row.get("valid", 1)) == 0:
            continue
        key = (
            int(row["robot"]),
            int(row["row_object"]),
            int(row["column_object"]),
        )
        result[key] = float(row["covariance"])
    return result


def covariance_relative_error(
    estimate_path: Path, reference_path: Path
) -> tuple[float, int]:
    estimate = _read_covariance(estimate_path)
    reference = _read_covariance(reference_path)
    keys = tuple(sorted(set(estimate) & set(reference)))
    if not keys:
        return float("nan"), 0
    difference = np.asarray(
        [estimate[key] - reference[key] for key in keys]
    )
    baseline = np.asarray([reference[key] for key in keys])
    denominator = float(np.linalg.norm(baseline))
    value = float(np.linalg.norm(difference))
    return (value / denominator if denominator > 0.0 else value), len(keys)


def edit_sign_agreement(
    estimate_path: Path, reference_path: Path
) -> tuple[float, int]:
    if not estimate_path.is_file() or not reference_path.is_file():
        return float("nan"), 0
    estimate = {
        row["edit_id"]: float(row["delta"]) for row in _read_csv(estimate_path)
    }
    reference = {
        row["edit_id"]: float(row["delta"])
        for row in _read_csv(reference_path)
    }
    keys = tuple(sorted(set(estimate) & set(reference)))
    if not keys:
        return float("nan"), 0

    def sign(value: float) -> int:
        return int(value > 0.0) - int(value < 0.0)

    return (
        sum(sign(estimate[key]) == sign(reference[key]) for key in keys)
        / len(keys),
        len(keys),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-association", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--lower-executable", type=Path, required=True)
    parser.add_argument("--lower-common-args", type=Path, required=True)
    parser.add_argument("--heldout-bundle", type=Path, required=True)
    parser.add_argument("--candidate-support", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (
        args.incumbent_association,
        args.metadata,
        args.topology,
        args.lower_executable,
        args.lower_common_args,
        args.heldout_bundle,
        args.candidate_support,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    common_args = json.loads(
        args.lower_common_args.read_text(encoding="utf-8")
    )
    if not isinstance(common_args, list) or not all(
        isinstance(value, str) for value in common_args
    ):
        raise ValueError("lower common args must be a JSON string list")
    association_rows = _read_csv(args.incumbent_association)
    if not association_rows:
        raise ValueError("incumbent association must contain a CSV header and rows")
    association_fields = tuple(association_rows[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology = _topology(args.topology)
    summary_rows = []
    for first, second in topology:
        edge_dir = args.output_dir / f"edge_{first}_{second}"
        rerun_dir = edge_dir / "nonlinear_rerun"
        edge_dir.mkdir(parents=True, exist_ok=True)
        removed_rows = remove_pair_relations(
            association_rows, (first, second)
        )
        association_path = edge_dir / "association_without_edge.csv"
        _write_csv(association_path, association_fields, removed_rows)
        command = build_lower_command(
            args.lower_executable,
            common_args,
            association_path,
            rerun_dir,
        )
        (edge_dir / "command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(command, check=True)
        heldout_dir = (
            args.heldout_bundle
            / f"leave_pair_cavity_{first}_{second}"
        )
        mean, maximum, mean_count = compare_means(
            rerun_dir / "final_pair_cavity_means.csv",
            heldout_dir / "final_pair_cavity_means.csv",
            robot_components=_robot_components(
                topology, (first, second)
            ),
        )
        covariance, covariance_count = covariance_relative_error(
            rerun_dir / "final_pair_cavity_covariance.csv",
            heldout_dir / "final_pair_cavity_covariance.csv",
        )
        agreement, edit_count = edit_sign_agreement(
            rerun_dir / "candidate_edit_scores.csv",
            heldout_dir / "candidate_edit_scores.csv",
        )
        row = {
            "first_robot": first,
            "second_robot": second,
            "removed_directed_rows": (
                len(association_rows) - len(removed_rows)
            ),
            "post_removal_robot_components": len(
                _robot_components(topology, (first, second))
            ),
            "mean_object_displacement": mean,
            "max_object_displacement": maximum,
            "compared_object_count": mean_count,
            "covariance_relative_error": covariance,
            "compared_covariance_entries": covariance_count,
            "edit_sign_agreement": agreement,
            "compared_edit_count": edit_count,
            "status": "complete",
        }
        summary_rows.append(row)
        (edge_dir / "audit.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_csv(
        args.output_dir / "summary.csv",
        tuple(summary_rows[0]),
        summary_rows,
    )
    finite_agreements = [
        float(row["edit_sign_agreement"])
        for row in summary_rows
        if np.isfinite(float(row["edit_sign_agreement"]))
    ]
    lines = [
        "# DBA leave-out fidelity audit",
        "",
        "This is a local quadratic fidelity audit, not a proof of exact nonlinear independence.",
        "",
        f"- Audited topology edges: {len(summary_rows)}",
        f"- Mean aligned object displacement: "
        f"{np.mean([row['mean_object_displacement'] for row in summary_rows]):.6g}",
        f"- Mean covariance relative error: "
        f"{np.nanmean([row['covariance_relative_error'] for row in summary_rows]):.6g}",
        f"- Mean edit-sign agreement: "
        f"{np.mean(finite_agreements):.6g}"
        if finite_agreements
        else "- Mean edit-sign agreement: unavailable",
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
