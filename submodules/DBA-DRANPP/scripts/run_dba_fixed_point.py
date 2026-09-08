#!/usr/bin/env python3
"""Repeat full DRAN++ lower responses and DBA upper solves to a fixed point."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dba_network_quotient_reference import read_scorer_inputs
from scripts.dba_protocol_manifest import DatasetConfig, load_dataset_manifest
from scripts.run_dba_collapsed_profile_partition import (
    ObjectCopy,
    posthoc_association_metrics,
    read_leave_pair_problems,
    read_object_attributes,
    run_inputs,
    write_result_bundle,
)


CopyKey = tuple[int, int]
CanonicalPartition = tuple[tuple[CopyKey, ...], ...]
LowerCallback = Callable[[int, Path, Path], None]
UpperCallback = Callable[[int, Path, Path, Path], Path]


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    input_association: Path
    lower_dir: Path
    upper_dir: Path
    output_association: Path
    input_partition_hash: str
    lower_partition_hash: str
    output_partition_hash: str
    status: str


@dataclass(frozen=True)
class ProtocolResult:
    status: str
    epochs: tuple[EpochResult, ...]
    final_association: Path
    final_lower_dir: Path | None
    attempted_outer_rounds: int
    scheduled_missing_response_count: int
    delivered_response_pattern: tuple[bool, ...]

    @property
    def delivered_response_count(self) -> int:
        return len(self.epochs)


def dataset_config(name: str, repo_root: Path) -> DatasetConfig:
    repo_root = Path(repo_root)
    if name not in ("05", "07"):
        raise ValueError(f"unsupported dataset: {name}")
    experiment_dir = repo_root / "experiments" / "dba"
    data_dir = experiment_dir / name / "data"
    return DatasetConfig(
        name=name,
        data_dir=data_dir,
        metadata=data_dir / "local_object_metadata.csv",
        lower_topology=experiment_dir / "lower_topology.csv",
        upper_topology=experiment_dir / "upper_topology.csv",
        ground_truth=data_dir / "gt_associations.csv",
        initial_association=experiment_dir / name / "initial_associations.csv",
        candidate_support=experiment_dir / name / "candidate_support.csv",
        num_robots=3,
        num_objects=259 if name == "05" else 168,
        lower_iterations=10,
        plane_axes="xy",
    )


def read_candidate_support(
    path: Path,
    copies_by_key: Mapping[CopyKey, ObjectCopy],
    topology: Iterable[tuple[int, int]],
) -> tuple[tuple[ObjectCopy, ObjectCopy], ...]:
    """Read a fixed, label-free candidate relation supplied by the front end."""
    supported_pairs = {tuple(sorted(map(int, edge))) for edge in topology}
    edges = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row.get("weight", 1.0)) <= 0.0:
                continue
            first_key = (
                int(row["receiver_robot"]),
                int(row["receiver_object"]),
            )
            second_key = (
                int(row["sender_robot"]),
                int(row["sender_object"]),
            )
            if first_key not in copies_by_key or second_key not in copies_by_key:
                raise ValueError("candidate support endpoint is absent from metadata")
            first = copies_by_key[first_key]
            second = copies_by_key[second_key]
            if first.robot == second.robot:
                raise ValueError("candidate support must be inter-robot")
            if tuple(sorted((first.robot, second.robot))) not in supported_pairs:
                raise ValueError("candidate support is outside upper topology")
            if first.semantic_class != second.semantic_class:
                raise ValueError("candidate support must be semantic-compatible")
            edges.add(tuple(sorted((first, second))))
    return tuple(sorted(edges))


def _topology_rows(path: Path) -> tuple[tuple[int, int], ...]:
    result = set()
    for row in _read_csv(path):
        if "first_robot" in row:
            first, second = int(row["first_robot"]), int(row["second_robot"])
        elif "robot_i" in row:
            first, second = int(row["robot_i"]), int(row["robot_j"])
        else:
            first = int(row["receiver_robot"])
            second = int(row["sender_robot"])
        result.add(tuple(sorted((first, second))))
    return tuple(sorted(result))


def build_lower_command(
    config: DatasetConfig,
    *,
    binary: Path,
    association: Path,
    output_dir: Path,
    cavity_export_leave_pair_states: bool = False,
) -> tuple[str, ...]:
    command = (
        str(Path(binary).resolve()),
        "--data_dir",
        str(config.data_dir.resolve()),
        "--num_robots",
        str(config.num_robots),
        "--num_objects",
        str(config.num_objects),
        "--iters",
        str(config.lower_iterations),
    )
    if config.lower_topology is not None:
        command += (
            "--topology_file",
            str(config.lower_topology.resolve()),
        )
    command += (
        "--association_file",
        str(Path(association).resolve()),
        "--association_transform_plane_axes",
        config.plane_axes,
        "--output_dir",
        str(Path(output_dir).resolve()),
        "--quiet",
        "true",
        "--beta",
        "1.0",
        "--topology",
        "ring",
        "--ring_hops",
        "1",
        "--communication_policy",
        "triggered",
        "--object_initialization",
        "neighbor_average",
        "--object_initialization_anchor_mode",
        "constant",
        "--object_orientation_observed",
        "false",
        "--association_aware_ted_cci_initialization",
        "false",
        "--association_lower_physical_copy_majorization",
        "false",
        "--object_interface_mode",
        "copy_baseline",
        "--local_solver",
        "reduced_rotation",
        "--local_max_iterations",
        "1",
        "--local_max_tcg_iterations",
        "10",
        "--reduced_rotation_preconditioner",
        "none",
        "--association_mode",
        "residual_soft",
        "--association_policy",
        "decentralized_feedback_uot_bilevel_v1",
        "--association_update_period",
        "1",
        "--association_freeze_upper_variables",
        "true",
        "--association_pair_cavity_diagnostic",
        "true",
        "--association_sinkhorn_iterations",
        "30",
        "--association_temperature",
        "0.5",
        "--association_row_capacity",
        "1.0",
        "--association_column_capacity",
        "1.0",
        "--association_max_candidates_per_edge",
        "10",
        "--association_sinkhorn_max_nonzeros",
        "0",
        "--association_geometry_scale",
        "1.0",
        "--association_require_observed_objects",
        "true",
        "--association_use_synthetic_metadata",
        "true",
        "--association_semantic_weight",
        "0.75",
        "--association_shape_weight",
        "0.5",
        "--association_descriptor_weight",
        "0.0",
        "--association_transform_dimension",
        "0",
        "--association_gauge_covariant_preserve_local_states",
        "true",
        "--association_post_update_lower_fixed_point_rounds",
        "1",
        "--association_lower_shared_latent_mode",
        "exact_weighted",
        "--association_object_observation_track_empirical_information",
        "true",
        "--association_local_cavity_feedback_sweep",
        "false",
        "--save",
        "true",
    )
    if cavity_export_leave_pair_states:
        command += (
            "--association_cavity_export_leave_pair_states",
            "true",
        )
    return command


def execute_lower(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)


def execute_upper(
    config: DatasetConfig,
    *,
    association: Path,
    lower_dir: Path,
    upper_dir: Path,
    upper_objective: str = "joint_cavity_leave_pair",
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> Path:
    if upper_objective == "frozen":
        upper_dir.mkdir(parents=True, exist_ok=True)
        output = upper_dir / "associations.csv"
        write_binary_partition_association(
            association, output, minimum_weight=0.5
        )
        initial_hash = partition_sha256(
            canonical_partition(_read_csv(output))
        )
        summary = {
            "method": "frozen_association",
            "mode": "frozen",
            "upper_objective": "frozen",
            "feedback_variant": "frozen",
            "component_intensity": (
                "empirical"
                if include_component_intensity
                else "disabled"
            ),
            "robot_covariance_mode": robot_covariance_mode,
            "candidate_support_path": (
                None
                if config.candidate_support is None
                else str(config.candidate_support)
            ),
            "candidate_support_sha256": (
                None
                if config.candidate_support is None
                else file_sha256(config.candidate_support)
            ),
            "initial_partition_sha256": initial_hash,
            "initial_component_count": len(
                canonical_partition(_read_csv(output))
            ),
            "final_component_count": len(
                canonical_partition(_read_csv(output))
            ),
            "accepted_edit_count": 0,
            "initial_objective": 0.0,
            "final_objective": 0.0,
            "distributed_selection_payload_bytes": 0,
            "distributed_canonical_state_sync_payload_bytes": 0,
            "ground_truth_used_by_optimizer": False,
        }
        (upper_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evaluated_pairs = {
            directed
            for first, second in _topology_rows(config.upper_topology)
            for directed in ((first, second), (second, first))
        }
        metrics = posthoc_association_metrics(
            _read_csv(output),
            _read_csv(config.ground_truth),
            evaluated_pairs=evaluated_pairs,
        )
        (upper_dir / "posthoc_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output
    inputs = read_scorer_inputs(
        lower_dir,
        config.metadata,
        association,
        config.upper_topology,
        config.plane_axes,
    )
    copies_by_key = {
        (copy.robot, copy.object_id): copy
        for copy in inputs.problem.partition.copies
    }
    candidate_edges = (
        None
        if config.candidate_support is None
        else read_candidate_support(
            config.candidate_support,
            copies_by_key,
            inputs.topology,
        )
    )
    attribute_names, keyed_attributes = read_object_attributes(
        config.metadata, set(copies_by_key)
    )
    attributes = {
        copies_by_key[key]: value for key, value in keyed_attributes.items()
    }
    leave_pair_problems = None
    if upper_objective in (
        "joint_cavity_leave_pair_pairwise",
        "joint_cavity_leave_pair",
    ):
        leave_pair_problems = read_leave_pair_problems(
            lower_dir,
            config.metadata,
            association,
            config.upper_topology,
            config.plane_axes,
            inputs.topology,
        )
    result = run_inputs(
        inputs,
        profile_transforms=True,
        attributes=attributes,
        attribute_names=attribute_names,
        empirical_component_intensity=True,
        upper_objective=upper_objective,
        leave_pair_problems=leave_pair_problems,
        candidate_edges=candidate_edges,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    write_result_bundle(
        result,
        upper_dir,
        input_manifest={
            "cavity_dir": str(lower_dir),
            "metadata": str(config.metadata),
            "association": str(association),
            "topology": str(config.upper_topology),
            "plane_axes": config.plane_axes,
            "independent_gauge_seed": None,
            "independent_gauges": (),
            "upper_objective": upper_objective,
            "feedback_variant": {
                "joint_cavity": "fitted_state",
                "joint_cavity_leave_pair_pairwise": "heldout_pairwise",
                "joint_cavity_leave_pair": "heldout_network",
            }.get(upper_objective, "legacy_collapsed"),
            "component_intensity": (
                "empirical"
                if include_component_intensity
                else "disabled"
            ),
            "robot_covariance_mode": robot_covariance_mode,
            "candidate_support": (
                None
                if config.candidate_support is None
                else str(config.candidate_support)
            ),
            "candidate_support_sha256": (
                None
                if config.candidate_support is None
                else file_sha256(config.candidate_support)
            ),
            "initial_partition_sha256": partition_sha256(
                canonical_partition(
                    _read_csv(association),
                    support=(
                        (copy.robot, copy.object_id)
                        for copy in inputs.problem.partition.copies
                    ),
                )
            ),
        },
    )
    evaluated_pairs = {
        directed
        for first, second in inputs.topology
        for directed in ((first, second), (second, first))
    }
    metrics = posthoc_association_metrics(
        result.association_rows,
        _read_csv(config.ground_truth),
        evaluated_pairs=evaluated_pairs,
    )
    (upper_dir / "posthoc_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return upper_dir / "associations.csv"


def _copy_key(row: Mapping[str, object], prefix: str) -> CopyKey:
    return int(row[f"{prefix}_robot"]), int(row[f"{prefix}_object"])


def canonical_partition(
    rows: Iterable[Mapping[str, object]],
    *,
    support: Iterable[CopyKey] = (),
    minimum_weight: float = 0.5,
) -> CanonicalPartition:
    """Return sorted components induced by selected directed association rows."""
    parent: dict[CopyKey, CopyKey] = {}

    def add(key: CopyKey) -> None:
        parent.setdefault(key, key)

    def find(key: CopyKey) -> CopyKey:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(first: CopyKey, second: CopyKey) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root))
        parent[high] = low

    for key in support:
        add((int(key[0]), int(key[1])))
    for row in rows:
        first = _copy_key(row, "receiver")
        second = _copy_key(row, "sender")
        add(first)
        add(second)
        if float(row.get("weight", 1.0)) >= minimum_weight:
            union(first, second)

    components: dict[CopyKey, list[CopyKey]] = {}
    for key in parent:
        components.setdefault(find(key), []).append(key)
    return tuple(
        sorted(tuple(sorted(component)) for component in components.values())
    )


def partition_sha256(partition: CanonicalPartition) -> str:
    payload = json.dumps(partition, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def write_binary_partition_association(
    source: Path,
    destination: Path,
    minimum_weight: float,
) -> None:
    """Serialize only the selected partition; frontend scores are not DBA variables."""
    required = (
        "receiver_robot",
        "sender_robot",
        "receiver_object",
        "sender_object",
    )
    selected: dict[tuple[int, int, int, int], dict[str, int]] = {}
    for row in _read_csv(source):
        if any(name not in row for name in required):
            raise ValueError(f"association row is missing required fields: {source}")
        if float(row.get("weight", 1.0)) < minimum_weight:
            continue
        key = tuple(int(row[name]) for name in required)
        selected[key] = {
            name: value for name, value in zip(required, key)
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=required + ("weight",))
        writer.writeheader()
        for key in sorted(selected):
            writer.writerow({**selected[key], "weight": 1})


def _association_hash(
    path: Path, support: tuple[CopyKey, ...], minimum_weight: float
) -> str:
    return partition_sha256(
        canonical_partition(
            _read_csv(path), support=support, minimum_weight=minimum_weight
        )
    )


def _last_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    return dict(rows[-1]) if rows else {}


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _numeric(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _epoch_trace_row(epoch: EpochResult) -> dict[str, object]:
    lower = _last_csv_row(epoch.lower_dir / "iteration_summary.csv")
    upper = _read_json(epoch.upper_dir / "summary.json")
    posthoc = _read_json(epoch.upper_dir / "posthoc_metrics.json")
    return {
        "epoch": epoch.epoch,
        "status": epoch.status,
        "input_partition_hash": epoch.input_partition_hash,
        "lower_partition_hash": epoch.lower_partition_hash,
        "output_partition_hash": epoch.output_partition_hash,
        "input_association": str(epoch.input_association),
        "output_association": str(epoch.output_association),
        "lower_dir": str(epoch.lower_dir),
        "upper_dir": str(epoch.upper_dir),
        "lower_measurement_cost": lower.get("measurement_cost", ""),
        "lower_augmented_cost": lower.get("augmented_cost", ""),
        "lower_gradient": lower.get("gradient", ""),
        "lower_cumulative_total_comm_mb": lower.get(
            "cumulative_total_comm_mb", ""
        ),
        "upper_initial_objective": upper.get("initial_objective", ""),
        "upper_final_objective": upper.get("final_objective", ""),
        "upper_accepted_edit_count": upper.get("accepted_edit_count", ""),
        "upper_initial_component_count": upper.get(
            "initial_component_count", ""
        ),
        "upper_final_component_count": upper.get("final_component_count", ""),
        "upper_objective": upper.get("upper_objective", ""),
        "upper_consensus_payload_bytes": upper.get(
            "distributed_selection_payload_bytes", ""
        ),
        "upper_canonical_state_sync_payload_bytes": upper.get(
            "distributed_canonical_state_sync_payload_bytes", ""
        ),
        "association_f1": posthoc.get("association_f1", ""),
        "network_partition_f1": posthoc.get("network_partition_f1", ""),
    }


def _write_outer_trace(output_dir: Path, epochs: tuple[EpochResult, ...]) -> None:
    rows = [_epoch_trace_row(epoch) for epoch in epochs]
    with (output_dir / "outer_trace.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_availability_trace(
    output_dir: Path, rows: list[dict[str, object]]
) -> None:
    fieldnames = (
        "attempted_round",
        "pattern_index",
        "delivered",
        "delivered_response_index",
        "status",
        "input_partition_hash",
        "output_partition_hash",
        "epoch_dir",
        "lower_dir",
        "upper_dir",
    )
    with (output_dir / "availability_trace.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_protocol_summary(
    output_dir: Path, result: ProtocolResult
) -> None:
    total_communication_mb = 0.0
    for epoch in result.epochs:
        lower = _last_csv_row(epoch.lower_dir / "iteration_summary.csv")
        upper = _read_json(epoch.upper_dir / "summary.json")
        total_communication_mb += _numeric(
            lower.get("cumulative_total_comm_mb")
        )
        total_communication_mb += _numeric(
            upper.get("distributed_selection_payload_bytes")
        ) / (1024.0 * 1024.0)
        total_communication_mb += _numeric(
            upper.get("distributed_canonical_state_sync_payload_bytes")
        ) / (1024.0 * 1024.0)
    partition_hashes = (
        [result.epochs[0].input_partition_hash]
        + [epoch.output_partition_hash for epoch in result.epochs]
        if result.epochs
        else []
    )
    upper_objectives = sorted(
        {
            str(_read_json(epoch.upper_dir / "summary.json").get(
                "upper_objective", ""
            ))
            for epoch in result.epochs
        }
        - {""}
    )
    upper_summaries = tuple(
        _read_json(epoch.upper_dir / "summary.json")
        for epoch in result.epochs
    )

    def distinct_summary_values(name: str) -> list[str]:
        return sorted(
            {
                str(value.get(name, ""))
                for value in upper_summaries
                if value.get(name) not in (None, "")
            }
        )

    summary = {
        "method": (
            "dba_joint_cavity_crossfit_fixed_point"
            if upper_objectives == ["joint_cavity_leave_pair"]
            else (
                "dba_joint_cavity_crossfit_pairwise_fixed_point"
                if upper_objectives
                == ["joint_cavity_leave_pair_pairwise"]
                else "dba_full_lower_fixed_point"
            )
        ),
        "status": result.status,
        "full_lower_response_count": len(result.epochs),
        "upper_solve_count": len(result.epochs),
        "attempted_outer_rounds": result.attempted_outer_rounds,
        "delivered_response_count": result.delivered_response_count,
        "scheduled_missing_response_count": (
            result.scheduled_missing_response_count
        ),
        "delivered_response_pattern": "".join(
            "1" if value else "0"
            for value in result.delivered_response_pattern
        ),
        "final_association": str(result.final_association),
        "final_lower_dir": (
            None if result.final_lower_dir is None else str(result.final_lower_dir)
        ),
        "partition_hashes": partition_hashes,
        "total_communication_mb": total_communication_mb,
        "ground_truth_used_by_optimizer": False,
        "upper_objectives": upper_objectives,
        "crossfit_block_surrogate": (
            upper_objectives
            in (
                ["joint_cavity_leave_pair_pairwise"],
                ["joint_cavity_leave_pair"],
            )
        ),
        "algorithmic_hyperparameters": {},
        "feedback_variants": distinct_summary_values(
            "feedback_variant"
        ),
        "component_intensity_modes": distinct_summary_values(
            "component_intensity"
        ),
        "robot_covariance_modes": distinct_summary_values(
            "robot_covariance_mode"
        ),
        "candidate_support_paths": distinct_summary_values(
            "candidate_support_path"
        ),
        "candidate_support_sha256": distinct_summary_values(
            "candidate_support_sha256"
        ),
        "initial_partition_sha256": (
            partition_hashes[0] if partition_hashes else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_fixed_point_protocol(
    *,
    initial_association: Path,
    support: Iterable[CopyKey],
    output_dir: Path,
    safety_max_outer_iterations: int,
    run_lower: LowerCallback,
    run_upper: UpperCallback,
    minimum_weight: float = 0.5,
    delivered_response_pattern: tuple[bool, ...] = (True,),
) -> ProtocolResult:
    """Apply the finite full-lower/DBA response map until it repeats."""
    if safety_max_outer_iterations <= 0:
        raise ValueError("safety max outer iterations must be positive")
    if not delivered_response_pattern or not any(delivered_response_pattern):
        raise ValueError("response pattern must deliver at least one response")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must not be nonempty")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "initial_association": str(Path(initial_association).resolve()),
                "minimum_weight": minimum_weight,
                "safety_max_outer_iterations": safety_max_outer_iterations,
                "delivered_response_pattern": "".join(
                    "1" if value else "0"
                    for value in delivered_response_pattern
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    support_keys = tuple(sorted(set(support)))
    current_association = output_dir / "initial_partition_associations.csv"
    write_binary_partition_association(
        Path(initial_association), current_association, minimum_weight
    )
    current_hash = _association_hash(
        current_association, support_keys, minimum_weight
    )
    seen_at_epoch = {current_hash: -1}
    epochs: list[EpochResult] = []
    availability_rows: list[dict[str, object]] = []
    scheduled_missing_response_count = 0

    for attempted_round in range(safety_max_outer_iterations):
        pattern_index = attempted_round % len(delivered_response_pattern)
        if not delivered_response_pattern[pattern_index]:
            scheduled_missing_response_count += 1
            availability_rows.append(
                {
                    "attempted_round": attempted_round,
                    "pattern_index": pattern_index,
                    "delivered": 0,
                    "delivered_response_index": "",
                    "status": "scheduled_missing_response",
                    "input_partition_hash": current_hash,
                    "output_partition_hash": current_hash,
                    "epoch_dir": "",
                    "lower_dir": "",
                    "upper_dir": "",
                }
            )
            _write_availability_trace(output_dir, availability_rows)
            continue

        epoch = len(epochs)
        epoch_dir = output_dir / f"epoch_{epoch:02d}"
        lower_dir = epoch_dir / "lower"
        upper_dir = epoch_dir / "upper"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        run_lower(epoch, current_association, lower_dir)
        lower_association = lower_dir / "final_association_weights.csv"
        if not lower_association.is_file():
            raise ValueError(
                f"lower response did not export association state: {lower_association}"
            )
        lower_hash = _association_hash(
            lower_association, support_keys, minimum_weight
        )
        if lower_hash != current_hash:
            raise ValueError(
                "lower response partition mismatch: "
                f"expected {current_hash} from {current_association}, "
                f"found {lower_hash} in {lower_association}"
            )
        proposed_association = Path(
            run_upper(
                epoch,
                current_association,
                lower_dir,
                upper_dir,
            )
        )
        normalized_response = epoch_dir / "partition_response.csv"
        write_binary_partition_association(
            proposed_association, normalized_response, minimum_weight
        )
        output_hash = _association_hash(
            normalized_response, support_keys, minimum_weight
        )
        if output_hash == current_hash:
            status = "partition_fixed_point"
            output_association = current_association
        elif output_hash in seen_at_epoch:
            status = "partition_cycle"
            output_association = normalized_response
        elif attempted_round + 1 == safety_max_outer_iterations:
            status = "safety_limit"
            output_association = normalized_response
        else:
            status = "continue"
            output_association = normalized_response
        epoch_result = EpochResult(
            epoch=epoch,
            input_association=current_association,
            lower_dir=lower_dir,
            upper_dir=upper_dir,
            output_association=output_association,
            input_partition_hash=current_hash,
            lower_partition_hash=lower_hash,
            output_partition_hash=output_hash,
            status=status,
        )
        epochs.append(epoch_result)
        (epoch_dir / "partition.sha256").write_text(
            output_hash + "\n", encoding="ascii"
        )
        _write_outer_trace(output_dir, tuple(epochs))
        availability_rows.append(
            {
                "attempted_round": attempted_round,
                "pattern_index": pattern_index,
                "delivered": 1,
                "delivered_response_index": epoch,
                "status": status,
                "input_partition_hash": current_hash,
                "output_partition_hash": output_hash,
                "epoch_dir": str(epoch_dir),
                "lower_dir": str(lower_dir),
                "upper_dir": str(upper_dir),
            }
        )
        _write_availability_trace(output_dir, availability_rows)

        if status != "continue":
            protocol_result = ProtocolResult(
                status=status,
                epochs=tuple(epochs),
                final_association=output_association,
                final_lower_dir=(
                    lower_dir if status == "partition_fixed_point" else None
                ),
                attempted_outer_rounds=attempted_round + 1,
                scheduled_missing_response_count=(
                    scheduled_missing_response_count
                ),
                delivered_response_pattern=delivered_response_pattern,
            )
            _write_protocol_summary(output_dir, protocol_result)
            return protocol_result
        seen_at_epoch[output_hash] = epoch
        current_association = output_association
        current_hash = output_hash

    protocol_result = ProtocolResult(
        status="safety_limit",
        epochs=tuple(epochs),
        final_association=current_association,
        final_lower_dir=None,
        attempted_outer_rounds=safety_max_outer_iterations,
        scheduled_missing_response_count=scheduled_missing_response_count,
        delivered_response_pattern=delivered_response_pattern,
    )
    _write_protocol_summary(output_dir, protocol_result)
    return protocol_result


def parse_response_pattern(value: str) -> tuple[bool, ...]:
    if not value or any(character not in "01" for character in value):
        raise ValueError("response pattern must contain only 0 and 1")
    pattern = tuple(character == "1" for character in value)
    if not any(pattern):
        raise ValueError("response pattern must deliver at least one response")
    return pattern


def _metadata_support(path: Path) -> tuple[CopyKey, ...]:
    rows = _read_csv(path)
    return tuple(
        sorted(
            {
                (int(row["robot_id"]), int(row["local_slot_id"]))
                for row in rows
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        choices=("05", "07"),
    )
    source.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--lower-iterations", type=int, default=10)
    parser.add_argument("--delivered-response-pattern", default="1")
    parser.add_argument("--initial-association", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=Path("build/bin/manual-object-dran-example")
    )
    parser.add_argument(
        "--safety-max-outer-iterations",
        type=int,
        default=20,
        help="Runtime guard only; reaching it is reported as nonconvergence.",
    )
    parser.add_argument(
        "--upper-objective",
        choices=(
            "frozen",
            "collapsed_independent",
            "joint_cavity",
            "joint_cavity_leave_pair_pairwise",
            "joint_cavity_leave_pair",
        ),
        default="joint_cavity_leave_pair",
    )
    parser.add_argument(
        "--disable-component-intensity",
        "--disable_component_intensity",
        action="store_true",
    )
    parser.add_argument(
        "--robot-covariance-mode",
        "--robot_covariance_mode",
        choices=("full", "diagonal"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delivered_response_pattern = parse_response_pattern(
        args.delivered_response_pattern
    )
    repo_root = Path(__file__).resolve().parents[1]
    config = (
        load_dataset_manifest(
            args.dataset_manifest,
            lower_iterations=args.lower_iterations,
        )
        if args.dataset_manifest is not None
        else replace(
            dataset_config(args.dataset, repo_root),
            lower_iterations=args.lower_iterations,
        )
    )
    initial_association = (
        config.initial_association
        if args.initial_association is None
        else args.initial_association
    )
    required = (
        Path(args.binary),
        config.data_dir,
        config.metadata,
        config.upper_topology,
        config.ground_truth,
        initial_association,
    )
    if config.candidate_support is not None:
        required += (config.candidate_support,)
    if config.lower_topology is not None:
        required += (config.lower_topology,)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing protocol inputs: " + ", ".join(missing))

    def lower_callback(epoch: int, association: Path, lower_dir: Path) -> None:
        del epoch
        execute_lower(
            build_lower_command(
                config,
                binary=args.binary,
                association=association,
                output_dir=lower_dir,
                cavity_export_leave_pair_states=(
                    args.upper_objective
                    in (
                        "joint_cavity_leave_pair_pairwise",
                        "joint_cavity_leave_pair",
                    )
                ),
            )
        )

    def upper_callback(
        epoch: int,
        association: Path,
        lower_dir: Path,
        upper_dir: Path,
    ) -> Path:
        del epoch
        return execute_upper(
            config,
            association=association,
            lower_dir=lower_dir,
            upper_dir=upper_dir,
            upper_objective=args.upper_objective,
            include_component_intensity=(
                not args.disable_component_intensity
            ),
            robot_covariance_mode=args.robot_covariance_mode,
        )

    result = run_fixed_point_protocol(
        initial_association=initial_association,
        support=_metadata_support(config.metadata),
        output_dir=args.output_dir,
        safety_max_outer_iterations=args.safety_max_outer_iterations,
        run_lower=lower_callback,
        run_upper=upper_callback,
        delivered_response_pattern=delivered_response_pattern,
    )
    print(
        json.dumps(
            _read_json(args.output_dir / "summary.json"), sort_keys=True
        )
    )
    if result.status != "partition_fixed_point":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
