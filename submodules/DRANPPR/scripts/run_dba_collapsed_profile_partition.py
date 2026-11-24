#!/usr/bin/env python3
"""Run variable-cardinality collapsed-profile DBA on a frozen lower state."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dba_collapsed_profile_partition import (
    GaussianObjectObservation,
    NetworkPartition,
    ObjectCopy,
    PartitionEdit,
    empirical_component_intensities,
    empirical_object_prior,
    optimize_partition,
    partition_association_rows,
    reciprocal_physical_candidate_edges,
    run_collapsed_profile_partitioning,
    transform_robot_observations,
)
from scripts.dba_network_quotient_reference import (
    NetworkQuotientProblem,
    ScorerInputs,
    apply_independent_robot_gauge,
    read_scorer_inputs,
)
from scripts.dba_joint_cavity_state_predictive import (
    relation_components,
    run_joint_cavity_leave_pair_pairwise,
    run_joint_cavity_partitioning,
)


@dataclass(frozen=True)
class RelationPartition:
    """Connected components of direct relations, including conflicts."""

    components: tuple[tuple[ObjectCopy, ...], ...]

    @property
    def copies(self) -> tuple[ObjectCopy, ...]:
        return tuple(
            sorted(copy for component in self.components for copy in component)
        )

    @property
    def copy_count(self) -> int:
        return len(self.copies)

    @property
    def component_count(self) -> int:
        return len(self.components)


@dataclass(frozen=True)
class DbaRunResult:
    mode: str
    partition: NetworkPartition | RelationPartition
    transforms: Mapping[int, tuple[float, np.ndarray]]
    objective_trace: tuple[float, ...]
    accepted_edits: tuple[PartitionEdit, ...]
    component_intensities: Mapping[int, float]
    component_intensity_model: str
    attribute_names: tuple[str, ...]
    attribute_noise: tuple[float, ...]
    consensus_rounds: int
    consensus_payload_bytes: int
    association_rows: tuple[dict[str, float | int], ...]
    initial_component_count: int
    candidate_edge_count: int
    candidate_edge_counts: tuple[int, ...]
    elapsed_seconds: float
    objective_name: str
    objective_evaluations: int
    objective_terms: Mapping[str, float]
    canonical_state_sync_messages: int
    canonical_state_sync_payload_bytes: int
    feedback_variant: str
    include_component_intensity: bool
    robot_covariance_mode: str
    network_feasible: bool
    duplicate_robot_component_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", MappingProxyType(dict(self.transforms)))
        object.__setattr__(
            self,
            "component_intensities",
            MappingProxyType(dict(self.component_intensities)),
        )
        object.__setattr__(
            self, "objective_terms", MappingProxyType(dict(self.objective_terms))
        )


def observations_from_problem(
    problem: NetworkQuotientProblem,
    attributes: Mapping[ObjectCopy, np.ndarray] | None = None,
    attribute_noise: np.ndarray | None = None,
) -> dict[ObjectCopy, GaussianObjectObservation]:
    """Extract each copy's isotropic marginal from the frozen lower state."""
    attributes = dict(attributes or {})
    support = set(problem.partition.copies)
    if attributes and set(attributes) != support:
        raise ValueError("attributes must cover the exact object-copy support")
    if attributes:
        dimension = next(iter(attributes.values())).size
        if any(np.asarray(value).shape != (dimension,) for value in attributes.values()):
            raise ValueError("attribute vectors must share a dimension")
        noise = np.asarray(attribute_noise, dtype=float).reshape(-1)
        if noise.shape != (dimension,) or np.any(noise <= 0.0):
            raise ValueError("attribute noise must be finite and positive")
    else:
        noise = np.empty(0)
    observations = {}
    for state in problem.robots:
        for index, copy in enumerate(state.copies):
            variance = float(state.covariance[index, index])
            mean = np.array(state.means[index], copy=True)
            covariance_diagonal = np.array([variance, variance])
            if attributes:
                mean = np.concatenate((mean, attributes[copy]))
                covariance_diagonal = np.concatenate((covariance_diagonal, noise))
            observations[copy] = GaussianObjectObservation(
                copy=copy,
                mean=mean,
                covariance=np.diag(covariance_diagonal),
            )
    return observations


def read_object_attributes(
    metadata_path: Path,
    observed_keys: set[tuple[int, int]],
) -> tuple[tuple[str, ...], dict[tuple[int, int], np.ndarray]]:
    rows = _read_csv(metadata_path)
    if not rows:
        raise ValueError("object metadata must not be empty")
    candidate_names = tuple(
        name for name in rows[0] if name.startswith("descriptor_")
    )
    keyed = {}
    for row in rows:
        key = (int(row["robot_id"]), int(row["local_slot_id"]))
        if key not in observed_keys:
            continue
        values = np.array([float(row[name]) for name in candidate_names])
        if not np.all(np.isfinite(values)):
            raise ValueError("object attributes must be finite")
        keyed[key] = values
    if set(keyed) != observed_keys:
        raise ValueError("metadata is missing an observed object's attributes")
    if not candidate_names:
        return (), {}
    matrix = np.stack([keyed[key] for key in sorted(keyed)])
    retained = np.var(matrix, axis=0) > 0.0
    names = tuple(
        name for name, keep in zip(candidate_names, retained) if keep
    )
    if not names:
        return (), {}
    return names, {
        key: value[retained] for key, value in keyed.items()
    }


def calibrate_attribute_noise(
    partition: NetworkPartition,
    attributes: Mapping[ObjectCopy, np.ndarray],
) -> np.ndarray:
    values = {copy: np.asarray(value, dtype=float) for copy, value in attributes.items()}
    if set(values) != set(partition.copies):
        raise ValueError("attributes must cover the exact partition support")
    dimension = next(iter(values.values())).size
    if dimension == 0:
        return np.empty(0)
    if any(value.shape != (dimension,) for value in values.values()):
        raise ValueError("attribute vectors must share a dimension")
    matrix = np.stack([values[copy] for copy in partition.copies])
    global_variance = np.var(matrix, axis=0)
    residual_sum = np.zeros(dimension)
    degrees_of_freedom = 0
    for component in partition.components:
        component_values = np.stack([values[copy] for copy in component])
        residual_sum += np.sum(
            (component_values - np.mean(component_values, axis=0)) ** 2,
            axis=0,
        )
        degrees_of_freedom += len(component) - 1
    if degrees_of_freedom == 0:
        return global_variance
    pooled_variance = residual_sum / degrees_of_freedom
    numerical_floor = np.sqrt(np.finfo(float).eps) * global_variance
    return np.maximum(pooled_variance, numerical_floor)


def _initial_transforms(
    inputs: ScorerInputs,
) -> dict[int, tuple[float, np.ndarray]]:
    robots = tuple(sorted(state.robot for state in inputs.problem.robots))
    if inputs.current_transforms is not None:
        return {
            robot: (
                float(inputs.current_transforms[robot][0]),
                np.array(inputs.current_transforms[robot][1], dtype=float, copy=True),
            )
            for robot in robots
        }
    return {robot: (0.0, np.zeros(2)) for robot in robots}


def read_leave_pair_problems(
    cavity_dir: Path,
    metadata_path: Path,
    association_path: Path,
    topology_path: Path,
    plane_axes: str,
    topology: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], NetworkQuotientProblem]:
    """Read the topology-indexed leave-one-pair-out cavity bundle."""
    result = {}
    for pair in topology:
        first, second = sorted(pair)
        pair_dir = Path(cavity_dir) / f"leave_pair_cavity_{first}_{second}"
        pair_inputs = read_scorer_inputs(
            pair_dir,
            metadata_path,
            association_path,
            topology_path,
            plane_axes,
        )
        if pair_inputs.topology != topology:
            raise ValueError("leave-pair topology does not match the base input")
        result[(first, second)] = pair_inputs.problem
    return result


def run_inputs(
    inputs: ScorerInputs,
    *,
    profile_transforms: bool,
    attributes: Mapping[ObjectCopy, np.ndarray] | None = None,
    attribute_names: tuple[str, ...] = (),
    empirical_component_intensity: bool = True,
    upper_objective: str = "collapsed_independent",
    leave_pair_problems: Mapping[
        tuple[int, int], NetworkQuotientProblem
    ] | None = None,
    candidate_edges: Iterable[tuple[ObjectCopy, ObjectCopy]] | None = None,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> DbaRunResult:
    start = time.perf_counter()
    supplied_attributes = dict(attributes or {})
    attribute_noise = (
        calibrate_attribute_noise(inputs.problem.partition, supplied_attributes)
        if supplied_attributes
        else np.empty(0)
    )
    observations = observations_from_problem(
        inputs.problem, supplied_attributes, attribute_noise
    )
    initial_transforms = _initial_transforms(inputs)
    initial_state = transform_robot_observations(observations, initial_transforms)
    prior = empirical_object_prior(initial_state)
    spatial_prior = None
    attribute_observations = None
    attribute_prior = None
    if supplied_attributes:
        spatial_state = transform_robot_observations(
            observations_from_problem(inputs.problem), initial_transforms
        )
        spatial_prior = empirical_object_prior(spatial_state)
        attribute_observations = {
            copy: GaussianObjectObservation(
                copy=copy,
                mean=observation.mean[2:],
                covariance=observation.covariance[2:, 2:],
            )
            for copy, observation in observations.items()
        }
        attribute_prior = empirical_object_prior(attribute_observations)
    if empirical_component_intensity:
        component_intensities = empirical_component_intensities(
            inputs.problem.partition
        )
        component_intensity_model = "empirical_semantic_ppp"
    else:
        component_intensities = {
            semantic_class: 1.0
            for semantic_class in {
                copy.semantic_class for copy in inputs.problem.partition.copies
            }
        }
        component_intensity_model = "uniform_partition"
    fixed_candidate_support = candidate_edges is not None
    candidate_edges = (
        reciprocal_physical_candidate_edges(initial_state, inputs.topology)
        if candidate_edges is None
        else tuple(sorted(set(candidate_edges)))
    )
    root_robot = min(initial_transforms)
    canonical_state_sync_messages = 0
    canonical_state_sync_payload_bytes = 0

    if upper_objective not in (
        "collapsed_independent",
        "joint_cavity",
        "joint_cavity_leave_pair_pairwise",
        "joint_cavity_leave_pair",
    ):
        raise ValueError(f"unsupported upper objective: {upper_objective}")
    joint_objectives = (
        "joint_cavity",
        "joint_cavity_leave_pair_pairwise",
        "joint_cavity_leave_pair",
    )
    if upper_objective in joint_objectives:
        if not profile_transforms:
            raise ValueError("joint-cavity upper requires transform profiling")
        if (
            upper_objective
            in ("joint_cavity_leave_pair_pairwise", "joint_cavity_leave_pair")
            and not leave_pair_problems
        ):
            raise ValueError(
                "joint-cavity leave-pair upper requires every topology cavity"
            )
        common_arguments = {
            "initial": inputs.problem.partition,
            "problem": inputs.problem,
            "prior": prior if spatial_prior is None else spatial_prior,
            "candidate_edges": tuple(candidate_edges),
            "root_robot": root_robot,
            "initial_transforms": initial_transforms,
            "component_intensities": component_intensities,
            "selection_topology": inputs.topology,
            "attribute_observations": attribute_observations,
            "attribute_prior": attribute_prior,
            "include_component_intensity": include_component_intensity,
            "robot_covariance_mode": robot_covariance_mode,
        }
        if upper_objective == "joint_cavity_leave_pair_pairwise":
            optimized = run_joint_cavity_leave_pair_pairwise(
                **common_arguments,
                leave_pair_problems=leave_pair_problems,
            )
            components = relation_components(
                inputs.problem.partition.copies, optimized.relation_edges
            )
            partition = (
                optimized.partition
                if optimized.partition is not None
                else RelationPartition(components)
            )
            association_rows = tuple(
                row
                for first, second in optimized.relation_edges
                for row in (
                    {
                        "receiver_robot": first.robot,
                        "sender_robot": second.robot,
                        "receiver_object": first.object_id,
                        "sender_object": second.object_id,
                        "weight": 1.0,
                    },
                    {
                        "receiver_robot": second.robot,
                        "sender_robot": first.robot,
                        "receiver_object": second.object_id,
                        "sender_object": first.object_id,
                        "weight": 1.0,
                    },
                )
            )
            mode = "joint_cavity_leave_pair_pairwise_profile"
            consensus_rounds = 0
            consensus_payload_bytes = 0
            network_feasible = optimized.network_feasible
            duplicate_robot_component_count = (
                optimized.duplicate_robot_component_count
            )
        else:
            optimized = run_joint_cavity_partitioning(
                **common_arguments,
                candidate_topology=(
                    None if fixed_candidate_support else inputs.topology
                ),
                leave_pair_problems=(
                    leave_pair_problems
                    if upper_objective == "joint_cavity_leave_pair"
                    else None
                ),
            )
            mode = (
                "joint_cavity_leave_pair_crossfit_profile"
                if upper_objective == "joint_cavity_leave_pair"
                else "joint_cavity_profile"
            )
            partition = optimized.partition
            association_rows = partition_association_rows(
                partition, inputs.topology
            )
            consensus_rounds = optimized.consensus_rounds
            consensus_payload_bytes = optimized.consensus_payload_bytes
            network_feasible = True
            duplicate_robot_component_count = 0
        transforms = optimized.transforms
        objective_trace = optimized.objective_trace
        accepted_edits = optimized.accepted_edits
        candidate_edge_counts = optimized.candidate_edge_counts
        objective_evaluations = optimized.objective_evaluations
        canonical_state_sync_messages = (
            optimized.canonical_state_sync_messages
        )
        canonical_state_sync_payload_bytes = (
            optimized.canonical_state_sync_payload_bytes
        )
        objective_terms = {
            "total": optimized.objective.total,
            "state_nll": optimized.objective.state_nll,
            "mahalanobis": optimized.objective.mahalanobis,
            "log_determinant": optimized.objective.log_determinant,
            "normalization": optimized.objective.normalization,
            "cardinality": optimized.objective.cardinality,
            "attribute_nll": optimized.objective.attribute_nll,
        }
    elif profile_transforms:
        optimized = run_collapsed_profile_partitioning(
            initial=inputs.problem.partition,
            observations=observations,
            prior=prior,
            candidate_edges=candidate_edges,
            root_robot=root_robot,
            initial_transforms=initial_transforms,
            component_intensities=component_intensities,
            selection_topology=inputs.topology,
            candidate_topology=(
                None if fixed_candidate_support else inputs.topology
            ),
        )
        mode = "collapsed_profile"
        partition = optimized.partition
        transforms = optimized.transforms
        objective_trace = optimized.objective_trace
        accepted_edits = optimized.accepted_edits
        consensus_rounds = optimized.consensus_rounds
        consensus_payload_bytes = optimized.consensus_payload_bytes
        candidate_edge_counts = optimized.candidate_edge_counts
        objective_evaluations = 0
        objective_terms = {"total": float(objective_trace[-1])}
        association_rows = partition_association_rows(
            partition, inputs.topology
        )
        network_feasible = True
        duplicate_robot_component_count = 0
    else:
        optimized = optimize_partition(
            initial=inputs.problem.partition,
            observations=initial_state,
            prior=prior,
            candidate_edges=candidate_edges,
            component_intensities=component_intensities,
            selection_topology=inputs.topology,
        )
        mode = "fixed_transforms"
        partition = optimized.partition
        transforms = initial_transforms
        objective_trace = optimized.objective_trace
        accepted_edits = optimized.accepted_edits
        consensus_rounds = optimized.consensus_rounds
        consensus_payload_bytes = optimized.consensus_payload_bytes
        candidate_edge_counts = (len(candidate_edges),)
        objective_evaluations = 0
        objective_terms = {"total": float(objective_trace[-1])}
        association_rows = partition_association_rows(
            partition, inputs.topology
        )
        network_feasible = True
        duplicate_robot_component_count = 0

    feedback_variant = {
        "joint_cavity": "fitted_state",
        "joint_cavity_leave_pair_pairwise": "heldout_pairwise",
        "joint_cavity_leave_pair": "heldout_network",
    }.get(upper_objective, "legacy_collapsed")

    return DbaRunResult(
        mode=mode,
        partition=partition,
        transforms=transforms,
        objective_trace=tuple(float(value) for value in objective_trace),
        accepted_edits=accepted_edits,
        component_intensities=component_intensities,
        component_intensity_model=component_intensity_model,
        attribute_names=tuple(attribute_names),
        attribute_noise=tuple(float(value) for value in attribute_noise),
        consensus_rounds=consensus_rounds,
        consensus_payload_bytes=consensus_payload_bytes,
        association_rows=association_rows,
        initial_component_count=inputs.problem.partition.component_count,
        candidate_edge_count=max(candidate_edge_counts),
        candidate_edge_counts=candidate_edge_counts,
        elapsed_seconds=time.perf_counter() - start,
        objective_name=upper_objective,
        objective_evaluations=objective_evaluations,
        objective_terms=objective_terms,
        canonical_state_sync_messages=canonical_state_sync_messages,
        canonical_state_sync_payload_bytes=canonical_state_sync_payload_bytes,
        feedback_variant=feedback_variant,
        include_component_intensity=bool(include_component_intensity),
        robot_covariance_mode=str(robot_covariance_mode),
        network_feasible=network_feasible,
        duplicate_robot_component_count=duplicate_robot_component_count,
    )


def posthoc_association_metrics(
    association_rows: Iterable[Mapping[str, object]],
    ground_truth_rows: Iterable[Mapping[str, object]],
    minimum_weight: float = 0.5,
    evaluated_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    rows = tuple(association_rows)
    ground_truth_rows = tuple(ground_truth_rows)
    tested_pairs = (
        {
            (int(row["receiver_robot"]), int(row["sender_robot"]))
            for row in rows
        }
        if evaluated_pairs is None
        else set(evaluated_pairs)
    )
    selected = {
        (
            int(row["receiver_robot"]),
            int(row["sender_robot"]),
            int(row["receiver_object"]),
            int(row["sender_object"]),
        )
        for row in rows
        if float(row.get("weight", 1.0)) >= minimum_weight
    }
    truth = {
        (
            int(row["receiver_robot"]),
            int(row["sender_robot"]),
            int(row["receiver_object"]),
            int(row["sender_object"]),
        )
        for row in ground_truth_rows
        if (int(row["receiver_robot"]), int(row["sender_robot"])) in tested_pairs
    }
    true_positive = len(selected & truth)
    false_positive = len(selected - truth)
    false_negative = len(truth - selected)
    precision = true_positive / len(selected) if selected else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    selected_partition = association_partition_closure(rows, minimum_weight)
    truth_partition = association_partition_closure(ground_truth_rows, 0.0)
    network_true_positive = len(selected_partition & truth_partition)
    network_false_positive = len(selected_partition - truth_partition)
    network_false_negative = len(truth_partition - selected_partition)
    network_precision = (
        network_true_positive / len(selected_partition)
        if selected_partition
        else 0.0
    )
    network_recall = (
        network_true_positive / len(truth_partition)
        if truth_partition
        else 0.0
    )
    network_f1 = (
        2.0 * network_precision * network_recall
        / (network_precision + network_recall)
        if network_precision + network_recall
        else 0.0
    )
    return {
        "association_selected": len(selected),
        "association_true_positive": true_positive,
        "association_false_positive": false_positive,
        "association_false_negative": false_negative,
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1,
        "network_partition_selected": len(selected_partition),
        "network_partition_truth": len(truth_partition),
        "network_partition_true_positive": network_true_positive,
        "network_partition_false_positive": network_false_positive,
        "network_partition_false_negative": network_false_negative,
        "network_partition_precision": network_precision,
        "network_partition_recall": network_recall,
        "network_partition_f1": network_f1,
    }


def association_partition_closure(
    rows: Iterable[Mapping[str, object]], minimum_weight: float
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root))
        parent[high] = low

    for row in rows:
        if float(row.get("weight", 1.0)) < minimum_weight:
            continue
        first = (int(row["receiver_robot"]), int(row["receiver_object"]))
        second = (int(row["sender_robot"]), int(row["sender_object"]))
        union(first, second)

    components: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)
    closure = set()
    for members in components.values():
        for first_index, first in enumerate(members):
            for second in members[first_index + 1 :]:
                if first[0] != second[0]:
                    closure.add(tuple(sorted((first, second))))
    return closure


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_result_bundle(
    result: DbaRunResult,
    output_dir: Path,
    *,
    input_manifest: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    association_fields = (
        "receiver_robot",
        "sender_robot",
        "receiver_object",
        "sender_object",
        "weight",
    )
    _write_csv(output_dir / "associations.csv", association_fields, result.association_rows)

    partition_rows = (
        {
            "component": component_index,
            "robot": copy.robot,
            "object": copy.object_id,
            "semantic_class": copy.semantic_class,
        }
        for component_index, component in enumerate(result.partition.components)
        for copy in component
    )
    _write_csv(
        output_dir / "partition.csv",
        ("component", "robot", "object", "semantic_class"),
        partition_rows,
    )
    transform_rows = (
        {
            "robot": robot,
            "yaw": transform[0],
            "translation_x": transform[1][0],
            "translation_y": transform[1][1],
        }
        for robot, transform in sorted(result.transforms.items())
    )
    _write_csv(
        output_dir / "transforms.csv",
        ("robot", "yaw", "translation_x", "translation_y"),
        transform_rows,
    )
    _write_csv(
        output_dir / "objective_trace.csv",
        ("step", "objective"),
        (
            {"step": index, "objective": objective}
            for index, objective in enumerate(result.objective_trace)
        ),
    )
    _write_csv(
        output_dir / "accepted_edits.csv",
        ("step", "kind", "robot", "object", "target_robot", "target_object"),
        (
            {
                "step": index,
                "kind": edit.kind,
                "robot": edit.copy.robot,
                "object": edit.copy.object_id,
                "target_robot": "" if edit.target is None else edit.target.robot,
                "target_object": "" if edit.target is None else edit.target.object_id,
            }
            for index, edit in enumerate(result.accepted_edits)
        ),
    )
    crossfit_block_surrogate = (
        result.objective_name
        in ("joint_cavity_leave_pair_pairwise", "joint_cavity_leave_pair")
    )
    method = (
        "distributed_joint_cavity_crossfit_pairwise"
        if result.objective_name == "joint_cavity_leave_pair_pairwise"
        else (
            "distributed_joint_cavity_crossfit_partition"
            if result.objective_name == "joint_cavity_leave_pair"
            else (
                "distributed_joint_cavity_partition"
                if result.objective_name == "joint_cavity"
                else "distributed_collapsed_profile_partition"
            )
        )
    )
    summary = {
        "method": method,
        "mode": result.mode,
        "upper_objective": result.objective_name,
        "crossfit_block_surrogate": crossfit_block_surrogate,
        "initial_component_count": result.initial_component_count,
        "final_component_count": result.partition.component_count,
        "copy_count": result.partition.copy_count,
        "candidate_edge_count": result.candidate_edge_count,
        "candidate_edge_counts": result.candidate_edge_counts,
        "accepted_edit_count": len(result.accepted_edits),
        "initial_objective": result.objective_trace[0],
        "final_objective": result.objective_trace[-1],
        "objective_decrease": result.objective_trace[0] - result.objective_trace[-1],
        "objective_evaluations": result.objective_evaluations,
        "final_objective_terms": dict(result.objective_terms),
        "elapsed_seconds": result.elapsed_seconds,
        "hyperparameters": {},
        "ground_truth_used_by_optimizer": False,
        "component_intensity_model": result.component_intensity_model,
        "component_intensity": (
            "empirical"
            if result.include_component_intensity
            else "disabled"
        ),
        "component_intensities": dict(result.component_intensities),
        "robot_covariance_mode": result.robot_covariance_mode,
        "feedback_variant": result.feedback_variant,
        "network_feasible": result.network_feasible,
        "duplicate_robot_component_count": (
            result.duplicate_robot_component_count
        ),
        "candidate_support_path": input_manifest.get(
            "candidate_support"
        ),
        "candidate_support_sha256": input_manifest.get(
            "candidate_support_sha256"
        ),
        "initial_partition_sha256": input_manifest.get(
            "initial_partition_sha256"
        ),
        "attribute_model": "collapsed_gaussian_empirical_noise",
        "attribute_names": result.attribute_names,
        "attribute_noise": result.attribute_noise,
        "distributed_selection_consensus_rounds": result.consensus_rounds,
        "distributed_selection_payload_bytes": result.consensus_payload_bytes,
        "distributed_canonical_state_sync_messages": (
            result.canonical_state_sync_messages
        ),
        "distributed_canonical_state_sync_payload_bytes": (
            result.canonical_state_sync_payload_bytes
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "input_manifest.json").write_text(
        json.dumps(dict(input_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cavity-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--plane-axes", choices=("xy", "xz", "yz"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-transforms", action="store_true")
    parser.add_argument("--uniform-partition", action="store_true")
    parser.add_argument(
        "--upper-objective",
        choices=(
            "collapsed_independent",
            "joint_cavity",
            "joint_cavity_leave_pair_pairwise",
            "joint_cavity_leave_pair",
        ),
        default="collapsed_independent",
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--independent-gauge-seed", type=int)
    parser.add_argument("--disable-component-intensity", action="store_true")
    parser.add_argument(
        "--robot-covariance-mode",
        choices=("full", "diagonal"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = read_scorer_inputs(
        args.cavity_dir,
        args.metadata,
        args.association,
        args.topology,
        args.plane_axes,
    )
    gauge_records = ()
    if args.independent_gauge_seed is not None:
        gauged_problem, gauge_records = apply_independent_robot_gauge(
            inputs.problem, args.independent_gauge_seed
        )
        inputs = ScorerInputs(
            gauged_problem,
            inputs.incumbent_edges,
            inputs.topology,
            inputs.metadata_by_key,
            None,
        )
    copies_by_key = {
        (copy.robot, copy.object_id): copy
        for copy in inputs.problem.partition.copies
    }
    attribute_names, keyed_attributes = read_object_attributes(
        args.metadata, set(copies_by_key)
    )
    attributes = {
        copies_by_key[key]: value for key, value in keyed_attributes.items()
    }
    leave_pair_problems = None
    if args.upper_objective in (
        "joint_cavity_leave_pair_pairwise",
        "joint_cavity_leave_pair",
    ):
        leave_pair_problems = read_leave_pair_problems(
            args.cavity_dir,
            args.metadata,
            args.association,
            args.topology,
            args.plane_axes,
            inputs.topology,
        )
    result = run_inputs(
        inputs,
        profile_transforms=not args.fixed_transforms,
        attributes=attributes,
        attribute_names=attribute_names,
        empirical_component_intensity=not args.uniform_partition,
        upper_objective=args.upper_objective,
        leave_pair_problems=leave_pair_problems,
        include_component_intensity=not args.disable_component_intensity,
        robot_covariance_mode=args.robot_covariance_mode,
    )
    manifest = {
        "cavity_dir": str(args.cavity_dir),
        "metadata": str(args.metadata),
        "association": str(args.association),
        "topology": str(args.topology),
        "plane_axes": args.plane_axes,
        "independent_gauge_seed": args.independent_gauge_seed,
        "independent_gauges": gauge_records,
        "upper_objective": args.upper_objective,
        "component_intensity": (
            "disabled"
            if args.disable_component_intensity
            else "empirical"
        ),
        "robot_covariance_mode": args.robot_covariance_mode,
    }
    write_result_bundle(result, args.output_dir, input_manifest=manifest)
    report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if args.ground_truth is not None:
        evaluated_pairs = {
            directed
            for first, second in inputs.topology
            for directed in ((first, second), (second, first))
        }
        metrics = posthoc_association_metrics(
            result.association_rows,
            _read_csv(args.ground_truth),
            evaluated_pairs=evaluated_pairs,
        )
        (args.output_dir / "posthoc_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["posthoc_metrics"] = metrics
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
