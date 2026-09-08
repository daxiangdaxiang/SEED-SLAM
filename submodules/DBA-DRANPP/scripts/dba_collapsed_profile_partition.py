#!/usr/bin/env python3
"""Collapsed Gaussian evidence for variable-cardinality DBA partitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import minimize

from scripts.dba_network_quotient_reference import (
    NetworkPartition,
    ObjectCopy,
    rotation2,
    run_min_consensus,
)


def _validated_spd(matrix: np.ndarray, dimension: int, name: str) -> np.ndarray:
    value = np.array(matrix, dtype=float, copy=True)
    if value.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    value = 0.5 * (value + value.T)
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class GaussianObjectObservation:
    copy: ObjectCopy
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.copy, ObjectCopy):
            raise ValueError("copy must be an ObjectCopy")
        mean = np.array(self.mean, dtype=float, copy=True).reshape(-1)
        if mean.size == 0 or not np.all(np.isfinite(mean)):
            raise ValueError("observation mean must be finite and nonempty")
        covariance = _validated_spd(
            self.covariance, mean.size, "observation covariance"
        )
        mean.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class GaussianObjectPrior:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=float, copy=True).reshape(-1)
        if mean.size == 0 or not np.all(np.isfinite(mean)):
            raise ValueError("prior mean must be finite and nonempty")
        covariance = _validated_spd(
            self.covariance, mean.size, "prior covariance"
        )
        mean.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class GaussianTransformPrior:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=float, copy=True).reshape(-1)
        if mean.shape != (3,) or not np.all(np.isfinite(mean)):
            raise ValueError("transform prior mean must be a finite 3-vector")
        covariance = _validated_spd(
            self.covariance, 3, "transform prior covariance"
        )
        mean.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class ComponentEvidencePacket:
    copies: tuple[ObjectCopy, ...]
    precision: np.ndarray
    information: np.ndarray
    constant: float

    def __post_init__(self) -> None:
        copies = tuple(self.copies)
        if not copies or len(set(copies)) != len(copies):
            raise ValueError("packet copies must be nonempty and unique")
        dimension = np.asarray(self.information).size
        precision = _validated_spd(
            self.precision, dimension, "packet precision"
        )
        information = np.array(
            self.information, dtype=float, copy=True
        ).reshape(-1)
        if information.shape != (dimension,) or not np.all(
            np.isfinite(information)
        ):
            raise ValueError("packet information must be finite")
        if not np.isfinite(self.constant):
            raise ValueError("packet constant must be finite")
        information.setflags(write=False)
        object.__setattr__(self, "copies", tuple(sorted(copies)))
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "information", information)
        object.__setattr__(self, "constant", float(self.constant))


@dataclass(frozen=True)
class PartitionOptimizationResult:
    partition: NetworkPartition
    objective_trace: tuple[float, ...]
    accepted_edits: tuple[PartitionEdit, ...]
    consensus_rounds: int = 0
    consensus_payload_bytes: int = 0

    @property
    def accepted_moves(self) -> tuple[tuple[ObjectCopy, ObjectCopy], ...]:
        return tuple(
            (edit.copy, edit.target)
            for edit in self.accepted_edits
            if edit.target is not None
        )


@dataclass(frozen=True)
class PartitionEdit:
    kind: str
    copy: ObjectCopy
    target: ObjectCopy | None


@dataclass(frozen=True)
class ProfiledPartitionResult:
    partition: NetworkPartition
    transforms: Mapping[int, tuple[float, np.ndarray]]
    transformed_observations: Mapping[ObjectCopy, GaussianObjectObservation]
    objective: float
    evaluations: int
    converged: bool

    def __post_init__(self) -> None:
        transforms = {
            int(robot): (
                float(transform[0]),
                np.array(transform[1], dtype=float, copy=True).reshape(2),
            )
            for robot, transform in self.transforms.items()
        }
        for _, translation in transforms.values():
            translation.setflags(write=False)
        object.__setattr__(self, "transforms", MappingProxyType(transforms))
        object.__setattr__(
            self,
            "transformed_observations",
            MappingProxyType(dict(self.transformed_observations)),
        )


@dataclass(frozen=True)
class CollapsedProfileResult:
    partition: NetworkPartition
    transforms: Mapping[int, tuple[float, np.ndarray]]
    objective_trace: tuple[float, ...]
    accepted_edits: tuple[PartitionEdit, ...]
    consensus_rounds: int = 0
    consensus_payload_bytes: int = 0
    candidate_edge_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", MappingProxyType(dict(self.transforms)))


@dataclass(frozen=True)
class ExactPartitionResult:
    partition: NetworkPartition
    objective: float
    feasible_partition_count: int


def _normalized_gaussian_canonical(
    mean: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    precision = np.linalg.solve(covariance, np.eye(mean.size))
    information = precision @ mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("Gaussian covariance must be positive definite")
    constant = 0.5 * (
        mean @ information
        + logdet
        + mean.size * np.log(2.0 * np.pi)
    )
    return precision, information, float(constant)


def make_component_packet(
    observation: GaussianObjectObservation,
) -> ComponentEvidencePacket:
    precision, information, constant = _normalized_gaussian_canonical(
        observation.mean, observation.covariance
    )
    return ComponentEvidencePacket(
        (observation.copy,), precision, information, constant
    )


def aggregate_component_packets(
    packets: Iterable[ComponentEvidencePacket],
) -> ComponentEvidencePacket:
    supplied = tuple(packets)
    if not supplied:
        raise ValueError("at least one component packet is required")
    dimension = supplied[0].information.size
    if any(packet.information.size != dimension for packet in supplied):
        raise ValueError("component packets must share a dimension")
    copies = tuple(copy for packet in supplied for copy in packet.copies)
    if len(set(copies)) != len(copies):
        raise ValueError("component packets must have disjoint copy support")
    if len({copy.robot for copy in copies}) != len(copies):
        raise ValueError("a component cannot contain two copies from one robot")
    return ComponentEvidencePacket(
        copies=tuple(sorted(copies)),
        precision=sum(
            (packet.precision for packet in supplied),
            np.zeros((dimension, dimension)),
        ),
        information=sum(
            (packet.information for packet in supplied),
            np.zeros(dimension),
        ),
        constant=sum(packet.constant for packet in supplied),
    )


def packet_component_negative_log_evidence(
    packet: ComponentEvidencePacket, prior: GaussianObjectPrior
) -> float:
    if packet.information.size != prior.mean.size:
        raise ValueError("packet and prior dimensions must match")
    prior_precision, prior_information, prior_constant = (
        _normalized_gaussian_canonical(prior.mean, prior.covariance)
    )
    precision = packet.precision + prior_precision
    information = packet.information + prior_information
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0.0:
        raise ValueError("component posterior precision must be positive definite")
    mean_term = information @ np.linalg.solve(precision, information)
    dimension = information.size
    return float(
        packet.constant
        + prior_constant
        - 0.5 * mean_term
        + 0.5 * logdet
        - 0.5 * dimension * np.log(2.0 * np.pi)
    )


def component_negative_log_evidence(
    observations: Iterable[GaussianObjectObservation],
    prior: GaussianObjectPrior,
) -> float:
    packets = tuple(make_component_packet(value) for value in observations)
    return packet_component_negative_log_evidence(
        aggregate_component_packets(packets), prior
    )


def component_evidence_with_mean_gradients(
    observations: Iterable[GaussianObjectObservation],
    prior: GaussianObjectPrior,
) -> tuple[float, dict[ObjectCopy, np.ndarray]]:
    values = tuple(observations)
    if not values:
        raise ValueError("component evidence requires observations")
    packets = tuple(make_component_packet(value) for value in values)
    aggregate = aggregate_component_packets(packets)
    prior_precision, prior_information, _ = _normalized_gaussian_canonical(
        prior.mean, prior.covariance
    )
    posterior_precision = aggregate.precision + prior_precision
    posterior_mean = np.linalg.solve(
        posterior_precision, aggregate.information + prior_information
    )
    gradients = {
        value.copy: packet.precision @ (value.mean - posterior_mean)
        for value, packet in zip(values, packets)
    }
    return (
        packet_component_negative_log_evidence(aggregate, prior),
        gradients,
    )


def partition_objective_with_mean_gradients(
    partition: NetworkPartition,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float] | None = None,
) -> tuple[float, dict[ObjectCopy, np.ndarray]]:
    values = _validated_observations(partition, observations)
    intensities = _validated_component_intensities(
        partition, component_intensities
    )
    objective = 0.0
    gradients = {}
    for component in partition.components:
        component_objective, component_gradients = (
            component_evidence_with_mean_gradients(
                tuple(values[copy] for copy in component), prior
            )
        )
        objective += component_objective - np.log(
            intensities[component[0].semantic_class]
        )
        gradients.update(component_gradients)
    return float(objective), gradients


def _validated_observations(
    partition: NetworkPartition,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
) -> dict[ObjectCopy, GaussianObjectObservation]:
    values = dict(observations)
    if set(values) != set(partition.copies):
        raise ValueError("observation support must match partition copies")
    for copy, observation in values.items():
        if observation.copy != copy:
            raise ValueError("observation key must match its object copy")
    dimensions = {observation.mean.size for observation in values.values()}
    if len(dimensions) != 1:
        raise ValueError("all observations must share a dimension")
    return values


def _validated_component_intensities(
    partition: NetworkPartition,
    component_intensities: Mapping[int, float] | None,
) -> dict[int, float]:
    semantic_classes = {
        copy.semantic_class for copy in partition.copies
    }
    if component_intensities is None:
        return {semantic_class: 1.0 for semantic_class in semantic_classes}
    intensities = {
        int(semantic_class): float(value)
        for semantic_class, value in component_intensities.items()
    }
    if set(intensities) != semantic_classes:
        raise ValueError("component intensities must cover every semantic class")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in intensities.values()
    ):
        raise ValueError("component intensities must be finite and positive")
    return intensities


def partition_objective(
    partition: NetworkPartition,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    log_partition_prior: float = 0.0,
    component_intensities: Mapping[int, float] | None = None,
) -> float:
    values = _validated_observations(partition, observations)
    intensities = _validated_component_intensities(
        partition, component_intensities
    )
    if not np.isfinite(log_partition_prior):
        raise ValueError("log partition prior must be finite")
    if next(iter(values.values())).mean.size != prior.mean.size:
        raise ValueError("observation and prior dimensions must match")
    return float(
        sum(
            component_negative_log_evidence(
                tuple(values[copy] for copy in component), prior
            )
            - np.log(intensities[component[0].semantic_class])
            for component in partition.components
        )
        - log_partition_prior
    )


def _component_containing(
    partition: NetworkPartition, copy: ObjectCopy
) -> tuple[ObjectCopy, ...]:
    for component in partition.components:
        if copy in component:
            return component
    raise ValueError("object copy is absent from partition")


def merge_partition_components(
    partition: NetworkPartition, first: ObjectCopy, second: ObjectCopy
) -> NetworkPartition:
    first_component = _component_containing(partition, first)
    second_component = _component_containing(partition, second)
    if first_component == second_component:
        return partition
    merged = tuple(sorted(first_component + second_component))
    robots = [copy.robot for copy in merged]
    if len(set(robots)) != len(robots):
        raise ValueError("cannot merge components containing the same robot")
    if len({copy.semantic_class for copy in merged}) != 1:
        raise ValueError("cannot merge components with different semantics")
    retained = tuple(
        component
        for component in partition.components
        if component not in (first_component, second_component)
    )
    return NetworkPartition.build(retained + (merged,))


def detach_partition_copy(
    partition: NetworkPartition, copy: ObjectCopy
) -> NetworkPartition:
    component = _component_containing(partition, copy)
    if len(component) == 1:
        return partition
    remainder = tuple(member for member in component if member != copy)
    retained = tuple(
        value for value in partition.components if value != component
    )
    return NetworkPartition.build(retained + (remainder, (copy,)))


def move_partition_copy(
    partition: NetworkPartition, copy: ObjectCopy, target: ObjectCopy
) -> NetworkPartition:
    source_component = _component_containing(partition, copy)
    target_component = _component_containing(partition, target)
    if source_component == target_component:
        return partition
    detached = detach_partition_copy(partition, copy)
    return merge_partition_components(detached, copy, target)


def merge_delta(
    partition: NetworkPartition,
    first: ObjectCopy,
    second: ObjectCopy,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float] | None = None,
) -> float:
    candidate = merge_partition_components(partition, first, second)
    return partition_objective(
        candidate,
        observations,
        prior,
        component_intensities=component_intensities,
    ) - (
        partition_objective(
            partition,
            observations,
            prior,
            component_intensities=component_intensities,
        )
    )


def _canonical_candidate_edge(
    edge: Iterable[ObjectCopy],
) -> tuple[ObjectCopy, ObjectCopy]:
    endpoints = tuple(edge)
    if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
        raise ValueError("candidate edge must have two distinct copies")
    return tuple(sorted(endpoints))


def enumerate_feasible_partition_edits(
    partition: NetworkPartition,
    candidate_edges: Iterable[Iterable[ObjectCopy]],
) -> tuple[PartitionEdit, ...]:
    """Enumerate topology-supported robot-exclusive edits without scoring."""
    support = set(partition.copies)
    edges = tuple(
        sorted({_canonical_candidate_edge(edge) for edge in candidate_edges})
    )
    if any(first not in support or second not in support for first, second in edges):
        raise ValueError("candidate edge is outside partition support")
    edits: dict[tuple[str, ObjectCopy, ObjectCopy | None], PartitionEdit] = {}
    for component in partition.components:
        if len(component) <= 1:
            continue
        for copy in component:
            edit = PartitionEdit("detach", copy, None)
            edits[(edit.kind, edit.copy, edit.target)] = edit
    for first, second in edges:
        first_component = _component_containing(partition, first)
        second_component = _component_containing(partition, second)
        if first_component == second_component:
            continue
        for moved, target, source_component, target_component in (
            (first, second, first_component, second_component),
            (second, first, second_component, first_component),
        ):
            if moved.robot in {copy.robot for copy in target_component}:
                continue
            kind = "merge" if len(source_component) == 1 else "move"
            edit = PartitionEdit(kind, moved, target)
            edits[(edit.kind, edit.copy, edit.target)] = edit
    return tuple(
        edits[key]
        for key in sorted(
            edits,
            key=lambda value: (
                value[0],
                value[1],
                (-1, -1, -1)
                if value[2] is None
                else (
                    value[2].robot,
                    value[2].object_id,
                    value[2].semantic_class,
                ),
            ),
        )
    )


def enumerate_robot_exclusive_partitions(
    copies: Iterable[ObjectCopy],
    candidate_edges: Iterable[Iterable[ObjectCopy]] | None = None,
) -> tuple[NetworkPartition, ...]:
    support = tuple(sorted(copies))
    if not support or len(set(support)) != len(support):
        raise ValueError("partition support must be nonempty and unique")
    if candidate_edges is None:
        edges = None
    else:
        edges = {
            _canonical_candidate_edge(edge) for edge in candidate_edges
        }
        if any(first not in support or second not in support for first, second in edges):
            raise ValueError("candidate edge is outside partition support")

    partitions = []

    def visit(
        index: int, components: tuple[tuple[ObjectCopy, ...], ...]
    ) -> None:
        if index == len(support):
            partitions.append(NetworkPartition.build(components))
            return
        copy = support[index]
        visit(index + 1, components + ((copy,),))
        for component_index, component in enumerate(components):
            if copy.semantic_class != component[0].semantic_class:
                continue
            if copy.robot in {member.robot for member in component}:
                continue
            if edges is not None and not any(
                tuple(sorted((copy, member))) in edges for member in component
            ):
                continue
            updated = list(components)
            updated[component_index] = tuple(sorted(component + (copy,)))
            visit(index + 1, tuple(updated))

    visit(0, ())
    return tuple(sorted(set(partitions), key=lambda value: value.components))


def solve_exact_partition(
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    candidate_edges: Iterable[Iterable[ObjectCopy]] | None = None,
    component_intensities: Mapping[int, float] | None = None,
) -> ExactPartitionResult:
    values = dict(observations)
    partitions = enumerate_robot_exclusive_partitions(
        values, candidate_edges=candidate_edges
    )
    scored = tuple(
        (
            partition_objective(
                partition,
                values,
                prior,
                component_intensities=component_intensities,
            ),
            partition,
        )
        for partition in partitions
    )
    objective, partition = min(
        scored, key=lambda value: (value[0], value[1].components)
    )
    return ExactPartitionResult(
        partition=partition,
        objective=float(objective),
        feasible_partition_count=len(partitions),
    )


def optimize_partition(
    initial: NetworkPartition,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    candidate_edges: Iterable[Iterable[ObjectCopy]],
    component_intensities: Mapping[int, float] | None = None,
    selection_topology: Iterable[tuple[int, int]] | None = None,
) -> PartitionOptimizationResult:
    values = _validated_observations(initial, observations)
    intensities = _validated_component_intensities(
        initial, component_intensities
    )
    edges = tuple(sorted({_canonical_candidate_edge(edge) for edge in candidate_edges}))
    if any(first not in values or second not in values for first, second in edges):
        raise ValueError("candidate edge is outside observation support")
    current = initial
    component_cost_cache: dict[tuple[ObjectCopy, ...], float] = {}

    def component_cost(component: tuple[ObjectCopy, ...]) -> float:
        canonical = tuple(sorted(component))
        if canonical not in component_cost_cache:
            component_cost_cache[canonical] = component_negative_log_evidence(
                tuple(values[copy] for copy in canonical), prior
            ) - np.log(intensities[canonical[0].semantic_class])
        return component_cost_cache[canonical]

    current_objective = sum(component_cost(value) for value in current.components)
    trace = [current_objective]
    accepted = []
    consensus_rounds = 0
    consensus_payload_bytes = 0
    robot_ids = tuple(sorted({copy.robot for copy in initial.copies}))

    def edit_id(edit: PartitionEdit) -> str:
        target_robot = -1 if edit.target is None else edit.target.robot
        target_object = -1 if edit.target is None else edit.target.object_id
        return (
            f"{edit.kind}:{edit.copy.robot}:{edit.copy.object_id}:"
            f"{target_robot}:{target_object}"
        )

    while True:
        proposals: list[tuple[float, str, PartitionEdit]] = []

        for component in current.components:
            if len(component) == 1:
                continue
            for copy in component:
                remainder = tuple(member for member in component if member != copy)
                delta = (
                    component_cost(remainder)
                    + component_cost((copy,))
                    - component_cost(component)
                )
                if delta < 0.0:
                    edit = PartitionEdit("detach", copy, None)
                    proposals.append((delta, edit_id(edit), edit))

        for first, second in edges:
            first_component = _component_containing(current, first)
            second_component = _component_containing(current, second)
            if first_component == second_component:
                continue
            for moved, target in ((first, second), (second, first)):
                source_component = (
                    first_component if moved == first else second_component
                )
                target_component = (
                    second_component if moved == first else first_component
                )
                if moved.robot in {copy.robot for copy in target_component}:
                    continue
                merged = tuple(sorted(target_component + (moved,)))
                delta = component_cost(merged) - component_cost(target_component)
                if len(source_component) == 1:
                    delta -= component_cost(source_component)
                else:
                    remainder = tuple(
                        copy for copy in source_component if copy != moved
                    )
                    delta += component_cost(remainder) - component_cost(
                        source_component
                    )
                if delta < 0.0:
                    kind = (
                        "merge"
                        if len(source_component) == 1
                        else "move"
                    )
                    edit = PartitionEdit(kind, moved, target)
                    proposals.append((delta, edit_id(edit), edit))

        by_id = {identifier: edit for _, identifier, edit in proposals}
        if selection_topology is None:
            if not proposals:
                break
            best_delta, _, best_edit = min(
                proposals, key=lambda value: (value[0], value[1])
            )
        else:
            robot_proposals = {
                robot: [(0.0, "noop")] for robot in robot_ids
            }
            for delta, identifier, edit in proposals:
                robot_proposals[edit.copy.robot].append((delta, identifier))
            consensus = run_min_consensus(
                robot_proposals, selection_topology
            )
            consensus_rounds += consensus.rounds
            consensus_payload_bytes += consensus.payload_bytes
            best_delta, selected_id = consensus.selected
            if selected_id == "noop":
                break
            best_edit = by_id[selected_id]
        if best_edit is None:
            break
        if best_edit.kind == "detach":
            current = detach_partition_copy(current, best_edit.copy)
        else:
            assert best_edit.target is not None
            current = move_partition_copy(
                current, best_edit.copy, best_edit.target
            )
        current_objective += best_delta
        trace.append(current_objective)
        accepted.append(best_edit)
    return PartitionOptimizationResult(
        partition=current,
        objective_trace=tuple(trace),
        accepted_edits=tuple(accepted),
        consensus_rounds=consensus_rounds,
        consensus_payload_bytes=consensus_payload_bytes,
    )


def affine_transform_model(
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    linear: np.ndarray,
    offset: np.ndarray,
) -> tuple[dict[ObjectCopy, GaussianObjectObservation], GaussianObjectPrior]:
    dimension = prior.mean.size
    linear = np.asarray(linear, dtype=float)
    offset = np.asarray(offset, dtype=float).reshape(-1)
    if linear.shape != (dimension, dimension) or offset.shape != (dimension,):
        raise ValueError("affine transform has incompatible dimension")
    if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(offset)):
        raise ValueError("affine transform must be finite")
    if np.linalg.matrix_rank(linear) != dimension:
        raise ValueError("affine transform must be invertible")
    transformed = {
        copy: GaussianObjectObservation(
            copy=copy,
            mean=linear @ observation.mean + offset,
            covariance=linear @ observation.covariance @ linear.T,
        )
        for copy, observation in observations.items()
    }
    transformed_prior = GaussianObjectPrior(
        mean=linear @ prior.mean + offset,
        covariance=linear @ prior.covariance @ linear.T,
    )
    return transformed, transformed_prior


def empirical_object_prior(
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
) -> GaussianObjectPrior:
    values = tuple(observations.values())
    if not values:
        raise ValueError("empirical prior requires observations")
    dimension = values[0].mean.size
    if any(value.mean.size != dimension for value in values):
        raise ValueError("empirical prior observations must share a dimension")
    means = np.stack([value.mean for value in values])
    mean = np.mean(means, axis=0)
    centered = means - mean
    spatial_covariance = centered.T @ centered / len(values)
    measurement_covariance = sum(
        (value.covariance for value in values),
        np.zeros((dimension, dimension)),
    ) / len(values)
    return GaussianObjectPrior(
        mean=mean,
        covariance=spatial_covariance + measurement_covariance,
    )


def empirical_component_intensities(
    partition: NetworkPartition,
) -> dict[int, float]:
    counts: dict[int, float] = {}
    for component in partition.components:
        semantic_class = component[0].semantic_class
        counts[semantic_class] = counts.get(semantic_class, 0.0) + 1.0
    return counts


def category_complete_candidate_edges(
    copies: Iterable[ObjectCopy],
    topology: Iterable[tuple[int, int]],
) -> tuple[tuple[ObjectCopy, ObjectCopy], ...]:
    support = tuple(sorted(copies))
    if len(set(support)) != len(support):
        raise ValueError("candidate support copies must be unique")
    by_robot: dict[int, list[ObjectCopy]] = {}
    for copy in support:
        by_robot.setdefault(copy.robot, []).append(copy)
    normalized_topology = set()
    for edge in topology:
        endpoints = tuple(edge)
        if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
            raise ValueError("topology edges must join distinct robots")
        first, second = sorted(map(int, endpoints))
        if first not in by_robot or second not in by_robot:
            raise ValueError("topology edge refers to an unobserved robot")
        normalized_topology.add((first, second))
    result = set()
    for first_robot, second_robot in normalized_topology:
        for first in by_robot[first_robot]:
            for second in by_robot[second_robot]:
                if first.semantic_class == second.semantic_class:
                    result.add(tuple(sorted((first, second))))
    return tuple(sorted(result))


def reciprocal_physical_candidate_edges(
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    topology: Iterable[tuple[int, int]],
) -> tuple[tuple[ObjectCopy, ObjectCopy], ...]:
    values = dict(observations)
    if not values or any(value.mean.size < 2 for value in values.values()):
        raise ValueError("physical candidate support requires 2D observations")
    complete = category_complete_candidate_edges(values, topology)
    best: dict[tuple[ObjectCopy, int], tuple[float, ObjectCopy]] = {}
    for first, second in complete:
        distance = float(
            np.linalg.norm(values[first].mean[:2] - values[second].mean[:2])
        )
        for source, target in ((first, second), (second, first)):
            key = (source, target.robot)
            candidate = (distance, target)
            if key not in best or candidate < best[key]:
                best[key] = candidate
    return tuple(
        edge
        for edge in complete
        if best[(edge[0], edge[1].robot)][1] == edge[1]
        and best[(edge[1], edge[0].robot)][1] == edge[0]
    )


def partition_association_rows(
    partition: NetworkPartition,
    topology: Iterable[tuple[int, int]],
) -> tuple[dict[str, float | int], ...]:
    topology_pairs = {tuple(sorted(map(int, edge))) for edge in topology}
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in topology_pairs):
        raise ValueError("topology edges must join distinct robots")
    rows = []
    for component in partition.components:
        by_robot = {copy.robot: copy for copy in component}
        for first_robot, second_robot in sorted(topology_pairs):
            if first_robot not in by_robot or second_robot not in by_robot:
                continue
            first = by_robot[first_robot]
            second = by_robot[second_robot]
            rows.extend(
                (
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
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row["receiver_robot"],
                row["sender_robot"],
                row["receiver_object"],
                row["sender_object"],
            ),
        )
    )


def transform_robot_observations(
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    transforms: Mapping[int, tuple[float, np.ndarray]],
) -> dict[ObjectCopy, GaussianObjectObservation]:
    result = {}
    robots = {copy.robot for copy in observations}
    if set(transforms) != robots:
        raise ValueError("transforms must cover every observed robot")
    for copy, observation in observations.items():
        yaw, translation = transforms[copy.robot]
        rotation = rotation2(float(yaw))
        translation = np.asarray(translation, dtype=float).reshape(-1)
        if observation.mean.size < 2 or translation.shape != (2,):
            raise ValueError("SE(2) profiling requires at least two dimensions")
        transform = np.eye(observation.mean.size)
        transform[:2, :2] = rotation
        mean = np.array(observation.mean, copy=True)
        mean[:2] = rotation @ observation.mean[:2] + translation
        covariance = transform @ observation.covariance @ transform.T
        covariance = 0.5 * (covariance + covariance.T)
        result[copy] = GaussianObjectObservation(
            copy=copy,
            mean=mean,
            covariance=covariance,
        )
    return result


def _pack_transforms(
    robot_ids: tuple[int, ...],
    root_robot: int,
    transforms: Mapping[int, tuple[float, np.ndarray]] | None,
) -> np.ndarray:
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    vector = np.zeros(3 * len(nonroot))
    if transforms is None:
        return vector
    if set(transforms) != set(robot_ids):
        raise ValueError("initial transforms must cover every robot")
    for index, robot in enumerate(nonroot):
        yaw, translation = transforms[robot]
        translation = np.asarray(translation, dtype=float).reshape(-1)
        if translation.shape != (2,) or not np.all(np.isfinite(translation)):
            raise ValueError("initial translations must be finite 2-vectors")
        vector[3 * index] = float(yaw)
        vector[3 * index + 1 : 3 * index + 3] = translation
    return vector


def _unpack_transforms(
    robot_ids: tuple[int, ...], root_robot: int, vector: np.ndarray
) -> dict[int, tuple[float, np.ndarray]]:
    transforms = {root_robot: (0.0, np.zeros(2))}
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    for index, robot in enumerate(nonroot):
        transforms[robot] = (
            float(vector[3 * index]),
            np.array(vector[3 * index + 1 : 3 * index + 3], copy=True),
        )
    return transforms


def profile_partition_transforms(
    partition: NetworkPartition,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    root_robot: int,
    initial_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
    transform_priors: Mapping[int, GaussianTransformPrior] | None = None,
    component_intensities: Mapping[int, float] | None = None,
) -> ProfiledPartitionResult:
    values = _validated_observations(partition, observations)
    robot_ids = tuple(sorted({copy.robot for copy in partition.copies}))
    if root_robot not in robot_ids:
        raise ValueError("root robot must be observed")
    transform_priors = dict(transform_priors or {})
    if not set(transform_priors).issubset(set(robot_ids) - {root_robot}):
        raise ValueError("transform priors must refer to nonroot observed robots")
    if any(
        not isinstance(value, GaussianTransformPrior)
        for value in transform_priors.values()
    ):
        raise ValueError("transform priors must be GaussianTransformPrior values")
    initial = _pack_transforms(robot_ids, root_robot, initial_transforms)

    def objective(vector: np.ndarray) -> float:
        transforms = _unpack_transforms(robot_ids, root_robot, vector)
        transformed = transform_robot_observations(values, transforms)
        return partition_objective(
            partition,
            transformed,
            prior,
            component_intensities=component_intensities,
        ) + (
            transform_prior_negative_log_density(transforms, transform_priors)
        )

    isotropic_covariances = all(
        np.allclose(
            value.covariance[:2, :2],
            np.eye(2) * np.trace(value.covariance[:2, :2]) / 2.0,
            atol=1e-12,
            rtol=1e-12,
        )
        and np.allclose(value.covariance[:2, 2:], 0.0, atol=1e-12, rtol=1e-12)
        for value in values.values()
    )

    def objective_with_gradient(
        vector: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        transforms = _unpack_transforms(robot_ids, root_robot, vector)
        transformed = transform_robot_observations(values, transforms)
        value, mean_gradients = partition_objective_with_mean_gradients(
            partition,
            transformed,
            prior,
            component_intensities=component_intensities,
        )
        value += transform_prior_negative_log_density(
            transforms, transform_priors
        )
        gradient = np.zeros_like(vector)
        skew = np.array([[0.0, -1.0], [1.0, 0.0]])
        nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
        offset_by_robot = {
            robot: 3 * index for index, robot in enumerate(nonroot)
        }
        for copy, observation in values.items():
            if copy.robot == root_robot:
                continue
            offset = offset_by_robot[copy.robot]
            rotation = rotation2(transforms[copy.robot][0])
            mean_gradient = mean_gradients[copy][:2]
            gradient[offset] += mean_gradient @ (
                skew @ rotation @ observation.mean[:2]
            )
            gradient[offset + 1 : offset + 3] += mean_gradient
        gradient += transform_prior_gradient(
            transforms, transform_priors, nonroot
        )
        return float(value), gradient

    if isotropic_covariances:
        result = minimize(
            objective_with_gradient,
            initial,
            method="BFGS",
            jac=True,
            options={"gtol": 1e-9, "maxiter": 500},
        )
    else:
        result = minimize(
            objective,
            initial,
            method="BFGS",
            options={"gtol": 1e-9, "maxiter": 500},
        )
    transforms = _unpack_transforms(robot_ids, root_robot, result.x)
    transformed = transform_robot_observations(values, transforms)
    return ProfiledPartitionResult(
        partition=partition,
        transforms=transforms,
        transformed_observations=transformed,
        objective=objective(result.x),
        evaluations=int(result.nfev),
        converged=bool(result.success),
    )


def run_collapsed_profile_partitioning(
    initial: NetworkPartition,
    observations: Mapping[ObjectCopy, GaussianObjectObservation],
    prior: GaussianObjectPrior,
    candidate_edges: Iterable[Iterable[ObjectCopy]],
    root_robot: int,
    initial_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
    transform_priors: Mapping[int, GaussianTransformPrior] | None = None,
    component_intensities: Mapping[int, float] | None = None,
    selection_topology: Iterable[tuple[int, int]] | None = None,
    candidate_topology: Iterable[tuple[int, int]] | None = None,
) -> CollapsedProfileResult:
    static_candidate_edges = tuple(candidate_edges)
    dynamic_candidate_topology = (
        None if candidate_topology is None else tuple(candidate_topology)
    )
    current = profile_partition_transforms(
        initial,
        observations,
        prior,
        root_robot,
        initial_transforms=initial_transforms,
        transform_priors=transform_priors,
        component_intensities=component_intensities,
    )
    trace = [current.objective]
    edits = []
    consensus_rounds = 0
    consensus_payload_bytes = 0
    candidate_edge_counts = []
    while True:
        active_candidate_edges = (
            static_candidate_edges
            if dynamic_candidate_topology is None
            else tuple(
                sorted(
                    set(static_candidate_edges)
                    | set(
                        reciprocal_physical_candidate_edges(
                            current.transformed_observations,
                            dynamic_candidate_topology,
                        )
                    )
                )
            )
        )
        candidate_edge_counts.append(len(active_candidate_edges))
        partition_step = optimize_partition(
            current.partition,
            current.transformed_observations,
            prior,
            active_candidate_edges,
            component_intensities=component_intensities,
            selection_topology=selection_topology,
        )
        edits.extend(partition_step.accepted_edits)
        consensus_rounds += partition_step.consensus_rounds
        consensus_payload_bytes += partition_step.consensus_payload_bytes
        transform_nll = transform_prior_negative_log_density(
            current.transforms, transform_priors or {}
        )
        trace.extend(
            value + transform_nll
            for value in partition_step.objective_trace[1:]
        )
        if partition_step.partition == current.partition:
            break
        profiled = profile_partition_transforms(
            partition_step.partition,
            observations,
            prior,
            root_robot,
            initial_transforms=current.transforms,
            transform_priors=transform_priors,
            component_intensities=component_intensities,
        )
        if profiled.objective > trace[-1] + 1e-9:
            raise RuntimeError("lower profile increased the shared objective")
        trace.append(profiled.objective)
        current = profiled
    return CollapsedProfileResult(
        partition=current.partition,
        transforms=current.transforms,
        objective_trace=tuple(trace),
        accepted_edits=tuple(edits),
        consensus_rounds=consensus_rounds,
        consensus_payload_bytes=consensus_payload_bytes,
        candidate_edge_counts=tuple(candidate_edge_counts),
    )


def transform_prior_negative_log_density(
    transforms: Mapping[int, tuple[float, np.ndarray]],
    priors: Mapping[int, GaussianTransformPrior],
) -> float:
    value = 0.0
    for robot, prior in priors.items():
        if robot not in transforms:
            raise ValueError("transform prior robot is absent from transforms")
        yaw, translation = transforms[robot]
        translation = np.asarray(translation, dtype=float).reshape(-1)
        if translation.shape != (2,):
            raise ValueError("transform translations must be 2-vectors")
        residual = np.array(
            [float(yaw), translation[0], translation[1]], dtype=float
        ) - prior.mean
        residual[0] = np.arctan2(np.sin(residual[0]), np.cos(residual[0]))
        sign, logdet = np.linalg.slogdet(prior.covariance)
        if sign <= 0.0:
            raise ValueError("transform prior covariance must be positive definite")
        value += 0.5 * (
            residual @ np.linalg.solve(prior.covariance, residual)
            + logdet
            + 3.0 * np.log(2.0 * np.pi)
        )
    return float(value)


def transform_prior_gradient(
    transforms: Mapping[int, tuple[float, np.ndarray]],
    priors: Mapping[int, GaussianTransformPrior],
    nonroot_robots: Iterable[int],
) -> np.ndarray:
    robots = tuple(nonroot_robots)
    offset_by_robot = {
        robot: 3 * index for index, robot in enumerate(robots)
    }
    gradient = np.zeros(3 * len(robots))
    for robot, prior in priors.items():
        if robot not in offset_by_robot or robot not in transforms:
            raise ValueError("transform prior robot is absent from nonroot state")
        yaw, translation = transforms[robot]
        translation = np.asarray(translation, dtype=float).reshape(2)
        residual = np.array(
            [float(yaw), translation[0], translation[1]], dtype=float
        ) - prior.mean
        residual[0] = np.arctan2(np.sin(residual[0]), np.cos(residual[0]))
        offset = offset_by_robot[robot]
        gradient[offset : offset + 3] += np.linalg.solve(
            prior.covariance, residual
        )
    return gradient
