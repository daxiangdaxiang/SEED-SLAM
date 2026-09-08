#!/usr/bin/env python3
"""Joint Gaussian evidence for state-predictive DBA partitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from scripts.dba_collapsed_profile_partition import (
    GaussianObjectObservation,
    GaussianObjectPrior,
    PartitionEdit,
    component_negative_log_evidence,
    detach_partition_copy,
    enumerate_feasible_partition_edits,
    move_partition_copy,
    reciprocal_physical_candidate_edges,
    transform_robot_observations,
)
from scripts.dba_kronecker_precision import KroneckerPrecisionFactor
from scripts.dba_network_quotient_reference import (
    NetworkPartition,
    NetworkQuotientProblem,
    ObjectCopy,
    partition_from_edges,
    rotation2,
    run_min_consensus,
)


@dataclass(frozen=True)
class JointCavityObjective:
    total: float
    state_nll: float
    mahalanobis: float
    log_determinant: float
    normalization: float
    cardinality: float
    attribute_nll: float = 0.0


class _AttributeEvidence:
    """Cached collapsed evidence for transform-invariant object attributes."""

    def __init__(
        self,
        copies: tuple[ObjectCopy, ...],
        observations: Mapping[ObjectCopy, GaussianObjectObservation],
        prior: GaussianObjectPrior,
    ) -> None:
        self.observations = dict(observations)
        if set(self.observations) != set(copies):
            raise ValueError("attribute observations must match partition support")
        if any(
            copy != observation.copy
            for copy, observation in self.observations.items()
        ):
            raise ValueError("attribute observation keys must match their copies")
        dimensions = {
            observation.mean.size for observation in self.observations.values()
        }
        if dimensions != {prior.mean.size}:
            raise ValueError("attribute observations and prior must share a dimension")
        self.prior = prior
        self.component_costs: dict[tuple[ObjectCopy, ...], float] = {}

    def evaluate(self, partition: NetworkPartition) -> float:
        if set(partition.copies) != set(self.observations):
            raise ValueError("attribute evidence partition support does not match")
        total = 0.0
        for component in partition.components:
            canonical = tuple(sorted(component))
            if canonical not in self.component_costs:
                self.component_costs[canonical] = component_negative_log_evidence(
                    tuple(self.observations[copy] for copy in canonical),
                    self.prior,
                )
            total += self.component_costs[canonical]
        return float(total)


@dataclass(frozen=True)
class JointCavityCanonicalPacket:
    robot: int
    component_ids: tuple[int, ...]
    precision: np.ndarray
    information: np.ndarray
    constant: float
    log_noise_determinant: float
    observation_count: int

    def __post_init__(self) -> None:
        component_ids = tuple(map(int, self.component_ids))
        if (
            not component_ids
            or tuple(sorted(set(component_ids))) != component_ids
        ):
            raise ValueError("packet component IDs must be sorted and unique")
        precision = np.array(self.precision, dtype=float, copy=True)
        information = np.array(self.information, dtype=float, copy=True)
        count = len(component_ids)
        if precision.shape != (count, count) or not np.allclose(
            precision, precision.T, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("packet precision must be symmetric")
        if information.shape != (count, 2):
            raise ValueError("packet information must have two columns")
        if not np.all(np.isfinite(precision)) or not np.all(
            np.isfinite(information)
        ):
            raise ValueError("packet arrays must be finite")
        if (
            not np.isfinite(self.constant)
            or not np.isfinite(self.log_noise_determinant)
            or int(self.observation_count) <= 0
        ):
            raise ValueError("packet scalars must be finite and valid")
        precision.setflags(write=False)
        information.setflags(write=False)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "information", information)
        object.__setattr__(self, "observation_count", int(self.observation_count))


@dataclass(frozen=True)
class DistributedJointCavityResult:
    objective: JointCavityObjective
    reduction_messages: int
    reduction_payload_bytes: int
    broadcast_messages: int
    broadcast_payload_bytes: int


@dataclass(frozen=True)
class JointCavityProfileResult:
    partition: NetworkPartition
    transforms: Mapping[int, tuple[float, np.ndarray]]
    objective: JointCavityObjective
    evaluations: int
    converged: bool
    gradient_norm: float
    status: int
    message: str

    def __post_init__(self) -> None:
        frozen = {
            int(robot): (
                float(transform[0]),
                np.array(transform[1], dtype=float, copy=True).reshape(2),
            )
            for robot, transform in self.transforms.items()
        }
        for _, translation in frozen.values():
            translation.setflags(write=False)
        object.__setattr__(self, "transforms", MappingProxyType(frozen))


@dataclass(frozen=True)
class JointCavityPartitionResult:
    partition: NetworkPartition
    transforms: Mapping[int, tuple[float, np.ndarray]]
    objective: JointCavityObjective
    objective_trace: tuple[float, ...]
    accepted_edits: tuple[PartitionEdit, ...]
    consensus_rounds: int
    consensus_payload_bytes: int
    candidate_edge_counts: tuple[int, ...]
    objective_evaluations: int
    canonical_state_sync_messages: int
    canonical_state_sync_payload_bytes: int

    def __post_init__(self) -> None:
        frozen = {
            int(robot): (
                float(transform[0]),
                np.array(transform[1], dtype=float, copy=True).reshape(2),
            )
            for robot, transform in self.transforms.items()
        }
        for _, translation in frozen.values():
            translation.setflags(write=False)
        object.__setattr__(self, "transforms", MappingProxyType(frozen))


@dataclass(frozen=True)
class PairwiseJointCavityResult:
    relation_edges: tuple[tuple[ObjectCopy, ObjectCopy], ...]
    partition: NetworkPartition | None
    transforms: Mapping[int, tuple[float, np.ndarray]]
    objective: JointCavityObjective
    objective_trace: tuple[float, ...]
    accepted_edits: tuple[PartitionEdit, ...]
    candidate_edge_counts: tuple[int, ...]
    objective_evaluations: int
    canonical_state_sync_messages: int
    canonical_state_sync_payload_bytes: int
    network_feasible: bool
    duplicate_robot_component_count: int

    def __post_init__(self) -> None:
        frozen = {
            int(robot): (
                float(transform[0]),
                np.array(transform[1], dtype=float, copy=True).reshape(2),
            )
            for robot, transform in self.transforms.items()
        }
        for _, translation in frozen.values():
            translation.setflags(write=False)
        object.__setattr__(self, "transforms", MappingProxyType(frozen))


def _validated_component_intensities(
    partition: NetworkPartition,
    supplied: Mapping[int, float],
) -> dict[int, float]:
    expected = {copy.semantic_class for copy in partition.copies}
    intensities = {
        int(semantic_class): float(value)
        for semantic_class, value in supplied.items()
    }
    if set(intensities) != expected:
        raise ValueError("component intensities must cover every semantic class")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in intensities.values()
    ):
        raise ValueError("component intensities must be finite and positive")
    return intensities


def _validated_robot_covariance_mode(mode: str) -> str:
    value = str(mode)
    if value not in ("full", "diagonal"):
        raise ValueError(
            "robot covariance mode must be 'full' or 'diagonal'"
        )
    return value


def _active_robot_covariance(
    covariance: np.ndarray, mode: str
) -> np.ndarray:
    value = _validated_robot_covariance_mode(mode)
    if value == "full":
        return covariance
    return np.diag(np.diag(covariance))


def _component_cardinality_nll(
    partition: NetworkPartition,
    intensities: Mapping[int, float],
    include_component_intensity: bool,
) -> float:
    if not include_component_intensity:
        return 0.0
    return float(
        -sum(
            np.log(intensities[component[0].semantic_class])
            for component in partition.components
        )
    )


def _validated_transforms(
    problem: NetworkQuotientProblem,
    supplied: Mapping[int, tuple[float, np.ndarray]],
) -> dict[int, tuple[float, np.ndarray]]:
    expected = {state.robot for state in problem.robots}
    if set(supplied) != expected:
        raise ValueError("transforms must cover every robot")
    transforms = {}
    for robot, transform in supplied.items():
        yaw = float(transform[0])
        translation = np.asarray(transform[1], dtype=float).reshape(-1)
        if (
            not np.isfinite(yaw)
            or translation.shape != (2,)
            or not np.all(np.isfinite(translation))
        ):
            raise ValueError("transforms must contain finite SE(2) values")
        transforms[int(robot)] = (yaw, translation)
    return transforms


def _observation_model(
    partition: NetworkPartition,
    problem: NetworkQuotientProblem,
    transforms: Mapping[int, tuple[float, np.ndarray]],
    prior: GaussianObjectPrior,
    *,
    robot_covariance_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if partition.copies != problem.partition.copies:
        raise ValueError("partition support must match the cavity problem")
    if prior.mean.shape != (2,) or prior.covariance.shape != (2, 2):
        raise ValueError("joint cavity evidence requires a two-dimensional prior")
    values = _validated_transforms(problem, transforms)
    copies = tuple(copy for state in problem.robots for copy in state.copies)
    component_by_copy: dict[ObjectCopy, int] = {
        copy: index
        for index, component in enumerate(partition.components)
        for copy in component
    }
    assignment = np.zeros((len(copies), partition.component_count))
    for row, copy in enumerate(copies):
        assignment[row, component_by_copy[copy]] = 1.0
    expanded_assignment = np.kron(assignment, np.eye(2))

    transformed_means = []
    covariance_blocks = []
    for state in problem.robots:
        yaw, translation = values[state.robot]
        rotation = rotation2(yaw)
        transformed_means.append(state.means @ rotation.T + translation)
        covariance_blocks.append(
            np.kron(
                _active_robot_covariance(
                    state.covariance, robot_covariance_mode
                ),
                np.eye(2),
            )
        )

    dimension = 2 * len(copies)
    noise_covariance = np.zeros((dimension, dimension))
    offset = 0
    for block in covariance_blocks:
        width = block.shape[0]
        noise_covariance[offset : offset + width, offset : offset + width] = block
        offset += width
    prior_mean = np.tile(prior.mean, partition.component_count)
    prior_covariance = np.kron(
        np.eye(partition.component_count), prior.covariance
    )
    return (
        np.concatenate(transformed_means).reshape(-1),
        expanded_assignment @ prior_mean,
        noise_covariance
        + expanded_assignment @ prior_covariance @ expanded_assignment.T,
    )


def joint_cavity_partition_objective(
    partition: NetworkPartition,
    problem: NetworkQuotientProblem,
    transforms: Mapping[int, tuple[float, np.ndarray]],
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float],
    attribute_evidence: _AttributeEvidence | None = None,
    *,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> JointCavityObjective:
    """Evaluate normalized joint cavity evidence for one partition."""
    if partition.copies != problem.partition.copies:
        raise ValueError("partition support must match the cavity problem")
    intensities = _validated_component_intensities(
        partition, component_intensities
    )
    objective, _ = _joint_objective_and_observation_gradient(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    attribute_nll = (
        0.0
        if attribute_evidence is None
        else attribute_evidence.evaluate(partition)
    )
    return JointCavityObjective(
        total=float(objective.total + attribute_nll),
        state_nll=objective.state_nll,
        mahalanobis=objective.mahalanobis,
        log_determinant=objective.log_determinant,
        normalization=objective.normalization,
        cardinality=objective.cardinality,
        attribute_nll=attribute_nll,
    )


def _joint_objective_and_observation_gradient(
    partition: NetworkPartition,
    problem: NetworkQuotientProblem,
    transforms: Mapping[int, tuple[float, np.ndarray]],
    prior: GaussianObjectPrior,
    intensities: Mapping[int, float],
    *,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> tuple[JointCavityObjective, np.ndarray]:
    observation, mean, covariance = _observation_model(
        partition,
        problem,
        transforms,
        prior,
        robot_covariance_mode=robot_covariance_mode,
    )
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("joint marginal covariance must be positive definite") from error
    residual = observation - mean
    whitened = np.linalg.solve(factor, residual)
    observation_gradient = np.linalg.solve(factor.T, whitened)
    mahalanobis = 0.5 * float(whitened @ whitened)
    log_determinant = float(np.sum(np.log(np.diag(factor))))
    normalization = 0.5 * residual.size * float(np.log(2.0 * np.pi))
    state_nll = mahalanobis + log_determinant + normalization
    cardinality = _component_cardinality_nll(
        partition, intensities, include_component_intensity
    )
    objective = JointCavityObjective(
        total=float(state_nll + cardinality),
        state_nll=float(state_nll),
        mahalanobis=mahalanobis,
        log_determinant=log_determinant,
        normalization=normalization,
        cardinality=float(cardinality),
    )
    return objective, observation_gradient


class _DensePrecisionFactor:
    """Dense test reference with the structured factor's solve interface."""

    def __init__(self, precision: np.ndarray) -> None:
        self.factor, self.lower = cho_factor(
            precision, lower=True, check_finite=False
        )
        self.half_log_determinant = float(
            np.log(np.diag(self.factor)).sum()
        )

    def solve(self, information: np.ndarray) -> np.ndarray:
        return cho_solve(
            (self.factor, self.lower), information, check_finite=False
        )


class _JointCavityCanonicalWorkspace:
    """Exact information-form marginal with partition-constant curvature."""

    def __init__(
        self,
        partition: NetworkPartition,
        problem: NetworkQuotientProblem,
        prior: GaussianObjectPrior,
        component_intensities: Mapping[int, float],
        problem_statistics: Mapping[
            int, tuple[np.ndarray, float]
        ] | None = None,
        attribute_evidence: _AttributeEvidence | None = None,
        *,
        precision_backend: str = "structured",
        include_component_intensity: bool = True,
        robot_covariance_mode: str = "full",
    ) -> None:
        if partition.copies != problem.partition.copies:
            raise ValueError("partition support must match the cavity problem")
        self.partition = partition
        self.problem = problem
        self.prior = prior
        self.intensities = _validated_component_intensities(
            partition, component_intensities
        )
        self.include_component_intensity = bool(
            include_component_intensity
        )
        self.robot_covariance_mode = _validated_robot_covariance_mode(
            robot_covariance_mode
        )
        self.component_by_copy = {
            copy: index
            for index, component in enumerate(partition.components)
            for copy in component
        }
        component_count = partition.component_count
        latent_dimension = 2 * component_count
        prior_precision = np.linalg.solve(prior.covariance, np.eye(2))
        prior_information = prior_precision @ prior.mean
        sign, prior_logdet = np.linalg.slogdet(prior.covariance)
        if sign <= 0.0:
            raise ValueError("object prior covariance must be positive definite")
        self.precisions = {}
        self.covariance_log_determinants = {}
        self.component_ids = {}
        component_precision = np.zeros((component_count, component_count))
        noise_logdet = 0.0
        observation_count = 0
        statistics = (
            _joint_cavity_problem_statistics(
                problem, robot_covariance_mode=self.robot_covariance_mode
            )
            if problem_statistics is None
            else dict(problem_statistics)
        )
        if set(statistics) != {state.robot for state in problem.robots}:
            raise ValueError("cavity statistics must cover every robot")
        for state in problem.robots:
            inverse, covariance_logdet = statistics[state.robot]
            self.precisions[state.robot] = inverse
            self.covariance_log_determinants[state.robot] = covariance_logdet
            ids = np.array(
                [self.component_by_copy[copy] for copy in state.copies],
                dtype=int,
            )
            if len(set(map(int, ids))) != len(ids):
                raise ValueError(
                    "a robot must contribute at most one copy per component"
                )
            self.component_ids[state.robot] = ids
            component_precision[np.ix_(ids, ids)] += inverse
            noise_logdet += 2.0 * covariance_logdet
            observation_count += len(state.copies)
        if precision_backend == "structured":
            self.precision_factor = KroneckerPrecisionFactor.build(
                component_precision, prior_precision
            )
        elif precision_backend == "dense":
            precision = np.kron(component_precision, np.eye(2))
            precision += np.kron(
                np.eye(component_count), prior_precision
            )
            self.precision_factor = _DensePrecisionFactor(precision)
        else:
            raise ValueError(
                "unsupported canonical precision backend: "
                f"{precision_backend}"
            )
        self.prior_information = np.tile(
            prior_information, component_count
        )
        prior_constant = 0.5 * (
            prior.mean @ prior_information
            + prior_logdet
            + 2.0 * np.log(2.0 * np.pi)
        )
        noise_constant = 0.5 * (
            noise_logdet
            + 2.0 * observation_count * np.log(2.0 * np.pi)
        )
        self.constant = (
            noise_constant
            + component_count * prior_constant
            - component_count * np.log(2.0 * np.pi)
            + self.precision_factor.half_log_determinant
        )
        self.cardinality = _component_cardinality_nll(
            partition,
            self.intensities,
            self.include_component_intensity,
        )
        self.attribute_nll = (
            0.0
            if attribute_evidence is None
            else attribute_evidence.evaluate(partition)
        )

    def evaluate(
        self,
        transforms: Mapping[int, tuple[float, np.ndarray]],
    ) -> tuple[float, dict[int, np.ndarray]]:
        values = _validated_transforms(self.problem, transforms)
        information = np.array(self.prior_information, copy=True)
        noise_quadratic = 0.0
        transformed_by_robot = {}
        for state in self.problem.robots:
            yaw, translation = values[state.robot]
            transformed = state.means @ rotation2(yaw).T + translation
            transformed_by_robot[state.robot] = transformed
            weighted = self.precisions[state.robot] @ transformed
            noise_quadratic += float(np.sum(transformed * weighted))
            component_information = np.zeros(
                (self.partition.component_count, 2)
            )
            component_information[self.component_ids[state.robot]] += weighted
            information += component_information.reshape(-1)
        posterior_mean = self.precision_factor.solve(information)
        total = (
            0.5 * noise_quadratic
            + self.constant
            - 0.5 * float(information @ posterior_mean)
            + self.cardinality
            + self.attribute_nll
        )
        gradients = {}
        latent_means = posterior_mean.reshape((-1, 2))
        for state in self.problem.robots:
            residual = transformed_by_robot[state.robot] - latent_means[
                self.component_ids[state.robot]
            ]
            gradients[state.robot] = self.precisions[state.robot] @ residual
        return float(total), gradients

    def canonical_packets(
        self,
        transforms: Mapping[int, tuple[float, np.ndarray]],
    ) -> tuple[JointCavityCanonicalPacket, ...]:
        values = _validated_transforms(self.problem, transforms)
        packets = []
        for state in self.problem.robots:
            yaw, translation = values[state.robot]
            transformed = state.means @ rotation2(yaw).T + translation
            precision = self.precisions[state.robot]
            weighted = precision @ transformed
            component_ids, local_ids = np.unique(
                self.component_ids[state.robot], return_inverse=True
            )
            local_precision = np.zeros(
                (len(component_ids), len(component_ids))
            )
            np.add.at(
                local_precision,
                (local_ids[:, np.newaxis], local_ids[np.newaxis, :]),
                precision,
            )
            local_information = np.zeros((len(component_ids), 2))
            np.add.at(local_information, local_ids, weighted)
            log_noise_determinant = (
                2.0 * self.covariance_log_determinants[state.robot]
            )
            constant = 0.5 * (
                float(np.sum(transformed * weighted))
                + log_noise_determinant
                + 2.0 * len(state.copies) * np.log(2.0 * np.pi)
            )
            packets.append(
                JointCavityCanonicalPacket(
                    robot=state.robot,
                    component_ids=tuple(map(int, component_ids)),
                    precision=local_precision,
                    information=local_information,
                    constant=constant,
                    log_noise_determinant=log_noise_determinant,
                    observation_count=len(state.copies),
                )
            )
        return tuple(packets)

    def distributed_evaluate(
        self,
        transforms: Mapping[int, tuple[float, np.ndarray]],
        topology: tuple[tuple[int, int], ...],
        root_robot: int,
    ) -> DistributedJointCavityResult:
        return _reduce_joint_cavity_packets(
            self.canonical_packets(transforms),
            self.partition,
            self.prior,
            self.intensities,
            topology,
            root_robot,
            include_component_intensity=self.include_component_intensity,
        )


def _joint_cavity_problem_statistics(
    problem: NetworkQuotientProblem,
    *,
    robot_covariance_mode: str = "full",
) -> dict[int, tuple[np.ndarray, float]]:
    statistics = {}
    for state in problem.robots:
        covariance = _active_robot_covariance(
            state.covariance, robot_covariance_mode
        )
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            raise ValueError("cavity covariance must be positive definite")
        statistics[state.robot] = (
            np.linalg.solve(covariance, np.eye(len(state.copies))),
            float(logdet),
        )
    return statistics


def _validated_reduction_tree(
    robot_ids: tuple[int, ...],
    topology: tuple[tuple[int, int], ...],
    root_robot: int,
) -> tuple[dict[int, tuple[int, ...]], dict[int, int], tuple[int, ...]]:
    if root_robot not in robot_ids:
        raise ValueError("reduction root must be present")
    edges = set()
    for supplied in topology:
        if len(supplied) != 2:
            raise ValueError("reduction topology edges must have two robots")
        first, second = map(int, supplied)
        if first == second or first not in robot_ids or second not in robot_ids:
            raise ValueError("reduction topology edge is invalid")
        edges.add(tuple(sorted((first, second))))
    if len(edges) != max(0, len(robot_ids) - 1):
        raise ValueError("canonical reduction requires a tree topology")
    neighbors = {robot: [] for robot in robot_ids}
    for first, second in sorted(edges):
        neighbors[first].append(second)
        neighbors[second].append(first)
    parent = {root_robot: root_robot}
    order = [root_robot]
    for robot in order:
        for neighbor in sorted(neighbors[robot]):
            if neighbor not in parent:
                parent[neighbor] = robot
                order.append(neighbor)
    if len(parent) != len(robot_ids):
        raise ValueError("canonical reduction topology must be connected")
    return (
        {robot: tuple(sorted(values)) for robot, values in neighbors.items()},
        parent,
        tuple(order),
    )


def joint_cavity_state_sync_accounting(
    problem: NetworkQuotientProblem,
    topology: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Count one tree all-gather of reusable copy-space canonical state."""
    robot_ids = tuple(state.robot for state in problem.robots)
    _validated_reduction_tree(robot_ids, topology, min(robot_ids))
    edge_count = len(robot_ids) - 1
    packet_bytes = 0
    for state in problem.robots:
        copy_count = len(state.copies)
        precision_entries = copy_count * (copy_count + 1) // 2
        packet_bytes += (
            48
            + 16 * copy_count
            + 16 * copy_count
            + 8 * precision_entries
        )
    return len(robot_ids) * edge_count, edge_count * packet_bytes


def _topology_partition_edges(
    partition: NetworkPartition,
    topology: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], tuple[tuple[ObjectCopy, ObjectCopy], ...]]:
    pairs = tuple(sorted({tuple(sorted(map(int, edge))) for edge in topology}))
    result = {pair: [] for pair in pairs}
    for component in partition.components:
        by_robot = {copy.robot: copy for copy in component}
        for pair in pairs:
            if pair[0] in by_robot and pair[1] in by_robot:
                result[pair].append(
                    tuple(sorted((by_robot[pair[0]], by_robot[pair[1]])))
                )
    return {pair: tuple(sorted(edges)) for pair, edges in result.items()}


def _crossfit_fold_partition(
    reference: NetworkPartition,
    candidate: NetworkPartition,
    fold_pair: tuple[int, int],
    topology: tuple[tuple[int, int], ...],
) -> NetworkPartition:
    if reference.copies != candidate.copies:
        raise ValueError("crossfit partitions must have identical support")
    reference_edges = _topology_partition_edges(reference, topology)
    candidate_edges = _topology_partition_edges(candidate, topology)
    return _crossfit_fold_partition_from_edges(
        reference.copies,
        reference_edges,
        candidate_edges,
        fold_pair,
    )


def _crossfit_fold_partition_from_edges(
    copies: tuple[ObjectCopy, ...],
    reference_edges: Mapping[
        tuple[int, int], tuple[tuple[ObjectCopy, ObjectCopy], ...]
    ],
    candidate_edges: Mapping[
        tuple[int, int], tuple[tuple[ObjectCopy, ObjectCopy], ...]
    ],
    fold_pair: tuple[int, int],
) -> NetworkPartition:
    """Build one fold from already-canonical topology edge statistics."""
    fold = tuple(sorted(map(int, fold_pair)))
    if fold not in reference_edges:
        raise ValueError("crossfit fold must belong to the topology")
    if set(candidate_edges) != set(reference_edges):
        raise ValueError("crossfit edge statistics must cover the same topology")
    edges = []
    for pair in sorted(reference_edges):
        edges.extend(
            candidate_edges[pair] if pair == fold else reference_edges[pair]
        )
    return _partition_from_validated_crossfit_edges(copies, edges)


def _partition_from_validated_crossfit_edges(
    copies: tuple[ObjectCopy, ...],
    edges: tuple[tuple[ObjectCopy, ObjectCopy], ...]
    | list[tuple[ObjectCopy, ObjectCopy]],
) -> NetworkPartition:
    """Build a partition from internally validated tree-supported edges."""
    support = tuple(copies)
    copy_index = {copy: index for index, copy in enumerate(support)}
    if len(copy_index) != len(support):
        raise ValueError("crossfit copy support must contain unique copies")
    parent = list(range(len(support)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    seen_edges = set()
    for first, second in edges:
        edge = tuple(sorted((first, second)))
        if edge in seen_edges:
            raise ValueError("crossfit edges must not contain duplicates")
        seen_edges.add(edge)
        if first not in copy_index or second not in copy_index:
            raise ValueError("crossfit edge endpoints must belong to support")
        if (
            first == second
            or first.robot == second.robot
            or first.semantic_class != second.semantic_class
        ):
            raise ValueError("crossfit edge is not association feasible")
        first_root = find(copy_index[first])
        second_root = find(copy_index[second])
        if first_root != second_root:
            low, high = sorted((first_root, second_root))
            parent[high] = low

    components: dict[int, list[ObjectCopy]] = {}
    for index, copy in enumerate(support):
        components.setdefault(find(index), []).append(copy)
    canonical = tuple(
        sorted(tuple(sorted(component)) for component in components.values())
    )
    for component in canonical:
        if len(component) != len({copy.robot for copy in component}):
            raise ValueError("crossfit component repeats a robot")
        if len({copy.semantic_class for copy in component}) != 1:
            raise ValueError("crossfit component has a semantic conflict")
    return NetworkPartition(canonical)


def _packet_accumulator(
    packet: JointCavityCanonicalPacket,
) -> dict[str, object]:
    precision = {}
    for row, first in enumerate(packet.component_ids):
        for column in range(row, len(packet.component_ids)):
            second = packet.component_ids[column]
            precision[(first, second)] = float(packet.precision[row, column])
    information = {
        component: np.array(packet.information[index], copy=True)
        for index, component in enumerate(packet.component_ids)
    }
    return {
        "precision": precision,
        "information": information,
        "constant": float(packet.constant),
        "log_noise_determinant": float(packet.log_noise_determinant),
        "observation_count": int(packet.observation_count),
    }


def _canonical_accumulator_payload_bytes(
    accumulator: Mapping[str, object],
) -> int:
    precision_count = len(accumulator["precision"])
    information_count = len(accumulator["information"])
    return 24 + 24 * precision_count + 24 * information_count


def _merge_canonical_accumulators(
    receiver: dict[str, object], sender: Mapping[str, object]
) -> None:
    receiver_precision = receiver["precision"]
    for key, value in sender["precision"].items():
        receiver_precision[key] = receiver_precision.get(key, 0.0) + value
    receiver_information = receiver["information"]
    for key, value in sender["information"].items():
        receiver_information.setdefault(key, np.zeros(2))
        receiver_information[key] += value
    receiver["constant"] += sender["constant"]
    receiver["log_noise_determinant"] += sender[
        "log_noise_determinant"
    ]
    receiver["observation_count"] += sender["observation_count"]


def _reduce_joint_cavity_packets(
    packets: tuple[JointCavityCanonicalPacket, ...],
    partition: NetworkPartition,
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float],
    topology: tuple[tuple[int, int], ...],
    root_robot: int,
    *,
    include_component_intensity: bool = True,
) -> DistributedJointCavityResult:
    robot_ids = tuple(sorted(packet.robot for packet in packets))
    if len(set(robot_ids)) != len(robot_ids) or not robot_ids:
        raise ValueError("canonical packets must cover unique robots")
    _, parent, order = _validated_reduction_tree(
        robot_ids, tuple(topology), root_robot
    )
    accumulators = {
        packet.robot: _packet_accumulator(packet) for packet in packets
    }
    reduction_payload_bytes = 0
    for robot in reversed(order[1:]):
        reduction_payload_bytes += _canonical_accumulator_payload_bytes(
            accumulators[robot]
        )
        _merge_canonical_accumulators(
            accumulators[parent[robot]], accumulators[robot]
        )
    root = accumulators[root_robot]

    component_count = partition.component_count
    dimension = 2 * component_count
    precision = np.zeros((dimension, dimension))
    for (first, second), value in root["precision"].items():
        first_slice = slice(2 * first, 2 * first + 2)
        second_slice = slice(2 * second, 2 * second + 2)
        block = value * np.eye(2)
        precision[first_slice, second_slice] += block
        if first != second:
            precision[second_slice, first_slice] += block
    prior_precision = np.linalg.solve(prior.covariance, np.eye(2))
    prior_information = prior_precision @ prior.mean
    information = np.zeros((component_count, 2))
    for component, value in root["information"].items():
        information[component] += value
    prior_sign, prior_logdet = np.linalg.slogdet(prior.covariance)
    if prior_sign <= 0.0:
        raise ValueError("object prior covariance must be positive definite")
    prior_constant = 0.5 * (
        prior.mean @ prior_information
        + prior_logdet
        + 2.0 * np.log(2.0 * np.pi)
    )
    for component in range(component_count):
        component_slice = slice(2 * component, 2 * component + 2)
        precision[component_slice, component_slice] += prior_precision
        information[component] += prior_information
    factor = np.linalg.cholesky(precision)
    flat_information = information.reshape(-1)
    intermediate = np.linalg.solve(factor, flat_information)
    posterior_mean = np.linalg.solve(factor.T, intermediate)
    state_nll = (
        float(root["constant"])
        + component_count * prior_constant
        - component_count * np.log(2.0 * np.pi)
        + float(np.sum(np.log(np.diag(factor))))
        - 0.5 * float(flat_information @ posterior_mean)
    )
    normalization = float(root["observation_count"]) * np.log(2.0 * np.pi)
    log_determinant = (
        0.5 * float(root["log_noise_determinant"])
        + 0.5 * component_count * prior_logdet
        + float(np.sum(np.log(np.diag(factor))))
    )
    mahalanobis = state_nll - log_determinant - normalization
    intensities = _validated_component_intensities(
        partition, component_intensities
    )
    cardinality = _component_cardinality_nll(
        partition, intensities, include_component_intensity
    )
    objective = JointCavityObjective(
        total=float(state_nll + cardinality),
        state_nll=float(state_nll),
        mahalanobis=float(mahalanobis),
        log_determinant=float(log_determinant),
        normalization=float(normalization),
        cardinality=float(cardinality),
    )
    edge_count = len(robot_ids) - 1
    return DistributedJointCavityResult(
        objective=objective,
        reduction_messages=edge_count,
        reduction_payload_bytes=reduction_payload_bytes,
        broadcast_messages=edge_count,
        broadcast_payload_bytes=8 * edge_count,
    )


def distributed_joint_cavity_partition_objective(
    partition: NetworkPartition,
    problem: NetworkQuotientProblem,
    transforms: Mapping[int, tuple[float, np.ndarray]],
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float],
    *,
    topology: tuple[tuple[int, int], ...],
    root_robot: int,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> DistributedJointCavityResult:
    """Evaluate exact evidence by topology-routed local canonical packets."""
    workspace = _JointCavityCanonicalWorkspace(
        partition,
        problem,
        prior,
        component_intensities,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    return workspace.distributed_evaluate(transforms, topology, root_robot)


def joint_cavity_transform_objective_with_gradient(
    partition: NetworkPartition,
    problem: NetworkQuotientProblem,
    transforms: Mapping[int, tuple[float, np.ndarray]],
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float],
    *,
    root_robot: int,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> tuple[float, np.ndarray]:
    """Return the objective and nonroot SE(2) transform gradient."""
    robot_ids = tuple(state.robot for state in problem.robots)
    if root_robot not in robot_ids:
        raise ValueError("root robot must be present in the cavity problem")
    intensities = _validated_component_intensities(
        partition, component_intensities
    )
    objective, observation_gradient = _joint_objective_and_observation_gradient(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    offsets = {robot: 3 * index for index, robot in enumerate(nonroot)}
    gradient = np.zeros(3 * len(nonroot))
    skew = np.array([[0.0, -1.0], [1.0, 0.0]])
    observation_offset = 0
    for state in problem.robots:
        count = len(state.copies)
        blocks = observation_gradient[
            observation_offset : observation_offset + 2 * count
        ].reshape((count, 2))
        observation_offset += 2 * count
        if state.robot == root_robot:
            continue
        offset = offsets[state.robot]
        rotation = rotation2(float(transforms[state.robot][0]))
        yaw_derivatives = state.means @ (skew @ rotation).T
        gradient[offset] = float(np.sum(blocks * yaw_derivatives))
        gradient[offset + 1 : offset + 3] = np.sum(blocks, axis=0)
    return objective.total, gradient


def _pack_transforms(
    robot_ids: tuple[int, ...],
    root_robot: int,
    transforms: Mapping[int, tuple[float, np.ndarray]],
    translation_scale: float,
) -> np.ndarray:
    values = _validated_transforms_for_ids(robot_ids, transforms)
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    vector = np.zeros(3 * len(nonroot))
    for index, robot in enumerate(nonroot):
        vector[3 * index] = values[robot][0]
        vector[3 * index + 1 : 3 * index + 3] = (
            values[robot][1] / translation_scale
        )
    return vector


def _validated_transforms_for_ids(
    robot_ids: tuple[int, ...],
    transforms: Mapping[int, tuple[float, np.ndarray]],
) -> dict[int, tuple[float, np.ndarray]]:
    if set(transforms) != set(robot_ids):
        raise ValueError("initial transforms must cover every robot")
    result = {}
    for robot in robot_ids:
        yaw = float(transforms[robot][0])
        translation = np.asarray(transforms[robot][1], dtype=float).reshape(-1)
        if (
            not np.isfinite(yaw)
            or translation.shape != (2,)
            or not np.all(np.isfinite(translation))
        ):
            raise ValueError("initial transforms must contain finite SE(2) values")
        result[robot] = (yaw, translation)
    return result


def _unpack_transforms(
    robot_ids: tuple[int, ...],
    root_robot: int,
    root_transform: tuple[float, np.ndarray],
    vector: np.ndarray,
    translation_scale: float,
) -> dict[int, tuple[float, np.ndarray]]:
    result = {
        root_robot: (
            float(root_transform[0]),
            np.array(root_transform[1], dtype=float, copy=True).reshape(2),
        )
    }
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    for index, robot in enumerate(nonroot):
        result[robot] = (
            float(vector[3 * index]),
            translation_scale
            * np.array(vector[3 * index + 1 : 3 * index + 3], copy=True),
        )
    return result


def profile_joint_cavity_transforms(
    partition: NetworkPartition,
    problem: NetworkQuotientProblem,
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float],
    *,
    root_robot: int,
    initial_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
    attribute_evidence: _AttributeEvidence | None = None,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> JointCavityProfileResult:
    """Profile robot transforms while descending the joint objective."""
    robot_ids = tuple(state.robot for state in problem.robots)
    if root_robot not in robot_ids:
        raise ValueError("root robot must be present in the cavity problem")
    initial = (
        {robot: (0.0, np.zeros(2)) for robot in robot_ids}
        if initial_transforms is None
        else _validated_transforms_for_ids(robot_ids, initial_transforms)
    )
    translation_scale = float(np.sqrt(np.trace(prior.covariance) / 2.0))
    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("transform translation scale must be finite and positive")
    vector = _pack_transforms(
        robot_ids, root_robot, initial, translation_scale
    )
    root_transform = initial[root_robot]
    workspace = _JointCavityCanonicalWorkspace(
        partition,
        problem,
        prior,
        component_intensities,
        attribute_evidence=attribute_evidence,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    states_by_robot = {state.robot: state for state in problem.robots}

    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray]:
        transforms = _unpack_transforms(
            robot_ids,
            root_robot,
            root_transform,
            value,
            translation_scale,
        )
        objective, observation_gradients = workspace.evaluate(transforms)
        gradient = np.zeros(3 * len(nonroot))
        skew = np.array([[0.0, -1.0], [1.0, 0.0]])
        for index, robot in enumerate(nonroot):
            state = states_by_robot[robot]
            rotation = rotation2(float(transforms[robot][0]))
            yaw_derivatives = state.means @ (skew @ rotation).T
            blocks = observation_gradients[robot]
            gradient[3 * index] = float(np.sum(blocks * yaw_derivatives))
            gradient[3 * index + 1 : 3 * index + 3] = (
                translation_scale * np.sum(blocks, axis=0)
            )
        return objective, gradient

    result = minimize(
        evaluate,
        vector,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-9, "maxiter": 500},
    )
    transforms = _unpack_transforms(
        robot_ids,
        root_robot,
        root_transform,
        result.x,
        translation_scale,
    )
    objective = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        component_intensities,
        attribute_evidence,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    initial_objective = joint_cavity_partition_objective(
        partition,
        problem,
        initial,
        prior,
        component_intensities,
        attribute_evidence,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    if objective.total > initial_objective.total + 1e-10:
        transforms = initial
        objective = initial_objective
    _, final_gradient = joint_cavity_transform_objective_with_gradient(
        partition,
        problem,
        transforms,
        prior,
        component_intensities,
        root_robot=root_robot,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    return JointCavityProfileResult(
        partition=partition,
        transforms=transforms,
        objective=objective,
        evaluations=int(result.nfev),
        converged=bool(result.success),
        gradient_norm=float(np.linalg.norm(final_gradient)),
        status=int(result.status),
        message=str(result.message),
    )


def profile_joint_cavity_leave_pair_composite_transforms(
    partition: NetworkPartition,
    leave_pair_problems: Mapping[
        tuple[int, int], NetworkQuotientProblem
    ],
    prior: GaussianObjectPrior,
    component_intensities: Mapping[int, float],
    *,
    root_robot: int,
    initial_transforms: Mapping[int, tuple[float, np.ndarray]],
    reference_partition: NetworkPartition | None = None,
    topology: tuple[tuple[int, int], ...] | None = None,
    attribute_evidence: _AttributeEvidence | None = None,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> JointCavityProfileResult:
    """Profile shared transforms in a cross-fitted network surrogate."""
    if not leave_pair_problems:
        raise ValueError("leave-pair evidence must not be empty")
    problems = {
        tuple(sorted(map(int, pair))): problem
        for pair, problem in leave_pair_problems.items()
    }
    active_topology = tuple(problems) if topology is None else tuple(topology)
    reference = partition if reference_partition is None else reference_partition
    if reference.copies != partition.copies or any(
        problem.partition.copies != partition.copies
        for problem in problems.values()
    ):
        raise ValueError("crossfit support must match every cavity problem")
    fold_partitions = {
        pair: _crossfit_fold_partition(
            reference, partition, pair, active_topology
        )
        for pair in problems
    }
    robot_ids = tuple(sorted({copy.robot for copy in partition.copies}))
    if root_robot not in robot_ids:
        raise ValueError("root robot must be present in the partition")
    initial = _validated_transforms_for_ids(robot_ids, initial_transforms)
    translation_scale = float(np.sqrt(np.trace(prior.covariance) / 2.0))
    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("transform translation scale must be finite and positive")
    root_transform = initial[root_robot]
    vector = _pack_transforms(
        robot_ids, root_robot, initial, translation_scale
    )
    nonroot = tuple(robot for robot in robot_ids if robot != root_robot)
    nonroot_index = {robot: index for index, robot in enumerate(nonroot)}
    workspaces = {}
    states_by_pair = {}
    for pair, problem in sorted(problems.items()):
        workspaces[pair] = _JointCavityCanonicalWorkspace(
            fold_partitions[pair],
            problem,
            prior,
            component_intensities,
            attribute_evidence=attribute_evidence,
            include_component_intensity=include_component_intensity,
            robot_covariance_mode=robot_covariance_mode,
        )
        states_by_pair[pair] = {
            state.robot: state for state in problem.robots
        }

    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray]:
        transforms = _unpack_transforms(
            robot_ids,
            root_robot,
            root_transform,
            value,
            translation_scale,
        )
        total = 0.0
        gradient = np.zeros(3 * len(nonroot))
        skew = np.array([[0.0, -1.0], [1.0, 0.0]])
        for pair, workspace in workspaces.items():
            objective, observation_gradients = workspace.evaluate(
                transforms
            )
            total += objective
            for robot in robot_ids:
                if robot == root_robot:
                    continue
                index = nonroot_index[robot]
                state = states_by_pair[pair][robot]
                rotation = rotation2(float(transforms[robot][0]))
                yaw_derivatives = state.means @ (skew @ rotation).T
                blocks = observation_gradients[robot]
                gradient[3 * index] += float(
                    np.sum(blocks * yaw_derivatives)
                )
                gradient[3 * index + 1 : 3 * index + 3] += (
                    translation_scale * np.sum(blocks, axis=0)
                )
        return float(total), gradient

    result = minimize(
        evaluate,
        vector,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-9, "maxiter": 500},
    )
    transforms = _unpack_transforms(
        robot_ids,
        root_robot,
        root_transform,
        result.x,
        translation_scale,
    )

    def objective_at(
        values: Mapping[int, tuple[float, np.ndarray]],
    ) -> JointCavityObjective:
        return _sum_joint_objectives(
            tuple(
                joint_cavity_partition_objective(
                    fold_partitions[pair],
                    problem,
                    values,
                    prior,
                    component_intensities,
                    attribute_evidence,
                    include_component_intensity=include_component_intensity,
                    robot_covariance_mode=robot_covariance_mode,
                )
                for pair, problem in sorted(problems.items())
            )
        )

    objective = objective_at(transforms)
    initial_objective = objective_at(initial)
    if objective.total > initial_objective.total + 1e-10:
        transforms = initial
        objective = initial_objective
    final_vector = _pack_transforms(
        robot_ids, root_robot, transforms, translation_scale
    )
    _, final_gradient = evaluate(final_vector)
    return JointCavityProfileResult(
        partition=partition,
        transforms=transforms,
        objective=objective,
        evaluations=int(result.nfev),
        converged=bool(result.success),
        gradient_norm=float(np.linalg.norm(final_gradient)),
        status=int(result.status),
        message=str(result.message),
    )


def _apply_partition_edit(
    partition: NetworkPartition, edit: PartitionEdit
) -> NetworkPartition:
    if edit.kind == "detach":
        return detach_partition_copy(partition, edit.copy)
    if edit.kind not in ("merge", "move") or edit.target is None:
        raise ValueError("unsupported partition edit")
    return move_partition_copy(partition, edit.copy, edit.target)


def _partition_edit_id(edit: PartitionEdit) -> str:
    target_robot = -1 if edit.target is None else edit.target.robot
    target_object = -1 if edit.target is None else edit.target.object_id
    return (
        f"{edit.kind}:{edit.copy.robot}:{edit.copy.object_id}:"
        f"{target_robot}:{target_object}"
    )


def _edit_component_footprint(
    partition: NetworkPartition, edit: PartitionEdit
) -> frozenset[int]:
    component_by_copy = {
        copy: index
        for index, component in enumerate(partition.components)
        for copy in component
    }
    footprint = {component_by_copy[edit.copy]}
    if edit.target is not None:
        footprint.add(component_by_copy[edit.target])
    return frozenset(footprint)


def _select_disjoint_negative_proposals(
    partition: NetworkPartition,
    proposals: tuple[
        tuple[float, str, PartitionEdit, NetworkPartition], ...
    ],
    robot_ids: tuple[int, ...],
    selection_topology: tuple[tuple[int, int], ...] | None,
    tolerance: float,
) -> tuple[
    tuple[tuple[float, str, PartitionEdit, NetworkPartition], ...], int, int
]:
    remaining = tuple(
        proposal for proposal in proposals if proposal[0] < -tolerance
    )
    selected = []
    blocked_components: set[int] = set()
    consensus_rounds = 0
    consensus_payload_bytes = 0

    while remaining or selection_topology is not None:
        if selection_topology is None:
            proposal = min(remaining, key=lambda value: (value[0], value[1]))
        else:
            robot_proposals = {robot: [(0.0, "noop")] for robot in robot_ids}
            for delta, identifier, edit, _ in remaining:
                robot_proposals[edit.copy.robot].append((delta, identifier))
            consensus = run_min_consensus(robot_proposals, selection_topology)
            consensus_rounds += consensus.rounds
            consensus_payload_bytes += consensus.payload_bytes
            _, identifier = consensus.selected
            if identifier == "noop":
                break
            proposal = next(
                value for value in remaining if value[1] == identifier
            )

        selected.append(proposal)
        blocked_components.update(
            _edit_component_footprint(partition, proposal[2])
        )
        remaining = tuple(
            value
            for value in remaining
            if _edit_component_footprint(partition, value[2]).isdisjoint(
                blocked_components
            )
        )

    return tuple(selected), consensus_rounds, consensus_payload_bytes


def _marginal_observations(
    problem: NetworkQuotientProblem,
) -> dict[ObjectCopy, GaussianObjectObservation]:
    result = {}
    for state in problem.robots:
        for index, copy in enumerate(state.copies):
            variance = float(state.covariance[index, index])
            result[copy] = GaussianObjectObservation(
                copy,
                state.means[index],
                variance * np.eye(2),
            )
    return result


def _sum_joint_objectives(
    objectives: tuple[JointCavityObjective, ...],
) -> JointCavityObjective:
    if not objectives:
        raise ValueError("at least one joint objective is required")
    return JointCavityObjective(
        total=float(sum(value.total for value in objectives)),
        state_nll=float(sum(value.state_nll for value in objectives)),
        mahalanobis=float(sum(value.mahalanobis for value in objectives)),
        log_determinant=float(
            sum(value.log_determinant for value in objectives)
        ),
        normalization=float(sum(value.normalization for value in objectives)),
        cardinality=float(sum(value.cardinality for value in objectives)),
        attribute_nll=float(sum(value.attribute_nll for value in objectives)),
    )


def relation_components(
    copies: tuple[ObjectCopy, ...],
    edges: tuple[tuple[ObjectCopy, ObjectCopy], ...],
) -> tuple[tuple[ObjectCopy, ...], ...]:
    """Return raw relation components without imposing network feasibility."""
    support = tuple(sorted(copies))
    if len(set(support)) != len(support):
        raise ValueError("relation support must contain unique copies")
    index = {copy: position for position, copy in enumerate(support)}
    parent = list(range(len(support)))

    def find(position: int) -> int:
        root = position
        while parent[root] != root:
            root = parent[root]
        while parent[position] != position:
            next_position = parent[position]
            parent[position] = root
            position = next_position
        return root

    for first, second in edges:
        if first not in index or second not in index:
            raise ValueError("relation edge is outside copy support")
        if (
            first == second
            or first.robot == second.robot
            or first.semantic_class != second.semantic_class
        ):
            raise ValueError("relation edge is not association feasible")
        first_root = find(index[first])
        second_root = find(index[second])
        if first_root != second_root:
            low, high = sorted((first_root, second_root))
            parent[high] = low
    components: dict[int, list[ObjectCopy]] = {}
    for position, copy in enumerate(support):
        components.setdefault(find(position), []).append(copy)
    return tuple(
        sorted(tuple(sorted(component)) for component in components.values())
    )


def relation_partition_diagnostics(
    copies: tuple[ObjectCopy, ...],
    edges: tuple[tuple[ObjectCopy, ObjectCopy], ...],
) -> tuple[bool, int]:
    components = relation_components(copies, edges)
    duplicate_count = sum(
        len(component) != len({copy.robot for copy in component})
        for component in components
    )
    semantic_conflict = any(
        len({copy.semantic_class for copy in component}) != 1
        for component in components
    )
    return duplicate_count == 0 and not semantic_conflict, duplicate_count


def _pairwise_relation_candidates(
    current: tuple[tuple[ObjectCopy, ObjectCopy], ...],
    candidates: tuple[tuple[ObjectCopy, ObjectCopy], ...],
) -> tuple[
    tuple[PartitionEdit, tuple[tuple[ObjectCopy, ObjectCopy], ...]], ...
]:
    proposals = {}
    current_set = set(current)
    for edge in current:
        edit = PartitionEdit("detach", edge[0], None)
        proposals[(edit.kind, edit.copy, edit.target)] = (
            edit,
            tuple(sorted(current_set - {edge})),
        )
    for first, second in candidates:
        retained = {
            edge
            for edge in current_set
            if first not in edge and second not in edge
        }
        retained.add(tuple(sorted((first, second))))
        relations = tuple(sorted(retained))
        if relations == current:
            continue
        kind = "merge" if not current else "move"
        edit = PartitionEdit(kind, first, second)
        proposals[(edit.kind, edit.copy, edit.target)] = (edit, relations)
    return tuple(proposals[key] for key in sorted(proposals))


def run_joint_cavity_leave_pair_pairwise(
    initial: NetworkPartition,
    problem: NetworkQuotientProblem,
    prior: GaussianObjectPrior,
    candidate_edges: tuple[tuple[ObjectCopy, ObjectCopy], ...],
    *,
    root_robot: int,
    initial_transforms: Mapping[int, tuple[float, np.ndarray]],
    component_intensities: Mapping[int, float],
    selection_topology: tuple[tuple[int, int], ...],
    leave_pair_problems: Mapping[
        tuple[int, int], NetworkQuotientProblem
    ],
    attribute_observations: Mapping[
        ObjectCopy, GaussianObjectObservation
    ] | None = None,
    attribute_prior: GaussianObjectPrior | None = None,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> PairwiseJointCavityResult:
    """Select held-out edge relations independently at shared transforms."""
    if initial.copies != problem.partition.copies:
        raise ValueError("initial partition support must match the cavity problem")
    if (attribute_observations is None) != (attribute_prior is None):
        raise ValueError(
            "attribute observations and prior must be supplied together"
        )
    topology = tuple(
        sorted({tuple(sorted(map(int, edge))) for edge in selection_topology})
    )
    folds = {
        tuple(sorted(map(int, pair))): value
        for pair, value in leave_pair_problems.items()
    }
    if set(folds) != set(topology):
        raise ValueError("leave-pair evidence must cover the selection topology")
    if any(value.partition.copies != initial.copies for value in folds.values()):
        raise ValueError("leave-pair cavity support must match the partition")
    attribute_evidence = (
        None
        if attribute_observations is None
        else _AttributeEvidence(
            initial.copies, attribute_observations, attribute_prior
        )
    )
    shared_profile = profile_joint_cavity_leave_pair_composite_transforms(
        initial,
        folds,
        prior,
        component_intensities,
        root_robot=root_robot,
        initial_transforms=initial_transforms,
        reference_partition=initial,
        topology=topology,
        attribute_evidence=attribute_evidence,
        include_component_intensity=include_component_intensity,
        robot_covariance_mode=robot_covariance_mode,
    )
    current_by_pair = _topology_partition_edges(initial, topology)
    candidate_by_pair = {pair: [] for pair in topology}
    for edge in sorted(set(candidate_edges)):
        pair = tuple(sorted((edge[0].robot, edge[1].robot)))
        if pair in candidate_by_pair:
            candidate_by_pair[pair].append(tuple(sorted(edge)))
    selected_by_pair = dict(current_by_pair)
    accepted = []
    initial_objectives = []
    final_objectives = []
    evaluations = shared_profile.evaluations + 1
    tolerance_scale = 0.0
    for pair in topology:
        current_partition = partition_from_edges(
            initial.copies, current_by_pair[pair]
        )
        current_objective = joint_cavity_partition_objective(
            current_partition,
            folds[pair],
            shared_profile.transforms,
            prior,
            component_intensities,
            attribute_evidence,
            include_component_intensity=include_component_intensity,
            robot_covariance_mode=robot_covariance_mode,
        )
        initial_objectives.append(current_objective)
        tolerance_scale += abs(current_objective.total)
        best = (current_objective.total, "", None, current_by_pair[pair])
        for edit, relations in _pairwise_relation_candidates(
            current_by_pair[pair], tuple(candidate_by_pair[pair])
        ):
            candidate_partition = partition_from_edges(
                initial.copies, relations
            )
            candidate_objective = joint_cavity_partition_objective(
                candidate_partition,
                folds[pair],
                shared_profile.transforms,
                prior,
                component_intensities,
                attribute_evidence,
                include_component_intensity=include_component_intensity,
                robot_covariance_mode=robot_covariance_mode,
            )
            evaluations += 1
            identifier = _partition_edit_id(edit)
            value = (
                candidate_objective.total,
                identifier,
                edit,
                relations,
            )
            if value[:2] < best[:2]:
                best = value
        tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(current_objective.total))
        )
        if best[2] is not None and best[0] < current_objective.total - tolerance:
            selected_by_pair[pair] = best[3]
            accepted.append(best[2])
        final_partition = partition_from_edges(
            initial.copies, selected_by_pair[pair]
        )
        final_objectives.append(
            joint_cavity_partition_objective(
                final_partition,
                folds[pair],
                shared_profile.transforms,
                prior,
                component_intensities,
                attribute_evidence,
                include_component_intensity=include_component_intensity,
                robot_covariance_mode=robot_covariance_mode,
            )
        )
        evaluations += 1
    relation_edges = tuple(
        sorted(
            {
                edge
                for relations in selected_by_pair.values()
                for edge in relations
            }
        )
    )
    network_feasible, duplicate_count = relation_partition_diagnostics(
        initial.copies, relation_edges
    )
    components = relation_components(initial.copies, relation_edges)
    partition = (
        NetworkPartition.build(components) if network_feasible else None
    )
    initial_objective = _sum_joint_objectives(tuple(initial_objectives))
    final_objective = _sum_joint_objectives(tuple(final_objectives))
    tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, tolerance_scale)
    )
    if final_objective.total > initial_objective.total + tolerance:
        raise RuntimeError("independent held-out edits increased pairwise objective")
    canonical_messages = 0
    canonical_payload = 0
    try:
        for value in folds.values():
            messages, payload = joint_cavity_state_sync_accounting(
                value, topology
            )
            canonical_messages += messages
            canonical_payload += payload
    except ValueError:
        # Pairwise diagnostics may use a cyclic association graph; the paper
        # protocol separately routes sufficient statistics on a spanning tree.
        canonical_messages = 0
        canonical_payload = 0
    return PairwiseJointCavityResult(
        relation_edges=relation_edges,
        partition=partition,
        transforms=shared_profile.transforms,
        objective=final_objective,
        objective_trace=(
            float(initial_objective.total),
            float(final_objective.total),
        ),
        accepted_edits=tuple(accepted),
        candidate_edge_counts=(len(candidate_edges),),
        objective_evaluations=evaluations,
        canonical_state_sync_messages=canonical_messages,
        canonical_state_sync_payload_bytes=canonical_payload,
        network_feasible=network_feasible,
        duplicate_robot_component_count=duplicate_count,
    )


def run_joint_cavity_partitioning(
    initial: NetworkPartition,
    problem: NetworkQuotientProblem,
    prior: GaussianObjectPrior,
    candidate_edges: tuple[tuple[ObjectCopy, ObjectCopy], ...],
    *,
    root_robot: int,
    initial_transforms: Mapping[int, tuple[float, np.ndarray]],
    component_intensities: Mapping[int, float],
    selection_topology: tuple[tuple[int, int], ...] | None = None,
    candidate_topology: tuple[tuple[int, int], ...] | None = None,
    leave_pair_problems: Mapping[
        tuple[int, int], NetworkQuotientProblem
    ] | None = None,
    attribute_observations: Mapping[
        ObjectCopy, GaussianObjectObservation
    ] | None = None,
    attribute_prior: GaussianObjectPrior | None = None,
    include_component_intensity: bool = True,
    robot_covariance_mode: str = "full",
) -> JointCavityPartitionResult:
    """Apply one causal partition block update for the current Lower cavity."""
    if initial.copies != problem.partition.copies:
        raise ValueError("initial partition support must match the cavity problem")
    if (attribute_observations is None) != (attribute_prior is None):
        raise ValueError(
            "attribute observations and prior must be supplied together"
        )
    attribute_evidence = (
        None
        if attribute_observations is None
        else _AttributeEvidence(
            initial.copies, attribute_observations, attribute_prior
        )
    )
    static_edges = tuple(sorted(set(candidate_edges)))
    full_leave_pair_problems = (
        None
        if leave_pair_problems is None
        else {
            tuple(sorted(map(int, pair))): value
            for pair, value in leave_pair_problems.items()
        }
    )
    if full_leave_pair_problems is not None and not full_leave_pair_problems:
        raise ValueError("leave-pair evidence must not be empty")
    if full_leave_pair_problems is not None and set(
        full_leave_pair_problems
    ) != {
        tuple(sorted(edge)) for edge in (selection_topology or ())
    }:
        raise ValueError("leave-pair evidence must cover the selection topology")
    if full_leave_pair_problems is not None and any(
        value.partition.copies != initial.copies
        for value in full_leave_pair_problems.values()
    ):
        raise ValueError("leave-pair cavity support must match the partition")
    evidence_problems = (
        {(0, 0): problem}
        if full_leave_pair_problems is None
        else full_leave_pair_problems
    )
    statistics_by_pair = {
        pair: _joint_cavity_problem_statistics(
            value, robot_covariance_mode=robot_covariance_mode
        )
        for pair, value in evidence_problems.items()
    }
    if full_leave_pair_problems is None:
        current_profile = profile_joint_cavity_transforms(
            initial,
            problem,
            prior,
            component_intensities,
            root_robot=root_robot,
            initial_transforms=initial_transforms,
            attribute_evidence=attribute_evidence,
            include_component_intensity=include_component_intensity,
            robot_covariance_mode=robot_covariance_mode,
        )
    else:
        current_profile = (
            profile_joint_cavity_leave_pair_composite_transforms(
                initial,
                full_leave_pair_problems,
                prior,
                component_intensities,
                root_robot=root_robot,
                initial_transforms=initial_transforms,
                reference_partition=initial,
                topology=selection_topology,
                attribute_evidence=attribute_evidence,
                include_component_intensity=include_component_intensity,
                robot_covariance_mode=robot_covariance_mode,
            )
        )
    current_objective = current_profile.objective
    current_partition = initial
    accepted_edits = []
    consensus_rounds = 0
    consensus_payload_bytes = 0
    canonical_state_sync_messages = 0
    canonical_state_sync_payload_bytes = 0
    if selection_topology is not None:
        for value in evidence_problems.values():
            messages, payload_bytes = joint_cavity_state_sync_accounting(
                value, selection_topology
            )
            canonical_state_sync_messages += messages
            canonical_state_sync_payload_bytes += payload_bytes
    candidate_edge_counts = []
    objective_evaluations = current_profile.evaluations + 1
    robot_ids = tuple(state.robot for state in problem.robots)
    base_observations = _marginal_observations(problem)
    objective_trace = [current_objective.total]

    while True:
        transformed = transform_robot_observations(
            base_observations, current_profile.transforms
        )
        active_edges = static_edges
        if candidate_topology is not None:
            active_edges = tuple(
                sorted(
                    set(static_edges)
                    | set(
                        reciprocal_physical_candidate_edges(
                            transformed, candidate_topology
                        )
                    )
                )
            )
        candidate_edge_counts.append(len(active_edges))
        edits = enumerate_feasible_partition_edits(
            current_partition, active_edges
        )
        reference_crossfit_edges = (
            None
            if full_leave_pair_problems is None
            else _topology_partition_edges(
                current_partition, selection_topology or ()
            )
        )
        crossfit_copies = (
            ()
            if full_leave_pair_problems is None
            else current_partition.copies
        )
        scored_by_partition: dict[
            tuple[tuple[ObjectCopy, ...], ...],
            tuple[float, str, PartitionEdit, NetworkPartition],
        ] = {}
        for edit in edits:
            candidate_partition = _apply_partition_edit(
                current_partition, edit
            )
            key = candidate_partition.components
            if key in scored_by_partition:
                continue
            candidate_crossfit_edges = (
                None
                if full_leave_pair_problems is None
                else _topology_partition_edges(
                    candidate_partition, selection_topology or ()
                )
            )
            candidate_total = 0.0
            for pair in sorted(evidence_problems):
                active_partition = candidate_partition
                intensities = component_intensities
                if full_leave_pair_problems is not None:
                    active_partition = _crossfit_fold_partition_from_edges(
                        crossfit_copies,
                        reference_crossfit_edges or {},
                        candidate_crossfit_edges or {},
                        pair,
                    )
                candidate_total += _JointCavityCanonicalWorkspace(
                    active_partition,
                    evidence_problems[pair],
                    prior,
                    intensities,
                    statistics_by_pair[pair],
                    attribute_evidence,
                    include_component_intensity=include_component_intensity,
                    robot_covariance_mode=robot_covariance_mode,
                ).evaluate(
                    {
                        robot: current_profile.transforms[robot]
                        for robot in robot_ids
                    }
                )[0]
            objective_evaluations += len(evidence_problems)
            delta = candidate_total - current_objective.total
            identifier = _partition_edit_id(edit)
            scored_by_partition[key] = (
                delta,
                identifier,
                edit,
                candidate_partition,
            )
        proposals = tuple(scored_by_partition.values())
        tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(current_objective.total))
        )

        if leave_pair_problems is None:
            if selection_topology is None:
                if not proposals:
                    break
                selected = min(
                    proposals, key=lambda value: (value[0], value[1])
                )
            else:
                robot_proposals = {
                    robot: [(0.0, "noop")] for robot in robot_ids
                }
                by_id = {value[1]: value for value in proposals}
                for delta, identifier, edit, _ in proposals:
                    robot_proposals[edit.copy.robot].append(
                        (delta, identifier)
                    )
                consensus = run_min_consensus(
                    robot_proposals, selection_topology
                )
                consensus_rounds += consensus.rounds
                consensus_payload_bytes += consensus.payload_bytes
                _, identifier = consensus.selected
                if identifier == "noop":
                    break
                selected = by_id[identifier]
            if selected[0] >= -tolerance:
                break
            selected_batch = (selected,)
            candidate_partition = selected[3]
        else:
            (
                selected_batch,
                batch_rounds,
                batch_payload_bytes,
            ) = _select_disjoint_negative_proposals(
                current_partition,
                proposals,
                robot_ids,
                selection_topology,
                tolerance,
            )
            consensus_rounds += batch_rounds
            consensus_payload_bytes += batch_payload_bytes
            if not selected_batch:
                break
            candidate_partition = current_partition
            for _, _, edit, _ in selected_batch:
                candidate_partition = _apply_partition_edit(
                    candidate_partition, edit
                )

        def profile_candidate(
            partition: NetworkPartition,
        ) -> JointCavityProfileResult:
            if full_leave_pair_problems is not None:
                return profile_joint_cavity_leave_pair_composite_transforms(
                    partition,
                    full_leave_pair_problems,
                    prior,
                    component_intensities,
                    root_robot=root_robot,
                    initial_transforms=current_profile.transforms,
                    reference_partition=current_partition,
                    topology=selection_topology,
                    attribute_evidence=attribute_evidence,
                    include_component_intensity=include_component_intensity,
                    robot_covariance_mode=robot_covariance_mode,
                )
            return profile_joint_cavity_transforms(
                partition,
                problem,
                prior,
                component_intensities,
                root_robot=root_robot,
                initial_transforms=current_profile.transforms,
                attribute_evidence=attribute_evidence,
                include_component_intensity=include_component_intensity,
                robot_covariance_mode=robot_covariance_mode,
            )

        candidate_profile = profile_candidate(candidate_partition)
        objective_evaluations += candidate_profile.evaluations + 1
        candidate_profiled_objective = candidate_profile.objective
        if (
            candidate_profiled_objective.total
            >= current_objective.total - tolerance
        ):
            candidate_partition = selected_batch[0][3]
            candidate_profile = profile_candidate(candidate_partition)
            objective_evaluations += candidate_profile.evaluations + 1
            candidate_profiled_objective = candidate_profile.objective
            selected_batch = selected_batch[:1]
            if (
                candidate_profiled_objective.total
                >= current_objective.total - tolerance
            ):
                raise RuntimeError(
                    "accepted edit did not decrease joint objective"
                )
        current_profile = candidate_profile
        current_objective = candidate_profiled_objective
        current_partition = candidate_partition
        objective_trace.append(current_objective.total)
        accepted_edits.extend(value[2] for value in selected_batch)
        # One inexact Upper block is evaluated per Lower response. The next
        # block is scored only after the Lower refreshes its cavity state.
        break

    return JointCavityPartitionResult(
        partition=current_partition,
        transforms=current_profile.transforms,
        objective=current_objective,
        objective_trace=tuple(objective_trace),
        accepted_edits=tuple(accepted_edits),
        consensus_rounds=consensus_rounds,
        consensus_payload_bytes=consensus_payload_bytes,
        candidate_edge_counts=tuple(candidate_edge_counts),
        objective_evaluations=objective_evaluations,
        canonical_state_sync_messages=canonical_state_sync_messages,
        canonical_state_sync_payload_bytes=canonical_state_sync_payload_bytes,
    )
