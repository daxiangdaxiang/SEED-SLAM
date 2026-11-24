#!/usr/bin/env python3
"""Gauge-invariant network quotient reference for DBA association."""

from __future__ import annotations

import argparse
import csv
from functools import cached_property
import itertools
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import OptimizeResult, least_squares


@dataclass(frozen=True, order=True)
class ObjectCopy:
    robot: int
    object_id: int
    semantic_class: int

    def __post_init__(self) -> None:
        identifiers = (self.robot, self.object_id, self.semantic_class)
        if any(
            isinstance(identifier, bool)
            or not isinstance(identifier, Integral)
            for identifier in identifiers
        ):
            raise ValueError("object copy identifiers must be integral")
        robot, object_id, semantic_class = map(int, identifiers)
        if robot < 0:
            raise ValueError("robot must be nonnegative")
        if object_id < 0:
            raise ValueError("object_id must be nonnegative")
        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "semantic_class", semantic_class)


@dataclass(frozen=True)
class NetworkPartition:
    components: tuple[tuple[ObjectCopy, ...], ...]

    @classmethod
    def build(
        cls, components: Iterable[Iterable[ObjectCopy]]
    ) -> NetworkPartition:
        canonical_components = []
        seen_copies = set()

        for component in components:
            members = tuple(component)
            if not members:
                raise ValueError("components must not be empty")

            component_copies = set()
            component_robots = set()
            semantic_class = members[0].semantic_class
            for copy in members:
                if copy in component_copies:
                    raise ValueError("copy is repeated within a component")
                if copy in seen_copies:
                    raise ValueError("copy appears in multiple components")
                if copy.robot in component_robots:
                    raise ValueError("component has more than one copy from a robot")
                if copy.semantic_class != semantic_class:
                    raise ValueError("component has a semantic class conflict")
                component_copies.add(copy)
                component_robots.add(copy.robot)

            seen_copies.update(component_copies)
            canonical_components.append(tuple(sorted(members)))

        return cls(tuple(sorted(canonical_components)))

    @cached_property
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


CanonicalEdge = tuple[ObjectCopy, ObjectCopy]


def canonical_edge(first: ObjectCopy, second: ObjectCopy) -> CanonicalEdge:
    if not isinstance(first, ObjectCopy) or not isinstance(second, ObjectCopy):
        raise ValueError("edge endpoints must be ObjectCopy values")
    if first == second:
        raise ValueError("edge endpoints must be distinct")
    if first.robot == second.robot:
        raise ValueError("edge endpoints must belong to different robots")
    if first.semantic_class != second.semantic_class:
        raise ValueError("edge endpoints must have the same semantic class")
    return tuple(sorted((first, second)))


def _normalize_edge(edge: Iterable[ObjectCopy]) -> CanonicalEdge:
    try:
        endpoints = tuple(edge)
    except TypeError as error:
        raise ValueError("an edge must contain two ObjectCopy values") from error
    if len(endpoints) != 2:
        raise ValueError("an edge must contain two ObjectCopy values")
    return canonical_edge(endpoints[0], endpoints[1])


def partition_from_edges(
    copies: Iterable[ObjectCopy],
    edges: Iterable[Iterable[ObjectCopy]],
) -> NetworkPartition:
    try:
        supplied_copies = tuple(copies)
    except TypeError as error:
        raise ValueError("copy support must be an iterable") from error
    if any(not isinstance(copy, ObjectCopy) for copy in supplied_copies):
        raise ValueError("copy support must contain only ObjectCopy values")
    if len(set(supplied_copies)) != len(supplied_copies):
        raise ValueError("copy support must contain unique copies")
    support = set(supplied_copies)

    try:
        supplied_edges = tuple(edges)
    except TypeError as error:
        raise ValueError("edges must be an iterable") from error
    normalized_edges = tuple(_normalize_edge(edge) for edge in supplied_edges)
    if len(set(normalized_edges)) != len(normalized_edges):
        raise ValueError("edges must not contain duplicates")
    if any(
        endpoint not in support
        for edge in normalized_edges
        for endpoint in edge
    ):
        raise ValueError("edge endpoints must belong to the exact copy support")

    parent = {copy: copy for copy in supplied_copies}

    def find(copy: ObjectCopy) -> ObjectCopy:
        root = copy
        while parent[root] != root:
            root = parent[root]
        while parent[copy] != copy:
            next_copy = parent[copy]
            parent[copy] = root
            copy = next_copy
        return root

    def union(first: ObjectCopy, second: ObjectCopy) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    for first, second in normalized_edges:
        union(first, second)

    components: dict[ObjectCopy, list[ObjectCopy]] = {}
    for copy in sorted(supplied_copies):
        components.setdefault(find(copy), []).append(copy)
    return NetworkPartition.build(components.values())


def partition_from_edge_components(
    edge_components: Iterable[Iterable[ObjectCopy]],
) -> NetworkPartition:
    try:
        components = tuple(edge_components)
    except TypeError as error:
        raise ValueError("edge components must be an iterable") from error
    copies = set()
    edges = []
    for component in components:
        try:
            members = tuple(component)
        except TypeError as error:
            raise ValueError(
                "edge components must contain one or two copies"
            ) from error
        if len(members) not in (1, 2):
            raise ValueError("edge components must contain one or two copies")
        if any(not isinstance(copy, ObjectCopy) for copy in members):
            raise ValueError(
                "edge components must contain only ObjectCopy values"
            )
        copies.update(members)
        if len(members) == 2:
            edges.append(_normalize_edge(members))
    return partition_from_edges(tuple(sorted(copies)), tuple(edges))


@dataclass(frozen=True)
class RobotCavityState:
    robot: int
    copies: tuple[ObjectCopy, ...]
    means: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        copies = tuple(self.copies)
        means = np.array(self.means, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        copy_count = len(copies)

        if self.robot < 0:
            raise ValueError("robot must be nonnegative")
        if not copies:
            raise ValueError("cavity copies must not be empty")
        if any(copy.robot != self.robot for copy in copies):
            raise ValueError("all cavity copies must belong to the stated robot")
        if len(set(copies)) != copy_count:
            raise ValueError("cavity copies must be unique")
        if means.shape != (copy_count, 2):
            raise ValueError("cavity means must have shape (n, 2)")
        if covariance.shape != (copy_count, copy_count):
            raise ValueError("cavity covariance must have shape (n, n)")
        if not np.all(np.isfinite(means)):
            raise ValueError("cavity means must be finite")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("cavity covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
            raise ValueError("cavity covariance must be symmetric")
        covariance = 0.5 * (covariance + covariance.T)
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "cavity covariance must be positive definite"
            ) from error

        means.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "copies", copies)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class NetworkQuotientProblem:
    robots: tuple[RobotCavityState, ...]
    partition: NetworkPartition

    def __post_init__(self) -> None:
        robots = tuple(self.robots)
        robot_ids = tuple(state.robot for state in robots)
        if len(set(robot_ids)) != len(robot_ids):
            raise ValueError("robot IDs must be unique")
        if robot_ids != tuple(sorted(robot_ids)):
            raise ValueError("robot IDs must be sorted")

        supplied_copies = tuple(
            sorted(copy for state in robots for copy in state.copies)
        )
        if supplied_copies != self.partition.copies:
            raise ValueError(
                "robot cavity copies must match the partition copies"
            )

        object.__setattr__(self, "robots", robots)


def relabel_robots(
    problem: NetworkQuotientProblem, mapping: Mapping[int, int]
) -> NetworkQuotientProblem:
    if not isinstance(mapping, Mapping):
        raise ValueError("robot mapping must be a mapping")
    robot_ids = tuple(state.robot for state in problem.robots)
    try:
        source_ids = tuple(mapping)
    except (KeyError, TypeError) as error:
        raise ValueError("robot mapping must cover every robot") from error
    if any(
        isinstance(robot, bool)
        or not isinstance(robot, Integral)
        for robot in source_ids
    ):
        raise ValueError("source robot IDs must be non-bool integers")
    normalized_sources = tuple(int(robot) for robot in source_ids)
    if (
        len(set(normalized_sources)) != len(normalized_sources)
        or set(normalized_sources) != set(robot_ids)
    ):
        raise ValueError("robot mapping must cover exactly every robot")
    try:
        mapped_robot = {
            int(source): mapping[source] for source in source_ids
        }
    except (KeyError, TypeError) as error:
        raise ValueError("robot mapping must cover every robot") from error
    destinations = tuple(mapped_robot[robot] for robot in robot_ids)
    if any(
        isinstance(robot, bool)
        or not isinstance(robot, Integral)
        or robot < 0
        for robot in destinations
    ):
        raise ValueError("mapped robot IDs must be nonnegative integers")
    if len(set(destinations)) != len(destinations):
        raise ValueError("mapped robot IDs must be unique")

    mapped_robot = {
        source: int(mapped_robot[source]) for source in robot_ids
    }
    mapped_copy = {
        copy: ObjectCopy(
            mapped_robot[copy.robot], copy.object_id, copy.semantic_class
        )
        for copy in problem.partition.copies
    }
    states = tuple(
        sorted(
            (
                RobotCavityState(
                    mapped_robot[state.robot],
                    tuple(mapped_copy[copy] for copy in state.copies),
                    state.means,
                    state.covariance,
                )
                for state in problem.robots
            ),
            key=lambda state: state.robot,
        )
    )
    partition = NetworkPartition.build(
        tuple(
            tuple(mapped_copy[copy] for copy in component)
            for component in problem.partition.components
        )
    )
    return NetworkQuotientProblem(states, partition)


def relabel_objects(
    problem: NetworkQuotientProblem,
    mapping: Mapping[ObjectCopy, ObjectCopy],
) -> NetworkQuotientProblem:
    if not isinstance(mapping, Mapping):
        raise ValueError("object mapping must be a mapping")
    support = problem.partition.copies
    try:
        sources = tuple(mapping)
    except (KeyError, TypeError) as error:
        raise ValueError("object mapping must cover every copy") from error
    if len(sources) != len(support) or set(sources) != set(support):
        raise ValueError("object mapping must cover exactly every copy")
    try:
        pairs = tuple((source, mapping[source]) for source in sources)
    except (KeyError, TypeError) as error:
        raise ValueError("object mapping must cover every copy") from error
    snapshot = MappingProxyType(dict(pairs))
    destinations = tuple(snapshot[copy] for copy in support)
    if any(not isinstance(copy, ObjectCopy) for copy in destinations):
        raise ValueError(
            "object mapping destinations must be ObjectCopy values"
        )
    for source, destination in zip(support, destinations):
        if destination.robot != source.robot:
            raise ValueError("object mapping must preserve robot IDs")
        if destination.semantic_class != source.semantic_class:
            raise ValueError("object mapping must preserve semantic classes")
    if len(set(destinations)) != len(destinations):
        raise ValueError("object mapping destinations must be unique")

    states = tuple(
        RobotCavityState(
            state.robot,
            tuple(snapshot[copy] for copy in state.copies),
            state.means,
            state.covariance,
        )
        for state in problem.robots
    )
    partition = NetworkPartition.build(
        tuple(
            tuple(snapshot[copy] for copy in component)
            for component in problem.partition.components
        )
    )
    return NetworkQuotientProblem(states, partition)


def apply_common_local_gauge(
    problem: NetworkQuotientProblem,
    yaw: float,
    translation: np.ndarray,
) -> NetworkQuotientProblem:
    try:
        yaw_array = np.asarray(yaw, dtype=float)
        translation_array = np.asarray(translation, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("gauge transform must be numeric") from error
    if yaw_array.shape != ():
        raise ValueError("gauge yaw must be a scalar")
    if translation_array.shape != (2,):
        raise ValueError("gauge translation must have shape (2,)")
    yaw_value = float(yaw_array)
    if not np.isfinite(yaw_value) or not np.all(
        np.isfinite(translation_array)
    ):
        raise ValueError("gauge transform must be finite")

    rotation = rotation2(yaw_value)
    states = tuple(
        RobotCavityState(
            state.robot,
            state.copies,
            state.means @ rotation.T + translation_array,
            state.covariance,
        )
        for state in problem.robots
    )
    return NetworkQuotientProblem(states, problem.partition)


@dataclass(frozen=True)
class VariableLayout:
    root_robot: int
    robots: tuple[int, ...]
    component_by_copy: Mapping[ObjectCopy, int]
    transform_offset: Mapping[int, int]
    component_offset: int
    dimension: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_by_copy",
            MappingProxyType(dict(self.component_by_copy)),
        )
        object.__setattr__(
            self,
            "transform_offset",
            MappingProxyType(dict(self.transform_offset)),
        )

    @classmethod
    def build(
        cls, problem: NetworkQuotientProblem, root_robot: int
    ) -> VariableLayout:
        robots = tuple(state.robot for state in problem.robots)
        if root_robot not in robots:
            raise ValueError("root robot must be present in the problem")

        nonroot_robots = tuple(robot for robot in robots if robot != root_robot)
        transform_offset = {
            robot: 3 * index for index, robot in enumerate(nonroot_robots)
        }
        component_offset = 3 * len(nonroot_robots)
        component_by_copy = {
            copy: component_index
            for component_index, component in enumerate(
                problem.partition.components
            )
            for copy in component
        }
        dimension = component_offset + 2 * problem.partition.component_count
        return cls(
            root_robot=root_robot,
            robots=robots,
            component_by_copy=component_by_copy,
            transform_offset=transform_offset,
            component_offset=component_offset,
            dimension=dimension,
        )


@dataclass(frozen=True)
class CanonicalEvaluation:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    residual_chi2: float


def _canonical_packet_bytes(dimension: int) -> int:
    return 8 * (
        4 + 2 * dimension + dimension * (dimension + 1) // 2
    )


@dataclass(frozen=True)
class CanonicalPacket:
    """Packed doubles plus pre-shared, untransmitted layout context.

    ``ambient_dimension`` and ``layout_signature`` identify protocol context
    already held by every participant, so ``serialized_bytes`` excludes them.
    """

    robot: int
    variable_ids: tuple[int, ...]
    packed_upper_hessian: np.ndarray
    gradient: np.ndarray
    quadratic_value: float
    normalizer: float
    serialized_bytes: int
    ambient_dimension: int
    layout_signature: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.robot, bool)
            or not isinstance(self.robot, Integral)
            or self.robot < 0
        ):
            raise ValueError("packet robot must be a nonnegative integer")
        if (
            isinstance(self.ambient_dimension, bool)
            or not isinstance(self.ambient_dimension, Integral)
            or self.ambient_dimension <= 0
        ):
            raise ValueError("ambient dimension must be a positive integer")
        ambient_dimension = int(self.ambient_dimension)
        try:
            layout_signature = tuple(self.layout_signature)
        except TypeError as error:
            raise ValueError("layout signature must be an iterable") from error
        if not layout_signature or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in layout_signature
        ):
            raise ValueError(
                "layout signature must contain non-bool integers"
            )
        layout_signature = tuple(int(value) for value in layout_signature)
        try:
            variable_ids = tuple(self.variable_ids)
        except TypeError as error:
            raise ValueError("variable IDs must be an iterable") from error
        if any(
            isinstance(variable_id, bool)
            or not isinstance(variable_id, Integral)
            or variable_id < 0
            for variable_id in variable_ids
        ) or any(
            left >= right
            for left, right in zip(variable_ids, variable_ids[1:])
        ):
            raise ValueError(
                "variable IDs must be strictly increasing nonnegative integers"
            )
        variable_ids = tuple(int(value) for value in variable_ids)
        if any(value >= ambient_dimension for value in variable_ids):
            raise ValueError("variable IDs exceed the ambient dimension")
        dimension = len(variable_ids)
        try:
            packed_upper = np.array(
                self.packed_upper_hessian, dtype=float, copy=True
            )
            gradient = np.array(self.gradient, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("packet arrays must be numeric") from error
        expected_upper = dimension * (dimension + 1) // 2
        if packed_upper.shape != (expected_upper,):
            raise ValueError(
                "packed upper Hessian has an inconsistent dimension"
            )
        if gradient.shape != (dimension,):
            raise ValueError("packet gradient has an inconsistent dimension")
        if not (
            np.all(np.isfinite(packed_upper))
            and np.all(np.isfinite(gradient))
        ):
            raise ValueError("packet arrays must be finite")
        try:
            quadratic_array = np.asarray(self.quadratic_value, dtype=float)
            normalizer_array = np.asarray(self.normalizer, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("packet scalars must be numeric") from error
        if quadratic_array.shape != () or normalizer_array.shape != ():
            raise ValueError("packet scalars must be scalar values")
        quadratic_value = float(quadratic_array)
        normalizer = float(normalizer_array)
        if not (
            np.isfinite(quadratic_value) and np.isfinite(normalizer)
        ):
            raise ValueError("packet scalars must be finite")
        expected_bytes = _canonical_packet_bytes(dimension)
        if (
            isinstance(self.serialized_bytes, bool)
            or not isinstance(self.serialized_bytes, Integral)
            or self.serialized_bytes != expected_bytes
        ):
            raise ValueError(
                "serialized bytes must equal the canonical packet schema"
            )

        packed_upper.setflags(write=False)
        gradient.setflags(write=False)
        object.__setattr__(self, "robot", int(self.robot))
        object.__setattr__(self, "variable_ids", variable_ids)
        object.__setattr__(self, "packed_upper_hessian", packed_upper)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "quadratic_value", quadratic_value)
        object.__setattr__(self, "normalizer", normalizer)
        object.__setattr__(self, "serialized_bytes", expected_bytes)
        object.__setattr__(self, "ambient_dimension", ambient_dimension)
        object.__setattr__(self, "layout_signature", layout_signature)


@dataclass(frozen=True)
class ReducedCanonicalModel:
    hessian: np.ndarray
    gradient: np.ndarray
    value: float
    link_payload_bytes: int

    def __post_init__(self) -> None:
        try:
            hessian = np.array(self.hessian, dtype=float, copy=True)
            gradient = np.array(self.gradient, dtype=float, copy=True)
            value = float(self.value)
        except (TypeError, ValueError) as error:
            raise ValueError("reduced model values must be numeric") from error
        if (
            hessian.ndim != 2
            or hessian.shape[0] != hessian.shape[1]
            or gradient.shape != (hessian.shape[0],)
        ):
            raise ValueError("reduced model has inconsistent dimensions")
        if not (
            np.all(np.isfinite(hessian))
            and np.all(np.isfinite(gradient))
            and np.isfinite(value)
        ):
            raise ValueError("reduced model values must be finite")
        if (
            isinstance(self.link_payload_bytes, bool)
            or not isinstance(self.link_payload_bytes, Integral)
            or self.link_payload_bytes < 0
        ):
            raise ValueError("link payload bytes must be a nonnegative integer")

        hessian.setflags(write=False)
        gradient.setflags(write=False)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self, "link_payload_bytes", int(self.link_payload_bytes)
        )


@dataclass(frozen=True)
class MinConsensusResult:
    selected: tuple[float, str]
    selected_by_robot: Mapping[int, tuple[float, str]]
    rounds: int
    payload_bytes: int

    def __post_init__(self) -> None:
        selected = _normalize_consensus_value(self.selected, "selected")
        if not isinstance(self.selected_by_robot, Mapping):
            raise ValueError("selected_by_robot must be a nonempty mapping")
        selected_items = tuple(self.selected_by_robot.items())
        if not selected_items:
            raise ValueError("selected_by_robot must be nonempty")
        selected_by_robot = {}
        for robot, value in selected_items:
            if (
                isinstance(robot, bool)
                or not isinstance(robot, Integral)
                or robot < 0
            ):
                raise ValueError(
                    "selected_by_robot robot IDs must be nonnegative integers"
                )
            robot = int(robot)
            if robot in selected_by_robot:
                raise ValueError("selected_by_robot robot IDs must be unique")
            normalized = _normalize_consensus_value(
                value, "selected_by_robot values"
            )
            if normalized != selected:
                raise ValueError(
                    "selected_by_robot values must agree with selected"
                )
            selected_by_robot[robot] = normalized
        if (
            isinstance(self.rounds, bool)
            or not isinstance(self.rounds, Integral)
            or self.rounds < 0
        ):
            raise ValueError("rounds must be a nonnegative integer")
        if (
            isinstance(self.payload_bytes, bool)
            or not isinstance(self.payload_bytes, Integral)
            or self.payload_bytes < 0
        ):
            raise ValueError("payload bytes must be a nonnegative integer")

        object.__setattr__(self, "selected", selected)
        object.__setattr__(
            self,
            "selected_by_robot",
            MappingProxyType(selected_by_robot),
        )
        object.__setattr__(self, "rounds", int(self.rounds))
        object.__setattr__(self, "payload_bytes", int(self.payload_bytes))


def _normalize_consensus_value(
    value: tuple[float, str], field: str
) -> tuple[float, str]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field} must be a (score, move ID) tuple")
    score, move_id = value
    if isinstance(score, (bool, np.bool_)):
        raise ValueError(f"{field} score must be numeric and non-bool")
    try:
        score_array = np.asarray(score, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} score must be numeric") from error
    if score_array.shape != ():
        raise ValueError(f"{field} score must be scalar")
    score = float(score_array)
    if not np.isfinite(score):
        raise ValueError(f"{field} score must be finite")
    if not isinstance(move_id, str):
        raise ValueError(f"{field} move ID must be a string")
    try:
        move_id.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{field} move ID must be UTF-8 encodable"
        ) from error
    return score, move_id


def rotation2(yaw: float) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]])


def _unpack(
    layout: VariableLayout, vector: np.ndarray
) -> tuple[dict[int, tuple[float, np.ndarray]], np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (layout.dimension,):
        raise ValueError(
            f"variable vector must have shape ({layout.dimension},)"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("variable vector must be finite")

    transforms = {layout.root_robot: (0.0, np.zeros(2))}
    for robot, offset in layout.transform_offset.items():
        transforms[robot] = (
            float(vector[offset]),
            vector[offset + 1 : offset + 3],
        )
    components = vector[layout.component_offset :].reshape((-1, 2))
    return transforms, components


def evaluate_local_canonical(
    state: RobotCavityState,
    layout: VariableLayout,
    vector: np.ndarray,
) -> CanonicalEvaluation:
    transforms, components = _unpack(layout, vector)
    if state.robot not in transforms:
        raise ValueError("cavity robot is absent from the variable layout")

    yaw, translation = transforms[state.robot]
    rotation = rotation2(yaw)
    rotated = state.means @ rotation.T
    residual = np.empty_like(rotated)
    jacobian = np.zeros((2 * len(state.copies), layout.dimension))
    skew = np.array([[0.0, -1.0], [1.0, 0.0]])

    for row, copy in enumerate(state.copies):
        try:
            component_index = layout.component_by_copy[copy]
        except KeyError as error:
            raise ValueError(
                "cavity copy is absent from the variable layout"
            ) from error
        residual[row] = rotated[row] + translation - components[component_index]
        residual_rows = slice(2 * row, 2 * row + 2)
        if state.robot != layout.root_robot:
            transform_offset = layout.transform_offset[state.robot]
            jacobian[residual_rows, transform_offset] = skew @ rotated[row]
            jacobian[
                residual_rows, transform_offset + 1 : transform_offset + 3
            ] = np.eye(2)
        component_offset = layout.component_offset + 2 * component_index
        jacobian[
            residual_rows, component_offset : component_offset + 2
        ] = -np.eye(2)

    flat_residual = residual.reshape(-1)
    information_residual_matrix = np.linalg.solve(state.covariance, residual)
    information_residual = information_residual_matrix.reshape(-1)
    residual_chi2 = float(np.sum(residual * information_residual_matrix))
    covariance_sign, covariance_logdet = np.linalg.slogdet(state.covariance)
    if not (
        covariance_sign == 1.0
        and np.isfinite(covariance_sign)
        and np.isfinite(covariance_logdet)
    ):
        raise ValueError("cavity covariance must have positive determinant")
    value = float(
        0.5 * residual_chi2
        + covariance_logdet
        + len(state.copies) * np.log(2.0 * np.pi)
    )
    gradient = jacobian.T @ information_residual
    information_jacobian = np.linalg.solve(
        state.covariance,
        jacobian.reshape((len(state.copies), 2 * layout.dimension)),
    ).reshape((2 * len(state.copies), layout.dimension))
    hessian = jacobian.T @ information_jacobian

    if state.robot != layout.root_robot:
        transform_offset = layout.transform_offset[state.robot]
        hessian[transform_offset, transform_offset] += np.sum(
            -rotated * information_residual_matrix
        )
    hessian = 0.5 * (hessian + hessian.T)
    return CanonicalEvaluation(value, gradient, hessian, residual_chi2)


def evaluate_network_canonical(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
    vector: np.ndarray,
) -> CanonicalEvaluation:
    value = 0.0
    gradient = np.zeros(layout.dimension)
    hessian = np.zeros((layout.dimension, layout.dimension))
    residual_chi2 = 0.0
    for state in problem.robots:
        local = evaluate_local_canonical(state, layout, vector)
        value += local.value
        gradient += local.gradient
        hessian += local.hessian
        residual_chi2 += local.residual_chi2
    return CanonicalEvaluation(value, gradient, hessian, residual_chi2)


def _layout_signature(layout: VariableLayout) -> tuple[int, ...]:
    signature = [
        1,
        layout.root_robot,
        layout.dimension,
        layout.component_offset,
        len(layout.robots),
        *layout.robots,
        len(layout.transform_offset),
    ]
    for robot, offset in sorted(layout.transform_offset.items()):
        signature.extend((robot, offset))
    component_items = sorted(layout.component_by_copy.items())
    signature.append(len(component_items))
    for copy, component in component_items:
        signature.extend(
            (
                copy.robot,
                copy.object_id,
                copy.semantic_class,
                component,
            )
        )
    return tuple(signature)


def _active_variable_ids(
    state: RobotCavityState, layout: VariableLayout
) -> tuple[int, ...]:
    active = set()
    if state.robot != layout.root_robot:
        try:
            transform_offset = layout.transform_offset[state.robot]
        except KeyError as error:
            raise ValueError(
                "cavity robot is absent from the variable layout"
            ) from error
        active.update(range(transform_offset, transform_offset + 3))
    for copy in state.copies:
        try:
            component = layout.component_by_copy[copy]
        except KeyError as error:
            raise ValueError(
                "cavity copy is absent from the variable layout"
            ) from error
        component_offset = layout.component_offset + 2 * component
        active.update((component_offset, component_offset + 1))
    return tuple(sorted(active))


def build_canonical_packets(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
    vector: np.ndarray,
) -> tuple[CanonicalPacket, ...]:
    packets = []
    layout_signature = _layout_signature(layout)
    for state in sorted(problem.robots, key=lambda item: item.robot):
        evaluation = evaluate_local_canonical(state, layout, vector)
        variable_ids = _active_variable_ids(state, layout)
        local_hessian = evaluation.hessian[
            np.ix_(variable_ids, variable_ids)
        ]
        packed_upper = local_hessian[
            np.triu_indices(len(variable_ids))
        ]
        gradient = evaluation.gradient[list(variable_ids)]
        covariance_sign, covariance_logdet = np.linalg.slogdet(
            state.covariance
        )
        if not (
            covariance_sign == 1.0
            and np.isfinite(covariance_sign)
            and np.isfinite(covariance_logdet)
        ):
            raise ValueError("cavity covariance must have positive determinant")
        normalizer = float(
            covariance_logdet
            + len(state.copies) * np.log(2.0 * np.pi)
        )
        packets.append(
            CanonicalPacket(
                robot=state.robot,
                variable_ids=variable_ids,
                packed_upper_hessian=packed_upper,
                gradient=gradient,
                quadratic_value=0.5 * evaluation.residual_chi2,
                normalizer=normalizer,
                serialized_bytes=_canonical_packet_bytes(
                    len(variable_ids)
                ),
                ambient_dimension=layout.dimension,
                layout_signature=layout_signature,
            )
        )
    return tuple(packets)


def _unpack_upper_hessian(packet: CanonicalPacket) -> np.ndarray:
    dimension = len(packet.variable_ids)
    hessian = np.zeros((dimension, dimension))
    upper_rows, upper_columns = np.triu_indices(dimension)
    hessian[upper_rows, upper_columns] = packet.packed_upper_hessian
    hessian[upper_columns, upper_rows] = packet.packed_upper_hessian
    return hessian


def _combine_canonical_packets(
    first: CanonicalPacket, second: CanonicalPacket
) -> CanonicalPacket:
    if (
        first.ambient_dimension != second.ambient_dimension
        or first.layout_signature != second.layout_signature
    ):
        raise ValueError("canonical packets have incompatible layout metadata")
    variable_ids = tuple(
        sorted(set(first.variable_ids) | set(second.variable_ids))
    )
    dimension = len(variable_ids)
    position = {
        variable_id: index
        for index, variable_id in enumerate(variable_ids)
    }
    hessian = np.zeros((dimension, dimension))
    gradient = np.zeros(dimension)
    for packet in (first, second):
        packet_positions = tuple(
            position[variable_id] for variable_id in packet.variable_ids
        )
        hessian[np.ix_(packet_positions, packet_positions)] += (
            _unpack_upper_hessian(packet)
        )
        gradient[list(packet_positions)] += packet.gradient
    return CanonicalPacket(
        robot=first.robot,
        variable_ids=variable_ids,
        packed_upper_hessian=hessian[np.triu_indices(dimension)],
        gradient=gradient,
        quadratic_value=first.quadratic_value + second.quadratic_value,
        normalizer=first.normalizer + second.normalizer,
        serialized_bytes=_canonical_packet_bytes(dimension),
        ambient_dimension=first.ambient_dimension,
        layout_signature=first.layout_signature,
    )


def _validated_topology(
    robot_ids: tuple[int, ...],
    topology: Iterable[tuple[int, int]],
    *,
    require_tree: bool,
    support_name: str,
) -> tuple[dict[int, tuple[int, ...]], tuple[tuple[int, int], ...]]:
    expected = set(robot_ids)
    neighbors = {robot: set() for robot in robot_ids}
    edges = []
    seen_edges = set()
    try:
        supplied_edges = tuple(topology)
    except TypeError as error:
        raise ValueError("topology must be an iterable of edges") from error
    for edge in supplied_edges:
        try:
            endpoints = tuple(edge)
        except TypeError as error:
            raise ValueError("topology edges must contain two robots") from error
        if len(endpoints) != 2:
            raise ValueError("topology edges must contain two robots")
        left, right = endpoints
        if any(
            isinstance(robot, bool)
            or not isinstance(robot, Integral)
            for robot in endpoints
        ):
            raise ValueError("topology robot IDs must be integers")
        left, right = int(left), int(right)
        if left == right:
            raise ValueError("topology must not contain a self-loop")
        if left not in expected or right not in expected:
            raise ValueError(
                f"topology must cover exactly {support_name} robots"
            )
        canonical_edge = (min(left, right), max(left, right))
        if canonical_edge in seen_edges:
            raise ValueError("topology must be a simple undirected graph")
        seen_edges.add(canonical_edge)
        edges.append(canonical_edge)
        neighbors[left].add(right)
        neighbors[right].add(left)

    incident = {robot for edge in edges for robot in edge}
    if len(robot_ids) > 1 and incident != expected:
        if require_tree:
            raise ValueError(
                f"topology must cover exactly {support_name} robots"
            )
        raise ValueError("topology must be connected")
    if robot_ids:
        reached = {robot_ids[0]}
        queue = [robot_ids[0]]
        for robot in queue:
            for neighbor in sorted(neighbors[robot]):
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if reached != expected:
            raise ValueError("topology must be connected")
    if require_tree and len(edges) != max(0, len(robot_ids) - 1):
        raise ValueError("topology must be a tree")
    return (
        {
            robot: tuple(sorted(robot_neighbors))
            for robot, robot_neighbors in neighbors.items()
        },
        tuple(sorted(edges)),
    )


def _reduce_packets_on_tree_with_trace(
    packets: Iterable[CanonicalPacket],
    topology: Iterable[tuple[int, int]],
    root_robot: int,
) -> tuple[ReducedCanonicalModel, dict[int, int]]:
    packets = tuple(packets)
    if any(not isinstance(packet, CanonicalPacket) for packet in packets):
        raise ValueError("all routed values must be canonical packets")
    robot_ids = tuple(packet.robot for packet in packets)
    if len(set(robot_ids)) != len(robot_ids):
        raise ValueError("tree reduction requires one packet per robot")
    if not packets:
        raise ValueError("tree reduction requires at least one packet")
    layout_metadata = (
        packets[0].ambient_dimension,
        packets[0].layout_signature,
    )
    if any(
        (packet.ambient_dimension, packet.layout_signature) != layout_metadata
        for packet in packets[1:]
    ):
        raise ValueError("canonical packets have incompatible layout metadata")
    robot_ids = tuple(sorted(robot_ids))
    if (
        isinstance(root_robot, bool)
        or not isinstance(root_robot, Integral)
        or int(root_robot) not in set(robot_ids)
    ):
        raise ValueError("routing root must be a packet robot")
    root_robot = int(root_robot)
    neighbors, _ = _validated_topology(
        robot_ids,
        topology,
        require_tree=True,
        support_name="packet",
    )

    parent = {root_robot: None}
    order = [root_robot]
    for robot in order:
        for neighbor in neighbors[robot]:
            if neighbor not in parent:
                parent[neighbor] = robot
                order.append(neighbor)

    aggregate = {packet.robot: packet for packet in packets}
    link_payload_bytes = 0
    bytes_by_transmitter = {robot: 0 for robot in robot_ids}
    for robot in reversed(order[1:]):
        transmitted = aggregate[robot]
        link_payload_bytes += transmitted.serialized_bytes
        bytes_by_transmitter[robot] += transmitted.serialized_bytes
        destination = parent[robot]
        aggregate[destination] = _combine_canonical_packets(
            aggregate[destination], transmitted
        )

    final = aggregate[root_robot]
    if not final.variable_ids:
        raise ValueError("cannot infer a model from an empty variable union")
    dimension = final.ambient_dimension
    final_positions = list(final.variable_ids)
    hessian = np.zeros((dimension, dimension))
    gradient = np.zeros(dimension)
    hessian[np.ix_(final_positions, final_positions)] = (
        _unpack_upper_hessian(final)
    )
    gradient[final_positions] = final.gradient
    for child in order[1:]:
        transmitter = parent[child]
        link_payload_bytes += final.serialized_bytes
        bytes_by_transmitter[transmitter] += final.serialized_bytes
    return (
        ReducedCanonicalModel(
            hessian=hessian,
            gradient=gradient,
            value=final.quadratic_value + final.normalizer,
            link_payload_bytes=link_payload_bytes,
        ),
        bytes_by_transmitter,
    )


def reduce_packets_on_tree(
    packets: Iterable[CanonicalPacket],
    topology: Iterable[tuple[int, int]],
    root_robot: int,
) -> ReducedCanonicalModel:
    return _reduce_packets_on_tree_with_trace(
        packets, topology, root_robot
    )[0]


def reduce_packets_on_tree_with_trace(
    packets: Iterable[CanonicalPacket],
    topology: Iterable[tuple[int, int]],
    root_robot: int,
) -> tuple[ReducedCanonicalModel, Mapping[int, int]]:
    reduced, trace = _reduce_packets_on_tree_with_trace(
        packets, topology, root_robot
    )
    return reduced, MappingProxyType(trace)


def _graph_eccentricity(
    source: int, neighbors: Mapping[int, tuple[int, ...]]
) -> int:
    distances = {source: 0}
    queue = [source]
    for robot in queue:
        for neighbor in neighbors[robot]:
            if neighbor not in distances:
                distances[neighbor] = distances[robot] + 1
                queue.append(neighbor)
    return max(distances.values(), default=0)


def run_min_consensus(
    proposals: Mapping[int, Iterable[tuple[float, str]]],
    topology: Iterable[tuple[int, int]],
) -> MinConsensusResult:
    if not isinstance(proposals, Mapping) or not proposals:
        raise ValueError("proposals must be a nonempty mapping")
    robot_ids = tuple(proposals)
    if any(
        isinstance(robot, bool)
        or not isinstance(robot, Integral)
        or robot < 0
        for robot in robot_ids
    ) or len(set(robot_ids)) != len(robot_ids):
        raise ValueError("proposal robot IDs must be unique nonnegative integers")
    robot_ids = tuple(sorted(int(robot) for robot in robot_ids))
    local_best = {}
    for robot in robot_ids:
        try:
            robot_proposals = tuple(proposals[robot])
        except (KeyError, TypeError) as error:
            raise ValueError("every robot must have a proposal") from error
        if not robot_proposals:
            raise ValueError("every robot must have at least one proposal")
        validated = [
            _normalize_consensus_value(proposal, "proposal")
            for proposal in robot_proposals
        ]
        local_best[robot] = min(validated)

    neighbors, edges = _validated_topology(
        robot_ids,
        topology,
        require_tree=False,
        support_name="proposal",
    )
    diameter = max(
        (_graph_eccentricity(robot, neighbors) for robot in robot_ids),
        default=0,
    )
    payload_bytes = 0
    current = local_best
    for _ in range(diameter):
        previous = current
        for left, right in edges:
            payload_bytes += 8 + len(previous[left][1].encode("utf-8"))
            payload_bytes += 8 + len(previous[right][1].encode("utf-8"))
        current = {
            robot: min(
                (previous[robot],)
                + tuple(previous[neighbor] for neighbor in neighbors[robot])
            )
            for robot in robot_ids
        }

    selected_values = set(current.values())
    if len(selected_values) != 1:
        raise ValueError("min consensus did not reach agreement")
    selected = next(iter(selected_values))
    return MinConsensusResult(
        selected=selected,
        selected_by_robot=current,
        rounds=diameter,
        payload_bytes=payload_bytes,
    )


def finite_difference_hessian(
    gradient: Callable[[np.ndarray], np.ndarray], vector: np.ndarray
) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    hessian = np.empty((vector.size, vector.size))
    base_step = np.finfo(float).eps ** 0.25
    for column in range(vector.size):
        step = base_step * (1.0 + abs(vector[column]))
        plus = vector.copy()
        minus = vector.copy()
        plus[column] += step
        minus[column] -= step
        hessian[:, column] = (gradient(plus) - gradient(minus)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    status: str


def validate_fixed_cardinality_candidate(
    current: NetworkPartition, candidate: NetworkPartition
) -> ValidationResult:
    if set(current.copies) != set(candidate.copies):
        return ValidationResult(valid=False, status="copy_support_mismatch")
    if current.component_count != candidate.component_count:
        return ValidationResult(
            valid=False, status="cardinality_change_unsupported"
        )
    return ValidationResult(valid=True, status="ok")


@dataclass(frozen=True)
class CutJoinMove:
    valid: bool
    status: str
    removed_edge: CanonicalEdge
    added_edge: CanonicalEdge
    candidate: NetworkPartition | None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("cut-and-join validity must be boolean")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("cut-and-join status must be a nonempty string")
        removed_edge = _normalize_edge(self.removed_edge)
        added_edge = _normalize_edge(self.added_edge)
        if self.candidate is not None and not isinstance(
            self.candidate, NetworkPartition
        ):
            raise ValueError("cut-and-join candidate must be a partition")
        if self.valid:
            if self.status != "ok" or self.candidate is None:
                raise ValueError(
                    "a valid cut-and-join candidate must have status ok"
                )
            if removed_edge == added_edge:
                raise ValueError(
                    "a valid cut-and-join move must use different edges"
                )
            candidate_support = set(self.candidate.copies)
            if any(
                endpoint not in candidate_support
                for edge in (removed_edge, added_edge)
                for endpoint in edge
            ):
                raise ValueError(
                    "a valid cut-and-join candidate must contain all endpoints"
                )
            component_by_copy = {
                copy: component_index
                for component_index, component in enumerate(
                    self.candidate.components
                )
                for copy in component
            }
            if (
                component_by_copy[removed_edge[0]]
                == component_by_copy[removed_edge[1]]
            ):
                raise ValueError(
                    "a valid move must separate the removed edge endpoints"
                )
            if (
                component_by_copy[added_edge[0]]
                != component_by_copy[added_edge[1]]
            ):
                raise ValueError(
                    "a valid move must join the added edge endpoints"
                )
        elif self.status == "ok":
            raise ValueError("an invalid cut-and-join status cannot be ok")

        object.__setattr__(self, "removed_edge", removed_edge)
        object.__setattr__(self, "added_edge", added_edge)


def make_cut_join_move(
    incumbent: NetworkPartition,
    incumbent_edges: Iterable[Iterable[ObjectCopy]],
    remove_edge: Iterable[ObjectCopy],
    add_edge: Iterable[ObjectCopy],
) -> CutJoinMove:
    if not isinstance(incumbent, NetworkPartition):
        raise ValueError("incumbent must be a NetworkPartition")
    removed = _normalize_edge(remove_edge)
    added = _normalize_edge(add_edge)
    try:
        supplied_edges = tuple(
            _normalize_edge(edge) for edge in incumbent_edges
        )
    except (TypeError, ValueError):
        return CutJoinMove(
            False,
            "incumbent_edges_not_canonical_forest",
            removed,
            added,
            None,
        )
    expected_edges = canonical_partition_forest(incumbent)
    if (
        len(set(supplied_edges)) != len(supplied_edges)
        or tuple(sorted(supplied_edges)) != expected_edges
    ):
        return CutJoinMove(
            False,
            "incumbent_edges_not_canonical_forest",
            removed,
            added,
            None,
        )
    edges = set(supplied_edges)
    if removed not in edges:
        return CutJoinMove(
            False, "missing_removed_edge", removed, added, None
        )
    if added == removed:
        return CutJoinMove(
            False,
            "added_edge_matches_removed_edge",
            removed,
            added,
            None,
        )
    if added in edges:
        return CutJoinMove(
            False, "existing_added_edge", removed, added, None
        )

    edges.remove(removed)
    edges.add(added)
    try:
        candidate = partition_from_edges(
            incumbent.copies, tuple(sorted(edges))
        )
    except ValueError:
        return CutJoinMove(
            False, "invalid_candidate", removed, added, None
        )
    if candidate == incumbent:
        return CutJoinMove(
            False, "unchanged_candidate", removed, added, None
        )
    validation = validate_fixed_cardinality_candidate(incumbent, candidate)
    return CutJoinMove(
        validation.valid,
        validation.status,
        removed,
        added,
        candidate,
    )


@dataclass(frozen=True)
class QuotientProfileState:
    vector: np.ndarray
    root_robot: int
    objective: float
    evaluations: int

    def __post_init__(self) -> None:
        vector = np.array(self.vector, dtype=float, copy=True)
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)


@dataclass(frozen=True)
class QuotientScore:
    valid: bool
    status: str
    score: float
    profile_chi2: float
    logdet: float
    root_scores: tuple[tuple[int, float], ...]
    root_score_spread: float
    stationarity_norm: float
    profiles: tuple[QuotientProfileState, ...]


@dataclass(frozen=True)
class _ReducedProfileWorkspace:
    layout: VariableLayout
    information_by_robot: tuple[np.ndarray, ...]
    covariance_factors_by_robot: tuple[np.ndarray, ...]
    component_rows_by_robot: tuple[np.ndarray, ...]
    component_information: np.ndarray
    component_scaling: np.ndarray
    component_factor: np.ndarray


def _build_reduced_profile_workspace(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
) -> _ReducedProfileWorkspace:
    information_by_robot = []
    covariance_factors_by_robot = []
    component_rows_by_robot = []
    component_information = np.zeros(
        (problem.partition.component_count, problem.partition.component_count)
    )
    for state in problem.robots:
        covariance_factor = np.linalg.cholesky(state.covariance)
        information = np.linalg.solve(
            state.covariance, np.eye(len(state.copies))
        )
        information = 0.5 * (information + information.T)
        component_rows = np.asarray(
            [layout.component_by_copy[copy] for copy in state.copies],
            dtype=int,
        )
        component_information[
            np.ix_(component_rows, component_rows)
        ] += information
        covariance_factors_by_robot.append(covariance_factor)
        information_by_robot.append(information)
        component_rows_by_robot.append(component_rows)

    component_information = 0.5 * (
        component_information + component_information.T
    )
    component_diagonal = np.diag(component_information)
    if (
        not np.all(np.isfinite(component_diagonal))
        or np.any(component_diagonal <= 0.0)
    ):
        raise ValueError(
            "component information must have positive finite diagonal"
        )
    component_scaling = 1.0 / np.sqrt(component_diagonal)
    component_factor = np.linalg.cholesky(
        component_scaling[:, None]
        * component_information
        * component_scaling[None, :]
    )
    return _ReducedProfileWorkspace(
        layout=layout,
        information_by_robot=tuple(information_by_robot),
        covariance_factors_by_robot=tuple(covariance_factors_by_robot),
        component_rows_by_robot=tuple(component_rows_by_robot),
        component_information=component_information,
        component_scaling=component_scaling,
        component_factor=component_factor,
    )


def _transforms_from_reduced_vector(
    layout: VariableLayout, transform_vector: np.ndarray
) -> dict[int, tuple[float, np.ndarray]]:
    vector = np.asarray(transform_vector, dtype=float)
    if vector.shape != (layout.component_offset,):
        raise ValueError(
            "transform vector must have shape "
            f"({layout.component_offset},)"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("transform vector must be finite")

    transforms = {layout.root_robot: (0.0, np.zeros(2))}
    for robot, offset in layout.transform_offset.items():
        transforms[robot] = (
            float(vector[offset]), vector[offset + 1 : offset + 3]
        )
    return transforms


def _transformed_means_for_reduced_profile(
    problem: NetworkQuotientProblem,
    workspace: _ReducedProfileWorkspace,
    transform_vector: np.ndarray,
) -> tuple[np.ndarray, ...]:
    transforms = _transforms_from_reduced_vector(
        workspace.layout, transform_vector
    )
    return tuple(
        state.means @ rotation2(transforms[state.robot][0]).T
        + transforms[state.robot][1]
        for state in problem.robots
    )


def _solve_reduced_component_system(
    workspace: _ReducedProfileWorkspace, rhs: np.ndarray
) -> np.ndarray:
    scaled_rhs = workspace.component_scaling[:, None] * rhs
    solution = solve_triangular(
        workspace.component_factor.T,
        solve_triangular(
            workspace.component_factor,
            scaled_rhs,
            lower=True,
            check_finite=False,
        ),
        lower=False,
        check_finite=False,
    )
    return workspace.component_scaling[:, None] * solution


def _profile_components_for_transforms(
    problem: NetworkQuotientProblem,
    workspace: _ReducedProfileWorkspace,
    transform_vector: np.ndarray,
) -> np.ndarray:
    transformed_means = _transformed_means_for_reduced_profile(
        problem, workspace, transform_vector
    )
    component_rhs = np.zeros((problem.partition.component_count, 2))
    for information, component_rows, means in zip(
        workspace.information_by_robot,
        workspace.component_rows_by_robot,
        transformed_means,
    ):
        np.add.at(component_rhs, component_rows, information @ means)
    components = _solve_reduced_component_system(workspace, component_rhs)
    return np.concatenate(
        (np.asarray(transform_vector, dtype=float), components.reshape(-1))
    )


def _reduced_profile_residual_and_jacobian(
    problem: NetworkQuotientProblem,
    workspace: _ReducedProfileWorkspace,
    transform_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = workspace.layout
    transforms = _transforms_from_reduced_vector(
        layout, transform_vector
    )
    transformed_means = tuple(
        state.means @ rotation2(transforms[state.robot][0]).T
        + transforms[state.robot][1]
        for state in problem.robots
    )
    transform_dimension = layout.component_offset
    component_rhs = np.zeros((problem.partition.component_count, 2))
    component_rhs_derivative = np.zeros(
        (problem.partition.component_count, 2, transform_dimension)
    )
    mean_derivatives = []
    skew = np.array([[0.0, -1.0], [1.0, 0.0]])
    for state, information, component_rows, means in zip(
        problem.robots,
        workspace.information_by_robot,
        workspace.component_rows_by_robot,
        transformed_means,
    ):
        derivative = np.zeros(
            (len(state.copies), 2, transform_dimension)
        )
        if state.robot != layout.root_robot:
            transform_offset = layout.transform_offset[state.robot]
            rotated = means - transforms[state.robot][1]
            derivative[:, :, transform_offset] = rotated @ skew.T
            derivative[
                :, :, transform_offset + 1 : transform_offset + 3
            ] = np.eye(2)
        np.add.at(component_rhs, component_rows, information @ means)
        local_rhs_derivative = np.einsum(
            "ab,bck->ack", information, derivative
        )
        np.add.at(
            component_rhs_derivative,
            component_rows,
            local_rhs_derivative,
        )
        mean_derivatives.append(derivative)

    components = _solve_reduced_component_system(
        workspace, component_rhs
    )
    component_derivative = _solve_reduced_component_system(
        workspace,
        component_rhs_derivative.reshape(
            (problem.partition.component_count, -1)
        ),
    ).reshape(
        (problem.partition.component_count, 2, transform_dimension)
    )
    residual_blocks = []
    jacobian_blocks = []
    for lower, component_rows, means, mean_derivative in zip(
        workspace.covariance_factors_by_robot,
        workspace.component_rows_by_robot,
        transformed_means,
        mean_derivatives,
    ):
        residual = means - components[component_rows]
        jacobian = mean_derivative - component_derivative[component_rows]
        residual_blocks.append(
            solve_triangular(
                lower, residual, lower=True, check_finite=False
            ).reshape(-1)
        )
        jacobian_blocks.append(
            solve_triangular(
                lower,
                jacobian.reshape((len(component_rows), -1)),
                lower=True,
                check_finite=False,
            ).reshape((2 * len(component_rows), transform_dimension))
        )
    full_vector = np.concatenate(
        (np.asarray(transform_vector, dtype=float), components.reshape(-1))
    )
    return (
        np.concatenate(residual_blocks),
        np.vstack(jacobian_blocks),
        full_vector,
    )


def profile_residual_and_jacobian(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
    vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transforms, components = _unpack(layout, vector)
    residual_blocks = []
    jacobian_blocks = []
    skew = np.array([[0.0, -1.0], [1.0, 0.0]])

    for state in problem.robots:
        yaw, translation = transforms[state.robot]
        rotated = state.means @ rotation2(yaw).T
        residual = np.empty_like(rotated)
        jacobian = np.zeros(
            (len(state.copies), 2, layout.dimension), dtype=float
        )
        for row, copy in enumerate(state.copies):
            component_index = layout.component_by_copy[copy]
            residual[row] = (
                rotated[row] + translation - components[component_index]
            )
            if state.robot != layout.root_robot:
                transform_offset = layout.transform_offset[state.robot]
                jacobian[row, :, transform_offset] = skew @ rotated[row]
                jacobian[
                    row, :, transform_offset + 1 : transform_offset + 3
                ] = np.eye(2)
            component_offset = layout.component_offset + 2 * component_index
            jacobian[
                row, :, component_offset : component_offset + 2
            ] = -np.eye(2)

        lower = np.linalg.cholesky(state.covariance)
        whitened_residual = solve_triangular(
            lower, residual, lower=True, check_finite=False
        )
        whitened_jacobian = solve_triangular(
            lower,
            jacobian.reshape((len(state.copies), 2 * layout.dimension)),
            lower=True,
            check_finite=False,
        ).reshape((len(state.copies), 2, layout.dimension))
        residual_blocks.append(whitened_residual.reshape(-1))
        jacobian_blocks.append(
            whitened_jacobian.reshape((2 * len(state.copies), layout.dimension))
        )

    return np.concatenate(residual_blocks), np.vstack(jacobian_blocks)


def _vector_from_transforms(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
    transforms: Mapping[int, tuple[float, np.ndarray]],
) -> np.ndarray:
    vector = np.zeros(layout.dimension)
    transformed_means: dict[ObjectCopy, np.ndarray] = {}
    for state in problem.robots:
        yaw, translation = transforms[state.robot]
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (2,):
            raise ValueError("transform translation must have shape (2,)")
        if not np.isfinite(yaw) or not np.all(np.isfinite(translation)):
            raise ValueError("transforms must be finite")
        if state.robot != layout.root_robot:
            offset = layout.transform_offset[state.robot]
            vector[offset] = yaw
            vector[offset + 1 : offset + 3] = translation
        means = state.means @ rotation2(yaw).T + translation
        transformed_means.update(zip(state.copies, means))

    for component_index, component in enumerate(problem.partition.components):
        position = np.mean(
            [transformed_means[copy] for copy in component], axis=0
        )
        offset = layout.component_offset + 2 * component_index
        vector[offset : offset + 2] = position
    return vector


def identity_profile_start(
    problem: NetworkQuotientProblem, layout: VariableLayout
) -> np.ndarray:
    transforms = {
        robot: (0.0, np.zeros(2, dtype=float)) for robot in layout.robots
    }
    return _vector_from_transforms(problem, layout, transforms)


def _weighted_procrustes(
    source: np.ndarray, target: np.ndarray, information: np.ndarray
) -> tuple[float, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    information = np.asarray(information, dtype=float)
    if source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError("source and target must have equal Nx2 shapes")
    if target.shape != source.shape:
        raise ValueError("source and target must have equal Nx2 shapes")
    if source.shape[0] < 2:
        raise ValueError("weighted Procrustes requires at least two points")
    if information.shape != (source.shape[0], source.shape[0]):
        raise ValueError("information must have shape (N, N)")

    ones = np.ones(source.shape[0])
    centroid_weights = information @ ones
    denominator = float(ones @ centroid_weights)
    source_centroid = source.T @ centroid_weights / denominator
    target_centroid = target.T @ centroid_weights / denominator
    centered_source = source - source_centroid
    centered_target = target - target_centroid
    cross = centered_target.T @ information @ centered_source
    left, _, right_transpose = np.linalg.svd(cross)
    determinant_correction = np.diag(
        [1.0, np.linalg.det(left @ right_transpose)]
    )
    rotation = left @ determinant_correction @ right_transpose
    translation = target_centroid - rotation @ source_centroid
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return yaw, translation


def incidence_profile_start(
    problem: NetworkQuotientProblem, layout: VariableLayout
) -> np.ndarray:
    states = {state.robot: state for state in problem.robots}
    component_copies: list[dict[int, ObjectCopy]] = []
    for component in problem.partition.components:
        component_copies.append({copy.robot: copy for copy in component})

    shared_components: dict[tuple[int, int], tuple[int, ...]] = {}
    for first_index, first_robot in enumerate(layout.robots):
        for second_robot in layout.robots[first_index + 1 :]:
            shared = tuple(
                index
                for index, copies in enumerate(component_copies)
                if first_robot in copies and second_robot in copies
            )
            if len(shared) >= 2:
                shared_components[(first_robot, second_robot)] = shared

    transforms: dict[int, tuple[float, np.ndarray]] = {
        layout.root_robot: (0.0, np.zeros(2, dtype=float))
    }
    queue = [layout.root_robot]
    while queue:
        parent = queue.pop(0)
        neighbors = []
        for robot in layout.robots:
            key = tuple(sorted((parent, robot)))
            if robot not in transforms and key in shared_components:
                neighbors.append(robot)
        for child in sorted(neighbors):
            key = tuple(sorted((parent, child)))
            shared = shared_components[key]
            child_state = states[child]
            parent_state = states[parent]
            child_row = {copy: row for row, copy in enumerate(child_state.copies)}
            parent_row = {
                copy: row for row, copy in enumerate(parent_state.copies)
            }
            source_rows = [
                child_row[component_copies[index][child]] for index in shared
            ]
            parent_rows = [
                parent_row[component_copies[index][parent]] for index in shared
            ]
            source = child_state.means[source_rows]
            parent_yaw, parent_translation = transforms[parent]
            target = (
                parent_state.means[parent_rows] @ rotation2(parent_yaw).T
                + parent_translation
            )
            child_information = np.linalg.solve(
                child_state.covariance, np.eye(len(child_state.copies))
            )
            information = child_information[np.ix_(source_rows, source_rows)]
            transforms[child] = _weighted_procrustes(
                source, target, information
            )
            queue.append(child)

    if len(transforms) != len(layout.robots):
        raise ValueError("partition transform graph is unobservable")
    return _vector_from_transforms(problem, layout, transforms)


def deterministic_profile_starts(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
) -> tuple[np.ndarray, ...]:
    starts = [incidence_profile_start(problem, layout)]
    if current_transforms is not None:
        if set(current_transforms) != set(layout.robots):
            raise ValueError("current transforms must cover every robot")
        root_yaw, root_translation = current_transforms[layout.root_robot]
        if root_yaw != 0.0 or not np.array_equal(
            np.asarray(root_translation, dtype=float), np.zeros(2)
        ):
            raise ValueError("root transform must be identity")
        current = _vector_from_transforms(problem, layout, current_transforms)
        if not any(np.array_equal(current, start) for start in starts):
            starts.append(current)
    return tuple(starts)


def _stationarity_norm(
    evaluation: CanonicalEvaluation, vector: np.ndarray
) -> float:
    return float(
        np.linalg.norm(evaluation.gradient)
        /
        (
            1.0
            + np.linalg.norm(evaluation.hessian, ord="fro")
            * np.linalg.norm(vector)
            + abs(evaluation.value)
        )
    )


def _equilibrated_spd_logdet(hessian: np.ndarray) -> float:
    matrix = np.asarray(hessian, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Hessian must be a finite square matrix")
    diagonal = np.diag(matrix)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("Hessian must have a positive finite diagonal")
    scaling = 1.0 / np.sqrt(diagonal)
    equilibrated = scaling[:, None] * matrix * scaling[None, :]
    factor = np.linalg.cholesky(equilibrated)
    logdet_equilibrated = float(2.0 * np.sum(np.log(np.diag(factor))))
    return float(logdet_equilibrated - 2.0 * np.sum(np.log(scaling)))


def _restore_full_profile_result(
    problem: NetworkQuotientProblem,
    workspace: _ReducedProfileWorkspace,
    reduced_result: OptimizeResult,
) -> OptimizeResult:
    _, _, full_vector = _reduced_profile_residual_and_jacobian(
        problem, workspace, reduced_result.x
    )
    residual, jacobian = profile_residual_and_jacobian(
        problem, workspace.layout, full_vector
    )
    result = OptimizeResult(reduced_result)
    result.x = full_vector
    result.fun = residual
    result.jac = jacobian
    result.cost = 0.5 * float(residual @ residual)
    result.grad = jacobian.T @ residual
    result.optimality = float(np.linalg.norm(result.grad, ord=np.inf))
    result.nfev = int(getattr(reduced_result, "nfev", 0))
    result.njev = int(getattr(reduced_result, "njev", 0))
    return result


def _exact_newton_polish(
    problem: NetworkQuotientProblem,
    layout: VariableLayout,
    result: OptimizeResult,
) -> OptimizeResult:
    result.exact_newton_polished = False
    evaluation = evaluate_network_canonical(problem, layout, result.x)
    if not (
        np.isfinite(evaluation.value)
        and np.all(np.isfinite(evaluation.gradient))
        and np.all(np.isfinite(evaluation.hessian))
    ):
        return result
    try:
        lower = np.linalg.cholesky(evaluation.hessian)
    except np.linalg.LinAlgError:
        return result
    delta = solve_triangular(
        lower.T,
        solve_triangular(
            lower, -evaluation.gradient, lower=True, check_finite=False
        ),
        lower=False,
        check_finite=False,
    )
    if not np.all(np.isfinite(delta)):
        return result
    candidate = result.x + delta
    candidate_evaluation = evaluate_network_canonical(
        problem, layout, candidate
    )
    candidate_stationarity = _stationarity_norm(
        candidate_evaluation, candidate
    )
    if not (
        np.isfinite(candidate_evaluation.value)
        and np.all(np.isfinite(candidate_evaluation.gradient))
        and np.all(np.isfinite(candidate_evaluation.hessian))
        and candidate_evaluation.value <= evaluation.value
        and candidate_stationarity <= _stationarity_norm(evaluation, result.x)
    ):
        return result

    polished_residual, polished_jacobian = profile_residual_and_jacobian(
        problem, layout, candidate
    )
    result.x = candidate
    result.fun = polished_residual
    result.jac = polished_jacobian
    result.cost = 0.5 * float(polished_residual @ polished_residual)
    result.grad = polished_jacobian.T @ polished_residual
    result.optimality = float(np.linalg.norm(result.grad, ord=np.inf))
    result.nfev += 1
    result.njev += 1
    result.exact_newton_polished = True
    return result


def profile_partition_for_root(
    problem: NetworkQuotientProblem,
    root_robot: int,
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
) -> tuple[VariableLayout, OptimizeResult]:
    layout = VariableLayout.build(problem, root_robot)
    workspace = _build_reduced_profile_workspace(problem, layout)
    full_starts = deterministic_profile_starts(
        problem, layout, current_transforms
    )
    starts = []
    for full_start in full_starts:
        reduced_start = np.array(
            full_start[: layout.component_offset], copy=True
        )
        if not any(np.array_equal(reduced_start, start) for start in starts):
            starts.append(reduced_start)
    tolerance = np.finfo(float).eps ** (2.0 / 3.0)
    successful_results = []

    def residual(vector: np.ndarray) -> np.ndarray:
        return _reduced_profile_residual_and_jacobian(
            problem, workspace, vector
        )[0]

    def jacobian(vector: np.ndarray) -> np.ndarray:
        return _reduced_profile_residual_and_jacobian(
            problem, workspace, vector
        )[1]

    for start_index, start in enumerate(starts):
        try:
            reduced_result = least_squares(
                residual,
                start,
                jac=jacobian,
                xtol=tolerance,
                ftol=tolerance,
                gtol=tolerance,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue
        if (
            reduced_result.success
            and np.isfinite(reduced_result.cost)
            and np.all(np.isfinite(reduced_result.x))
        ):
            result = _restore_full_profile_result(
                problem, workspace, reduced_result
            )
            result = _exact_newton_polish(problem, layout, result)
            successful_results.append((float(result.cost), start_index, result))
    if not successful_results:
        raise ValueError("partition profiling failed")
    _, _, best = min(successful_results, key=lambda item: (item[0], item[1]))
    return layout, best


def _invalid_quotient_score(
    status: str,
    profiles: tuple[QuotientProfileState, ...] = (),
    root_scores: tuple[tuple[int, float], ...] = (),
) -> QuotientScore:
    return QuotientScore(
        valid=False,
        status=status,
        score=np.nan,
        profile_chi2=np.nan,
        logdet=np.nan,
        root_scores=root_scores,
        root_score_spread=np.nan,
        stationarity_norm=np.nan,
        profiles=profiles,
    )


def _validate_current_transforms(
    problem: NetworkQuotientProblem,
    current_transforms: Mapping[int, tuple[float, np.ndarray]],
) -> dict[int, tuple[float, np.ndarray]]:
    robot_ids = {state.robot for state in problem.robots}
    try:
        supplied_robot_ids = set(current_transforms)
    except TypeError as error:
        raise ValueError("current transforms must be a mapping") from error
    if supplied_robot_ids != robot_ids:
        raise ValueError("current transforms must cover every robot")
    parsed = {}
    for robot in sorted(robot_ids):
        try:
            yaw, translation = current_transforms[robot]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "each current transform must contain yaw and translation"
            ) from error
        try:
            yaw_array = np.asarray(yaw, dtype=float)
            translation = np.asarray(translation, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("current transforms must be numeric") from error
        if yaw_array.shape != ():
            raise ValueError("transform yaw must be a scalar")
        if translation.shape != (2,):
            raise ValueError("transform translation must have shape (2,)")
        yaw_value = float(yaw_array)
        if not np.isfinite(yaw_value) or not np.all(np.isfinite(translation)):
            raise ValueError("transforms must be finite")
        parsed[robot] = (yaw_value, translation.copy())
    return parsed


def _regauge_current_transforms(
    problem: NetworkQuotientProblem,
    root_robot: int,
    current_transforms: Mapping[int, tuple[float, np.ndarray]],
) -> dict[int, tuple[float, np.ndarray]]:
    root_yaw, root_translation = current_transforms[root_robot]
    root_rotation_transpose = rotation2(root_yaw).T
    root_translation = np.asarray(root_translation, dtype=float)
    regauged = {}
    for state in problem.robots:
        yaw, translation = current_transforms[state.robot]
        rotation = root_rotation_transpose @ rotation2(yaw)
        regauged[state.robot] = (
            float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            root_rotation_transpose
            @ (np.asarray(translation, dtype=float) - root_translation),
        )
    regauged[root_robot] = (0.0, np.zeros(2, dtype=float))
    return regauged


def _center_profile_problem(
    problem: NetworkQuotientProblem,
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None,
) -> tuple[
    NetworkQuotientProblem,
    dict[int, tuple[float, np.ndarray]] | None,
]:
    centers = {
        state.robot: np.mean(state.means, axis=0) for state in problem.robots
    }
    centered = NetworkQuotientProblem(
        tuple(
            RobotCavityState(
                state.robot,
                state.copies,
                state.means - centers[state.robot],
                state.covariance,
            )
            for state in problem.robots
        ),
        problem.partition,
    )
    if current_transforms is None:
        return centered, None
    centered_transforms = {
        robot: (
            yaw,
            np.asarray(translation, dtype=float)
            + rotation2(yaw) @ centers[robot],
        )
        for robot, (yaw, translation) in current_transforms.items()
    }
    return centered, centered_transforms


def _restore_original_profile_vector(
    original_problem: NetworkQuotientProblem,
    layout: VariableLayout,
    centered_vector: np.ndarray,
) -> np.ndarray:
    centers = {
        state.robot: np.mean(state.means, axis=0)
        for state in original_problem.robots
    }
    root_center = centers[layout.root_robot]
    restored = np.array(centered_vector, dtype=float, copy=True)
    for state in original_problem.robots:
        if state.robot == layout.root_robot:
            continue
        offset = layout.transform_offset[state.robot]
        yaw = restored[offset]
        restored[offset + 1 : offset + 3] += (
            root_center - rotation2(yaw) @ centers[state.robot]
        )
    for component_index in range(original_problem.partition.component_count):
        offset = layout.component_offset + 2 * component_index
        restored[offset : offset + 2] += root_center
    return restored


def _component_scalar_information(
    problem: NetworkQuotientProblem,
) -> np.ndarray:
    component_by_copy = {
        copy: index
        for index, component in enumerate(problem.partition.components)
        for copy in component
    }
    dimension = problem.partition.component_count
    information = np.zeros((dimension, dimension))
    for state in problem.robots:
        local_information = np.linalg.solve(
            state.covariance, np.eye(len(state.copies))
        )
        local_information = 0.5 * (
            local_information + local_information.T
        )
        components = [component_by_copy[copy] for copy in state.copies]
        information[np.ix_(components, components)] += local_information
    return 0.5 * (information + information.T)


def _root_consistent_logdet(
    problem: NetworkQuotientProblem,
    root_models: Iterable[tuple[VariableLayout, CanonicalEvaluation]],
) -> tuple[float, int, tuple[tuple[int, float], ...]]:
    models = tuple(root_models)
    if not models:
        raise ValueError("root determinant requires root models")
    component_information = _component_scalar_information(problem)
    component_diagonal = np.diag(component_information)
    if (
        not np.all(np.isfinite(component_diagonal))
        or np.any(component_diagonal <= 0.0)
    ):
        raise ValueError("component information must have positive finite diagonal")
    component_scaling = 1.0 / np.sqrt(component_diagonal)
    equilibrated_component = (
        component_scaling[:, None]
        * component_information
        * component_scaling[None, :]
    )
    component_factor = np.linalg.cholesky(equilibrated_component)
    component_logdet = float(
        2.0 * np.sum(np.log(np.diag(component_factor)))
        - 2.0 * np.sum(np.log(component_scaling))
    )

    def solve_component(rhs: np.ndarray) -> np.ndarray:
        scaled_rhs = component_scaling[:, None] * rhs
        solution = solve_triangular(
            component_factor.T,
            solve_triangular(
                component_factor,
                scaled_rhs,
                lower=True,
                check_finite=False,
            ),
            lower=False,
            check_finite=False,
        )
        return component_scaling[:, None] * solution

    schur_by_root = {}
    conditions = []
    for layout, evaluation in models:
        transform_dimension = layout.component_offset
        transform_hessian = evaluation.hessian[
            :transform_dimension, :transform_dimension
        ]
        coupling = evaluation.hessian[:transform_dimension, transform_dimension:]
        coupling_x = coupling[:, 0::2]
        coupling_y = coupling[:, 1::2]
        schur = (
            transform_hessian
            - coupling_x @ solve_component(coupling_x.T)
            - coupling_y @ solve_component(coupling_y.T)
        )
        schur = 0.5 * (schur + schur.T)
        diagonal = np.diag(schur)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise ValueError("root Schur model must have positive finite diagonal")
        scaling = 1.0 / np.sqrt(diagonal)
        equilibrated = scaling[:, None] * schur * scaling[None, :]
        condition = float(np.linalg.cond(equilibrated, p=2))
        if not np.isfinite(condition):
            raise ValueError("root Schur condition must be finite")
        schur_by_root[layout.root_robot] = schur
        conditions.append((layout.root_robot, condition))
    selected_root, _ = min(conditions, key=lambda item: (item[1], item[0]))
    schur_logdet = _equilibrated_spd_logdet(schur_by_root[selected_root])
    return (
        float(2.0 * component_logdet + schur_logdet),
        selected_root,
        tuple(conditions),
    )


def score_network_partition(
    problem: NetworkQuotientProblem,
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
) -> QuotientScore:
    if current_transforms is not None:
        try:
            parsed_current_transforms = _validate_current_transforms(
                problem, current_transforms
            )
        except (TypeError, ValueError):
            return _invalid_quotient_score("invalid_current_transforms")
    else:
        parsed_current_transforms = None
    original_problem = problem
    problem, parsed_current_transforms = _center_profile_problem(
        problem, parsed_current_transforms
    )
    profiles = []
    root_models = []
    root_scores = []
    root_score_values = []
    stationarity_norms = []
    first_chi2 = np.nan
    first_logdet = np.nan
    for state in problem.robots:
        root_current_transforms = (
            None
            if parsed_current_transforms is None
            else _regauge_current_transforms(
                problem, state.robot, parsed_current_transforms
            )
        )
        try:
            layout, result = profile_partition_for_root(
                problem, state.robot, root_current_transforms
            )
        except ValueError as error:
            status = (
                "partition_transform_graph_unobservable"
                if str(error) == "partition transform graph is unobservable"
                else "profile_failure"
            )
            return _invalid_quotient_score(
                status, tuple(profiles), tuple(root_scores)
            )

        evaluation = evaluate_network_canonical(problem, layout, result.x)
        if not (
            np.isfinite(evaluation.value)
            and np.isfinite(evaluation.residual_chi2)
            and np.all(np.isfinite(evaluation.gradient))
            and np.all(np.isfinite(evaluation.hessian))
        ):
            return _invalid_quotient_score(
                "nonfinite_profile", tuple(profiles), tuple(root_scores)
            )
        stationarity = _stationarity_norm(evaluation, result.x)
        profiles.append(
            QuotientProfileState(
                vector=_restore_original_profile_vector(
                    original_problem, layout, result.x
                ),
                root_robot=state.robot,
                objective=float(result.cost),
                evaluations=int(result.nfev),
            )
        )
        stationarity_norms.append(stationarity)
        root_models.append((layout, evaluation))
        if len(profiles) == 1:
            first_chi2 = evaluation.residual_chi2
    try:
        logdet, _, _ = _root_consistent_logdet(problem, root_models)
    except (ValueError, np.linalg.LinAlgError):
        status = (
            "profile_not_stationary"
            if max(stationarity_norms) > np.sqrt(np.finfo(float).eps)
            else "indefinite_exact_hessian"
        )
        return QuotientScore(
            False,
            status,
            np.nan,
            first_chi2,
            first_logdet,
            tuple(root_scores),
            np.nan,
            max(stationarity_norms),
            tuple(profiles),
        )
    first_logdet = logdet
    for layout, evaluation in root_models:
        root_score = float(
            evaluation.value
            + 0.5 * logdet
            - 0.5 * layout.dimension * np.log(2.0 * np.pi)
        )
        if not np.isfinite(logdet + root_score):
            status = (
                "profile_not_stationary"
                if max(stationarity_norms) > np.sqrt(np.finfo(float).eps)
                else "nonfinite_score"
            )
            return QuotientScore(
                False,
                status,
                np.nan,
                first_chi2,
                first_logdet,
                tuple(root_scores),
                np.nan,
                max(stationarity_norms),
                tuple(profiles),
            )
        root_scores.append((layout.root_robot, root_score))
        root_score_values.append(root_score)

    root_scores_tuple = tuple(root_scores)
    profile_tuple = tuple(profiles)
    stationarity_norm = max(stationarity_norms)
    mean_root_score = float(np.mean(root_score_values))
    root_score_spread = float(
        max(root_score_values) - min(root_score_values)
    )
    if stationarity_norm > np.sqrt(np.finfo(float).eps):
        return QuotientScore(
            False,
            "profile_not_stationary",
            mean_root_score,
            first_chi2,
            first_logdet,
            root_scores_tuple,
            root_score_spread,
            stationarity_norm,
            profile_tuple,
        )
    if root_score_spread >= 1e-8:
        return QuotientScore(
            False,
            "root_invariance_failure",
            mean_root_score,
            first_chi2,
            first_logdet,
            root_scores_tuple,
            root_score_spread,
            stationarity_norm,
            profile_tuple,
        )
    return QuotientScore(
        True,
        "ok",
        mean_root_score,
        first_chi2,
        first_logdet,
        root_scores_tuple,
        root_score_spread,
        stationarity_norm,
        profile_tuple,
    )


def canonical_partition_forest(
    partition: NetworkPartition,
) -> tuple[CanonicalEdge, ...]:
    if not isinstance(partition, NetworkPartition):
        raise ValueError("canonical forest requires a NetworkPartition")
    edges = []
    for component in partition.components:
        center = component[0]
        edges.extend(canonical_edge(center, copy) for copy in component[1:])
    return tuple(sorted(edges))


def _finite_scalar(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a scalar non-bool numeric value")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be a scalar non-bool numeric value"
        ) from error
    if array.shape != ():
        raise ValueError(f"{field} must be a scalar non-bool numeric value")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{field} must be finite")
    return scalar


def _snapshot_partition_problems(
    problem_by_partition: Mapping[
        NetworkPartition, NetworkQuotientProblem
    ],
) -> dict[NetworkPartition, NetworkQuotientProblem]:
    if not isinstance(problem_by_partition, Mapping):
        raise ValueError("partition problems must be a mapping")
    try:
        items = tuple(problem_by_partition.items())
    except (TypeError, ValueError) as error:
        raise ValueError(
            "partition problems must expose mapping items"
        ) from error
    snapshot = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("partition mapping items must be key-value pairs")
        partition, problem = item
        if not isinstance(partition, NetworkPartition):
            raise ValueError("partition mapping keys must be NetworkPartition")
        if not isinstance(problem, NetworkQuotientProblem):
            raise ValueError(
                "partition mapping values must be NetworkQuotientProblem"
            )
        if problem.partition != partition:
            raise ValueError("problem partition must match its mapping key")
        if partition in snapshot:
            raise ValueError("partition mapping keys must be unique")
        snapshot[partition] = problem
    return snapshot


def _validate_fixed_candidate_set(
    problems: Mapping[NetworkPartition, NetworkQuotientProblem],
    incumbent: NetworkPartition,
) -> None:
    if incumbent not in problems:
        raise ValueError("incumbent must be a supplied partition")
    for partition in problems:
        if not validate_fixed_cardinality_candidate(
            incumbent, partition
        ).valid:
            raise ValueError(
                "all candidates must preserve fixed cardinality and support"
            )


def _normalize_ranked_candidates(
    ranked: Iterable[tuple[NetworkPartition, QuotientScore]],
    incumbent: NetworkPartition,
    problems: Mapping[NetworkPartition, NetworkQuotientProblem] | None = None,
) -> tuple[tuple[NetworkPartition, QuotientScore], ...]:
    try:
        items = tuple(ranked)
    except TypeError as error:
        raise ValueError("ranked candidates must be an iterable") from error
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "ranked candidates must be partition-score pairs"
            )
        partition, score = item
        if not isinstance(partition, NetworkPartition):
            raise ValueError(
                "ranked candidates must contain NetworkPartition values"
            )
        if not isinstance(score, QuotientScore):
            raise ValueError(
                "ranked candidates must contain valid partition scores"
            )
        if not isinstance(score.valid, bool) or not score.valid:
            raise ValueError("ranked quotient scores must be valid")
        _finite_scalar(score.score, "ranked quotient score")
        if partition in seen:
            raise ValueError("ranked candidate partitions must be unique")
        seen.add(partition)
        if problems is not None and partition not in problems:
            raise ValueError(
                "every ranked partition must be a supplied partition"
            )
        if not validate_fixed_cardinality_candidate(
            incumbent, partition
        ).valid:
            raise ValueError(
                "ranked candidates must preserve fixed cardinality and support"
            )
        normalized.append((partition, score))
    normalized.sort(
        key=lambda item: (
            _finite_scalar(item[1].score, "ranked quotient score"),
            item[0].components,
        )
    )
    return tuple(normalized)


def _has_common_robot_cavity_data(
    first: NetworkQuotientProblem,
    second: NetworkQuotientProblem,
) -> bool:
    if len(first.robots) != len(second.robots):
        return False
    for first_state, second_state in zip(first.robots, second.robots):
        if (
            first_state.robot != second_state.robot
            or first_state.copies != second_state.copies
            or not np.array_equal(first_state.means, second_state.means)
            or not np.array_equal(
                first_state.covariance, second_state.covariance
            )
        ):
            return False
    return True


@dataclass(frozen=True)
class ExhaustiveOracleResult:
    best_partition: NetworkPartition
    best_score: float
    second_score: float
    ranked: tuple[tuple[NetworkPartition, QuotientScore], ...]
    local_descent_partition: NetworkPartition
    local_descent_score: float
    local_descent_regret: float

    def __post_init__(self) -> None:
        if not isinstance(self.best_partition, NetworkPartition):
            raise ValueError("best partition must be a NetworkPartition")
        if not isinstance(self.local_descent_partition, NetworkPartition):
            raise ValueError(
                "local-descent partition must be a NetworkPartition"
            )
        best_score = _finite_scalar(self.best_score, "best score")
        second_score = _finite_scalar(self.second_score, "second score")
        local_score = _finite_scalar(
            self.local_descent_score, "local-descent score"
        )
        regret = _finite_scalar(
            self.local_descent_regret, "local-descent regret"
        )
        if second_score < best_score:
            raise ValueError("second score must not precede the best score")
        if regret < 0.0:
            raise ValueError("local-descent regret must be nonnegative")
        ranked = _normalize_ranked_candidates(
            self.ranked, self.best_partition
        )
        if len(ranked) < 2:
            raise ValueError("ranked candidates need two valid partitions")
        if (
            self.best_partition != ranked[0][0]
            or best_score
            != _finite_scalar(ranked[0][1].score, "ranked quotient score")
            or second_score
            != _finite_scalar(ranked[1][1].score, "ranked quotient score")
        ):
            raise ValueError("best and second scores must match the ranking")
        ranked_by_partition = dict(ranked)
        if self.local_descent_partition not in ranked_by_partition:
            raise ValueError("local-descent partition must be ranked")
        if local_score != _finite_scalar(
            ranked_by_partition[self.local_descent_partition].score,
            "ranked quotient score",
        ):
            raise ValueError("local-descent score must match the ranking")
        expected_regret = max(0.0, local_score - best_score)
        if not np.isclose(regret, expected_regret, rtol=0.0, atol=1e-12):
            raise ValueError("local-descent regret is inconsistent")

        object.__setattr__(self, "best_score", best_score)
        object.__setattr__(self, "second_score", second_score)
        object.__setattr__(self, "ranked", ranked)
        object.__setattr__(self, "local_descent_score", local_score)
        object.__setattr__(self, "local_descent_regret", regret)


def _local_cut_join_descent_from_snapshot(
    problems: Mapping[NetworkPartition, NetworkQuotientProblem],
    incumbent: NetworkPartition,
    ranked: tuple[tuple[NetworkPartition, QuotientScore], ...],
) -> tuple[NetworkPartition, float]:
    score_by_partition = {
        partition: _finite_scalar(score.score, "ranked quotient score")
        for partition, score in ranked
    }
    if incumbent not in score_by_partition:
        raise ValueError("incumbent must have a valid quotient score")

    current = incumbent
    current_score = score_by_partition[current]
    while True:
        current_forest = set(canonical_partition_forest(current))
        improving = []
        for candidate, candidate_score in score_by_partition.items():
            if candidate == current or candidate_score >= current_score:
                continue
            candidate_forest = set(canonical_partition_forest(candidate))
            if (
                len(current_forest - candidate_forest) == 1
                and len(candidate_forest - current_forest) == 1
            ):
                improving.append(
                    (candidate_score, candidate.components, candidate)
                )
        if not improving:
            return current, float(current_score)
        current_score, _, current = min(improving)


def local_cut_join_descent(
    problem_by_partition: Mapping[
        NetworkPartition, NetworkQuotientProblem
    ],
    incumbent: NetworkPartition,
    ranked: Iterable[tuple[NetworkPartition, QuotientScore]],
) -> tuple[NetworkPartition, float]:
    if not isinstance(incumbent, NetworkPartition):
        raise ValueError("incumbent must be a NetworkPartition")
    problems = _snapshot_partition_problems(problem_by_partition)
    _validate_fixed_candidate_set(problems, incumbent)
    reference_problem = problems[incumbent]
    if any(
        not _has_common_robot_cavity_data(reference_problem, problem)
        for problem in problems.values()
    ):
        raise ValueError(
            "local candidates must share common robot cavity data"
        )
    normalized_ranked = _normalize_ranked_candidates(
        ranked, incumbent, problems
    )
    return _local_cut_join_descent_from_snapshot(
        problems, incumbent, normalized_ranked
    )


def exhaustive_partition_oracle(
    problem_by_partition: Mapping[
        NetworkPartition, NetworkQuotientProblem
    ],
    incumbent: NetworkPartition,
) -> ExhaustiveOracleResult:
    if not isinstance(incumbent, NetworkPartition):
        raise ValueError("incumbent must be a NetworkPartition")
    problems = _snapshot_partition_problems(problem_by_partition)
    _validate_fixed_candidate_set(problems, incumbent)
    reference_problem = problems[incumbent]
    if any(
        not _has_common_robot_cavity_data(reference_problem, problem)
        for problem in problems.values()
    ):
        raise ValueError(
            "oracle candidates must share common robot cavity data"
        )

    ranked = []
    for partition, problem in sorted(
        problems.items(), key=lambda item: item[0].components
    ):
        score = score_network_partition(problem)
        if not isinstance(score, QuotientScore) or not isinstance(
            score.valid, bool
        ):
            raise ValueError("scorer must return a QuotientScore")
        if not score.valid:
            continue
        _finite_scalar(score.score, "valid quotient score")
        ranked.append((partition, score))
    ranked = list(
        _normalize_ranked_candidates(ranked, incumbent, problems)
    )
    if len(ranked) < 2:
        raise ValueError("exhaustive oracle needs two valid partitions")

    local_partition, local_score = _local_cut_join_descent_from_snapshot(
        problems, incumbent, tuple(ranked)
    )
    best_score = _finite_scalar(ranked[0][1].score, "ranked quotient score")
    regret = max(0.0, float(local_score - best_score))
    return ExhaustiveOracleResult(
        best_partition=ranked[0][0],
        best_score=best_score,
        second_score=_finite_scalar(
            ranked[1][1].score, "ranked quotient score"
        ),
        ranked=tuple(ranked),
        local_descent_partition=local_partition,
        local_descent_score=local_score,
        local_descent_regret=regret,
    )


# The command-line protocol below deliberately owns its parsing and artifact
# schema.  It is an offline reference, not a compatibility layer for the
# evaluator, and consequently has no ground-truth input surface.
SCORE_ABLATIONS = (
    "full",
    "common-frame",
    "diagonal-covariance",
    "gauss-newton",
    "pairwise-sum",
)
CANDIDATE_CSV_COLUMNS = (
    "candidate",
    "group",
    "valid",
    "status",
    "component_count",
    "score",
    "delta_from_incumbent",
    "profile_chi2",
    "logdet",
    "root_score_spread",
    "stationarity_norm",
    "optimizer_evaluations",
    "packet_matrix_relative_error",
    "packet_link_payload_bytes",
    "wall_time_s",
)
PACKET_CSV_COLUMNS = (
    "candidate",
    "group",
    "robot",
    "variable_count",
    "serialized_bytes",
    "routed_link_bytes",
)
_MANIFEST_KEYS = {"candidates"}
_CANDIDATE_KEYS = {
    "name",
    "group",
    "remove_edges",
    "add_edges",
}
_FORBIDDEN_MANIFEST_FIELD_FRAGMENTS = ("gt", "truth", "ate", "rmse")


@dataclass(frozen=True)
class ScorerInputs:
    problem: NetworkQuotientProblem
    incumbent_edges: tuple[CanonicalEdge, ...]
    topology: tuple[tuple[int, int], ...]
    metadata_by_key: Mapping[tuple[int, int], int]
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None

    def __post_init__(self) -> None:
        if not isinstance(self.problem, NetworkQuotientProblem):
            raise ValueError("problem must be a NetworkQuotientProblem")
        try:
            incumbent_edges = tuple(
                sorted(_normalize_edge(edge) for edge in self.incumbent_edges)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("incumbent edges must be canonical edges") from error
        if len(incumbent_edges) != len(set(incumbent_edges)):
            raise ValueError("incumbent edges must be unique")
        try:
            reconstructed = partition_from_edges(
                self.problem.partition.copies, incumbent_edges
            )
        except ValueError as error:
            raise ValueError(
                "incumbent edges must reconstruct the problem partition"
            ) from error
        if reconstructed != self.problem.partition:
            raise ValueError(
                "incumbent edges must reconstruct the problem partition"
            )

        robot_ids = tuple(state.robot for state in self.problem.robots)
        try:
            _, topology = _validated_topology(
                robot_ids,
                self.topology,
                require_tree=True,
                support_name="problem",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "topology must be a connected tree over problem robots"
            ) from error

        if not isinstance(self.metadata_by_key, Mapping):
            raise ValueError("metadata must be a mapping")
        try:
            metadata_items = tuple(self.metadata_by_key.items())
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must expose mapping items") from error
        metadata = {}
        for key, semantic_class in metadata_items:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, Integral)
                    for value in key
                )
                or isinstance(semantic_class, bool)
                or not isinstance(semantic_class, Integral)
            ):
                raise ValueError(
                    "metadata keys and values must be non-bool integral values"
                )
            normalized_key = (int(key[0]), int(key[1]))
            if min(normalized_key) < 0 or normalized_key in metadata:
                raise ValueError(
                    "metadata keys must be unique nonnegative identifiers"
                )
            metadata[normalized_key] = int(semantic_class)
        expected_metadata = {
            (copy.robot, copy.object_id): copy.semantic_class
            for copy in self.problem.partition.copies
        }
        if set(metadata) != set(expected_metadata):
            raise ValueError("metadata support must match observed copies exactly")
        if metadata != expected_metadata:
            raise ValueError("metadata semantic classes must match observed copies")

        if self.current_transforms is None:
            current_transforms = None
        else:
            try:
                parsed = _validate_current_transforms(
                    self.problem, self.current_transforms
                )
            except (TypeError, ValueError) as error:
                raise ValueError("invalid current transforms") from error
            frozen = {
                robot: (yaw, np.array(translation, dtype=float, copy=True))
                for robot, (yaw, translation) in parsed.items()
            }
            for _, translation in frozen.values():
                translation.setflags(write=False)
            current_transforms = MappingProxyType(frozen)

        object.__setattr__(self, "incumbent_edges", incumbent_edges)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self, "metadata_by_key", MappingProxyType(metadata)
        )
        object.__setattr__(self, "current_transforms", current_transforms)


@dataclass(frozen=True)
class CandidateApplication:
    valid: bool
    status: str
    partition: NetworkPartition
    edges: tuple[CanonicalEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ValueError("candidate validity must be a strict bool")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("candidate status must be a nonempty string")
        if self.valid != (self.status == "ok"):
            raise ValueError("candidate valid/status semantics are inconsistent")
        if not isinstance(self.partition, NetworkPartition):
            raise ValueError("candidate partition must be a NetworkPartition")
        try:
            edges = tuple(sorted(_normalize_edge(edge) for edge in self.edges))
        except (TypeError, ValueError) as error:
            raise ValueError("candidate edges must be canonical edges") from error
        if len(edges) != len(set(edges)):
            raise ValueError("candidate edges must be unique")
        try:
            reconstructed = partition_from_edges(self.partition.copies, edges)
        except ValueError as error:
            raise ValueError(
                "candidate edges must reconstruct the candidate partition"
            ) from error
        if reconstructed != self.partition:
            raise ValueError(
                "candidate edges must reconstruct the candidate partition"
            )
        object.__setattr__(self, "edges", edges)


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if str(value).strip() != str(parsed):
        raise ValueError(f"{field} must be an integer")
    return parsed


def _strict_float(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _read_csv(path: Path | str, required: Iterable[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            missing = sorted(set(required) - set(reader.fieldnames or ()))
            raise ValueError(f"{path} is missing CSV columns: {missing}")
        return list(reader)


def read_observed_metadata(path: Path | str) -> dict[tuple[int, int], int]:
    """Read only the three scorer-visible metadata columns."""
    _reject_ground_truth_path(path, "metadata")
    rows = _read_csv(path, ("robot_id", "local_slot_id", "category"))
    result: dict[tuple[int, int], int] = {}
    order = []
    for row in rows:
        robot = _strict_int(row["robot_id"], "robot_id")
        object_id = _strict_int(row["local_slot_id"], "local_slot_id")
        category = _strict_int(row["category"], "category")
        if robot < 0 or object_id < 0:
            raise ValueError("metadata identifiers must be nonnegative")
        key = (robot, object_id)
        if key in result:
            raise ValueError("metadata object keys must be unique")
        result[key] = category
        order.append(key)
    if order != sorted(order):
        raise ValueError("metadata object order must be canonical")
    return result


def canonical_metadata_bytes(
    metadata: Mapping[tuple[int, int], int],
) -> bytes:
    payload = [
        [int(robot), int(object_id), int(category)]
        for (robot, object_id), category in sorted(metadata.items())
    ]
    return json.dumps(payload, separators=(",", ":")).encode("ascii")


def _metadata_copies(
    metadata: Mapping[tuple[int, int], int],
) -> dict[tuple[int, int], ObjectCopy]:
    return {
        key: ObjectCopy(key[0], key[1], semantic_class)
        for key, semantic_class in metadata.items()
    }


def read_association_edges(
    path: Path | str,
    metadata: Mapping[tuple[int, int], int],
    *,
    observed_keys: set[tuple[int, int]] | None = None,
) -> tuple[CanonicalEdge, ...]:
    _reject_ground_truth_path(path, "association")
    rows = _read_csv(
        path,
        (
            "receiver_robot",
            "sender_robot",
            "receiver_object",
            "sender_object",
            "weight",
        ),
    )
    copies = _metadata_copies(metadata)
    edges = set()
    for row in rows:
        weight = _strict_float(row["weight"], "association weight")
        if weight <= 0.5:
            continue
        first_key = (
            _strict_int(row["receiver_robot"], "receiver_robot"),
            _strict_int(row["receiver_object"], "receiver_object"),
        )
        second_key = (
            _strict_int(row["sender_robot"], "sender_robot"),
            _strict_int(row["sender_object"], "sender_object"),
        )
        if first_key not in copies or second_key not in copies:
            raise ValueError("active association edge has missing metadata")
        if observed_keys is not None and (
            first_key not in observed_keys or second_key not in observed_keys
        ):
            raise ValueError("active association edge refers to an unobserved copy")
        edges.add(canonical_edge(copies[first_key], copies[second_key]))
    return tuple(sorted(edges))


def _read_cavity_states(
    cavity_dir: Path | str,
    metadata: Mapping[tuple[int, int], int],
    plane_axes: str,
) -> tuple[RobotCavityState, ...]:
    _reject_ground_truth_path(cavity_dir, "cavity")
    if plane_axes not in ("xy", "xz", "yz"):
        raise ValueError("plane axes must be xy, xz, or yz")
    cavity_dir = Path(cavity_dir)
    mean_rows = _read_csv(
        cavity_dir / "final_pair_cavity_means.csv",
        ("robot", "valid", "status", "object_id", "x", "y", "z"),
    )
    axes = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}[plane_axes]
    means_by_robot: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    observed_order = []
    for row in mean_rows:
        if _strict_int(row["valid"], "mean valid") != 1:
            continue
        robot = _strict_int(row["robot"], "mean robot")
        object_id = _strict_int(row["object_id"], "mean object_id")
        key = (robot, object_id)
        if key not in metadata:
            raise ValueError("metadata is missing an observed object")
        point = (
            _strict_float(row[axes[0]], f"mean {axes[0]}"),
            _strict_float(row[axes[1]], f"mean {axes[1]}"),
        )
        means_by_robot.setdefault(robot, []).append((object_id, point))
        observed_order.append(key)
    if not observed_order or observed_order != sorted(observed_order):
        raise ValueError("valid cavity mean object order must be canonical")
    if len(set(observed_order)) != len(observed_order):
        raise ValueError("valid cavity means must be unique")

    covariance_rows = _read_csv(
        cavity_dir / "final_pair_cavity_covariance.csv",
        (
            "robot",
            "valid",
            "status",
            "row_object",
            "column_object",
            "covariance",
        ),
    )
    covariance_values: dict[tuple[int, int, int], float] = {}
    for row in covariance_rows:
        if _strict_int(row["valid"], "covariance valid") != 1:
            continue
        key = (
            _strict_int(row["robot"], "covariance robot"),
            _strict_int(row["row_object"], "row_object"),
            _strict_int(row["column_object"], "column_object"),
        )
        if key in covariance_values:
            raise ValueError("cavity covariance entries must be unique")
        covariance_values[key] = _strict_float(
            row["covariance"], "covariance"
        )
    if {key[0] for key in covariance_values} != set(means_by_robot):
        raise ValueError("cavity covariance robot support must match valid means")

    states = []
    for robot, object_points in sorted(means_by_robot.items()):
        object_ids = tuple(object_id for object_id, _ in object_points)
        expected = {
            (robot, row, column)
            for row in object_ids
            for column in object_ids
        }
        present = {key for key in covariance_values if key[0] == robot}
        if present != expected:
            raise ValueError("cavity covariance must be complete on valid means")
        covariance = np.array(
            [
                [covariance_values[(robot, row, column)] for column in object_ids]
                for row in object_ids
            ]
        )
        copies = tuple(
            ObjectCopy(robot, object_id, metadata[(robot, object_id)])
            for object_id in object_ids
        )
        states.append(
            RobotCavityState(
                robot,
                copies,
                np.asarray([point for _, point in object_points]),
                covariance,
            )
        )
    return tuple(states)


def read_topology(
    path: Path | str, robot_ids: Iterable[int]
) -> tuple[tuple[int, int], ...]:
    _reject_ground_truth_path(path, "topology")
    rows = _read_csv(path, ("first_robot", "second_robot"))
    edges = tuple(
        (
            _strict_int(row["first_robot"], "first_robot"),
            _strict_int(row["second_robot"], "second_robot"),
        )
        for row in rows
    )
    _, canonical = _validated_topology(
        tuple(sorted(robot_ids)), edges, require_tree=True, support_name="cavity"
    )
    return canonical


def _matrix_to_plane_transform(
    values: np.ndarray, plane_axes: str
) -> tuple[float, np.ndarray]:
    matrix = values.reshape((3, 4))
    axis_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane_axes]
    first, second = axis_indices
    rotation = matrix[np.ix_(axis_indices, axis_indices)]
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-6, rtol=0.0):
        raise ValueError("association transform has an invalid plane rotation")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("association transform has a reflected plane rotation")
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    translation = np.array((matrix[first, 3], matrix[second, 3]))
    return yaw, translation


def _compose_se2(
    first: tuple[float, np.ndarray], second: tuple[float, np.ndarray]
) -> tuple[float, np.ndarray]:
    first_yaw, first_translation = first
    second_yaw, second_translation = second
    rotation = rotation2(first_yaw) @ rotation2(second_yaw)
    return (
        float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        rotation2(first_yaw) @ second_translation + first_translation,
    )


def read_optional_current_transforms(
    cavity_dir: Path | str,
    topology: Iterable[tuple[int, int]],
    robot_ids: Iterable[int],
    plane_axes: str,
) -> dict[int, tuple[float, np.ndarray]] | None:
    _reject_ground_truth_path(cavity_dir, "cavity transform")
    path = Path(cavity_dir) / "final_association_transforms.csv"
    if not path.exists():
        return None
    try:
        rows = _read_csv(
            path,
            ("receiver_robot", "sender_robot", "valid", "rows", "cols")
            + tuple(f"value_{index}" for index in range(12)),
        )
        direct = {}
        for row in rows:
            if _strict_int(row["valid"], "transform valid") != 1:
                continue
            if (
                _strict_int(row["rows"], "transform rows") != 3
                or _strict_int(row["cols"], "transform cols") != 4
            ):
                continue
            receiver = _strict_int(row["receiver_robot"], "receiver_robot")
            sender = _strict_int(row["sender_robot"], "sender_robot")
            values = np.array(
                [
                    _strict_float(row[f"value_{index}"], f"value_{index}")
                    for index in range(12)
                ]
            )
            if (receiver, sender) in direct:
                return None
            direct[(receiver, sender)] = _matrix_to_plane_transform(
                values, plane_axes
            )

        robot_ids = tuple(sorted(robot_ids))
        neighbors, _ = _validated_topology(
            robot_ids,
            topology,
            require_tree=True,
            support_name="transform",
        )
        root = robot_ids[0]
        transforms = {root: (0.0, np.zeros(2))}
        queue = [root]
        for parent in queue:
            for child in neighbors[parent]:
                if child in transforms:
                    continue
                if (parent, child) in direct:
                    edge_transform = direct[(parent, child)]
                elif (child, parent) in direct:
                    yaw, translation = direct[(child, parent)]
                    inverse_rotation = rotation2(yaw).T
                    edge_transform = (
                        -yaw,
                        -inverse_rotation @ translation,
                    )
                else:
                    return None
                transforms[child] = _compose_se2(
                    transforms[parent], edge_transform
                )
                queue.append(child)
        return transforms if len(transforms) == len(robot_ids) else None
    except (OSError, ValueError, csv.Error):
        return None


def read_scorer_inputs(
    cavity_dir: Path | str,
    metadata_path: Path | str,
    association_path: Path | str,
    topology_path: Path | str,
    plane_axes: str,
) -> ScorerInputs:
    _reject_ground_truth_path(cavity_dir, "cavity")
    _reject_ground_truth_path(association_path, "association")
    metadata = read_observed_metadata(metadata_path)
    states = _read_cavity_states(cavity_dir, metadata, plane_axes)
    observed_keys = {
        (copy.robot, copy.object_id) for state in states for copy in state.copies
    }
    if any(key not in metadata for key in observed_keys):
        raise ValueError("metadata is missing an observed object")
    edges = read_association_edges(
        association_path, metadata, observed_keys=observed_keys
    )
    copies = tuple(copy for state in states for copy in state.copies)
    partition = partition_from_edges(copies, edges)
    problem = NetworkQuotientProblem(states, partition)
    topology = read_topology(topology_path, (state.robot for state in states))
    current = read_optional_current_transforms(
        cavity_dir,
        topology,
        (state.robot for state in states),
        plane_axes,
    )
    observed_metadata = {key: metadata[key] for key in sorted(observed_keys)}
    return ScorerInputs(problem, edges, topology, observed_metadata, current)


def _forbid_manifest_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("candidate manifest field names must be strings")
            lower = key.lower()
            if key not in (_MANIFEST_KEYS | _CANDIDATE_KEYS) and any(
                fragment in lower
                for fragment in _FORBIDDEN_MANIFEST_FIELD_FRAGMENTS
            ):
                raise ValueError(f"candidate manifest has forbidden field: {key}")
            _forbid_manifest_fields(child)
    elif isinstance(value, list):
        for child in value:
            _forbid_manifest_fields(child)


def _normalize_manifest_endpoint(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("candidate edge endpoints must be [robot, object_id]")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError("candidate endpoint identifiers must be integers")
    endpoint = (int(value[0]), int(value[1]))
    if min(endpoint) < 0:
        raise ValueError("candidate endpoint identifiers must be nonnegative")
    return endpoint


def _normalize_manifest_edge(value: object) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("candidate edges must contain two endpoints")
    endpoints = tuple(sorted(_normalize_manifest_endpoint(item) for item in value))
    if endpoints[0] == endpoints[1] or endpoints[0][0] == endpoints[1][0]:
        raise ValueError("candidate edge endpoints must be distinct robots")
    return [list(endpoints[0]), list(endpoints[1])]


def _normalize_candidate_record(record: object) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != _CANDIDATE_KEYS:
        raise ValueError("candidate records must match the exact schema")
    name = record["name"]
    group = record["group"]
    if not isinstance(name, str) or not name or not isinstance(group, str) or not group:
        raise ValueError("candidate name and group must be nonempty strings")
    normalized = {"name": name, "group": group}
    for field in ("remove_edges", "add_edges"):
        values = record[field]
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        normalized[field] = sorted(_normalize_manifest_edge(value) for value in values)
    return normalized


def read_candidate_manifest(path: Path | str) -> dict[str, list[dict[str, object]]]:
    _reject_ground_truth_path(path, "candidate manifest")
    with Path(path).open() as handle:
        manifest = json.load(handle)
    _forbid_manifest_fields(manifest)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("candidate manifest must match the exact schema")
    candidates = manifest["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate manifest must contain candidates")
    normalized = [_normalize_candidate_record(candidate) for candidate in candidates]
    names = [candidate["name"] for candidate in normalized]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    return {"candidates": normalized}


def _edge_from_manifest(
    value: object, metadata: Mapping[tuple[int, int], int]
) -> CanonicalEdge:
    normalized = _normalize_manifest_edge(value)
    endpoints = []
    for robot, object_id in normalized:
        key = (robot, object_id)
        if key not in metadata:
            raise ValueError("candidate edge endpoint is not observed")
        endpoints.append(ObjectCopy(robot, object_id, metadata[key]))
    return canonical_edge(*endpoints)


def apply_candidate_batch(
    incumbent: NetworkPartition,
    incumbent_edges: Iterable[CanonicalEdge],
    candidate: Mapping[str, object],
    metadata: Mapping[tuple[int, int], int],
) -> CandidateApplication:
    """Apply all edits transactionally; failures return the untouched incumbent."""
    try:
        normalized = _normalize_candidate_record(dict(candidate))
        base_edges = tuple(_normalize_edge(edge) for edge in incumbent_edges)
        if len(base_edges) != len(set(base_edges)):
            raise ValueError("incumbent edges must be unique")
    except ValueError:
        return CandidateApplication(
            False, "invalid_candidate_schema", incumbent, tuple(incumbent_edges)
        )
    try:
        removals = tuple(
            _edge_from_manifest(edge, metadata)
            for edge in normalized["remove_edges"]
        )
    except ValueError:
        return CandidateApplication(False, "nonexistent_removal", incumbent, base_edges)
    try:
        additions = tuple(
            _edge_from_manifest(edge, metadata)
            for edge in normalized["add_edges"]
        )
    except ValueError:
        return CandidateApplication(False, "conflicting_addition", incumbent, base_edges)
    if len(removals) != len(set(removals)):
        return CandidateApplication(False, "duplicate_removal", incumbent, base_edges)
    if len(additions) != len(set(additions)):
        return CandidateApplication(False, "duplicate_addition", incumbent, base_edges)
    if set(additions) & set(removals):
        return CandidateApplication(False, "conflicting_addition", incumbent, base_edges)
    base = set(base_edges)
    if any(edge not in base for edge in removals):
        return CandidateApplication(False, "nonexistent_removal", incumbent, base_edges)
    remaining = base - set(removals)
    if any(edge in remaining for edge in additions):
        return CandidateApplication(False, "conflicting_addition", incumbent, base_edges)
    edited = tuple(sorted(remaining | set(additions)))
    try:
        partition = partition_from_edges(incumbent.copies, edited)
    except ValueError:
        return CandidateApplication(False, "conflicting_addition", incumbent, base_edges)
    validation = validate_fixed_cardinality_candidate(incumbent, partition)
    if not validation.valid:
        return CandidateApplication(False, validation.status, incumbent, base_edges)
    return CandidateApplication(True, "ok", partition, edited)


def _key_edge(first: tuple[int, int], second: tuple[int, int]) -> list[list[int]]:
    return [list(endpoint) for endpoint in sorted((first, second))]


def _candidate_record(
    name: str,
    group: str,
    removals: Iterable[tuple[tuple[int, int], tuple[int, int]]],
    additions: Iterable[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[str, object]:
    return {
        "name": name,
        "group": group,
        "remove_edges": sorted(_key_edge(*edge) for edge in removals),
        "add_edges": sorted(_key_edge(*edge) for edge in additions),
    }


def _pair_mapping(
    edges: Iterable[CanonicalEdge], first_robot: int, second_robot: int
) -> dict[tuple[int, int], tuple[int, int]]:
    mapping = {}
    for edge in edges:
        by_robot = {copy.robot: (copy.robot, copy.object_id) for copy in edge}
        if set(by_robot) != {first_robot, second_robot}:
            continue
        first = by_robot[first_robot]
        second = by_robot[second_robot]
        if first in mapping and mapping[first] != second:
            raise ValueError("association is not unit-capacity on the edit pair")
        if second in mapping.values() and mapping.get(first) != second:
            raise ValueError("association is not unit-capacity on the edit pair")
        mapping[first] = second
    return mapping


def _mapping_edit(
    current: Mapping[tuple[int, int], tuple[int, int]],
    proposed: Mapping[tuple[int, int], tuple[int, int]],
) -> tuple[
    tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    tuple[tuple[tuple[int, int], tuple[int, int]], ...],
]:
    old = {tuple(sorted((first, second))) for first, second in current.items()}
    new = {tuple(sorted((first, second))) for first, second in proposed.items()}
    return tuple(sorted(old - new)), tuple(sorted(new - old))


def _iter_permutation_choices(
    by_class: tuple[
        tuple[list[tuple[int, int]], list[tuple[int, int]]], ...
    ]
) -> Iterable[tuple[tuple[tuple[int, int], ...], ...]]:
    def recurse(
        index: int,
        prefix: tuple[tuple[tuple[int, int], ...], ...],
    ) -> Iterable[tuple[tuple[tuple[int, int], ...], ...]]:
        if index == len(by_class):
            yield prefix
            return
        for permutation in itertools.permutations(by_class[index][1]):
            yield from recurse(index + 1, prefix + (permutation,))

    return recurse(0, ())


def build_candidate_manifest(
    mode: str,
    metadata_path: Path | str,
    association_path: Path | str,
    edit_robot_pair: tuple[int, int],
    *,
    fixed_copy: str | None = None,
    semantic_classes: Iterable[int] | None = None,
    frozen_alternative_association: Path | str | None = None,
) -> dict[str, list[dict[str, object]]]:
    _reject_ground_truth_path(association_path, "association")
    if frozen_alternative_association is not None:
        _reject_ground_truth_path(
            frozen_alternative_association,
            "frozen alternative association",
        )
    if mode not in (
        "single-column",
        "cyclic-by-class",
        "symmetric-difference-permutations",
    ):
        raise ValueError("unknown candidate manifest builder")
    if len(edit_robot_pair) != 2 or edit_robot_pair[0] == edit_robot_pair[1]:
        raise ValueError("edit robot pair must contain two distinct robots")
    first_robot, second_robot = map(int, edit_robot_pair)
    metadata = read_observed_metadata(metadata_path)
    edges = read_association_edges(association_path, metadata)
    current = _pair_mapping(edges, first_robot, second_robot)
    candidates = []

    if mode == "single-column":
        if fixed_copy is None:
            raise ValueError("single-column requires --fixed-copy")
        pieces = fixed_copy.split(":")
        if len(pieces) != 2:
            raise ValueError("fixed copy must be ROBOT:OBJECT")
        fixed = (_strict_int(pieces[0], "fixed robot"), _strict_int(pieces[1], "fixed object"))
        if fixed[0] not in (first_robot, second_robot) or fixed not in metadata:
            raise ValueError("fixed copy must be observed on the edit pair")
        group = f"fixed_{fixed[0]}_{fixed[1]}"
        candidates.append(_candidate_record("incumbent", group, (), ()))
        other_robot = second_robot if fixed[0] == first_robot else first_robot
        category = metadata[fixed]
        eligible = [
            key for key, value in sorted(metadata.items())
            if key[0] == other_robot and value == category
        ]
        inverse = {value: key for key, value in current.items()}
        for other in eligible:
            desired_first, desired_second = (
                (fixed, other)
                if fixed[0] == first_robot
                else (other, fixed)
            )
            old_second = current.get(desired_first)
            old_first = inverse.get(desired_second)
            if old_second == desired_second:
                continue
            if old_second is None and old_first is None:
                continue
            proposed = dict(current)
            if old_second is not None:
                del proposed[desired_first]
            if old_first is not None:
                del proposed[old_first]
            proposed[desired_first] = desired_second
            if old_second is not None and old_first is not None:
                proposed[old_first] = old_second
            removals, additions = _mapping_edit(current, proposed)
            if removals or additions:
                candidates.append(
                    _candidate_record(
                        f"pair_{other_robot}_{other[1]}", group, removals, additions
                    )
                )

    elif mode == "cyclic-by-class":
        if semantic_classes is None:
            raise ValueError("cyclic-by-class requires semantic classes")
        for category in sorted(set(map(int, semantic_classes))):
            group = f"category_{category}"
            candidates.append(_candidate_record(f"incumbent_{group}", group, (), ()))
            active = sorted(
                (first, second)
                for first, second in current.items()
                if metadata[first] == category and metadata[second] == category
            )
            if len(active) < 2:
                continue
            firsts = [item[0] for item in active]
            seconds = [item[1] for item in active]
            for shift in range(1, len(active)):
                proposed = dict(current)
                for index, first in enumerate(firsts):
                    proposed[first] = seconds[(index + shift) % len(seconds)]
                removals, additions = _mapping_edit(current, proposed)
                candidates.append(
                    _candidate_record(
                        f"cyclic_{category}_{shift}", group, removals, additions
                    )
                )

    else:
        if frozen_alternative_association is None:
            raise ValueError(
                "symmetric-difference-permutations requires a frozen alternative"
            )
        alternative_edges = read_association_edges(
            frozen_alternative_association, metadata
        )
        alternative = _pair_mapping(
            alternative_edges, first_robot, second_robot
        )
        old_edges = {tuple(sorted(item)) for item in current.items()}
        alternative_set = {tuple(sorted(item)) for item in alternative.items()}
        touched = {endpoint for edge in old_edges ^ alternative_set for endpoint in edge}
        by_class: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        for category in sorted({metadata[key] for key in touched}):
            firsts = sorted(key for key in touched if key[0] == first_robot and metadata[key] == category)
            seconds = sorted(key for key in touched if key[0] == second_robot and metadata[key] == category)
            if len(firsts) != len(seconds):
                raise ValueError("edit zone is not a same-class bijection")
            by_class.append((firsts, seconds))
        group = f"edit_zone_{len(touched)}"
        candidates.append(_candidate_record("incumbent", group, (), ()))
        for permutation_choice in _iter_permutation_choices(tuple(by_class)):
            proposed = dict(current)
            for (firsts, _), permuted in zip(by_class, permutation_choice):
                for first, second in zip(firsts, permuted):
                    proposed[first] = second
            removals, additions = _mapping_edit(current, proposed)
            if not removals and not additions:
                continue
            candidates.append(
                _candidate_record(
                    f"permutation_{len(candidates)}", group, removals, additions
                )
            )

    candidates.sort(
        key=lambda candidate: (
            candidate["group"],
            candidate["remove_edges"],
            candidate["add_edges"],
            candidate["name"],
        )
    )
    # Make the incumbent deterministic and first within every comparison group.
    candidates.sort(
        key=lambda candidate: (
            candidate["group"],
            bool(candidate["remove_edges"] or candidate["add_edges"]),
            candidate["remove_edges"],
            candidate["add_edges"],
        )
    )
    names = set()
    for index, candidate in enumerate(candidates):
        name = str(candidate["name"])
        if name in names:
            candidate["name"] = f"{name}_{index}"
        names.add(str(candidate["name"]))
    return {"candidates": candidates}


def apply_independent_robot_gauge(
    problem: NetworkQuotientProblem, seed: int
) -> tuple[NetworkQuotientProblem, tuple[dict[str, object], ...]]:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("independent gauge seed must be an integer")
    generator = np.random.default_rng(int(seed))
    states = []
    records = []
    for state in sorted(problem.robots, key=lambda item: item.robot):
        yaw = float(generator.uniform(-np.pi, np.pi))
        translation = generator.normal(0.0, 20.0, size=2)
        means = state.means @ rotation2(yaw).T + translation
        states.append(
            RobotCavityState(
                state.robot, state.copies, means, state.covariance
            )
        )
        records.append(
            {
                "robot": state.robot,
                "yaw": yaw,
                "translation": [float(value) for value in translation],
            }
        )
    return NetworkQuotientProblem(tuple(states), problem.partition), tuple(records)


def _score_with_hessian_mode(
    problem: NetworkQuotientProblem,
    *,
    gauss_newton: bool,
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
) -> QuotientScore:
    parsed = None
    if current_transforms is not None:
        try:
            parsed = _validate_current_transforms(problem, current_transforms)
        except ValueError:
            return _invalid_quotient_score("invalid_current_transforms")
    original_problem = problem
    problem, parsed = _center_profile_problem(problem, parsed)
    profiles = []
    root_scores = []
    chi2 = np.nan
    first_logdet = np.nan
    stationarity_values = []
    for state in problem.robots:
        current = None if parsed is None else _regauge_current_transforms(
            problem, state.robot, parsed
        )
        try:
            layout, result = profile_partition_for_root(
                problem, state.robot, current
            )
        except ValueError:
            return _invalid_quotient_score("profile_failure", tuple(profiles), tuple(root_scores))
        evaluation = evaluate_network_canonical(problem, layout, result.x)
        residual, jacobian = profile_residual_and_jacobian(problem, layout, result.x)
        hessian = jacobian.T @ jacobian if gauss_newton else evaluation.hessian
        stationarity = _stationarity_norm(evaluation, result.x)
        stationarity_values.append(stationarity)
        profiles.append(
            QuotientProfileState(
                _restore_original_profile_vector(
                    original_problem, layout, result.x
                ),
                state.robot,
                float(result.cost),
                int(result.nfev),
            )
        )
        try:
            logdet = _equilibrated_spd_logdet(hessian)
        except (ValueError, np.linalg.LinAlgError):
            return _invalid_quotient_score(
                "indefinite_hessian", tuple(profiles), tuple(root_scores)
            )
        score = float(
            evaluation.value
            + 0.5 * logdet
            - 0.5 * layout.dimension * np.log(2.0 * np.pi)
        )
        root_scores.append((state.robot, score))
        if len(root_scores) == 1:
            chi2 = float(residual @ residual)
            first_logdet = logdet
    values = [value for _, value in root_scores]
    spread = float(max(values) - min(values))
    stationarity = max(stationarity_values)
    valid = stationarity <= np.sqrt(np.finfo(float).eps)
    return QuotientScore(
        valid,
        "ok" if valid else "profile_not_stationary",
        float(np.mean(values)),
        chi2,
        first_logdet,
        tuple(root_scores),
        spread,
        stationarity,
        tuple(profiles),
    )


def _score_common_frame(problem: NetworkQuotientProblem) -> QuotientScore:
    component_by_copy = {
        copy: index
        for index, component in enumerate(problem.partition.components)
        for copy in component
    }
    dimension = 2 * problem.partition.component_count
    residual_parts = []
    jacobian_parts = []
    normalizer = 0.0
    for state in problem.robots:
        jacobian = np.zeros((2 * len(state.copies), dimension))
        for row, copy in enumerate(state.copies):
            offset = 2 * component_by_copy[copy]
            jacobian[2 * row : 2 * row + 2, offset : offset + 2] = -np.eye(2)
        lower = np.linalg.cholesky(state.covariance)
        residual_parts.append(
            solve_triangular(lower, state.means, lower=True).reshape(-1)
        )
        jacobian_parts.append(
            solve_triangular(
                lower,
                jacobian.reshape((len(state.copies), 2 * dimension)),
                lower=True,
            ).reshape((2 * len(state.copies), dimension))
        )
        normalizer += float(
            np.linalg.slogdet(state.covariance)[1]
            + len(state.copies) * np.log(2.0 * np.pi)
        )
    initial = np.concatenate(residual_parts)
    jacobian = np.vstack(jacobian_parts)
    vector, _, _, _ = np.linalg.lstsq(jacobian, -initial, rcond=None)
    residual = initial + jacobian @ vector
    hessian = jacobian.T @ jacobian
    try:
        logdet = _equilibrated_spd_logdet(hessian)
    except (ValueError, np.linalg.LinAlgError):
        return _invalid_quotient_score("indefinite_hessian")
    chi2 = float(residual @ residual)
    value = 0.5 * chi2 + normalizer
    score = float(value + 0.5 * logdet - 0.5 * dimension * np.log(2.0 * np.pi))
    stationarity = float(np.linalg.norm(jacobian.T @ residual) / (1.0 + abs(value)))
    root = problem.robots[0].robot
    profile = QuotientProfileState(vector, root, 0.5 * chi2, 1)
    return QuotientScore(
        True, "ok", score, chi2, logdet, ((root, score),), 0.0,
        stationarity, (profile,),
    )


def _restrict_problem_to_pair(
    problem: NetworkQuotientProblem, first_robot: int, second_robot: int
) -> NetworkQuotientProblem | None:
    pair_components = [
        component
        for component in problem.partition.components
        if {copy.robot for copy in component}.issuperset({first_robot, second_robot})
    ]
    if not pair_components:
        return None
    selected_by_robot = {
        robot: [
            next(copy for copy in component if copy.robot == robot)
            for component in pair_components
        ]
        for robot in (first_robot, second_robot)
    }
    states = []
    for state in problem.robots:
        if state.robot not in selected_by_robot:
            continue
        index = {copy: row for row, copy in enumerate(state.copies)}
        rows = [index[copy] for copy in selected_by_robot[state.robot]]
        states.append(
            RobotCavityState(
                state.robot,
                tuple(selected_by_robot[state.robot]),
                state.means[rows],
                state.covariance[np.ix_(rows, rows)],
            )
        )
    partition = NetworkPartition.build(
        tuple(
            tuple(copy for copy in component if copy.robot in (first_robot, second_robot))
            for component in pair_components
        )
    )
    return NetworkQuotientProblem(tuple(states), partition)


def _score_pairwise_sum(problem: NetworkQuotientProblem) -> QuotientScore:
    scores = []
    profiles = []
    roots = []
    represented_pairs = sorted(
        {
            tuple(sorted((first.robot, second.robot)))
            for component in problem.partition.components
            for first, second in itertools.combinations(component, 2)
        }
    )
    for first_robot, second_robot in represented_pairs:
        pair = _restrict_problem_to_pair(problem, first_robot, second_robot)
        if pair is None:
            continue
        score = score_network_partition(pair)
        if not score.valid:
            return _invalid_quotient_score(
                f"pairwise_pair_invalid_{score.status}"
            )
        scores.append(score)
        profiles.extend(score.profiles)
        roots.extend(score.root_scores)
    if not scores:
        return _invalid_quotient_score("pairwise_graph_unobservable")
    total_score = float(sum(score.score for score in scores))
    return QuotientScore(
        True,
        "ok",
        total_score,
        float(sum(score.profile_chi2 for score in scores)),
        float(sum(score.logdet for score in scores)),
        tuple(roots),
        float(max(score.root_score_spread for score in scores)),
        float(max(score.stationarity_norm for score in scores)),
        tuple(profiles),
    )


def score_partition_ablation(
    problem: NetworkQuotientProblem,
    label: str,
    current_transforms: Mapping[int, tuple[float, np.ndarray]] | None = None,
) -> QuotientScore:
    if label not in SCORE_ABLATIONS:
        raise ValueError("unknown score ablation")
    if label == "full":
        return score_network_partition(problem, current_transforms)
    if label == "common-frame":
        return _score_common_frame(problem)
    if label == "diagonal-covariance":
        diagonal = tuple(
            RobotCavityState(
                state.robot,
                state.copies,
                state.means,
                np.diag(np.diag(state.covariance)),
            )
            for state in problem.robots
        )
        return score_network_partition(
            NetworkQuotientProblem(diagonal, problem.partition), current_transforms
        )
    if label == "gauss-newton":
        return _score_with_hessian_mode(
            problem, gauss_newton=True, current_transforms=current_transforms
        )
    return _score_pairwise_sum(problem)


def _packet_diagnostics(
    problem: NetworkQuotientProblem,
    topology: tuple[tuple[int, int], ...],
    score: QuotientScore,
) -> tuple[
    float,
    int,
    tuple[CanonicalPacket, ...],
    Mapping[int, int],
]:
    if not score.valid or not score.profiles:
        return np.nan, 0, (), MappingProxyType({})
    profile = score.profiles[0]
    layout = VariableLayout.build(problem, profile.root_robot)
    packets = build_canonical_packets(problem, layout, profile.vector)
    reduced, routing_trace = reduce_packets_on_tree_with_trace(
        packets, topology, profile.root_robot
    )
    direct = evaluate_network_canonical(problem, layout, profile.vector)
    numerator = np.sqrt(
        np.linalg.norm(reduced.hessian - direct.hessian, ord="fro") ** 2
        + np.linalg.norm(reduced.gradient - direct.gradient) ** 2
        + (reduced.value - direct.value) ** 2
    )
    denominator = max(
        1.0,
        np.sqrt(
            np.linalg.norm(direct.hessian, ord="fro") ** 2
            + np.linalg.norm(direct.gradient) ** 2
            + direct.value ** 2
        ),
    )
    return (
        float(numerator / denominator),
        reduced.link_payload_bytes,
        packets,
        routing_trace,
    )


def _float_csv(value: float) -> str:
    return format(float(value), ".17g")


def canonical_move_id(
    remove_edges: Iterable[object], add_edges: Iterable[object]
) -> str:
    try:
        removals = sorted(_normalize_manifest_edge(edge) for edge in remove_edges)
        additions = sorted(_normalize_manifest_edge(edge) for edge in add_edges)
    except TypeError as error:
        raise ValueError("move edge batches must be iterable") from error
    return json.dumps(
        {"add_edges": additions, "remove_edges": removals},
        sort_keys=True,
        separators=(",", ":"),
    )


def consensus_select_move(
    candidates: Iterable[Mapping[str, object]],
    topology: Iterable[tuple[int, int]],
    robot_ids: Iterable[int],
) -> dict[str, object]:
    records = tuple(candidates)
    supplied_robots = tuple(robot_ids)
    if any(
        isinstance(robot, bool)
        or not isinstance(robot, Integral)
        or robot < 0
        for robot in supplied_robots
    ) or len(set(supplied_robots)) != len(supplied_robots):
        raise ValueError("consensus robot IDs must be unique nonnegative integers")
    robots = tuple(sorted(int(robot) for robot in supplied_robots))
    if not records or not robots:
        raise ValueError("consensus selection requires candidates and robots")
    moves: dict[str, dict[str, object]] = {}
    seen_candidates = set()
    for record in records:
        candidate = record.get("candidate")
        if (
            not isinstance(candidate, str)
            or not candidate
            or candidate in seen_candidates
        ):
            raise ValueError("consensus candidate IDs must be unique strings")
        seen_candidates.add(candidate)
        valid_flag = record.get("valid")
        if valid_flag not in (False, True, 0, 1):
            raise ValueError("consensus candidate validity must be Boolean")
        if not bool(valid_flag):
            continue
        score = _finite_scalar(record.get("score"), "candidate score")
        move_id = canonical_move_id(
            record.get("remove_edges", ()), record.get("add_edges", ())
        )
        if move_id not in moves:
            moves[move_id] = {"score": score, "records": [record]}
        else:
            if score != moves[move_id]["score"]:
                raise ValueError(
                    "duplicate semantic moves must have a consistent score"
                )
            moves[move_id]["records"].append(record)
    if not moves:
        raise ValueError("consensus selection requires a valid candidate")
    proposals = {
        robot: tuple(
            (float(move["score"]), move_id)
            for move_id, move in sorted(moves.items())
        )
        for robot in robots
    }
    consensus = run_min_consensus(proposals, topology)
    selected_score, selected_move_id = consensus.selected
    selected_records = moves[selected_move_id]["records"]
    selected_record = min(
        selected_records,
        key=lambda record: (
            str(record.get("group", "")), str(record["candidate"])
        ),
    )
    return {
        "candidate": selected_record["candidate"],
        "canonical_move_id": selected_move_id,
        "group": selected_record.get("group"),
        "score": selected_score,
        "remove_edges": selected_record.get("remove_edges", []),
        "add_edges": selected_record.get("add_edges", []),
        "selected_by_robot": dict(consensus.selected_by_robot),
        "rounds": consensus.rounds,
        "payload_bytes": consensus.payload_bytes,
    }


def _reject_ground_truth_path(path: Path | str, role: str) -> None:
    normalized = Path(path).expanduser().resolve(strict=False)
    tokens = {
        token
        for component in normalized.parts
        for token in re.findall(r"[a-z0-9]+", component.lower())
    }
    metric_or_truth = {"gt", "truth", "groundtruth", "ate", "rmse"}
    concatenated_suffixes = (
        "pose",
        "poses",
        "association",
        "associations",
        "edge",
        "edges",
        "transform",
        "transforms",
    )
    suspicious = any(
        "truth" in token
        or token in metric_or_truth
        or any(
            token == prefix + suffix
            for prefix in ("gt", "ate", "rmse")
            for suffix in concatenated_suffixes
        )
        for token in tokens
    )
    if suspicious:
        raise ValueError(f"scorer refuses ground-truth {role} paths")


def score_candidate_manifest(
    inputs: ScorerInputs,
    manifest: Mapping[str, object],
    output_dir: Path | str,
    *,
    plane_axes: str,
    independent_gauge_seed: int | None,
    score_ablation: str,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(manifest["candidates"])
    base_problem = inputs.problem
    gauge_records: tuple[dict[str, object], ...] = ()
    current_transforms = inputs.current_transforms
    if independent_gauge_seed is not None:
        base_problem, gauge_records = apply_independent_robot_gauge(
            base_problem, independent_gauge_seed
        )
        current_transforms = None

    candidate_rows = []
    packet_rows = []
    scored_records = []
    candidate_root_scores = {}
    incumbent_score = None
    total_start = time.perf_counter()
    for candidate in candidates:
        start = time.perf_counter()
        applied = apply_candidate_batch(
            base_problem.partition,
            inputs.incumbent_edges,
            candidate,
            inputs.metadata_by_key,
        )
        score = _invalid_quotient_score(applied.status)
        full_score = score
        packet_error = np.nan
        packet_bytes = 0
        packets: tuple[CanonicalPacket, ...] = ()
        routing_trace: Mapping[int, int] = MappingProxyType({})
        if applied.valid:
            problem = NetworkQuotientProblem(
                base_problem.robots, applied.partition
            )
            score = score_partition_ablation(
                problem, score_ablation, current_transforms
            )
            full_score = (
                score
                if score_ablation == "full"
                else score_network_partition(problem, current_transforms)
            )
            (
                packet_error,
                packet_bytes,
                packets,
                routing_trace,
            ) = _packet_diagnostics(problem, inputs.topology, full_score)
        is_incumbent = not candidate["remove_edges"] and not candidate["add_edges"]
        if is_incumbent and score.valid and incumbent_score is None:
            incumbent_score = score.score
        elapsed = time.perf_counter() - start
        optimizer_evaluations = sum(profile.evaluations for profile in score.profiles)
        record = {
            "candidate": candidate["name"],
            "group": candidate["group"],
            "valid": int(bool(applied.valid and score.valid)),
            "status": score.status if applied.valid else applied.status,
            "component_count": applied.partition.component_count,
            "score": score.score,
            "delta_from_incumbent": np.nan,
            "profile_chi2": score.profile_chi2,
            "logdet": score.logdet,
            "root_score_spread": score.root_score_spread,
            "stationarity_norm": score.stationarity_norm,
            "optimizer_evaluations": optimizer_evaluations,
            "packet_matrix_relative_error": packet_error,
            "packet_link_payload_bytes": packet_bytes,
            "wall_time_s": elapsed,
            "remove_edges": candidate["remove_edges"],
            "add_edges": candidate["add_edges"],
        }
        candidate_rows.append(record)
        scored_records.append((record, score))
        candidate_root_scores[str(candidate["name"])] = [
            [int(root), float(root_score)]
            for root, root_score in score.root_scores
        ]
        if packets:
            for packet in packets:
                packet_rows.append(
                    {
                        "candidate": candidate["name"],
                        "group": candidate["group"],
                        "robot": packet.robot,
                        "variable_count": len(packet.variable_ids),
                        "serialized_bytes": packet.serialized_bytes,
                        "routed_link_bytes": routing_trace[packet.robot],
                    }
                )
        else:
            for state in base_problem.robots:
                packet_rows.append(
                    {
                        "candidate": candidate["name"],
                        "group": candidate["group"],
                        "robot": state.robot,
                        "variable_count": 0,
                        "serialized_bytes": 0,
                        "routed_link_bytes": 0,
                    }
                )
    if incumbent_score is None:
        raise ValueError("manifest requires a valid unchanged incumbent candidate")
    for record, score in scored_records:
        if score.valid:
            record["delta_from_incumbent"] = score.score - incumbent_score

    with (output_dir / "network_quotient_candidates.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_COLUMNS)
        writer.writeheader()
        for record in candidate_rows:
            writer.writerow(
                {
                    field: (
                        _float_csv(record[field])
                        if isinstance(record[field], (float, np.floating))
                        else record[field]
                    )
                    for field in CANDIDATE_CSV_COLUMNS
                }
            )
    with (output_dir / "network_quotient_packets.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKET_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(packet_rows)

    valid_packet_errors = [
        row["packet_matrix_relative_error"]
        for row in candidate_rows
        if np.isfinite(row["packet_matrix_relative_error"])
    ]
    robot_ids = tuple(state.robot for state in base_problem.robots)
    network_selection = consensus_select_move(
        candidate_rows, inputs.topology, robot_ids
    )
    groups = sorted({str(row["group"]) for row in candidate_rows})
    group_selections = {
        group: consensus_select_move(
            tuple(row for row in candidate_rows if row["group"] == group),
            inputs.topology,
            robot_ids,
        )
        for group in groups
    }
    frozen_partition = inputs.problem.partition
    observed_copies = [
        [copy.robot, copy.object_id, copy.semantic_class]
        for copy in frozen_partition.copies
    ]
    incumbent_edges = [
        [[copy.robot, copy.object_id] for copy in edge]
        for edge in inputs.incumbent_edges
    ]
    incumbent_components = [
        [[copy.robot, copy.object_id] for copy in component]
        for component in frozen_partition.components
    ]
    topology_edges = [list(edge) for edge in inputs.topology]
    summary = {
        "candidate_count": len(candidate_rows),
        "candidate_order": [row["candidate"] for row in candidate_rows],
        "candidate_root_scores": candidate_root_scores,
        "cardinality_change_candidates_scored": 0,
        "configured_anchors": 0,
        "configured_damping_parameters": 0,
        "configured_null_constants": 0,
        "current_lower_start_used": bool(current_transforms is not None),
        "ground_truth_inputs": 0,
        "observed_copies": observed_copies,
        "incumbent_edges": incumbent_edges,
        "incumbent_components": incumbent_components,
        "incumbent_component_count": frozen_partition.component_count,
        "topology_edges": topology_edges,
        "independent_gauge_seed": independent_gauge_seed,
        "independent_gauge_transforms": list(gauge_records),
        "max_packet_matrix_relative_error": (
            max(valid_packet_errors) if valid_packet_errors else None
        ),
        "network_selection": network_selection,
        "packet_candidate_count": len(valid_packet_errors),
        "packet_link_payload_bytes": int(
            sum(row["packet_link_payload_bytes"] for row in candidate_rows)
        ),
        "plane_axes": plane_axes,
        "score_ablation": score_ablation,
        "score_gate_eligible": score_ablation == "full",
        "scored_candidate_count": sum(row["valid"] for row in candidate_rows),
        "group_selections": group_selections,
        "wall_time_s": time.perf_counter() - total_start,
    }
    with (output_dir / "network_quotient_summary.json").open("w") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return summary


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score fixed-cardinality network association partitions"
    )
    parser.add_argument("--cavity-dir", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plane-axes", choices=("xy", "xz", "yz"), default="xy")
    parser.add_argument("--independent-gauge-seed", type=int)
    parser.add_argument("--score-ablation", choices=SCORE_ABLATIONS, default="full")
    parser.add_argument(
        "--build-candidate-manifest",
        choices=(
            "single-column",
            "cyclic-by-class",
            "symmetric-difference-permutations",
        ),
    )
    parser.add_argument("--edit-robot-pair", nargs=2, type=int)
    parser.add_argument("--fixed-copy")
    parser.add_argument("--semantic-classes", nargs="+", type=int)
    parser.add_argument("--frozen-alternative-association", type=Path)
    return parser


def run_cli(argv: Iterable[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(
        None if argv is None else tuple(argv)
    )
    _reject_ground_truth_path(arguments.association, "association")
    if arguments.build_candidate_manifest is not None:
        if arguments.edit_robot_pair is None:
            raise ValueError("manifest construction requires --edit-robot-pair")
        if arguments.frozen_alternative_association is not None:
            _reject_ground_truth_path(
                arguments.frozen_alternative_association,
                "frozen alternative association",
            )
        manifest = build_candidate_manifest(
            arguments.build_candidate_manifest,
            arguments.metadata,
            arguments.association,
            tuple(arguments.edit_robot_pair),
            fixed_copy=arguments.fixed_copy,
            semantic_classes=arguments.semantic_classes,
            frozen_alternative_association=arguments.frozen_alternative_association,
        )
        arguments.candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        with arguments.candidate_manifest.open("w") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
        return 0
    if any(
        value is None
        for value in (arguments.cavity_dir, arguments.topology, arguments.output_dir)
    ):
        raise ValueError(
            "scoring requires --cavity-dir, --topology, and --output-dir"
        )
    for path, role in (
        (arguments.cavity_dir, "cavity"),
        (arguments.metadata, "metadata"),
        (arguments.association, "association"),
        (arguments.candidate_manifest, "candidate manifest"),
        (arguments.topology, "topology"),
    ):
        _reject_ground_truth_path(path, role)
    inputs = read_scorer_inputs(
        arguments.cavity_dir,
        arguments.metadata,
        arguments.association,
        arguments.topology,
        arguments.plane_axes,
    )
    manifest = read_candidate_manifest(arguments.candidate_manifest)
    score_candidate_manifest(
        inputs,
        manifest,
        arguments.output_dir,
        plane_axes=arguments.plane_axes,
        independent_gauge_seed=arguments.independent_gauge_seed,
        score_ablation=arguments.score_ablation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
