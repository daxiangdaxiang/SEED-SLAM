"""Ground-truth-free distributed fixed-yaw Pose2-Point2 initialization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from benchmarks.slideslam.materialize import CanonicalGraph
from benchmarks.slideslam.schema import (
    ObjectVertex,
    OdometryFactor,
    PoseObjectFactor,
    PoseVertex,
)
from benchmarks.slideslam.se3 import SE3

from .evaluate import lifted_objective_breakdown


class Point2InitializationError(ValueError):
    """Raised when the fixed-yaw Point2 subproblem is malformed."""


@dataclass(frozen=True, eq=False)
class ObjectSufficientStatistic:
    weighted_position: np.ndarray
    information: float


@dataclass(frozen=True)
class InitializationCommunication:
    logical_bytes: int
    payload_blocks: int
    messages: int

    def __add__(self, other: "InitializationCommunication") -> "InitializationCommunication":
        return InitializationCommunication(
            logical_bytes=self.logical_bytes + other.logical_bytes,
            payload_blocks=self.payload_blocks + other.payload_blocks,
            messages=self.messages + other.messages,
        )


@dataclass(frozen=True)
class Point2InitializationRound:
    iteration: int
    cost: float
    mean_factor_cost: float
    max_object_delta_m: float
    solver_wall_time_s: float
    round_wall_time_s: float
    communication: InitializationCommunication


@dataclass(frozen=True)
class Point2InitializationResult:
    state: Mapping[int, SE3]
    initial_cost: float
    rounds: tuple[Point2InitializationRound, ...]
    communication: InitializationCommunication
    total_wall_time_s: float


def fixed_yaw_point2_cost(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> float:
    """Evaluate the frozen public objective without constructing its gradient."""

    return lifted_objective_breakdown(graph, state).total_cost


def _rotation2(pose: SE3) -> np.ndarray:
    yaw = math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _rotation2_derivative(pose: SE3) -> np.ndarray:
    yaw = math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array([[-sine, -cosine], [cosine, -sine]], dtype=np.float64)


def planar_odometry_linearization(
    pose_i: SE3,
    pose_j: SE3,
    measurement: SE3,
    *,
    kappa: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact planar chordal residual and analytic Pose2 Jacobians."""

    root_kappa = math.sqrt(float(kappa))
    root_tau = math.sqrt(float(tau))
    rotation_i = _rotation2(pose_i)
    rotation_j = _rotation2(pose_j)
    derivative_i = _rotation2_derivative(pose_i)
    derivative_j = _rotation2_derivative(pose_j)
    measurement_rotation = _rotation2(measurement)
    measurement_translation = measurement.translation[:2]
    rotation_residual = rotation_j - rotation_i @ measurement_rotation
    translation_residual = (
        pose_j.translation[:2]
        - pose_i.translation[:2]
        - rotation_i @ measurement_translation
    )
    residual = np.concatenate(
        [
            root_kappa * rotation_residual.reshape(4),
            root_tau * translation_residual,
        ]
    )
    jacobian_i = np.zeros((6, 3), dtype=np.float64)
    jacobian_j = np.zeros((6, 3), dtype=np.float64)
    jacobian_i[:4, 2] = (
        -root_kappa * derivative_i @ measurement_rotation
    ).reshape(4)
    jacobian_j[:4, 2] = (root_kappa * derivative_j).reshape(4)
    jacobian_i[4:, :2] = -root_tau * np.eye(2)
    jacobian_j[4:, :2] = root_tau * np.eye(2)
    jacobian_i[4:, 2] = -root_tau * derivative_i @ measurement_translation
    return residual, jacobian_i, jacobian_j


def planar_observation_linearization(
    pose: SE3,
    object_position: np.ndarray,
    measurement: SE3,
    *,
    kappa: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Point2-embedded chordal residual and its Pose2 Jacobian."""

    point = np.asarray(object_position, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise Point2InitializationError("object position must be a finite Point2")
    root_kappa = math.sqrt(float(kappa))
    root_tau = math.sqrt(float(tau))
    rotation = _rotation2(pose)
    derivative = _rotation2_derivative(pose)
    measurement_rotation = _rotation2(measurement)
    measurement_translation = measurement.translation[:2]
    rotation_residual = np.eye(2) - rotation @ measurement_rotation
    translation_residual = (
        point - pose.translation[:2] - rotation @ measurement_translation
    )
    residual = np.concatenate(
        [
            root_kappa * rotation_residual.reshape(4),
            root_tau * translation_residual,
        ]
    )
    jacobian = np.zeros((6, 3), dtype=np.float64)
    jacobian[:4, 2] = (
        -root_kappa * derivative @ measurement_rotation
    ).reshape(4)
    jacobian[4:, :2] = -root_tau * np.eye(2)
    jacobian[4:, 2] = -root_tau * derivative @ measurement_translation
    return residual, jacobian


def _planar_pose(rotation_source: SE3, translation_xy: np.ndarray) -> SE3:
    yaw = math.atan2(
        float(rotation_source.rotation[1, 0]),
        float(rotation_source.rotation[0, 0]),
    )
    return SE3.from_rotvec(
        [0.0, 0.0, yaw],
        [float(translation_xy[0]), float(translation_xy[1]), 0.0],
    )


def _pose_vertices(graph: CanonicalGraph) -> dict[int, PoseVertex]:
    return {
        vertex.global_id: vertex
        for vertex in graph.vertices
        if isinstance(vertex, PoseVertex)
    }


def solve_robot_translations(
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    *,
    robot_id: int,
) -> dict[int, np.ndarray]:
    """Solve one robot's exact weighted translation problem for fixed yaws/objects."""

    pose_vertices = _pose_vertices(graph)
    robot_poses = tuple(
        sorted(
            (
                vertex
                for vertex in pose_vertices.values()
                if vertex.robot_id == robot_id
            ),
            key=lambda vertex: vertex.local_index,
        )
    )
    if not robot_poses:
        raise Point2InitializationError(f"robot {robot_id} has no poses")
    missing = [vertex.global_id for vertex in graph.vertices if vertex.global_id not in state]
    if missing:
        raise Point2InitializationError(f"state is missing variable {missing[0]}")

    active = [vertex for vertex in robot_poses if not vertex.fixed]
    offsets = {vertex.global_id: index for index, vertex in enumerate(active)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    right_hand_side: list[float] = []

    def append_equation(
        coefficients: Mapping[int, float], target: np.ndarray, weight: float
    ) -> None:
        root_weight = math.sqrt(float(weight))
        for dimension in range(2):
            row = len(right_hand_side)
            rhs = float(target[dimension])
            for variable_id, coefficient in coefficients.items():
                if variable_id in offsets:
                    rows.append(row)
                    columns.append(2 * offsets[variable_id] + dimension)
                    values.append(root_weight * coefficient)
                else:
                    rhs -= coefficient * float(state[variable_id].translation[dimension])
            right_hand_side.append(root_weight * rhs)

    robot_pose_ids = {vertex.global_id for vertex in robot_poses}
    for factor in graph.factors:
        if isinstance(factor, OdometryFactor):
            if factor.from_id not in robot_pose_ids:
                continue
            if factor.to_id not in robot_pose_ids:
                raise Point2InitializationError("odometry crosses robot ownership")
            target = _rotation2(state[factor.from_id]) @ factor.measurement.translation[:2]
            append_equation(
                {factor.from_id: -1.0, factor.to_id: 1.0},
                target,
                factor.tau,
            )
        elif isinstance(factor, PoseObjectFactor) and factor.from_id in robot_pose_ids:
            target = (
                state[factor.to_id].translation[:2]
                - _rotation2(state[factor.from_id]) @ factor.measurement.translation[:2]
            )
            append_equation({factor.from_id: 1.0}, target, factor.tau)

    if active and not right_hand_side:
        raise Point2InitializationError(f"robot {robot_id} has no translation factors")
    solved = {
        vertex.global_id: np.array(state[vertex.global_id].translation[:2], copy=True)
        for vertex in robot_poses
    }
    if not active:
        return solved

    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(right_hand_side), 2 * len(active)),
        dtype=np.float64,
    ).tocsr()
    solution = lsqr(
        matrix,
        np.asarray(right_hand_side, dtype=np.float64),
        atol=1e-12,
        btol=1e-12,
        iter_lim=max(100, 4 * matrix.shape[1]),
    )
    if solution[1] not in (1, 2) or not np.all(np.isfinite(solution[0])):
        raise Point2InitializationError(
            f"robot {robot_id} translation solve failed with LSQR status {solution[1]}"
        )
    for vertex in active:
        offset = 2 * offsets[vertex.global_id]
        solved[vertex.global_id] = np.array(solution[0][offset : offset + 2], copy=True)
    return solved


def _robot_local_residuals(
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    robot_pose_ids: set[int],
) -> list[np.ndarray]:
    residuals = []
    for factor in graph.factors:
        if isinstance(factor, OdometryFactor) and factor.from_id in robot_pose_ids:
            residuals.append(
                planar_odometry_linearization(
                    state[factor.from_id],
                    state[factor.to_id],
                    factor.measurement,
                    kappa=factor.kappa,
                    tau=factor.tau,
                )[0]
            )
        elif isinstance(factor, PoseObjectFactor) and factor.from_id in robot_pose_ids:
            residuals.append(
                planar_observation_linearization(
                    state[factor.from_id],
                    state[factor.to_id].translation[:2],
                    factor.measurement,
                    kappa=factor.kappa,
                    tau=factor.tau,
                )[0]
            )
    return residuals


def solve_robot_pose2_step(
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    *,
    robot_id: int,
) -> dict[int, SE3]:
    """Apply one locally decreasing sparse Pose2 Gauss-Newton step."""

    robot_poses = tuple(
        sorted(
            (
                vertex
                for vertex in _pose_vertices(graph).values()
                if vertex.robot_id == robot_id
            ),
            key=lambda vertex: vertex.local_index,
        )
    )
    if not robot_poses:
        raise Point2InitializationError(f"robot {robot_id} has no poses")
    active = [vertex for vertex in robot_poses if not vertex.fixed]
    if not active:
        return {vertex.global_id: state[vertex.global_id] for vertex in robot_poses}
    offsets = {vertex.global_id: index for index, vertex in enumerate(active)}
    robot_pose_ids = {vertex.global_id for vertex in robot_poses}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    right_hand_side: list[float] = []

    def append_block(variable_id: int, jacobian: np.ndarray, row_offset: int) -> None:
        if variable_id not in offsets:
            return
        column_offset = 3 * offsets[variable_id]
        nonzero_rows, nonzero_columns = np.nonzero(jacobian)
        rows.extend((row_offset + nonzero_rows).tolist())
        columns.extend((column_offset + nonzero_columns).tolist())
        values.extend(jacobian[nonzero_rows, nonzero_columns].tolist())

    for factor in graph.factors:
        if isinstance(factor, OdometryFactor) and factor.from_id in robot_pose_ids:
            if factor.to_id not in robot_pose_ids:
                raise Point2InitializationError("odometry crosses robot ownership")
            residual, jacobian_from, jacobian_to = planar_odometry_linearization(
                state[factor.from_id],
                state[factor.to_id],
                factor.measurement,
                kappa=factor.kappa,
                tau=factor.tau,
            )
            row_offset = len(right_hand_side)
            right_hand_side.extend((-residual).tolist())
            append_block(factor.from_id, jacobian_from, row_offset)
            append_block(factor.to_id, jacobian_to, row_offset)
        elif isinstance(factor, PoseObjectFactor) and factor.from_id in robot_pose_ids:
            residual, jacobian = planar_observation_linearization(
                state[factor.from_id],
                state[factor.to_id].translation[:2],
                factor.measurement,
                kappa=factor.kappa,
                tau=factor.tau,
            )
            row_offset = len(right_hand_side)
            right_hand_side.extend((-residual).tolist())
            append_block(factor.from_id, jacobian, row_offset)
    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(right_hand_side), 3 * len(active)),
        dtype=np.float64,
    ).tocsr()
    solution = lsqr(
        matrix,
        np.asarray(right_hand_side, dtype=np.float64),
        atol=1e-10,
        btol=1e-10,
        iter_lim=max(100, 6 * matrix.shape[1]),
    )
    if solution[1] not in (1, 2) or not np.all(np.isfinite(solution[0])):
        raise Point2InitializationError(
            f"robot {robot_id} Pose2 solve failed with LSQR status {solution[1]}"
        )
    current_residuals = _robot_local_residuals(graph, state, robot_pose_ids)
    current_cost = sum(float(residual @ residual) for residual in current_residuals)
    for step_size in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        candidate = dict(state)
        for vertex in active:
            offset = 3 * offsets[vertex.global_id]
            delta = step_size * solution[0][offset : offset + 3]
            pose = state[vertex.global_id]
            yaw = math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0]))
            candidate[vertex.global_id] = SE3.from_rotvec(
                [0.0, 0.0, yaw + float(delta[2])],
                [
                    float(pose.translation[0] + delta[0]),
                    float(pose.translation[1] + delta[1]),
                    0.0,
                ],
            )
        candidate_residuals = _robot_local_residuals(
            graph, candidate, robot_pose_ids
        )
        candidate_cost = sum(
            float(residual @ residual) for residual in candidate_residuals
        )
        if candidate_cost <= current_cost + 1e-10 * max(1.0, current_cost):
            return {
                vertex.global_id: candidate[vertex.global_id]
                for vertex in robot_poses
            }
    return {vertex.global_id: state[vertex.global_id] for vertex in robot_poses}


def object_sufficient_statistics(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> dict[int, dict[int, ObjectSufficientStatistic]]:
    """Return each robot's exact Point2 normal-equation sufficient statistics."""

    pose_vertices = _pose_vertices(graph)
    weighted: dict[int, dict[int, np.ndarray]] = {}
    information: dict[int, dict[int, float]] = {}
    for factor in graph.factors:
        if not isinstance(factor, PoseObjectFactor):
            continue
        try:
            robot_id = pose_vertices[factor.from_id].robot_id
            pose = state[factor.from_id]
        except KeyError as error:
            raise Point2InitializationError("observation state is incomplete") from error
        implied = (
            pose.translation[:2]
            + _rotation2(pose) @ factor.measurement.translation[:2]
        )
        weighted.setdefault(robot_id, {}).setdefault(
            factor.to_id, np.zeros(2, dtype=np.float64)
        )
        information.setdefault(robot_id, {}).setdefault(factor.to_id, 0.0)
        weighted[robot_id][factor.to_id] += factor.tau * implied
        information[robot_id][factor.to_id] += factor.tau

    result: dict[int, dict[int, ObjectSufficientStatistic]] = {}
    for robot_id, objects in sorted(weighted.items()):
        result[robot_id] = {}
        for object_id, position in sorted(objects.items()):
            value = np.array(position, copy=True)
            value.setflags(write=False)
            result[robot_id][object_id] = ObjectSufficientStatistic(
                weighted_position=value,
                information=information[robot_id][object_id],
            )
    return result


def complete_exchange_accounting(
    statistics: Mapping[int, Mapping[int, ObjectSufficientStatistic]],
    *,
    num_robots: int,
) -> InitializationCommunication:
    if num_robots < 1 or set(statistics) - set(range(num_robots)):
        raise Point2InitializationError("invalid Complete-topology robot IDs")
    directed_pairs = num_robots - 1
    blocks = directed_pairs * sum(len(objects) for objects in statistics.values())
    active_senders = sum(bool(objects) for objects in statistics.values())
    messages = active_senders * directed_pairs
    return InitializationCommunication(
        logical_bytes=24 * blocks,
        payload_blocks=blocks,
        messages=messages,
    )


def _updated_objects(
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    statistics: Mapping[int, Mapping[int, ObjectSufficientStatistic]],
) -> tuple[dict[int, SE3], float]:
    updates = dict(state)
    maximum_delta = 0.0
    for vertex in graph.vertices:
        if not isinstance(vertex, ObjectVertex):
            continue
        numerator = np.zeros(2, dtype=np.float64)
        denominator = 0.0
        for objects in statistics.values():
            statistic = objects.get(vertex.global_id)
            if statistic is None:
                continue
            numerator += statistic.weighted_position
            denominator += statistic.information
        if denominator <= 0.0:
            raise Point2InitializationError(
                f"object {vertex.global_id} has no observation information"
            )
        position = numerator / denominator
        maximum_delta = max(
            maximum_delta,
            float(np.linalg.norm(position - state[vertex.global_id].translation[:2])),
        )
        updates[vertex.global_id] = SE3.from_rotvec(
            [0.0, 0.0, 0.0], [float(position[0]), float(position[1]), 0.0]
        )
    return updates, maximum_delta


def _run_initializer(
    graph: CanonicalGraph, *, rounds: int, pose2_updates: bool
) -> Point2InitializationResult:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise Point2InitializationError("rounds must be a positive integer")
    robot_ids = sorted(
        {vertex.robot_id for vertex in graph.vertices if isinstance(vertex, PoseVertex)}
    )
    if robot_ids != list(range(len(robot_ids))):
        raise Point2InitializationError("robot IDs must be contiguous from zero")
    state = {
        vertex.global_id: _planar_pose(vertex.initial_pose, vertex.initial_pose.translation[:2])
        if isinstance(vertex, PoseVertex)
        else SE3.from_rotvec(
            [0.0, 0.0, 0.0],
            [float(vertex.initial_pose.translation[0]), float(vertex.initial_pose.translation[1]), 0.0],
        )
        for vertex in graph.vertices
    }
    initial_cost = fixed_yaw_point2_cost(graph, state)
    previous_cost = initial_cost
    total_communication = InitializationCommunication(0, 0, 0)
    records = []
    start = time.perf_counter()
    for iteration in range(1, rounds + 1):
        round_start = time.perf_counter()
        solve_times = []
        translated_state = dict(state)
        for robot_id in robot_ids:
            solve_start = time.perf_counter()
            if pose2_updates:
                robot_state = solve_robot_pose2_step(
                    graph, state, robot_id=robot_id
                )
            else:
                translations = solve_robot_translations(
                    graph, state, robot_id=robot_id
                )
                robot_state = {
                    variable_id: _planar_pose(state[variable_id], position)
                    for variable_id, position in translations.items()
                }
            solve_times.append(time.perf_counter() - solve_start)
            translated_state.update(robot_state)
        statistics = object_sufficient_statistics(graph, translated_state)
        communication = complete_exchange_accounting(
            statistics, num_robots=len(robot_ids)
        )
        candidate, maximum_delta = _updated_objects(
            graph, translated_state, statistics
        )
        candidate_cost = fixed_yaw_point2_cost(graph, candidate)
        tolerance = 1e-9 * max(1.0, previous_cost)
        if not math.isfinite(candidate_cost) or candidate_cost > previous_cost + tolerance:
            raise Point2InitializationError(
                "fixed-yaw alternating round increased the frozen Point2 objective"
            )
        state = candidate
        previous_cost = candidate_cost
        total_communication = total_communication + communication
        records.append(
            Point2InitializationRound(
                iteration=iteration,
                cost=candidate_cost,
                mean_factor_cost=candidate_cost / len(graph.factors),
                max_object_delta_m=maximum_delta,
                solver_wall_time_s=max(solve_times, default=0.0),
                round_wall_time_s=time.perf_counter() - round_start,
                communication=communication,
            )
        )
    return Point2InitializationResult(
        state=state,
        initial_cost=initial_cost,
        rounds=tuple(records),
        communication=total_communication,
        total_wall_time_s=time.perf_counter() - start,
    )


def run_fixed_yaw_initializer(
    graph: CanonicalGraph, *, rounds: int
) -> Point2InitializationResult:
    return _run_initializer(graph, rounds=rounds, pose2_updates=False)


def run_pose2_initializer(
    graph: CanonicalGraph, *, rounds: int
) -> Point2InitializationResult:
    return _run_initializer(graph, rounds=rounds, pose2_updates=True)


__all__ = [
    "InitializationCommunication",
    "ObjectSufficientStatistic",
    "Point2InitializationError",
    "Point2InitializationResult",
    "Point2InitializationRound",
    "complete_exchange_accounting",
    "fixed_yaw_point2_cost",
    "object_sufficient_statistics",
    "planar_observation_linearization",
    "planar_odometry_linearization",
    "run_fixed_yaw_initializer",
    "run_pose2_initializer",
    "solve_robot_pose2_step",
    "solve_robot_translations",
]
