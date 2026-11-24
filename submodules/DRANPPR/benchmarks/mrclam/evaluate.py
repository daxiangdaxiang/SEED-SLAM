"""Physical MR.CLAM evaluation kept separate from graph initialization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from benchmarks.slideslam.evaluate import StateEvaluation, evaluate_state
from benchmarks.slideslam.materialize import CanonicalGraph
from benchmarks.slideslam.schema import (
    ObjectVertex,
    OdometryFactor,
    PoseObjectFactor,
    PoseVertex,
)
from benchmarks.slideslam.se3 import SE3, chordal_residual

from .extract import PlanarPose, wrap_angle
from .materialize import estimate_se2_registration, se3_to_planar_pose
from .source import MRCLAMManifest, read_numeric_table


class MRCLAMEvaluationError(ValueError):
    """Raised when physical MR.CLAM evaluation input is incomplete."""


@dataclass(frozen=True, eq=False)
class RobotGroundTruthSeries:
    robot_subject: int
    timestamps_s: np.ndarray
    positions_xy: np.ndarray
    yaws_rad: np.ndarray
    unwrapped_yaws_rad: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        positions = np.asarray(self.positions_xy, dtype=np.float64)
        yaws = np.asarray(self.yaws_rad, dtype=np.float64)
        if (
            timestamps.ndim != 1
            or timestamps.size < 2
            or positions.shape != (timestamps.size, 2)
            or yaws.shape != (timestamps.size,)
            or not np.all(np.isfinite(timestamps))
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(yaws))
            or np.any(np.diff(timestamps) <= 0.0)
        ):
            raise MRCLAMEvaluationError(
                "robot ground truth must be finite, aligned, and strictly monotonic"
            )
        timestamps = np.array(timestamps, copy=True)
        positions = np.array(positions, copy=True)
        yaws = np.array(yaws, copy=True)
        timestamps.setflags(write=False)
        positions.setflags(write=False)
        yaws.setflags(write=False)
        unwrapped_yaws = np.unwrap(yaws)
        unwrapped_yaws.setflags(write=False)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "positions_xy", positions)
        object.__setattr__(self, "yaws_rad", yaws)
        object.__setattr__(self, "unwrapped_yaws_rad", unwrapped_yaws)

    def interpolate(self, timestamp_s: float) -> tuple[np.ndarray, float]:
        timestamp = float(timestamp_s)
        boundary_tolerance_s = 0.01
        if (
            not math.isfinite(timestamp)
            or timestamp < self.timestamps_s[0] - boundary_tolerance_s
            or timestamp > self.timestamps_s[-1] + boundary_tolerance_s
        ):
            raise MRCLAMEvaluationError(
                f"robot {self.robot_subject} ground truth does not support {timestamp}"
            )
        timestamp = min(max(timestamp, self.timestamps_s[0]), self.timestamps_s[-1])
        x = float(np.interp(timestamp, self.timestamps_s, self.positions_xy[:, 0]))
        y = float(np.interp(timestamp, self.timestamps_s, self.positions_xy[:, 1]))
        yaw = wrap_angle(
            float(np.interp(timestamp, self.timestamps_s, self.unwrapped_yaws_rad))
        )
        return np.array([x, y]), yaw

    def supports(self, timestamp_s: float) -> bool:
        timestamp = float(timestamp_s)
        return bool(
            math.isfinite(timestamp)
            and self.timestamps_s[0] - 0.01 <= timestamp
            and timestamp <= self.timestamps_s[-1] + 0.01
        )


@dataclass(frozen=True)
class GroundTruthData:
    robots: Mapping[int, RobotGroundTruthSeries]
    landmarks: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class PhysicalMetrics:
    robot_translation_rmse_m: float
    robot_yaw_rmse_rad: float
    object_position_rmse_m: float
    object_position_permutation_invariant_rmse_m: float
    robot_pose_count: int
    object_count: int
    ground_truth_from_estimate: PlanarPose


@dataclass(frozen=True)
class LiftedObjectiveBreakdown:
    odometry_cost: float
    observation_translation_cost: float
    observation_rotation_regularizer_cost: float
    total_cost: float


def project_state_to_point2(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> dict[int, SE3]:
    """Fix the global SE(3) gauge, then remove non-Point2 motion."""

    _validate_state(graph, state)
    fixed_poses = [
        vertex
        for vertex in graph.vertices
        if isinstance(vertex, PoseVertex) and vertex.fixed
    ]
    if len(fixed_poses) != 1:
        raise MRCLAMEvaluationError(
            "Point2 projection requires exactly one fixed trajectory root"
        )
    anchor = fixed_poses[0]
    world_from_solver = anchor.initial_pose.compose(
        state[anchor.global_id].inverse()
    )
    result = {}
    for variable_id, pose in state.items():
        planar = se3_to_planar_pose(world_from_solver.compose(pose))
        result[variable_id] = SE3.from_rotvec(
            [0.0, 0.0, planar.yaw], [planar.x, planar.y, 0.0]
        )
    return result


def evaluate_point2_state(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> StateEvaluation:
    """Evaluate the MR.CLAM Point2 model in its canonical SE(3) embedding."""

    return evaluate_state(graph, project_state_to_point2(graph, state))


def read_imesa_point_state(
    path: Path, graph: CanonicalGraph
) -> dict[int, SE3]:
    """Fuse native Point2 copies and supply only the evaluator's nuisance yaw."""

    pose_vertices = {
        vertex.global_id: vertex
        for vertex in graph.vertices
        if isinstance(vertex, PoseVertex)
    }
    object_vertices = {
        vertex.global_id: vertex
        for vertex in graph.vertices
        if isinstance(vertex, ObjectVertex)
    }
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MRCLAMEvaluationError(f"cannot read iMESA state {path}: {error}") from error
    expected_header = "type robot_id canonical_id x y yaw".split()
    if not lines or lines[0].split() != expected_header:
        raise MRCLAMEvaluationError(f"invalid iMESA state header: {path}")

    state: dict[int, SE3] = {}
    object_copies: dict[int, dict[int, np.ndarray]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(expected_header):
            raise MRCLAMEvaluationError(
                f"invalid iMESA state row at {path}:{line_number}"
            )
        try:
            kind = fields[0]
            robot_id = int(fields[1])
            variable_id = int(fields[2])
            x, y, yaw = (float(value) for value in fields[3:])
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ValueError("non-finite state")
        except ValueError as error:
            raise MRCLAMEvaluationError(
                f"invalid iMESA state row at {path}:{line_number}"
            ) from error
        if kind == "trajectory":
            vertex = pose_vertices.get(variable_id)
            if vertex is None or vertex.robot_id != robot_id or variable_id in state:
                raise MRCLAMEvaluationError(
                    f"invalid iMESA trajectory ownership at {path}:{line_number}"
                )
            state[variable_id] = SE3.from_rotvec(
                [0.0, 0.0, yaw], [x, y, 0.0]
            )
        elif kind == "object":
            vertex = object_vertices.get(variable_id)
            if vertex is None or robot_id not in vertex.observing_robots:
                raise MRCLAMEvaluationError(
                    f"invalid iMESA object ownership at {path}:{line_number}"
                )
            copies = object_copies.setdefault(variable_id, {})
            if robot_id in copies:
                raise MRCLAMEvaluationError(
                    f"duplicate iMESA object copy at {path}:{line_number}"
                )
            copies[robot_id] = np.array([x, y], dtype=np.float64)
        else:
            raise MRCLAMEvaluationError(
                f"invalid iMESA state type at {path}:{line_number}"
            )

    missing_poses = sorted(set(pose_vertices) - set(state))
    if missing_poses:
        raise MRCLAMEvaluationError(
            f"iMESA state is missing trajectory IDs {missing_poses[:10]}"
        )
    incident_rotations: dict[int, list[tuple[float, float]]] = {}
    for factor in graph.factors:
        if not isinstance(factor, PoseObjectFactor):
            continue
        pose_yaw = se3_to_planar_pose(state[factor.from_id]).yaw
        measurement_yaw = se3_to_planar_pose(factor.measurement).yaw
        incident_rotations.setdefault(factor.to_id, []).append(
            (factor.kappa, pose_yaw + measurement_yaw)
        )
    for variable_id, vertex in object_vertices.items():
        copies = object_copies.get(variable_id, {})
        missing = sorted(set(vertex.observing_robots) - set(copies))
        if missing:
            raise MRCLAMEvaluationError(
                f"iMESA object {variable_id} is missing observer copies {missing}"
            )
        position = np.mean(
            np.vstack([copies[robot] for robot in sorted(copies)]), axis=0
        )
        rotations = incident_rotations.get(variable_id, [])
        cosine = sum(weight * math.cos(yaw) for weight, yaw in rotations)
        sine = sum(weight * math.sin(yaw) for weight, yaw in rotations)
        nuisance_yaw = math.atan2(sine, cosine) if cosine or sine else 0.0
        state[variable_id] = SE3.from_rotvec(
            [0.0, 0.0, nuisance_yaw], [position[0], position[1], 0.0]
        )
    _validate_state(graph, state)
    return state


def read_dgs_point2_state(
    path: Path, graph: CanonicalGraph
) -> dict[int, SE3]:
    """Read the extension's canonical Point2 state using the common fusion rule."""

    return read_imesa_point_state(path, graph)


def load_ground_truth(
    manifest: MRCLAMManifest, dataset_directory: Path
) -> GroundTruthData:
    dataset_directory = Path(dataset_directory)
    robots = {}
    for subject in manifest.robot_subjects:
        table = read_numeric_table(
            dataset_directory / f"Robot{subject}_Groundtruth.dat",
            4,
            f"robot {subject} ground truth",
        )
        # Duplicate Vicon timestamps, if ever present, are reduced deterministically.
        unique_times, first_indices, counts = np.unique(
            table[:, 0], return_index=True, return_counts=True
        )
        positions = np.add.reduceat(table[:, 1:3], first_indices, axis=0)
        positions /= counts[:, None]
        yaw_vectors = np.exp(1j * table[:, 3])
        yaw_means = np.add.reduceat(yaw_vectors, first_indices) / counts
        yaws = np.angle(yaw_means)
        robots[subject] = RobotGroundTruthSeries(
            subject, unique_times, positions, yaws
        )
    landmark_table = read_numeric_table(
        dataset_directory / "Landmark_Groundtruth.dat",
        5,
        "landmark ground truth",
    )
    landmarks = {
        int(row[0]): np.asarray(row[1:3], dtype=np.float64)
        for row in landmark_table
    }
    if set(landmarks) != set(manifest.static_landmark_subjects):
        raise MRCLAMEvaluationError(
            "landmark ground truth does not match the manifest subjects"
        )
    return GroundTruthData(robots=robots, landmarks=landmarks)


def _validate_state(graph: CanonicalGraph, state: Mapping[int, SE3]) -> None:
    expected = {vertex.global_id for vertex in graph.vertices}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise MRCLAMEvaluationError(
            f"state IDs do not match graph; missing={missing}, extra={extra}"
        )
    if any(not isinstance(state[variable_id], SE3) for variable_id in expected):
        raise MRCLAMEvaluationError("physical evaluation state must contain SE3 poses")


def _pose_ground_truth(
    vertex: PoseVertex, ground_truth: GroundTruthData
) -> tuple[np.ndarray, float, int]:
    metadata = vertex.reference_metadata
    if not isinstance(metadata, Mapping):
        raise MRCLAMEvaluationError(
            f"pose {vertex.global_id} has no MR.CLAM reference metadata"
        )
    subject = metadata.get("mrclam_robot_subject")
    timestamp = metadata.get("timestamp_s")
    if isinstance(subject, bool) or not isinstance(subject, int):
        raise MRCLAMEvaluationError("pose reference robot subject is invalid")
    if subject not in ground_truth.robots:
        raise MRCLAMEvaluationError(f"ground truth is missing robot {subject}")
    position, yaw = ground_truth.robots[subject].interpolate(float(timestamp))
    return position, yaw, subject


def _landmark_subject(vertex: ObjectVertex) -> int:
    match = re.fullmatch(r"mrclam_cylinder_(\d+)", vertex.semantic_label)
    if match is None:
        raise MRCLAMEvaluationError(
            f"object {vertex.global_id} has no MR.CLAM landmark subject"
        )
    return int(match.group(1))


def physical_metrics(
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    ground_truth: GroundTruthData,
) -> PhysicalMetrics:
    _validate_state(graph, state)
    state = project_state_to_point2(graph, state)
    if graph.metadata.get("ground_truth_used_for_initialization") is not False:
        raise MRCLAMEvaluationError(
            "canonical provenance does not prove ground-truth-free initialization"
        )
    pose_vertices = sorted(
        (item for item in graph.vertices if isinstance(item, PoseVertex)),
        key=lambda item: item.global_id,
    )
    object_vertices = sorted(
        (item for item in graph.vertices if isinstance(item, ObjectVertex)),
        key=lambda item: item.global_id,
    )
    estimated_positions = []
    truth_positions = []
    truth_yaws = []
    estimated_yaws = []
    for vertex in pose_vertices:
        metadata = vertex.reference_metadata
        if not isinstance(metadata, Mapping):
            raise MRCLAMEvaluationError(
                f"pose {vertex.global_id} has no MR.CLAM reference metadata"
            )
        subject = metadata.get("mrclam_robot_subject")
        timestamp = metadata.get("timestamp_s")
        if (
            isinstance(subject, bool)
            or not isinstance(subject, int)
            or subject not in ground_truth.robots
        ):
            raise MRCLAMEvaluationError("pose reference robot subject is invalid")
        if not ground_truth.robots[subject].supports(float(timestamp)):
            continue
        truth_position, truth_yaw, _ = _pose_ground_truth(vertex, ground_truth)
        estimated = se3_to_planar_pose(state[vertex.global_id])
        estimated_positions.append([estimated.x, estimated.y])
        estimated_yaws.append(estimated.yaw)
        truth_positions.append(truth_position)
        truth_yaws.append(truth_yaw)
    alignment, _ = estimate_se2_registration(
        estimated_positions, truth_positions
    )
    aligned_positions = np.vstack(
        [alignment.transform_point(point) for point in estimated_positions]
    )
    truth_position_array = np.vstack(truth_positions)
    robot_translation_rmse = float(
        np.sqrt(
            np.mean(np.sum((aligned_positions - truth_position_array) ** 2, axis=1))
        )
    )
    yaw_errors = np.array(
        [
            wrap_angle(alignment.yaw + estimated - truth)
            for estimated, truth in zip(estimated_yaws, truth_yaws)
        ]
    )
    robot_yaw_rmse = float(np.sqrt(np.mean(yaw_errors**2)))

    raw_label_map = graph.metadata.get(
        "evaluation_object_ground_truth_label_map", {}
    )
    if not isinstance(raw_label_map, dict):
        raise MRCLAMEvaluationError(
            "object ground-truth label map must be an object"
        )
    ground_truth_label_map = {}
    for raw_subject, raw_target in raw_label_map.items():
        try:
            subject = int(raw_subject)
        except (TypeError, ValueError) as error:
            raise MRCLAMEvaluationError(
                "object ground-truth label map keys must be integer labels"
            ) from error
        if str(subject) != str(raw_subject) or isinstance(raw_target, bool) or not isinstance(
            raw_target, int
        ):
            raise MRCLAMEvaluationError(
                "object ground-truth label map values must be integer labels"
            )
        ground_truth_label_map[subject] = int(raw_target)
    if set(ground_truth_label_map) != set(ground_truth_label_map.values()):
        raise MRCLAMEvaluationError(
            "object ground-truth label map must be a closed label permutation"
        )

    aligned_object_positions = []
    raw_object_truth_positions = []
    corrected_object_truth_positions = []
    for vertex in object_vertices:
        subject = _landmark_subject(vertex)
        corrected_subject = ground_truth_label_map.get(subject, subject)
        if subject not in ground_truth.landmarks or corrected_subject not in ground_truth.landmarks:
            raise MRCLAMEvaluationError(
                f"ground truth is missing landmark subject {subject} or {corrected_subject}"
            )
        estimated = state[vertex.global_id].translation[:2]
        aligned = alignment.transform_point(estimated)
        aligned_object_positions.append(aligned)
        raw_object_truth_positions.append(ground_truth.landmarks[subject])
        corrected_object_truth_positions.append(
            ground_truth.landmarks[corrected_subject]
        )
    if not aligned_object_positions:
        raise MRCLAMEvaluationError("physical evaluation graph has no objects")
    aligned_object_array = np.vstack(aligned_object_positions)
    raw_object_truth_array = np.vstack(raw_object_truth_positions)
    corrected_object_truth_array = np.vstack(corrected_object_truth_positions)
    raw_squared_distances = np.sum(
        (aligned_object_array - raw_object_truth_array) ** 2, axis=1
    )
    corrected_squared_distances = np.sum(
        (aligned_object_array - corrected_object_truth_array) ** 2, axis=1
    )
    return PhysicalMetrics(
        robot_translation_rmse_m=robot_translation_rmse,
        robot_yaw_rmse_rad=robot_yaw_rmse,
        object_position_rmse_m=float(
            np.sqrt(np.mean(raw_squared_distances))
        ),
        object_position_permutation_invariant_rmse_m=float(
            np.sqrt(np.mean(corrected_squared_distances))
        ),
        robot_pose_count=len(estimated_positions),
        object_count=len(object_vertices),
        ground_truth_from_estimate=alignment,
    )


def lifted_objective_breakdown(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> LiftedObjectiveBreakdown:
    _validate_state(graph, state)
    state = project_state_to_point2(graph, state)
    odometry = 0.0
    observation_translation = 0.0
    observation_rotation = 0.0
    for factor in graph.factors:
        residual = chordal_residual(
            state[factor.from_id],
            state[factor.to_id],
            factor.measurement,
            factor.kappa,
            factor.tau,
        )
        if isinstance(factor, OdometryFactor):
            odometry += float(residual @ residual)
        else:
            observation_rotation += float(residual[:9] @ residual[:9])
            observation_translation += float(residual[9:] @ residual[9:])
    total = odometry + observation_translation + observation_rotation
    return LiftedObjectiveBreakdown(
        odometry_cost=odometry,
        observation_translation_cost=observation_translation,
        observation_rotation_regularizer_cost=observation_rotation,
        total_cost=total,
    )


__all__ = [
    "GroundTruthData",
    "LiftedObjectiveBreakdown",
    "MRCLAMEvaluationError",
    "PhysicalMetrics",
    "RobotGroundTruthSeries",
    "evaluate_point2_state",
    "lifted_objective_breakdown",
    "load_ground_truth",
    "physical_metrics",
    "project_state_to_point2",
    "read_imesa_point_state",
    "read_dgs_point2_state",
]
