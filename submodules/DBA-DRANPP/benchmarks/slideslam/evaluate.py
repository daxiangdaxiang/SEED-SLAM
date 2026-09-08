"""Common external evaluation for SlideSLAM backend benchmark runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .materialize import CanonicalGraph, inspect_g2o
from .schema import ObjectVertex, PoseObjectFactor, PoseVertex
from .se3 import SE3, chordal_jacobians, chordal_residual


class EvaluationError(ValueError):
    """Raised when solver output cannot be evaluated without ambiguity."""


@dataclass(frozen=True, eq=False)
class StateEvaluation:
    variable_ids: tuple[int, ...]
    cost: float
    gradient: np.ndarray
    gradient_norm: float
    odometry_cost: float
    observation_cost: float
    residual_count: int
    max_residual_norm: float
    finite: bool


@dataclass(frozen=True)
class ReferenceMetrics:
    trajectory_translation_rmse_to_centralized_reference: float
    trajectory_rotation_rmse_to_centralized_reference: float
    object_translation_rmse_to_centralized_reference: float
    object_rotation_rmse_to_centralized_reference: float


@dataclass(frozen=True)
class IterationRecord:
    method: str
    iteration: int
    external_cost: float
    external_gradient_norm: float
    native_objective: Optional[float]
    logical_bytes: int
    payload_blocks: int
    messages: int
    solver_wall_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise EvaluationError("iteration method must be a non-empty string")
        if isinstance(self.iteration, bool) or self.iteration < 1:
            raise EvaluationError("iteration must be a positive integer")
        for name in (
            "external_cost",
            "external_gradient_norm",
            "solver_wall_time_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise EvaluationError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.native_objective is not None:
            native = float(self.native_objective)
            if not math.isfinite(native):
                raise EvaluationError("native_objective must be finite when present")
            object.__setattr__(self, "native_objective", native)
        for name in ("logical_bytes", "payload_blocks", "messages"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvaluationError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    iteration: Optional[int]
    cap: int


def _validated_state(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> tuple[tuple[int, ...], dict[int, SE3]]:
    expected = tuple(sorted(vertex.global_id for vertex in graph.vertices))
    expected_set = set(expected)
    actual_set = set(state)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        raise EvaluationError(f"state is missing IDs: {', '.join(map(str, missing))}")
    if extra:
        raise EvaluationError(
            f"state has unexpected IDs: {', '.join(map(str, extra))}"
        )
    result = {}
    for variable_id in expected:
        pose = state[variable_id]
        if not isinstance(pose, SE3):
            raise EvaluationError(f"state ID {variable_id} is not an SE3 value")
        result[variable_id] = pose
    return expected, result


def evaluate_state(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> StateEvaluation:
    """Evaluate the frozen scalar chordal objective and its 6D gradients."""

    variable_ids, values = _validated_state(graph, state)
    offsets = {variable_id: index for index, variable_id in enumerate(variable_ids)}
    gradient = np.zeros(6 * len(variable_ids), dtype=float)
    odometry_cost = 0.0
    observation_cost = 0.0
    max_residual_norm = 0.0
    for factor in graph.factors:
        residual = chordal_residual(
            values[factor.from_id],
            values[factor.to_id],
            factor.measurement,
            factor.kappa,
            factor.tau,
        )
        jacobian_from, jacobian_to = chordal_jacobians(
            values[factor.from_id],
            values[factor.to_id],
            factor.measurement,
            factor.kappa,
            factor.tau,
        )
        factor_cost = float(residual @ residual)
        from_slice = slice(6 * offsets[factor.from_id], 6 * (offsets[factor.from_id] + 1))
        to_slice = slice(6 * offsets[factor.to_id], 6 * (offsets[factor.to_id] + 1))
        gradient[from_slice] += 2.0 * jacobian_from.T @ residual
        gradient[to_slice] += 2.0 * jacobian_to.T @ residual
        if factor.kind == "odometry":
            odometry_cost += factor_cost
        else:
            observation_cost += factor_cost
        max_residual_norm = max(max_residual_norm, float(np.linalg.norm(residual)))

    fixed_ids = {
        vertex.global_id
        for vertex in graph.vertices
        if isinstance(vertex, PoseVertex) and vertex.fixed
    }
    active_gradient = np.concatenate(
        [
            gradient[6 * offsets[variable_id] : 6 * (offsets[variable_id] + 1)]
            for variable_id in variable_ids
            if variable_id not in fixed_ids
        ]
    )
    cost = odometry_cost + observation_cost
    gradient_norm = float(np.linalg.norm(active_gradient))
    finite = bool(
        math.isfinite(cost)
        and math.isfinite(max_residual_norm)
        and np.all(np.isfinite(gradient))
        and math.isfinite(gradient_norm)
    )
    gradient.setflags(write=False)
    return StateEvaluation(
        variable_ids=variable_ids,
        cost=cost,
        gradient=gradient,
        gradient_norm=gradient_norm,
        odometry_cost=odometry_cost,
        observation_cost=observation_cost,
        residual_count=len(graph.factors),
        max_residual_norm=max_residual_norm,
        finite=finite,
    )


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right_transpose = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_transpose)
    return left @ correction @ right_transpose


def alignment_to_reference(
    graph: CanonicalGraph,
    estimated: Mapping[int, SE3],
    reference: Mapping[int, SE3],
) -> SE3:
    """Find one rigid trajectory alignment mapping estimate into reference."""

    _, estimate = _validated_state(graph, estimated)
    _, target = _validated_state(graph, reference)
    pose_ids = [
        vertex.global_id for vertex in graph.vertices if isinstance(vertex, PoseVertex)
    ]
    if not pose_ids:
        raise EvaluationError("reference alignment requires trajectory poses")
    source_points = np.vstack([estimate[item].translation for item in pose_ids])
    target_points = np.vstack([target[item].translation for item in pose_ids])
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    if len(pose_ids) >= 3 and np.linalg.matrix_rank(source_centered) >= 2:
        left, _, right_transpose = np.linalg.svd(
            source_centered.T @ target_centered
        )
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(right_transpose.T @ left.T)
        rotation = right_transpose.T @ correction @ left.T
    else:
        rotation = _project_rotation(
            sum(
                (target[item].rotation @ estimate[item].rotation.T for item in pose_ids),
                np.zeros((3, 3)),
            )
        )
    translation = target_center - rotation @ source_center
    return SE3(rotation, translation)


def align_state_to_reference(
    graph: CanonicalGraph,
    estimated: Mapping[int, SE3],
    reference: Mapping[int, SE3],
) -> dict[int, SE3]:
    alignment = alignment_to_reference(graph, estimated, reference)
    _, values = _validated_state(graph, estimated)
    return {
        variable_id: alignment.compose(pose)
        for variable_id, pose in values.items()
    }


def _rmse(values: Sequence[float]) -> float:
    if not values:
        raise EvaluationError("cannot compute RMSE for an empty variable group")
    return math.sqrt(float(np.mean(np.square(values))))


def _geodesic_rotation_error(rotation: np.ndarray) -> float:
    sine = 0.5 * float(
        np.linalg.norm(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    )
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    return math.atan2(sine, cosine)


def reference_metrics(
    graph: CanonicalGraph,
    estimated: Mapping[int, SE3],
    reference: Mapping[int, SE3],
) -> ReferenceMetrics:
    _, target = _validated_state(graph, reference)
    aligned = align_state_to_reference(graph, estimated, reference)
    pose_ids = [
        vertex.global_id for vertex in graph.vertices if isinstance(vertex, PoseVertex)
    ]
    object_ids = [
        vertex.global_id for vertex in graph.vertices if isinstance(vertex, ObjectVertex)
    ]

    def translation_errors(ids: Sequence[int]) -> list[float]:
        return [
            float(np.linalg.norm(aligned[item].translation - target[item].translation))
            for item in ids
        ]

    def rotation_errors(ids: Sequence[int]) -> list[float]:
        return [
            _geodesic_rotation_error(
                target[item].rotation.T @ aligned[item].rotation
            )
            for item in ids
        ]

    return ReferenceMetrics(
        trajectory_translation_rmse_to_centralized_reference=_rmse(
            translation_errors(pose_ids)
        ),
        trajectory_rotation_rmse_to_centralized_reference=_rmse(
            rotation_errors(pose_ids)
        ),
        object_translation_rmse_to_centralized_reference=_rmse(
            translation_errors(object_ids)
        ),
        object_rotation_rmse_to_centralized_reference=_rmse(
            rotation_errors(object_ids)
        ),
    )


def relative_cost_gap(cost: float, reference_cost: float) -> float:
    cost = float(cost)
    reference_cost = float(reference_cost)
    if not math.isfinite(cost) or not math.isfinite(reference_cost):
        raise EvaluationError("cost gap inputs must be finite")
    return (cost - reference_cost) / max(abs(reference_cost), 1e-12)


def read_larpg_state(path: Path, graph: CanonicalGraph) -> dict[int, SE3]:
    state = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"cannot read LARPG state: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            variable_id = int(value["id"])
            pose = SE3.from_dict(value["pose"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EvaluationError(
                f"invalid LARPG state record at {path}:{line_number}"
            ) from error
        if variable_id in state:
            raise EvaluationError(f"duplicate LARPG state ID {variable_id}")
        state[variable_id] = pose
    _, validated = _validated_state(graph, state)
    return validated


def _read_dran_root_transforms(
    path: Path, robot_ids: Sequence[int], common_frame_robot: int
) -> dict[int, SE3]:
    robot_set = set(robot_ids)
    adjacency: dict[int, list[tuple[int, SE3]]] = {
        robot_id: [] for robot_id in robot_ids
    }
    directed_edges: set[tuple[int, int]] = set()
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise EvaluationError(f"cannot read DRAN frame transforms: {path}") from error
    for row_number, row in enumerate(rows, start=2):
        try:
            receiver = int(row["receiver_robot"])
            sender = int(row["sender_robot"])
            valid = int(row["valid"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(
                f"invalid DRAN frame transform at {path}:{row_number}"
            ) from error
        if valid != 1:
            continue
        if receiver == sender or receiver not in robot_set or sender not in robot_set:
            raise EvaluationError(
                f"invalid DRAN frame-transform edge at {path}:{row_number}"
            )
        edge = (receiver, sender)
        if edge in directed_edges:
            raise EvaluationError(
                f"duplicate DRAN frame transform for edge {receiver}->{sender}"
            )
        directed_edges.add(edge)
        try:
            rows_count = int(row["rows"])
            cols_count = int(row["cols"])
            values = np.asarray(
                [float(row[f"value_{index}"]) for index in range(12)],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(
                f"invalid DRAN frame transform at {path}:{row_number}"
            ) from error
        if (rows_count, cols_count) != (3, 4) or not np.all(np.isfinite(values)):
            raise EvaluationError(
                "DRAN common-frame evaluation requires finite full SE(3) transforms"
            )
        matrix = values.reshape(3, 4)
        rotation = matrix[:, :3]
        if (
            np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-6
            or np.linalg.det(rotation) <= 0.0
        ):
            raise EvaluationError(
                f"invalid DRAN root-frame rotation for robot {sender}"
            )
        receiver_from_sender = SE3(
            _project_rotation(rotation), matrix[:, 3]
        )
        adjacency[receiver].append((sender, receiver_from_sender))
        adjacency[sender].append((receiver, receiver_from_sender.inverse()))

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: item[0])
    transforms = {common_frame_robot: SE3.identity()}
    queue = [common_frame_robot]
    for receiver in queue:
        root_from_receiver = transforms[receiver]
        for sender, receiver_from_sender in adjacency[receiver]:
            if sender in transforms:
                continue
            transforms[sender] = root_from_receiver.compose(receiver_from_sender)
            queue.append(sender)
    missing = sorted(set(robot_ids) - set(transforms))
    if missing:
        raise EvaluationError(
            "DRAN frame transforms do not connect all robots to root "
            f"{common_frame_robot}: missing {missing}"
        )
    return transforms


def _average_dran_object_copies(copies: Sequence[SE3]) -> SE3:
    if not copies:
        raise EvaluationError("cannot fuse an object without observer copies")
    return SE3(
        _project_rotation(
            sum((pose.rotation for pose in copies), np.zeros((3, 3)))
        ),
        np.mean(np.vstack([pose.translation for pose in copies]), axis=0),
    )


def read_dran_state(
    path: Path,
    adapter_manifest: Path,
    graph: CanonicalGraph,
    *,
    transform_path: Optional[Path] = None,
    common_frame_robot: int = 0,
    point_objects: bool = False,
) -> dict[int, SE3]:
    """Restore canonical IDs and fuse observer copies in one common frame.

    Object translations use an arithmetic mean over all observing robots. The
    projected chordal rotation mean supplies the default object orientation.
    With ``point_objects``, object yaw is instead eliminated analytically from
    the incident pose-object rotation regularizers.
    """

    try:
        manifest = json.loads(Path(adapter_manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationError(
            f"cannot read DRAN adapter manifest: {adapter_manifest}"
        ) from error
    trajectory_map = {}
    object_map = {}
    try:
        for robot in manifest["robots"]:
            robot_id = int(robot["robot_id"])
            for local_id, global_id in robot["trajectory_local_to_global"].items():
                trajectory_map[(robot_id, int(local_id))] = int(global_id)
            for local_id, global_id in robot["object_local_to_global"].items():
                object_map[(robot_id, int(local_id))] = int(global_id)
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError("invalid DRAN adapter manifest mappings") from error
    object_observers = {
        vertex.global_id: frozenset(vertex.observing_robots)
        for vertex in graph.vertices
        if isinstance(vertex, ObjectVertex)
    }
    robot_ids = sorted({robot_id for robot_id, _ in trajectory_map})
    if common_frame_robot not in robot_ids:
        raise EvaluationError(
            f"DRAN common-frame robot {common_frame_robot} is absent"
        )
    frame_transforms = (
        {robot_id: SE3.identity() for robot_id in robot_ids}
        if transform_path is None
        else _read_dran_root_transforms(
            Path(transform_path), robot_ids, common_frame_robot
        )
    )
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"cannot read DRAN state: {path}") from error
    if not lines:
        raise EvaluationError(f"DRAN state is empty: {path}")
    expected_header = (
        "type robot_id local_vertex_id object_id x y z qx qy qz qw".split()
    )
    if lines[0].split() != expected_header:
        raise EvaluationError(f"invalid DRAN state header: {path}")
    state = {}
    object_copies: dict[int, dict[int, SE3]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(expected_header):
            raise EvaluationError(f"invalid DRAN state row at {path}:{line_number}")
        try:
            kind = fields[0]
            robot_id = int(fields[1])
            local_id = int(fields[2])
            pose = SE3.from_quaternion_xyzw(
                [float(value) for value in fields[7:11]],
                [float(value) for value in fields[4:7]],
            )
            pose = frame_transforms[robot_id].compose(pose)
            if kind == "trajectory":
                variable_id = trajectory_map[(robot_id, local_id)]
            elif kind == "object":
                variable_id = object_map[(robot_id, local_id)]
            else:
                raise ValueError("unsupported state kind")
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(
                f"invalid DRAN state row at {path}:{line_number}"
            ) from error
        if kind == "object":
            if robot_id not in object_observers[variable_id]:
                continue
            copies = object_copies.setdefault(variable_id, {})
            if robot_id in copies:
                raise EvaluationError(
                    f"duplicate DRAN object copy {variable_id} from robot {robot_id}"
                )
            copies[robot_id] = pose
            continue
        if variable_id in state:
            raise EvaluationError(f"duplicate canonical DRAN state ID {variable_id}")
        state[variable_id] = pose
    for variable_id, observers in object_observers.items():
        copies = object_copies.get(variable_id, {})
        missing = sorted(observers - set(copies))
        if missing:
            raise EvaluationError(
                f"DRAN object {variable_id} is missing observer copies {missing}"
            )
        fused = _average_dran_object_copies(
            [copies[robot_id] for robot_id in sorted(observers)]
        )
        if point_objects:
            cosine = 0.0
            sine = 0.0
            for factor in graph.factors:
                if not isinstance(factor, PoseObjectFactor):
                    continue
                if factor.to_id != variable_id or factor.from_id not in state:
                    continue
                pose_rotation = state[factor.from_id].rotation
                measurement_rotation = factor.measurement.rotation
                pose_yaw = math.atan2(pose_rotation[1, 0], pose_rotation[0, 0])
                measurement_yaw = math.atan2(
                    measurement_rotation[1, 0], measurement_rotation[0, 0]
                )
                implied_yaw = pose_yaw + measurement_yaw
                cosine += factor.kappa * math.cos(implied_yaw)
                sine += factor.kappa * math.sin(implied_yaw)
            if cosine != 0.0 or sine != 0.0:
                fused = SE3.from_rotvec(
                    [0.0, 0.0, math.atan2(sine, cosine)], fused.translation
                )
        state[variable_id] = fused
    _, validated = _validated_state(graph, state)
    return validated


def read_g2o_state(path: Path, graph: CanonicalGraph) -> dict[int, SE3]:
    try:
        state = dict(inspect_g2o(path).states)
    except (OSError, ValueError) as error:
        raise EvaluationError(f"cannot read g2o state: {path}") from error
    _, validated = _validated_state(graph, state)
    return validated


def read_distributed_mapper_state(
    seed_root: Path, adapter_manifest: Path, graph: CanonicalGraph
) -> dict[int, SE3]:
    """Read per-robot Distributed Mapper outputs and fuse object copies.

    The benchmark adapter writes full 64-bit Symbol keys in each per-robot
    output. Reading those files preserves the owner/copy identity and avoids
    both duplicate numeric landmark IDs and the official evaluator's one-to-one
    mapping limitation. The accompanying ``fullGraph.g2o`` is intentionally the
    canonical physical graph; copy-consensus factors exist only in the local
    Distributed Mapper partitions.
    """

    try:
        manifest = json.loads(Path(adapter_manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationError(
            f"cannot read Distributed Mapper adapter manifest: {adapter_manifest}"
        ) from error
    try:
        variables = {
            int(item["key"]): item for item in manifest["variables"]
        }
        num_robots = int(manifest["num_robots"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(
            "invalid Distributed Mapper object adapter manifest"
        ) from error

    state: dict[int, SE3] = {}
    object_copies: dict[int, dict[int, SE3]] = {}
    root = Path(seed_root)
    for robot_id in range(num_robots):
        output_path = root / f"{robot_id}_optimized.g2o"
        try:
            parsed = inspect_g2o(output_path)
        except (OSError, ValueError) as error:
            raise EvaluationError(
                f"cannot read Distributed Mapper output: {output_path}"
            ) from error
        for key, pose in parsed.states.items():
            variable = variables.get(int(key))
            if variable is None:
                raise EvaluationError(
                    f"Distributed Mapper output contains unknown key {key}"
                )
            owner = int(variable["robot_id"])
            if owner != robot_id:
                raise EvaluationError(
                    f"Distributed Mapper output {output_path} contains key owned by robot {owner}"
                )
            global_id = int(variable["global_id"])
            if variable["kind"] == "object":
                copies = object_copies.setdefault(global_id, {})
                if robot_id in copies:
                    raise EvaluationError(
                        f"duplicate Distributed Mapper object copy {global_id} from robot {robot_id}"
                    )
                copies[robot_id] = pose
            else:
                if global_id in state:
                    raise EvaluationError(
                        f"duplicate Distributed Mapper trajectory state {global_id}"
                    )
                state[global_id] = pose

    object_observers = {
        vertex.global_id: frozenset(vertex.observing_robots)
        for vertex in graph.vertices
        if isinstance(vertex, ObjectVertex)
    }
    for global_id, observers in object_observers.items():
        copies = object_copies.get(global_id, {})
        missing = sorted(observers - set(copies))
        if missing:
            raise EvaluationError(
                f"Distributed Mapper object {global_id} is missing copies {missing}"
            )
        state[global_id] = _average_dran_object_copies(
            [copies[robot_id] for robot_id in sorted(observers)]
        )
    _, validated = _validated_state(graph, state)
    return validated


def read_sesync_state(path: Path, graph: CanonicalGraph) -> dict[int, SE3]:
    try:
        dense = np.loadtxt(path)
    except (OSError, ValueError) as error:
        raise EvaluationError(f"cannot read SE-Sync state: {path}") from error
    expected_count = len(graph.vertices)
    if dense.ndim != 2:
        raise EvaluationError("SE-Sync state must be a dense matrix")
    if dense.shape == (3, 4 * expected_count):
        translations = dense[:, :expected_count]
        rotations = [
            dense[:, expected_count + 3 * index : expected_count + 3 * (index + 1)]
            for index in range(expected_count)
        ]
    elif dense.shape == (4 * expected_count, 3):
        translations = dense[:expected_count, :].T
        rotations = [
            dense[
                expected_count + 3 * index : expected_count + 3 * (index + 1),
                :,
            ].T
            for index in range(expected_count)
        ]
    else:
        raise EvaluationError(
            f"SE-Sync state shape {dense.shape} does not match {expected_count} variables"
        )
    state = {
        variable_id: SE3(
            _project_rotation(rotations[variable_id]),
            translations[:, variable_id],
        )
        for variable_id in range(expected_count)
    }
    _, validated = _validated_state(graph, state)
    return validated


def _read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter=delimiter))
    except OSError as error:
        raise EvaluationError(f"cannot read iteration log: {path}") from error
    if not rows:
        raise EvaluationError(f"iteration log is empty: {path}")
    return rows


def _integer(row: Mapping[str, str], name: str, context: str) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"invalid {name} in {context}") from error
    if value < 0:
        raise EvaluationError(f"invalid {name} in {context}")
    return value


def _number(row: Mapping[str, str], name: str, context: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"invalid {name} in {context}") from error
    if not math.isfinite(value):
        raise EvaluationError(f"non-finite {name} in {context}")
    return value


def _rows_by_iteration(
    rows: Sequence[Mapping[str, str]],
    *,
    field: str,
    one_based_offset: int,
    context: str,
) -> dict[int, Mapping[str, str]]:
    result = {}
    for row in rows:
        iteration = _integer(row, field, context) + one_based_offset
        if iteration in result:
            raise EvaluationError(f"duplicate iteration {iteration} in {context}")
        result[iteration] = row
    return result


def _iteration_record(
    *,
    method: str,
    iteration: int,
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    native_objective: Optional[float],
    logical_bytes: int,
    payload_blocks: int,
    messages: int,
    solver_wall_time_s: float,
) -> IterationRecord:
    external = evaluate_state(graph, state)
    if not external.finite:
        raise EvaluationError(f"non-finite external evaluation at iteration {iteration}")
    return IterationRecord(
        method=method,
        iteration=iteration,
        external_cost=external.cost,
        external_gradient_norm=external.gradient_norm,
        native_objective=native_objective,
        logical_bytes=logical_bytes,
        payload_blocks=payload_blocks,
        messages=messages,
        solver_wall_time_s=solver_wall_time_s,
    )


def parse_larpg_iterations(
    path: Path,
    graph: CanonicalGraph,
    states_by_iteration: Mapping[int, Mapping[int, SE3]],
) -> tuple[IterationRecord, ...]:
    rows = _rows_by_iteration(
        _read_rows(path, ","),
        field="iteration",
        one_based_offset=0,
        context="LARPG iterations",
    )
    records = []
    for iteration in sorted(states_by_iteration):
        if iteration not in rows:
            raise EvaluationError(f"LARPG is missing iteration {iteration}")
        row = rows[iteration]
        schur_blocks = _integer(
            row, "uploaded_schur_blocks", "LARPG iterations"
        )
        gradient_blocks = _integer(
            row, "uploaded_gradient_blocks", "LARPG iterations"
        )
        broadcast_blocks = _integer(
            row, "broadcast_object_blocks", "LARPG iterations"
        )
        upload_bytes = _integer(row, "logical_upload_bytes", "LARPG iterations")
        broadcast_bytes = _integer(
            row, "logical_broadcast_bytes", "LARPG iterations"
        )
        if gradient_blocks != schur_blocks or upload_bytes != 42 * 8 * schur_blocks:
            raise EvaluationError(
                f"LARPG upload accounting mismatch at iteration {iteration}"
            )
        if broadcast_bytes != 6 * 8 * broadcast_blocks:
            raise EvaluationError(
                f"LARPG broadcast accounting mismatch at iteration {iteration}"
            )
        records.append(
            _iteration_record(
                method="larpg",
                iteration=iteration,
                graph=graph,
                state=states_by_iteration[iteration],
                native_objective=_number(
                    row, "objective_after", "LARPG iterations"
                ),
                logical_bytes=upload_bytes + broadcast_bytes,
                payload_blocks=schur_blocks + gradient_blocks + broadcast_blocks,
                messages=schur_blocks + broadcast_blocks,
                solver_wall_time_s=_number(
                    row, "wall_seconds", "LARPG iterations"
                ),
            )
        )
    return tuple(records)


def parse_dran_iterations(
    path: Path,
    graph: CanonicalGraph,
    states_by_iteration: Mapping[int, Mapping[int, SE3]],
) -> tuple[IterationRecord, ...]:
    rows = _rows_by_iteration(
        _read_rows(path, ","),
        field="iter",
        one_based_offset=1,
        context="DRAN iterations",
    )
    records = []
    for iteration in sorted(states_by_iteration):
        if iteration not in rows:
            raise EvaluationError(f"DRAN is missing iteration {iteration}")
        row = rows[iteration]
        sent = _integer(row, "object_comm_poses", "DRAN iterations")
        expected = _integer(
            row, "expected_full_object_comm_poses", "DRAN iterations"
        )
        if sent != expected:
            raise EvaluationError(
                f"DRAN full communication mismatch at iteration {iteration}: "
                f"expected {expected}, observed {sent}"
            )
        logical_bytes = int(
            round(_number(row, "comm_mb", "DRAN iterations") * 1024 * 1024)
        )
        records.append(
            _iteration_record(
                method="dran",
                iteration=iteration,
                graph=graph,
                state=states_by_iteration[iteration],
                native_objective=_number(
                    row, "chordal_measurement_cost", "DRAN iterations"
                ),
                logical_bytes=logical_bytes,
                payload_blocks=_integer(
                    row, "object_comm_payload_blocks", "DRAN iterations"
                ),
                messages=sent,
                solver_wall_time_s=_number(
                    row, "iter_time_sec", "DRAN iterations"
                ),
            )
        )
    return tuple(records)


def parse_distributed_mapper_iterations(
    loss_path: Path,
    communication_path: Path,
    graph: CanonicalGraph,
    states_by_iteration: Mapping[int, Mapping[int, SE3]],
    *,
    solver_wall_times: Optional[Mapping[int, float]] = None,
) -> tuple[IterationRecord, ...]:
    loss_rows = _rows_by_iteration(
        _read_rows(loss_path, "\t"),
        field="iter",
        one_based_offset=1,
        context="Distributed Mapper loss",
    )
    communication_rows = _read_rows(communication_path, "\t")
    communication: dict[int, tuple[int, int]] = {}
    for row in communication_rows:
        iteration = _integer(
            row, "iter", "Distributed Mapper communication"
        ) + 1
        old_messages, old_bytes = communication.get(iteration, (0, 0))
        communication[iteration] = (
            old_messages
            + _integer(row, "messages", "Distributed Mapper communication"),
            old_bytes + _integer(row, "bytes", "Distributed Mapper communication"),
        )
    records = []
    wall_times = solver_wall_times or {}
    for iteration in sorted(states_by_iteration):
        if iteration not in loss_rows:
            raise EvaluationError(
                f"Distributed Mapper is missing iteration {iteration}"
            )
        row = loss_rows[iteration]
        logged_blocks = _integer(
            row, "iter_comm_poses", "Distributed Mapper loss"
        )
        messages, logical_bytes = communication.get(iteration, (0, 0))
        if logged_blocks != messages:
            raise EvaluationError(
                "Distributed Mapper communication accounting mismatch at "
                f"iteration {iteration}"
            )
        records.append(
            _iteration_record(
                method="distributed_mapper",
                iteration=iteration,
                graph=graph,
                state=states_by_iteration[iteration],
                native_objective=_number(
                    row, "cost", "Distributed Mapper loss"
                ),
                logical_bytes=logical_bytes,
                payload_blocks=logged_blocks,
                messages=messages,
                solver_wall_time_s=float(wall_times.get(iteration, 0.0)),
            )
        )
    return tuple(records)


def require_checkpoints(
    records: Sequence[IterationRecord], checkpoints: Sequence[int]
) -> tuple[IterationRecord, ...]:
    by_iteration = {}
    for record in records:
        if record.iteration in by_iteration:
            raise EvaluationError(f"duplicate checkpoint {record.iteration}")
        by_iteration[record.iteration] = record
    missing = [checkpoint for checkpoint in checkpoints if checkpoint not in by_iteration]
    if missing:
        details = "; ".join(f"checkpoint {item}" for item in missing)
        raise EvaluationError(f"missing {details}")
    return tuple(by_iteration[checkpoint] for checkpoint in checkpoints)


def detect_convergence(
    records: Sequence[IterationRecord],
    *,
    relative_cost_tolerance: float,
    gradient_tolerance: float,
    consecutive: int,
) -> ConvergenceResult:
    if not records:
        raise EvaluationError("convergence requires at least one iteration")
    if (
        not math.isfinite(relative_cost_tolerance)
        or relative_cost_tolerance < 0.0
        or not math.isfinite(gradient_tolerance)
        or gradient_tolerance < 0.0
        or isinstance(consecutive, bool)
        or consecutive < 1
    ):
        raise EvaluationError("invalid convergence thresholds")
    ordered = sorted(records, key=lambda item: item.iteration)
    if len({record.iteration for record in ordered}) != len(ordered):
        raise EvaluationError("convergence records contain duplicate iterations")
    streak = 0
    previous = ordered[0]
    for record in ordered[1:]:
        relative_change = abs(record.external_cost - previous.external_cost) / max(
            abs(previous.external_cost), 1e-12
        )
        if (
            relative_change <= relative_cost_tolerance
            and record.external_gradient_norm <= gradient_tolerance
        ):
            streak += 1
        else:
            streak = 0
        if streak >= consecutive:
            return ConvergenceResult(True, record.iteration, ordered[-1].iteration)
        previous = record
    return ConvergenceResult(False, None, ordered[-1].iteration)


__all__ = [
    "ConvergenceResult",
    "EvaluationError",
    "IterationRecord",
    "ReferenceMetrics",
    "StateEvaluation",
    "align_state_to_reference",
    "alignment_to_reference",
    "detect_convergence",
    "evaluate_state",
    "parse_distributed_mapper_iterations",
    "parse_dran_iterations",
    "parse_larpg_iterations",
    "read_distributed_mapper_state",
    "read_dran_state",
    "read_g2o_state",
    "read_larpg_state",
    "read_sesync_state",
    "reference_metrics",
    "relative_cost_gap",
    "require_checkpoints",
]
