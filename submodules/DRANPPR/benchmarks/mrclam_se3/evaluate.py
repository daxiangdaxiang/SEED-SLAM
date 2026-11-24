"""Common physical and objective evaluator for the derived Pose3 benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from benchmarks.slideslam.evaluate import StateEvaluation, evaluate_state
from benchmarks.slideslam.materialize import CanonicalGraph
from benchmarks.slideslam.schema import ObjectVertex, PoseVertex
from benchmarks.slideslam.se3 import SE3, rotation_angle


class MRCLAMSE3EvaluationError(ValueError):
  """Raised when a state or planted reference cannot be evaluated fairly."""


@dataclass(frozen=True)
class PhysicalMetrics:
  robot_translation_rmse_m: float
  robot_rotation_rmse_deg: float
  object_translation_rmse_m: float
  object_rotation_rmse_deg: float
  robot_pose_count: int
  object_count: int
  truth_from_estimate: SE3


@dataclass(frozen=True)
class BenchmarkEvaluation:
  total_cost: float
  mean_factor_cost: float
  physical: PhysicalMetrics
  objective: StateEvaluation


def _validate_state(
  graph: CanonicalGraph,
  state: Mapping[int, SE3],
  reference: Mapping[int, SE3],
) -> None:
  expected = {vertex.global_id for vertex in graph.vertices}
  if set(state) != expected or set(reference) != expected:
    raise MRCLAMSE3EvaluationError(
      "state and planted reference must cover every canonical variable"
    )
  if any(
    not isinstance(state[variable_id], SE3)
    or not isinstance(reference[variable_id], SE3)
    for variable_id in expected
  ):
    raise MRCLAMSE3EvaluationError("state and reference values must be SE3 poses")
  if graph.metadata.get("ground_truth_used_for_initialization") is not False:
    raise MRCLAMSE3EvaluationError(
      "canonical provenance does not prove factor-only initialization"
    )


def _fit_truth_from_estimate(
  estimated_positions: np.ndarray, truth_positions: np.ndarray
) -> SE3:
  if (
    estimated_positions.shape != truth_positions.shape
    or estimated_positions.ndim != 2
    or estimated_positions.shape[1] != 3
    or estimated_positions.shape[0] < 4
    or not np.all(np.isfinite(estimated_positions))
    or not np.all(np.isfinite(truth_positions))
  ):
    raise MRCLAMSE3EvaluationError(
      "SE3 alignment requires at least four finite 3D robot positions"
    )
  estimate_center = np.mean(estimated_positions, axis=0)
  truth_center = np.mean(truth_positions, axis=0)
  estimate_zero = estimated_positions - estimate_center
  truth_zero = truth_positions - truth_center
  singular_values = np.linalg.svd(estimate_zero, compute_uv=False)
  tolerance = max(estimate_zero.shape) * np.finfo(float).eps * singular_values[0]
  if np.count_nonzero(singular_values > tolerance) < 3:
    raise MRCLAMSE3EvaluationError(
      "robot positions are rank deficient for one global SE3 alignment"
    )
  u, _, vt = np.linalg.svd(estimate_zero.T @ truth_zero)
  rotation = vt.T @ u.T
  if np.linalg.det(rotation) < 0.0:
    vt[-1, :] *= -1.0
    rotation = vt.T @ u.T
  translation = truth_center - rotation @ estimate_center
  return SE3(rotation, translation)


def _rmse(values: list[float]) -> float:
  if not values:
    raise MRCLAMSE3EvaluationError("cannot evaluate an empty variable class")
  return math.sqrt(float(np.mean(np.square(values))))


def physical_metrics(
  graph: CanonicalGraph,
  state: Mapping[int, SE3],
  planted_reference: Mapping[int, SE3],
) -> PhysicalMetrics:
  """Evaluate robots and objects using one alignment fitted from robots only."""

  _validate_state(graph, state, planted_reference)
  pose_vertices = sorted(
    (item for item in graph.vertices if isinstance(item, PoseVertex)),
    key=lambda item: item.global_id,
  )
  object_vertices = sorted(
    (item for item in graph.vertices if isinstance(item, ObjectVertex)),
    key=lambda item: item.global_id,
  )
  alignment = _fit_truth_from_estimate(
    np.vstack([state[item.global_id].translation for item in pose_vertices]),
    np.vstack(
      [planted_reference[item.global_id].translation for item in pose_vertices]
    ),
  )

  robot_translation_errors = []
  robot_rotation_errors = []
  for vertex in pose_vertices:
    aligned = alignment.compose(state[vertex.global_id])
    truth = planted_reference[vertex.global_id]
    robot_translation_errors.append(
      float(np.linalg.norm(aligned.translation - truth.translation))
    )
    robot_rotation_errors.append(
      math.degrees(rotation_angle(truth.rotation.T @ aligned.rotation))
    )
  object_translation_errors = []
  object_rotation_errors = []
  for vertex in object_vertices:
    aligned = alignment.compose(state[vertex.global_id])
    truth = planted_reference[vertex.global_id]
    object_translation_errors.append(
      float(np.linalg.norm(aligned.translation - truth.translation))
    )
    object_rotation_errors.append(
      math.degrees(rotation_angle(truth.rotation.T @ aligned.rotation))
    )
  return PhysicalMetrics(
    robot_translation_rmse_m=_rmse(robot_translation_errors),
    robot_rotation_rmse_deg=_rmse(robot_rotation_errors),
    object_translation_rmse_m=_rmse(object_translation_errors),
    object_rotation_rmse_deg=_rmse(object_rotation_errors),
    robot_pose_count=len(pose_vertices),
    object_count=len(object_vertices),
    truth_from_estimate=alignment,
  )


def evaluate_benchmark_state(
  graph: CanonicalGraph,
  state: Mapping[int, SE3],
  planted_reference: Mapping[int, SE3],
) -> BenchmarkEvaluation:
  if not graph.factors:
    raise MRCLAMSE3EvaluationError("mean factor cost requires at least one factor")
  objective = evaluate_state(graph, state)
  return BenchmarkEvaluation(
    total_cost=objective.cost,
    mean_factor_cost=objective.cost / len(graph.factors),
    physical=physical_metrics(graph, state, planted_reference),
    objective=objective,
  )


def read_planted_reference(path: str | Path) -> dict[int, SE3]:
  path = Path(path)
  result = {}
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except OSError as error:
    raise MRCLAMSE3EvaluationError(
      f"cannot read planted reference {path}: {error}"
    ) from error
  for line_number, line in enumerate(lines, start=1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
      if value.get("record_type") != "planted_reference":
        raise ValueError("invalid record type")
      variable_id = int(value["variable_id"])
      if variable_id < 0 or variable_id in result:
        raise ValueError("invalid or duplicate variable ID")
      if value.get("variable_kind") not in ("pose", "object"):
        raise ValueError("invalid variable kind")
      result[variable_id] = SE3.from_dict(value["pose"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
      raise MRCLAMSE3EvaluationError(
        f"invalid planted reference row at {path}:{line_number}"
      ) from error
  if not result:
    raise MRCLAMSE3EvaluationError("planted reference is empty")
  return dict(sorted(result.items()))


__all__ = [
  "BenchmarkEvaluation",
  "MRCLAMSE3EvaluationError",
  "PhysicalMetrics",
  "evaluate_benchmark_state",
  "physical_metrics",
  "read_planted_reference",
]
