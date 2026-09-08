"""Ground-truth-free common initialization for the synthetic Pose3 graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from benchmarks.slideslam.schema import (
  AssociationRecord,
  Factor,
  ObjectVertex,
  OdometryFactor,
  PoseObjectFactor,
  PoseVertex,
  Vertex,
)
from benchmarks.slideslam.se3 import SE3, rotation_angle


class MRCLAMSE3InitializationError(ValueError):
  """Raised when the Pose3 factors do not admit a common initialization."""


@dataclass(frozen=True)
class PairRegistration:
  target_robot_id: int
  source_robot_id: int
  target_from_source: SE3
  common_object_ids: tuple[int, ...]
  residual_rms: float


@dataclass(frozen=True)
class InitializationResult:
  states: Mapping[int, SE3]
  robot_alignments: Mapping[int, SE3]
  local_object_poses: Mapping[int, Mapping[int, SE3]]
  pair_registrations: tuple[PairRegistration, ...]
  spanning_tree: tuple[tuple[int, int], ...]


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
  u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
  rotation = u @ vt
  if np.linalg.det(rotation) < 0.0:
    u[:, -1] *= -1.0
    rotation = u @ vt
  return rotation


def _mean_pose(poses: Sequence[SE3]) -> SE3:
  if not poses:
    raise MRCLAMSE3InitializationError("cannot average an empty pose collection")
  rotation = _project_rotation(sum((pose.rotation for pose in poses), np.zeros((3, 3))))
  translation = np.mean([pose.translation for pose in poses], axis=0)
  return SE3(rotation, translation)


def _local_trajectories(
  pose_vertices: Sequence[PoseVertex], odometry_factors: Sequence[OdometryFactor]
) -> dict[int, dict[int, SE3]]:
  poses_by_robot: dict[int, list[PoseVertex]] = {}
  pose_owner = {}
  for vertex in pose_vertices:
    poses_by_robot.setdefault(vertex.robot_id, []).append(vertex)
    pose_owner[vertex.global_id] = vertex.robot_id

  adjacency: dict[int, list[tuple[int, SE3]]] = {
    vertex.global_id: [] for vertex in pose_vertices
  }
  for factor in odometry_factors:
    if factor.from_id not in pose_owner or factor.to_id not in pose_owner:
      raise MRCLAMSE3InitializationError(
        f"odometry factor {factor.factor_id} references a non-pose vertex"
      )
    if pose_owner[factor.from_id] != pose_owner[factor.to_id]:
      raise MRCLAMSE3InitializationError(
        f"odometry factor {factor.factor_id} crosses robot ownership"
      )
    adjacency[factor.from_id].append((factor.to_id, factor.measurement))
    adjacency[factor.to_id].append((factor.from_id, factor.measurement.inverse()))

  result = {}
  for robot_id, vertices in sorted(poses_by_robot.items()):
    ordered = sorted(vertices, key=lambda item: (item.local_index, item.global_id))
    if len({item.local_index for item in ordered}) != len(ordered):
      raise MRCLAMSE3InitializationError(
        f"robot {robot_id} has duplicate local pose indices"
      )
    local_states = {ordered[0].global_id: SE3.identity()}
    frontier = [ordered[0].global_id]
    while frontier:
      current = frontier.pop(0)
      for neighbor, current_from_neighbor in sorted(
        adjacency[current], key=lambda item: item[0]
      ):
        if neighbor in local_states:
          continue
        local_states[neighbor] = local_states[current].compose(current_from_neighbor)
        frontier.append(neighbor)
    expected = {item.global_id for item in ordered}
    if set(local_states) != expected:
      raise MRCLAMSE3InitializationError(
        f"odometry graph for robot {robot_id} is disconnected"
      )
    result[robot_id] = local_states
  return result


def _local_object_maps(
  object_vertices: Sequence[ObjectVertex],
  pose_object_factors: Sequence[PoseObjectFactor],
  trajectories: Mapping[int, Mapping[int, SE3]],
) -> dict[int, dict[int, SE3]]:
  pose_owner = {
    pose_id: robot_id
    for robot_id, states in trajectories.items()
    for pose_id in states
  }
  object_ids = {vertex.global_id for vertex in object_vertices}
  samples: dict[int, dict[int, list[SE3]]] = {}
  for factor in pose_object_factors:
    if factor.from_id not in pose_owner or factor.to_id not in object_ids:
      raise MRCLAMSE3InitializationError(
        f"pose-object factor {factor.factor_id} has invalid endpoint types"
      )
    robot_id = pose_owner[factor.from_id]
    sample = trajectories[robot_id][factor.from_id].compose(factor.measurement)
    samples.setdefault(robot_id, {}).setdefault(factor.to_id, []).append(sample)
  return {
    robot_id: {
      object_id: _mean_pose(object_samples)
      for object_id, object_samples in sorted(robot_samples.items())
    }
    for robot_id, robot_samples in sorted(samples.items())
  }


def _pair_registration(
  target_robot: int,
  source_robot: int,
  local_maps: Mapping[int, Mapping[int, SE3]],
) -> PairRegistration | None:
  common = tuple(
    sorted(set(local_maps.get(target_robot, {})) & set(local_maps.get(source_robot, {})))
  )
  if not common:
    return None
  target_poses = local_maps[target_robot]
  source_poses = local_maps[source_robot]
  rotation = _project_rotation(
    sum(
      (
        target_poses[object_id].rotation
        @ source_poses[object_id].rotation.T
        for object_id in common
      ),
      np.zeros((3, 3)),
    )
  )
  translation = np.mean(
    [
      target_poses[object_id].translation
      - rotation @ source_poses[object_id].translation
      for object_id in common
    ],
    axis=0,
  )
  transform = SE3(rotation, translation)
  squared_residuals = []
  for object_id in common:
    predicted = transform.compose(source_poses[object_id])
    translation_error = np.linalg.norm(
      predicted.translation - target_poses[object_id].translation
    )
    rotation_error = rotation_angle(
      predicted.rotation.T @ target_poses[object_id].rotation
    )
    squared_residuals.append(translation_error**2 + rotation_error**2)
  return PairRegistration(
    target_robot,
    source_robot,
    transform,
    common,
    math.sqrt(float(np.mean(squared_residuals))),
  )


def initialize_from_factors(
  vertices: Sequence[Vertex],
  factors: Sequence[Factor],
  associations: Sequence[AssociationRecord],
  *,
  root_robot_id: int,
) -> InitializationResult:
  """Initialize every Pose3 variable using only factors and fixed associations."""

  pose_vertices = tuple(item for item in vertices if isinstance(item, PoseVertex))
  object_vertices = tuple(item for item in vertices if isinstance(item, ObjectVertex))
  odometry = tuple(item for item in factors if isinstance(item, OdometryFactor))
  observations = tuple(item for item in factors if isinstance(item, PoseObjectFactor))
  if len(pose_vertices) + len(object_vertices) != len(vertices):
    raise MRCLAMSE3InitializationError("unsupported vertex type")
  if len(odometry) + len(observations) != len(factors):
    raise MRCLAMSE3InitializationError("unsupported factor type")

  robot_ids = tuple(sorted({item.robot_id for item in pose_vertices}))
  if not robot_ids or root_robot_id not in robot_ids:
    raise MRCLAMSE3InitializationError("root robot is missing from the graph")
  object_ids = {item.global_id for item in object_vertices}
  association_pairs = {
    (item.robot_id, item.global_object_id) for item in associations
  }
  observed_pairs = {
    (pose.robot_id, factor.to_id)
    for factor in observations
    for pose in pose_vertices
    if pose.global_id == factor.from_id
  }
  if observed_pairs - association_pairs:
    raise MRCLAMSE3InitializationError(
      "fixed associations do not cover every pose-object factor"
    )
  if any(object_id not in object_ids for _, object_id in association_pairs):
    raise MRCLAMSE3InitializationError("association references a missing object")

  trajectories = _local_trajectories(pose_vertices, odometry)
  local_maps = _local_object_maps(object_vertices, observations, trajectories)
  registrations = []
  adjacency: dict[int, list[tuple[int, SE3, int, float]]] = {
    robot_id: [] for robot_id in robot_ids
  }
  for target_index, target_robot in enumerate(robot_ids):
    for source_robot in robot_ids[target_index + 1 :]:
      registration = _pair_registration(target_robot, source_robot, local_maps)
      if registration is None:
        continue
      registrations.append(registration)
      support = len(registration.common_object_ids)
      adjacency[target_robot].append(
        (
          source_robot,
          registration.target_from_source,
          support,
          registration.residual_rms,
        )
      )
      adjacency[source_robot].append(
        (
          target_robot,
          registration.target_from_source.inverse(),
          support,
          registration.residual_rms,
        )
      )

  alignments = {root_robot_id: SE3.identity()}
  spanning_tree = []
  while len(alignments) < len(robot_ids):
    candidates = []
    for parent in sorted(alignments):
      for child, parent_from_child, support, residual in adjacency[parent]:
        if child not in alignments:
          stable_residual = 0.0 if residual < 1e-7 else residual
          candidates.append(
            (-support, stable_residual, parent, child, parent_from_child)
          )
    if not candidates:
      raise MRCLAMSE3InitializationError(
        "shared-object registration graph does not connect all robots"
      )
    _, _, parent, child, parent_from_child = min(
      candidates, key=lambda item: item[:4]
    )
    alignments[child] = alignments[parent].compose(parent_from_child)
    spanning_tree.append((parent, child))

  states = {}
  for robot_id, local_states in trajectories.items():
    for variable_id, local_pose in local_states.items():
      states[variable_id] = alignments[robot_id].compose(local_pose)
  for object_id in sorted(object_ids):
    object_samples = [
      alignments[robot_id].compose(robot_map[object_id])
      for robot_id, robot_map in local_maps.items()
      if object_id in robot_map
    ]
    if not object_samples:
      raise MRCLAMSE3InitializationError(
        f"object {object_id} has no observation factor"
      )
    states[object_id] = _mean_pose(object_samples)

  if set(states) != {item.global_id for item in vertices}:
    raise MRCLAMSE3InitializationError("initializer did not produce every state")
  return InitializationResult(
    states=dict(sorted(states.items())),
    robot_alignments=dict(sorted(alignments.items())),
    local_object_poses={
      robot_id: dict(sorted(robot_map.items()))
      for robot_id, robot_map in sorted(local_maps.items())
    },
    pair_registrations=tuple(
      sorted(
        registrations,
        key=lambda item: (item.target_robot_id, item.source_robot_id),
      )
    ),
    spanning_tree=tuple(spanning_tree),
  )


__all__ = [
  "InitializationResult",
  "MRCLAMSE3InitializationError",
  "PairRegistration",
  "initialize_from_factors",
]
