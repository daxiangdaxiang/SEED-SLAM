"""Independent Point2 audit of the MR.CLAM common chordal objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from benchmarks.slideslam.materialize import CanonicalGraph
from benchmarks.slideslam.schema import OdometryFactor, PoseVertex
from benchmarks.slideslam.se3 import SE3


@dataclass(frozen=True)
class PlanarFactorCost:
    factor_id: str
    kind: str
    rotation_cost: float
    translation_cost: float

    @property
    def total_cost(self) -> float:
        return self.rotation_cost + self.translation_cost


@dataclass(frozen=True)
class PlanarCostAudit:
    raw_cost: float
    mean_factor_cost: float
    odometry_cost: float
    observation_cost: float
    factor_count: int
    odometry_factor_count: int
    observation_factor_count: int
    factor_costs: tuple[PlanarFactorCost, ...]


def _yaw(rotation: np.ndarray, *, context: str) -> float:
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    expected = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if not np.allclose(rotation, expected, rtol=0.0, atol=1e-7):
        raise ValueError(f"{context} is not a planar z-axis rotation")
    return yaw


def _validate_state(graph: CanonicalGraph, state: Mapping[int, SE3]) -> None:
    expected = {vertex.global_id for vertex in graph.vertices}
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"state ID mismatch: missing={missing}, extra={extra}")
    for variable_id, pose in state.items():
        if not isinstance(pose, SE3):
            raise TypeError(f"state ID {variable_id} is not an SE3 value")


def _project_point2(pose: SE3) -> SE3:
    yaw = math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0]))
    return SE3(
        np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([pose.translation[0], pose.translation[1], 0.0]),
    )


def _cost_audit(
    graph: CanonicalGraph,
    state: Mapping[int, SE3],
    *,
    project_to_point2: bool,
) -> PlanarCostAudit:
    _validate_state(graph, state)
    if project_to_point2:
        fixed_poses = [
            vertex
            for vertex in graph.vertices
            if isinstance(vertex, PoseVertex) and vertex.fixed
        ]
        if len(fixed_poses) != 1:
            raise ValueError("Point2 cost audit requires exactly one fixed root")
        anchor = fixed_poses[0]
        world_from_solver = anchor.initial_pose.compose(
            state[anchor.global_id].inverse()
        )
        values = {
            variable_id: _project_point2(world_from_solver.compose(pose))
            for variable_id, pose in state.items()
        }
    else:
        values = state
    factor_costs = []
    odometry_cost = 0.0
    observation_cost = 0.0
    odometry_count = 0
    observation_count = 0
    for factor in graph.factors:
        pose_from = values[factor.from_id]
        pose_to = values[factor.to_id]
        measurement = (
            _project_point2(factor.measurement)
            if project_to_point2
            else factor.measurement
        )
        rotation_error = (
            pose_to.rotation - pose_from.rotation @ measurement.rotation
        )
        rotation_cost = factor.kappa * float(
            np.sum(rotation_error * rotation_error)
        )
        translation_error = (
            pose_to.translation
            - pose_from.translation
            - pose_from.rotation @ measurement.translation
        )
        translation_cost = factor.tau * float(translation_error @ translation_error)
        item = PlanarFactorCost(
            factor_id=factor.factor_id,
            kind=factor.kind,
            rotation_cost=rotation_cost,
            translation_cost=translation_cost,
        )
        factor_costs.append(item)
        if isinstance(factor, OdometryFactor):
            odometry_cost += item.total_cost
            odometry_count += 1
        else:
            observation_cost += item.total_cost
            observation_count += 1

    factor_count = len(factor_costs)
    raw_cost = odometry_cost + observation_cost
    return PlanarCostAudit(
        raw_cost=raw_cost,
        mean_factor_cost=raw_cost / factor_count if factor_count else 0.0,
        odometry_cost=odometry_cost,
        observation_cost=observation_cost,
        factor_count=factor_count,
        odometry_factor_count=odometry_count,
        observation_factor_count=observation_count,
        factor_costs=tuple(factor_costs),
    )


def independent_planar_cost_audit(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> PlanarCostAudit:
    """Project solver output to Point2 and independently recompute the cost.

    For planar rotations, ``||R_j - R_i R_ij||_F^2`` is exactly
    ``4 (1 - cos(theta_j - theta_i - theta_ij))``. Projection prevents an
    SE(3)-capable baseline from exploiting roll, pitch, or vertical motion that
    is absent from the MR.CLAM measurement model.
    """

    return _cost_audit(graph, state, project_to_point2=True)


def independent_embedded_cost_audit(
    graph: CanonicalGraph, state: Mapping[int, SE3]
) -> PlanarCostAudit:
    """Independently recompute the existing full SE(3)-embedded objective."""

    return _cost_audit(graph, state, project_to_point2=False)


__all__ = [
    "PlanarCostAudit",
    "PlanarFactorCost",
    "independent_embedded_cost_audit",
    "independent_planar_cost_audit",
]
