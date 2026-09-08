"""SE(3) conventions shared by all SlideSLAM benchmark adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = result.copy()
    result.setflags(write=False)
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} precision must be finite and positive")
    return result


@dataclass(frozen=True, eq=False)
class SE3:
    """World-from-local rigid transform with an active rotation."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("SE3 rotation must have shape (3, 3)")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("SE3 rotation must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-9):
            raise ValueError("SE3 rotation must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not math.isfinite(determinant) or abs(determinant - 1.0) > 1e-9:
            raise ValueError("SE3 rotation determinant must equal one")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        translation = _finite_vector(self.translation, 3, "SE3 translation")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SE3):
            return NotImplemented
        return bool(
            np.array_equal(self.rotation, other.rotation)
            and np.array_equal(self.translation, other.translation)
        )

    @staticmethod
    def identity() -> "SE3":
        return SE3(np.eye(3), np.zeros(3))

    @staticmethod
    def from_rotvec(rotvec: Sequence[float], translation: Sequence[float]) -> "SE3":
        vector = _finite_vector(rotvec, 3, "rotation vector")
        return SE3(Rotation.from_rotvec(np.array(vector, copy=True)).as_matrix(), translation)

    @staticmethod
    def from_quaternion_xyzw(
        quaternion: Sequence[float], translation: Sequence[float]
    ) -> "SE3":
        value = _finite_vector(quaternion, 4, "quaternion")
        norm = float(np.linalg.norm(value))
        if norm <= np.finfo(float).eps:
            raise ValueError("quaternion norm must be positive")
        value = value / norm
        return SE3(Rotation.from_quat(np.array(value, copy=True)).as_matrix(), translation)

    @staticmethod
    def from_matrix(matrix: Sequence[Sequence[float]]) -> "SE3":
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("SE3 homogeneous matrix must be finite and 4 by 4")
        if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
            raise ValueError("SE3 homogeneous matrix has an invalid final row")
        return SE3(value[:3, :3], value[:3, 3])

    def quaternion_xyzw(self) -> np.ndarray:
        quaternion = Rotation.from_matrix(np.array(self.rotation, copy=True)).as_quat()
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        elif quaternion[3] == 0.0:
            nonzero = np.flatnonzero(quaternion[:3])
            if len(nonzero) and quaternion[nonzero[0]] < 0.0:
                quaternion = -quaternion
        quaternion[np.abs(quaternion) < 1e-16] = 0.0
        return quaternion

    def as_matrix(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.translation
        return result

    def compose(self, other: "SE3") -> "SE3":
        return SE3(
            self.rotation @ other.rotation,
            self.translation + self.rotation @ other.translation,
        )

    def inverse(self) -> "SE3":
        inverse_rotation = self.rotation.T
        return SE3(inverse_rotation, -(inverse_rotation @ self.translation))

    def between(self, other: "SE3") -> "SE3":
        return self.inverse().compose(other)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "translation": self.translation.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw().tolist(),
        }

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> "SE3":
        if not isinstance(value, Mapping):
            raise ValueError("SE3 JSON value must be an object")
        return SE3.from_quaternion_xyzw(
            value.get("quaternion_xyzw", ()), value.get("translation", ())
        )


def skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = _finite_vector(vector, 3, "skew vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_angle(rotation: Sequence[Sequence[float]]) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation angle input must be a finite 3 by 3 matrix")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def retract_pose(pose: SE3, delta: Sequence[float]) -> SE3:
    """Retract world-frame translation and right-local rotation increments."""

    value = _finite_vector(delta, 6, "SE3 increment")
    increment_rotation = Rotation.from_rotvec(np.array(value[3:], copy=True)).as_matrix()
    return SE3(pose.rotation @ increment_rotation, pose.translation + value[:3])


def _validate_precisions(kappa: float, tau: float) -> tuple[float, float]:
    return _finite_positive(kappa, "rotation"), _finite_positive(tau, "translation")


def chordal_residual(
    x_i: SE3, x_j: SE3, measurement_ij: SE3, kappa: float, tau: float
) -> np.ndarray:
    """Return nine row-major rotation and three translation residual entries."""

    kappa, tau = _validate_precisions(kappa, tau)
    rotation = x_j.rotation - x_i.rotation @ measurement_ij.rotation
    translation = (
        x_j.translation
        - x_i.translation
        - x_i.rotation @ measurement_ij.translation
    )
    return np.concatenate(
        [math.sqrt(kappa) * rotation.reshape(9), math.sqrt(tau) * translation]
    )


def chordal_jacobians(
    x_i: SE3, x_j: SE3, measurement_ij: SE3, kappa: float, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Jacobians for :func:`retract_pose` increments."""

    kappa, tau = _validate_precisions(kappa, tau)
    root_kappa = math.sqrt(kappa)
    root_tau = math.sqrt(tau)
    jacobian_i = np.zeros((12, 6))
    jacobian_j = np.zeros((12, 6))
    jacobian_i[9:, :3] = -root_tau * np.eye(3)
    jacobian_j[9:, :3] = root_tau * np.eye(3)
    for column in range(3):
        basis = np.zeros(3)
        basis[column] = 1.0
        generator = skew(basis)
        jacobian_i[:9, 3 + column] = (
            -root_kappa
            * (x_i.rotation @ generator @ measurement_ij.rotation).reshape(9)
        )
        jacobian_j[:9, 3 + column] = (
            root_kappa * (x_j.rotation @ generator).reshape(9)
        )
        jacobian_i[9:, 3 + column] = (
            -root_tau * x_i.rotation @ generator @ measurement_ij.translation
        )
    return jacobian_i, jacobian_j


def g2o_information(kappa: float, tau: float) -> np.ndarray:
    """Map the common chordal scalar precisions to g2o Pose3 ordering."""

    kappa, tau = _validate_precisions(kappa, tau)
    return np.diag([tau, tau, tau, 2.0 * kappa, 2.0 * kappa, 2.0 * kappa])


def _sample_bounded_ball(rng: np.random.Generator, radius: float) -> np.ndarray:
    if radius == 0.0:
        return np.zeros(3)
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    while norm <= np.finfo(float).eps:
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
    return radius * float(rng.random()) ** (1.0 / 3.0) * direction / norm


def perturb_pose(
    pose: SE3,
    variable_id: int,
    seed: int,
    translation_bound_m: float = 0.1,
    rotation_bound_deg: float = 3.0,
) -> SE3:
    """Apply a deterministic bounded perturbation shared by every backend."""

    if isinstance(variable_id, bool) or not isinstance(variable_id, int) or variable_id < 0:
        raise ValueError("variable_id must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    translation_bound_m = float(translation_bound_m)
    rotation_bound_deg = float(rotation_bound_deg)
    if (
        not math.isfinite(translation_bound_m)
        or translation_bound_m < 0.0
        or not math.isfinite(rotation_bound_deg)
        or rotation_bound_deg < 0.0
    ):
        raise ValueError("perturbation bounds must be finite and non-negative")
    if seed == 0:
        return SE3(pose.rotation, pose.translation)
    rng = np.random.default_rng(np.random.SeedSequence([seed, variable_id]))
    translation = _sample_bounded_ball(rng, translation_bound_m)
    rotvec = _sample_bounded_ball(rng, math.radians(rotation_bound_deg))
    return SE3(
        pose.rotation @ Rotation.from_rotvec(rotvec).as_matrix(),
        pose.translation + translation,
    )


__all__ = [
    "SE3",
    "chordal_jacobians",
    "chordal_residual",
    "g2o_information",
    "perturb_pose",
    "retract_pose",
    "rotation_angle",
    "skew",
]
