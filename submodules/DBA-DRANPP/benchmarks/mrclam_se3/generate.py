"""Pose3 lift preserving MR.CLAM planar residual coordinates."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.slideslam.schema import OdometryFactor, PoseObjectFactor
from benchmarks.slideslam.se3 import SE3, skew

from .config import GenerationConfig, MRCLAMSE3ConfigError, nonplanar_profile


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise MRCLAMSE3ConfigError(f"{name} must be a finite {size}-vector")
    return result


def _left_jacobian_so3(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    generator = skew(rotvec)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * generator + (1.0 / 6.0) * generator @ generator
    theta2 = theta * theta
    return (
        np.eye(3)
        + (1.0 - math.cos(theta)) / theta2 * generator
        + (theta - math.sin(theta)) / (theta2 * theta) * generator @ generator
    )


def _left_jacobian_so3_inverse(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    generator = skew(rotvec)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * generator + (1.0 / 12.0) * generator @ generator
    coefficient = (
        1.0 / (theta * theta)
        - (1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta))
    )
    return np.eye(3) - 0.5 * generator + coefficient * generator @ generator


def se3_exp(tangent: Sequence[float]) -> SE3:
    """Exponential map with tangent order tx,ty,tz,rx,ry,rz."""

    value = _vector(tangent, 6, "SE3 tangent")
    translation = _left_jacobian_so3(value[3:]) @ value[:3]
    return SE3(Rotation.from_rotvec(value[3:]).as_matrix(), translation)


def se3_log(pose: SE3) -> np.ndarray:
    """Logarithm map inverse to :func:`se3_exp` near the identity."""

    rotvec = Rotation.from_matrix(np.array(pose.rotation, copy=True)).as_rotvec()
    translation = _left_jacobian_so3_inverse(rotvec) @ pose.translation
    return np.concatenate([translation, rotvec])


def _stable_rng(config: GenerationConfig, identity: str, channel: str) -> np.random.Generator:
    payload = (
        f"{config.generator_seed}|{config.dataset_number}|{identity}|{channel}"
    ).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def planted_robot_pose(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    normalized_time: float,
    robot_subject: int,
    config: GenerationConfig,
) -> SE3:
    values = (float(x_m), float(y_m), float(yaw_rad), float(normalized_time))
    if not all(math.isfinite(value) for value in values):
        raise MRCLAMSE3ConfigError("planted robot inputs must be finite")
    profile = nonplanar_profile(config, robot_subject, normalized_time)
    rotation = Rotation.from_euler(
        "xyz", [profile.roll_rad, profile.pitch_rad, yaw_rad]
    ).as_matrix()
    return SE3(rotation, [x_m, y_m, profile.z_m])


def planted_object_poses(
    landmark_truth: Mapping[int, Sequence[float]], config: GenerationConfig
) -> dict[int, SE3]:
    result = {}
    height_span = config.object_height_max_m - config.object_height_min_m
    for subject, point_value in sorted(landmark_truth.items()):
        if isinstance(subject, bool) or not isinstance(subject, int) or subject < 1:
            raise MRCLAMSE3ConfigError("landmark subject must be a positive integer")
        point = _vector(point_value, 2, "landmark position")
        rng = _stable_rng(config, f"landmark-{subject}", "object-pose")
        height = config.object_height_min_m + height_span * float(rng.random())
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = math.radians(20.0 + 120.0 * float(rng.random()))
        result[subject] = SE3(
            Rotation.from_rotvec(angle * axis).as_matrix(),
            [point[0], point[1], height],
        )
    if not result:
        raise MRCLAMSE3ConfigError("at least one landmark is required")
    return result


def _translation_sigma(tau: float) -> float:
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise MRCLAMSE3ConfigError("translation precision must be positive")
    return 1.0 / math.sqrt(tau)


def _rotation_sigma(kappa: float) -> float:
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise MRCLAMSE3ConfigError("rotation precision must be positive")
    return 1.0 / math.sqrt(2.0 * kappa)


def _source_odometry_residual(
    factor: OdometryFactor, source_truth_i: SE3, source_truth_j: SE3
) -> np.ndarray:
    ideal = source_truth_i.between(source_truth_j)
    rotation_residual = Rotation.from_matrix(
        ideal.rotation.T @ factor.measurement.rotation
    ).as_rotvec()
    result = np.zeros(6)
    result[:2] = factor.measurement.translation[:2] - ideal.translation[:2]
    result[5] = rotation_residual[2]
    return result


def lift_odometry_factor(
    factor: OdometryFactor,
    source_truth_i: SE3,
    source_truth_j: SE3,
    planted_i: SE3,
    planted_j: SE3,
    config: GenerationConfig,
) -> OdometryFactor:
    residual = _source_odometry_residual(factor, source_truth_i, source_truth_j)
    rng = _stable_rng(config, factor.factor_id, "odometry-nonplanar")
    residual[2] = float(rng.normal(scale=_translation_sigma(factor.tau)))
    residual[3:5] = rng.normal(scale=_rotation_sigma(factor.kappa), size=2)
    measurement = planted_i.between(planted_j).compose(se3_exp(residual))
    return OdometryFactor(
        factor.factor_id,
        factor.from_id,
        factor.to_id,
        measurement,
        factor.kappa,
        factor.tau,
        factor.source_covariance,
    )


def lift_pose_object_factor(
    factor: PoseObjectFactor,
    source_robot_truth: SE3,
    source_object_position: Sequence[float],
    planted_robot: SE3,
    planted_object: SE3,
    config: GenerationConfig,
) -> PoseObjectFactor:
    point = _vector(source_object_position, 2, "source object position")
    expected = source_robot_truth.rotation.T @ (
        np.array([point[0], point[1], 0.0]) - source_robot_truth.translation
    )
    residual = np.zeros(6)
    residual[:2] = factor.measurement.translation[:2] - expected[:2]
    rng = _stable_rng(config, factor.factor_id, "object-nonplanar")
    residual[2] = float(rng.normal(scale=_translation_sigma(factor.tau)))
    residual[3:] = rng.normal(scale=config.object_rotation_sigma_rad, size=3)
    ideal = planted_robot.between(planted_object)
    rotation_noise = Rotation.from_rotvec(residual[3:]).as_matrix()
    measurement = SE3(
        ideal.rotation @ rotation_noise,
        ideal.translation + residual[:3],
    )
    object_kappa = 1.0 / (2.0 * config.object_rotation_sigma_rad**2)
    return PoseObjectFactor(
        factor.factor_id,
        factor.from_id,
        factor.to_id,
        measurement,
        object_kappa,
        factor.tau,
        factor.source_covariance,
    )


__all__ = [
    "lift_odometry_factor",
    "lift_pose_object_factor",
    "planted_object_poses",
    "planted_robot_pose",
    "se3_exp",
    "se3_log",
]
