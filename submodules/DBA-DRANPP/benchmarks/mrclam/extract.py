"""Planar MR.CLAM parsing, odometry integration, and object aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .source import DatasetAudit, MRCLAMManifest, read_numeric_table


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite(self.x, "pose x"))
        object.__setattr__(self, "y", _finite(self.y, "pose y"))
        object.__setattr__(self, "yaw", wrap_angle(_finite(self.yaw, "pose yaw")))

    @classmethod
    def identity(cls) -> "PlanarPose":
        return cls(0.0, 0.0, 0.0)

    def compose(self, other: "PlanarPose") -> "PlanarPose":
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        return PlanarPose(
            self.x + cosine * other.x - sine * other.y,
            self.y + sine * other.x + cosine * other.y,
            self.yaw + other.yaw,
        )

    def inverse(self) -> "PlanarPose":
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        return PlanarPose(
            -cosine * self.x - sine * self.y,
            sine * self.x - cosine * self.y,
            -self.yaw,
        )

    def between(self, other: "PlanarPose") -> "PlanarPose":
        return self.inverse().compose(other)

    def transform_point(self, point: Sequence[float]) -> np.ndarray:
        value = np.asarray(point, dtype=np.float64)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("planar point must contain two finite values")
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        return np.array(
            [
                self.x + cosine * value[0] - sine * value[1],
                self.y + sine * value[0] + cosine * value[1],
            ]
        )


@dataclass(frozen=True)
class OdometrySample:
    timestamp_s: float
    forward_velocity_mps: float
    angular_velocity_radps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "odometry timestamp"))
        object.__setattr__(
            self,
            "forward_velocity_mps",
            _finite(self.forward_velocity_mps, "forward velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity_radps",
            _finite(self.angular_velocity_radps, "angular velocity"),
        )


@dataclass(frozen=True)
class RangeBearingObservation:
    timestamp_s: float
    landmark_subject: int
    range_m: float
    bearing_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "observation timestamp"))
        if isinstance(self.landmark_subject, bool) or not isinstance(
            self.landmark_subject, int
        ):
            raise ValueError("landmark subject must be an integer")
        range_m = _finite(self.range_m, "observation range")
        if range_m <= 0.0:
            raise ValueError("observation range must be positive")
        object.__setattr__(self, "range_m", range_m)
        object.__setattr__(self, "bearing_rad", _finite(self.bearing_rad, "observation bearing"))


@dataclass(frozen=True)
class RobotSensorData:
    robot_subject: int
    odometry: tuple[OdometrySample, ...]
    observations: tuple[RangeBearingObservation, ...]
    rejected_unknown_barcodes: int
    excluded_teammate_observations: int


@dataclass(frozen=True)
class Keyframe:
    local_index: int
    timestamp_s: float
    pose: PlanarPose


@dataclass(frozen=True)
class PlanarOdometryFactor:
    from_keyframe_index: int
    to_keyframe_index: int
    measurement: PlanarPose


@dataclass(frozen=True)
class AggregatedObjectObservation:
    robot_subject: int
    keyframe_index: int
    landmark_subject: int
    translation_body: tuple[float, float]
    source_count: int
    translation_dispersion_m: float
    first_timestamp_s: float
    last_timestamp_s: float


@dataclass(frozen=True)
class RobotExtraction:
    robot_subject: int
    keyframes: tuple[Keyframe, ...]
    odometry_factors: tuple[PlanarOdometryFactor, ...]
    object_observations: tuple[AggregatedObjectObservation, ...]
    raw_static_observation_count: int
    rejected_unknown_barcodes: int
    excluded_teammate_observations: int


def exact_unicycle_increment(
    forward_velocity_mps: float, angular_velocity_radps: float, duration_s: float
) -> PlanarPose:
    velocity = _finite(forward_velocity_mps, "forward velocity")
    omega = _finite(angular_velocity_radps, "angular velocity")
    duration = _finite(duration_s, "integration duration")
    if duration < 0.0:
        raise ValueError("integration duration must be non-negative")
    angle = omega * duration
    if abs(omega) <= 1e-12:
        return PlanarPose(velocity * duration, 0.0, 0.0)
    radius = velocity / omega
    return PlanarPose(
        radius * math.sin(angle),
        radius * (1.0 - math.cos(angle)),
        angle,
    )


def _merge_duplicate_odometry(
    samples: Sequence[OdometrySample],
) -> tuple[OdometrySample, ...]:
    if len(samples) < 2:
        raise ValueError("at least two odometry samples are required")
    ordered = sorted(samples, key=lambda item: item.timestamp_s)
    if any(
        ordered[index].timestamp_s < ordered[index - 1].timestamp_s
        for index in range(1, len(ordered))
    ):
        raise ValueError("odometry timestamps must be monotonic")
    grouped: list[OdometrySample] = []
    index = 0
    while index < len(ordered):
        timestamp = ordered[index].timestamp_s
        values = []
        while index < len(ordered) and ordered[index].timestamp_s == timestamp:
            values.append(ordered[index])
            index += 1
        grouped.append(
            OdometrySample(
                timestamp,
                float(np.mean([item.forward_velocity_mps for item in values])),
                float(np.mean([item.angular_velocity_radps for item in values])),
            )
        )
    if len(grouped) < 2:
        raise ValueError("odometry must contain at least two distinct timestamps")
    return tuple(grouped)


def integrate_odometry(
    samples: Sequence[OdometrySample],
    query_times_s: Sequence[float],
    *,
    reference_time_s: float,
) -> tuple[PlanarPose, ...]:
    merged = _merge_duplicate_odometry(samples)
    reference = _finite(reference_time_s, "reference time")
    queries = tuple(_finite(value, "query time") for value in query_times_s)
    requested = tuple(sorted(set((reference, *queries))))
    if not requested:
        return ()
    first_time = merged[0].timestamp_s
    last_time = merged[-1].timestamp_s
    tolerance = 1e-9
    if requested[0] < first_time - tolerance or requested[-1] > last_time + tolerance:
        raise ValueError(
            f"query interval [{requested[0]}, {requested[-1]}] is outside odometry support "
            f"[{first_time}, {last_time}]"
        )

    pose = PlanarPose.identity()
    current_time = first_time
    control_index = 0
    absolute: dict[float, PlanarPose] = {}
    for target in requested:
        target = min(max(target, first_time), last_time)
        while (
            control_index + 1 < len(merged)
            and merged[control_index + 1].timestamp_s <= target + tolerance
        ):
            next_time = merged[control_index + 1].timestamp_s
            duration = max(0.0, next_time - current_time)
            control = merged[control_index]
            pose = pose.compose(
                exact_unicycle_increment(
                    control.forward_velocity_mps,
                    control.angular_velocity_radps,
                    duration,
                )
            )
            current_time = next_time
            control_index += 1
        if target > current_time:
            control = merged[control_index]
            pose = pose.compose(
                exact_unicycle_increment(
                    control.forward_velocity_mps,
                    control.angular_velocity_radps,
                    target - current_time,
                )
            )
            current_time = target
        absolute[target] = pose

    reference_inverse = absolute[reference].inverse()
    return tuple(
        reference_inverse.compose(absolute[query])
        for query in queries
    )


def load_robot_sensor_data(
    dataset_directory: Path,
    robot_subject: int,
    barcode_to_subject: Mapping[int, int],
    *,
    robot_subjects: Sequence[int],
    static_landmark_subjects: Sequence[int],
) -> RobotSensorData:
    dataset_directory = Path(dataset_directory)
    odometry_table = read_numeric_table(
        dataset_directory / f"Robot{robot_subject}_Odometry.dat",
        3,
        f"robot {robot_subject} odometry",
    )
    measurement_table = read_numeric_table(
        dataset_directory / f"Robot{robot_subject}_Measurement.dat",
        4,
        f"robot {robot_subject} measurements",
    )
    odometry = tuple(
        OdometrySample(float(timestamp), float(velocity), float(omega))
        for timestamp, velocity, omega in odometry_table
    )
    static_subjects = set(static_landmark_subjects)
    teammate_subjects = set(robot_subjects)
    observations = []
    unknown = 0
    teammates = 0
    for timestamp, raw_barcode, range_m, bearing in measurement_table:
        barcode = int(raw_barcode)
        if raw_barcode != barcode:
            raise ValueError(f"robot {robot_subject} has a non-integer barcode")
        measured_subject = barcode_to_subject.get(barcode)
        if measured_subject in static_subjects:
            observations.append(
                RangeBearingObservation(
                    float(timestamp), int(measured_subject), float(range_m), float(bearing)
                )
            )
        elif measured_subject in teammate_subjects:
            teammates += 1
        else:
            unknown += 1
    return RobotSensorData(
        robot_subject=robot_subject,
        odometry=odometry,
        observations=tuple(observations),
        rejected_unknown_barcodes=unknown,
        excluded_teammate_observations=teammates,
    )


def _keyframe_times(start: float, end: float, period: float) -> tuple[float, ...]:
    duration = end - start
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("common sensor interval must be finite and positive")
    period = _finite(period, "keyframe period")
    if period <= 0.0:
        raise ValueError("keyframe period must be positive")
    count = int(math.floor(duration / period + 1e-12)) + 1
    return tuple(start + index * period for index in range(count))


def extract_robot(
    sensor: RobotSensorData,
    *,
    common_start_s: float,
    common_end_s: float,
    keyframe_period_s: float,
) -> RobotExtraction:
    keyframe_times = _keyframe_times(
        float(common_start_s), float(common_end_s), float(keyframe_period_s)
    )
    retained_observations = tuple(
        item
        for item in sensor.observations
        if common_start_s <= item.timestamp_s <= common_end_s
    )
    query_times = (*keyframe_times, *(item.timestamp_s for item in retained_observations))
    query_poses = integrate_odometry(
        sensor.odometry, query_times, reference_time_s=common_start_s
    )
    keyframe_poses = query_poses[: len(keyframe_times)]
    observation_poses = query_poses[len(keyframe_times) :]
    keyframes = tuple(
        Keyframe(index, timestamp, keyframe_poses[index])
        for index, timestamp in enumerate(keyframe_times)
    )
    odometry_factors = tuple(
        PlanarOdometryFactor(
            index,
            index + 1,
            keyframes[index].pose.between(keyframes[index + 1].pose),
        )
        for index in range(len(keyframes) - 1)
    )

    grouped: dict[tuple[int, int], list[tuple[float, np.ndarray]]] = {}
    for observation, measurement_pose in zip(retained_observations, observation_poses):
        relative_index = (observation.timestamp_s - common_start_s) / keyframe_period_s
        keyframe_index = int(math.floor(relative_index + 0.5))
        keyframe_index = max(0, min(keyframe_index, len(keyframes) - 1))
        body_point = np.array(
            [
                observation.range_m * math.cos(observation.bearing_rad),
                observation.range_m * math.sin(observation.bearing_rad),
            ]
        )
        local_world_point = measurement_pose.transform_point(body_point)
        compensated = keyframes[keyframe_index].pose.inverse().transform_point(
            local_world_point
        )
        grouped.setdefault((keyframe_index, observation.landmark_subject), []).append(
            (observation.timestamp_s, compensated)
        )

    object_observations = []
    for (keyframe_index, landmark_subject), values in sorted(grouped.items()):
        timestamps = [item[0] for item in values]
        points = np.vstack([item[1] for item in values])
        median = np.median(points, axis=0)
        dispersion = float(
            np.sqrt(np.mean(np.sum((points - median) ** 2, axis=1)) / 2.0)
        )
        object_observations.append(
            AggregatedObjectObservation(
                robot_subject=sensor.robot_subject,
                keyframe_index=keyframe_index,
                landmark_subject=landmark_subject,
                translation_body=(float(median[0]), float(median[1])),
                source_count=len(values),
                translation_dispersion_m=dispersion,
                first_timestamp_s=min(timestamps),
                last_timestamp_s=max(timestamps),
            )
        )
    return RobotExtraction(
        robot_subject=sensor.robot_subject,
        keyframes=keyframes,
        odometry_factors=odometry_factors,
        object_observations=tuple(object_observations),
        raw_static_observation_count=len(retained_observations),
        rejected_unknown_barcodes=sensor.rejected_unknown_barcodes,
        excluded_teammate_observations=sensor.excluded_teammate_observations,
    )


def extract_dataset(
    manifest: MRCLAMManifest, audit: DatasetAudit
) -> tuple[RobotExtraction, ...]:
    if not audit.native_multi_robot_eligible:
        raise ValueError("MR.CLAM source audit is not native-team eligible")
    result = []
    for subject in manifest.robot_subjects:
        sensor = load_robot_sensor_data(
            audit.dataset_directory,
            subject,
            audit.barcode_to_subject,
            robot_subjects=manifest.robot_subjects,
            static_landmark_subjects=manifest.static_landmark_subjects,
        )
        result.append(
            extract_robot(
                sensor,
                common_start_s=audit.common_sensor_start_s,
                common_end_s=audit.common_sensor_end_s,
                keyframe_period_s=float(manifest.extraction["keyframe_period_s"]),
            )
        )
    return tuple(result)


__all__ = [
    "AggregatedObjectObservation",
    "Keyframe",
    "OdometrySample",
    "PlanarOdometryFactor",
    "PlanarPose",
    "RangeBearingObservation",
    "RobotExtraction",
    "RobotSensorData",
    "exact_unicycle_increment",
    "extract_dataset",
    "extract_robot",
    "integrate_odometry",
    "load_robot_sensor_data",
    "wrap_angle",
]
