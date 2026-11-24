"""Deterministic odometry, keyframe, and cuboid extraction from SlideSLAM bags."""

from __future__ import annotations

import dataclasses
import itertools
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .acquire import BenchmarkManifest, SourceSpec
from .se3 import SE3, rotation_angle


class ExtractionError(ValueError):
    """Raised when decoded source data cannot form a valid extraction."""


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ExtractionError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ExtractionError(f"{name} must be non-negative")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ExtractionError(f"{name} must be finite and positive")
    return result


def _covariance(value: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != 36 or not np.all(np.isfinite(result)):
        raise ExtractionError("odometry covariance must contain 36 finite values")
    return result


@dataclass(frozen=True)
class KeyframeThresholds:
    translation_m: float
    rotation_deg: float
    elapsed_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation_m", _finite_positive(self.translation_m, "translation threshold")
        )
        object.__setattr__(
            self, "rotation_deg", _finite_positive(self.rotation_deg, "rotation threshold")
        )
        object.__setattr__(
            self, "elapsed_s", _finite_positive(self.elapsed_s, "elapsed threshold")
        )


@dataclass(frozen=True)
class CuboidObservation:
    robot_id: int
    timestamp_ns: int
    observation_index: int
    pose_body_object: SE3
    robot_pose_local: SE3
    dimensions: tuple[float, float, float]
    semantic_label: int
    source_topic: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_id", _integer(self.robot_id, "robot_id"))
        object.__setattr__(
            self, "timestamp_ns", _integer(self.timestamp_ns, "cuboid timestamp")
        )
        object.__setattr__(
            self,
            "observation_index",
            _integer(self.observation_index, "cuboid observation index"),
        )
        if not isinstance(self.pose_body_object, SE3) or not isinstance(
            self.robot_pose_local, SE3
        ):
            raise ExtractionError("cuboid poses must be finite SE3 values")
        dimensions = tuple(float(item) for item in self.dimensions)
        if (
            len(dimensions) != 3
            or not np.all(np.isfinite(dimensions))
            or any(item <= 0.0 for item in dimensions)
        ):
            raise ExtractionError("cuboid dimensions must contain 3 finite positive values")
        object.__setattr__(self, "dimensions", dimensions)
        if isinstance(self.semantic_label, bool) or not isinstance(
            self.semantic_label, (int, np.integer)
        ):
            raise ExtractionError("cuboid semantic label must be an integer")
        object.__setattr__(self, "semantic_label", int(self.semantic_label))
        if not isinstance(self.source_topic, str) or not self.source_topic:
            raise ExtractionError("cuboid source topic must be non-empty")

    @property
    def pose_local_object(self) -> SE3:
        return self.robot_pose_local.compose(self.pose_body_object)

    @property
    def centroid_local(self) -> np.ndarray:
        return self.pose_local_object.translation


@dataclass(frozen=True)
class DecodedSample:
    robot_id: int
    timestamp_ns: int
    odometry_pose: SE3
    odometry_covariance: tuple[float, ...]
    cuboids: tuple[CuboidObservation, ...]
    is_semantic: bool
    source_topic: str
    source_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_id", _integer(self.robot_id, "robot_id"))
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "sample timestamp"))
        if not isinstance(self.odometry_pose, SE3):
            raise ExtractionError("sample odometry pose must be SE3")
        object.__setattr__(self, "odometry_covariance", _covariance(self.odometry_covariance))
        cuboids = tuple(self.cuboids)
        if any(
            cuboid.robot_id != self.robot_id or cuboid.timestamp_ns != self.timestamp_ns
            for cuboid in cuboids
        ):
            raise ExtractionError("sample cuboids must share its robot and timestamp")
        object.__setattr__(self, "cuboids", cuboids)
        if not isinstance(self.is_semantic, bool):
            raise ExtractionError("is_semantic must be boolean")
        if not isinstance(self.source_topic, str) or not self.source_topic:
            raise ExtractionError("sample source topic must be non-empty")
        object.__setattr__(
            self, "source_sequence", _integer(self.source_sequence, "source sequence")
        )


@dataclass(frozen=True)
class Keyframe:
    robot_id: int
    local_index: int
    timestamp_ns: int
    odometry_pose: SE3
    odometry_covariance: tuple[float, ...]
    cuboids: tuple[CuboidObservation, ...]
    is_semantic: bool
    source_topic: str


@dataclass(frozen=True)
class ExtractedOdometryFactor:
    from_local_index: int
    to_local_index: int
    measurement: SE3
    kappa: float
    tau: float
    source_covariance: tuple[float, ...]


@dataclass(frozen=True)
class LocalCuboidTrack:
    track_id: str
    robot_id: int
    observations: tuple[CuboidObservation, ...]
    semantic_label: int
    representative_pose: SE3
    representative_dimensions: tuple[float, float, float]

    @property
    def centroid_local(self) -> np.ndarray:
        return self.representative_pose.translation


@dataclass(frozen=True)
class RejectedLocalTrack:
    track: LocalCuboidTrack
    reason: str
    maximum_centroid_deviation_m: float


@dataclass(frozen=True)
class LocalTrackResult:
    accepted: tuple[LocalCuboidTrack, ...]
    rejected: tuple[RejectedLocalTrack, ...]


@dataclass(frozen=True)
class RobotExtraction:
    robot_id: int
    keyframes: tuple[Keyframe, ...]
    odometry_factors: tuple[ExtractedOdometryFactor, ...]
    tracks: LocalTrackResult
    source_sample_count: int
    semantic_sample_count: int
    cuboid_observation_count: int
    discarded_prefix_message_count: int = 0


@dataclass(frozen=True)
class MessageRegistration:
    official_commit: str
    registered_types: tuple[str, ...]
    message_directory: Path


@dataclass(frozen=True)
class BagReadResult:
    samples: tuple[DecodedSample, ...]
    discarded_prefix_message_count: int


def register_slideslam_messages(
    manifest: BenchmarkManifest, repository_root: Path
) -> MessageRegistration:
    """Register the pinned custom ROS1 definitions with the installed rosbags API."""

    checkout_value = manifest.source.get("official_checkout")
    expected_commit = manifest.source.get("official_commit")
    if not isinstance(checkout_value, str) or not isinstance(expected_commit, str):
        raise ExtractionError("manifest lacks official SlideSLAM checkout provenance")
    checkout = Path(repository_root) / checkout_value
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExtractionError(f"cannot inspect official SlideSLAM checkout: {error}") from error
    actual_commit = completed.stdout.strip()
    if actual_commit != expected_commit:
        raise ExtractionError(
            f"official SlideSLAM checkout commit {actual_commit} does not match "
            f"manifest {expected_commit}"
        )
    message_directory = checkout / "backend" / "sloam_msgs" / "msg"
    message_files = sorted(message_directory.glob("*.msg"))
    if not message_files:
        raise ExtractionError(f"no SlideSLAM message definitions under {message_directory}")

    from rosbags.typesys import get_types_from_msg, register_types

    definitions = {}
    try:
        for path in message_files:
            definitions.update(
                get_types_from_msg(
                    path.read_text(encoding="utf-8"),
                    f"sloam_msgs/msg/{path.stem}",
                )
            )
        register_types(definitions)
    except Exception as error:
        raise ExtractionError(f"cannot register SlideSLAM ROS messages: {error}") from error
    required = {
        "sloam_msgs/msg/SemanticMeasSyncOdom",
        "sloam_msgs/msg/ROSCube",
    }
    if not required.issubset(definitions):
        raise ExtractionError("pinned checkout lacks required SlideSLAM message definitions")
    return MessageRegistration(
        official_commit=actual_commit,
        registered_types=tuple(sorted(definitions)),
        message_directory=message_directory.resolve(),
    )


def _sample_preference(sample: DecodedSample) -> tuple[int, int, str]:
    return (1 if sample.is_semantic else 0, sample.source_sequence, sample.source_topic)


def select_keyframes(
    samples: Iterable[DecodedSample], thresholds: KeyframeThresholds
) -> tuple[Keyframe, ...]:
    """Select inclusive motion/time thresholds and force semantic timestamps."""

    ordered = sorted(
        samples,
        key=lambda item: (item.timestamp_ns, item.source_sequence, item.source_topic),
    )
    if not ordered:
        raise ExtractionError("cannot select keyframes from an empty sample stream")
    robot_ids = {sample.robot_id for sample in ordered}
    if len(robot_ids) != 1:
        raise ExtractionError("one extraction stream must contain exactly one robot")

    deduplicated: list[DecodedSample] = []
    for _, group_iterator in itertools.groupby(ordered, key=lambda item: item.timestamp_ns):
        group = list(group_iterator)
        deduplicated.append(max(group, key=_sample_preference))

    selected: list[DecodedSample] = [deduplicated[0]]
    translation_threshold = thresholds.translation_m
    rotation_threshold = math.radians(thresholds.rotation_deg)
    elapsed_threshold_ns = int(round(thresholds.elapsed_s * 1e9))
    tolerance = 1e-12
    for sample in deduplicated[1:]:
        previous = selected[-1]
        translation = float(
            np.linalg.norm(sample.odometry_pose.translation - previous.odometry_pose.translation)
        )
        rotation = rotation_angle(
            previous.odometry_pose.rotation.T @ sample.odometry_pose.rotation
        )
        elapsed_ns = sample.timestamp_ns - previous.timestamp_ns
        if elapsed_ns < 0:
            raise ExtractionError("sample timestamps are not monotonic")
        if (
            sample.is_semantic
            or translation + tolerance >= translation_threshold
            or rotation + tolerance >= rotation_threshold
            or elapsed_ns >= elapsed_threshold_ns
        ):
            selected.append(sample)

    return tuple(
        Keyframe(
            robot_id=sample.robot_id,
            local_index=index,
            timestamp_ns=sample.timestamp_ns,
            odometry_pose=sample.odometry_pose,
            odometry_covariance=sample.odometry_covariance,
            cuboids=sample.cuboids,
            is_semantic=sample.is_semantic,
            source_topic=sample.source_topic,
        )
        for index, sample in enumerate(selected)
    )


def make_odometry_factors(
    keyframes: Sequence[Keyframe], kappa: float, tau: float
) -> tuple[ExtractedOdometryFactor, ...]:
    kappa = _finite_positive(kappa, "odometry kappa")
    tau = _finite_positive(tau, "odometry tau")
    if not keyframes:
        raise ExtractionError("cannot form odometry factors without keyframes")
    factors = []
    for previous, current in zip(keyframes, keyframes[1:]):
        if previous.robot_id != current.robot_id:
            raise ExtractionError("odometry factor cannot cross robots")
        factors.append(
            ExtractedOdometryFactor(
                from_local_index=previous.local_index,
                to_local_index=current.local_index,
                measurement=previous.odometry_pose.between(current.odometry_pose),
                kappa=kappa,
                tau=tau,
                source_covariance=current.odometry_covariance,
            )
        )
    return tuple(factors)


def _representative_track(
    track_id: str, observations: Sequence[CuboidObservation]
) -> LocalCuboidTrack:
    ordered = tuple(
        sorted(observations, key=lambda item: (item.timestamp_ns, item.observation_index))
    )
    robot_ids = {item.robot_id for item in ordered}
    if not ordered or len(robot_ids) != 1:
        raise ExtractionError("a local cuboid track must contain one robot's observations")
    centroids = np.vstack([item.centroid_local for item in ordered])
    median = np.median(centroids, axis=0)
    representative_index = min(
        range(len(ordered)),
        key=lambda index: (float(np.linalg.norm(centroids[index] - median)), index),
    )
    representative_observation = ordered[representative_index]
    representative_pose = SE3(
        representative_observation.pose_local_object.rotation,
        median,
    )
    dimensions = tuple(
        float(value) for value in np.median(
            np.asarray([item.dimensions for item in ordered]), axis=0
        )
    )
    labels, counts = np.unique(
        np.asarray([item.semantic_label for item in ordered], dtype=np.int64),
        return_counts=True,
    )
    maximum_count = int(np.max(counts))
    semantic_label = int(np.min(labels[counts == maximum_count]))
    return LocalCuboidTrack(
        track_id=track_id,
        robot_id=ordered[0].robot_id,
        observations=ordered,
        semantic_label=semantic_label,
        representative_pose=representative_pose,
        representative_dimensions=dimensions,
    )


def track_local_cuboids(
    observations: Sequence[CuboidObservation],
    max_distance: float,
    consistency_distance: float,
) -> LocalTrackResult:
    """Apply SlideSLAM's centroid distance with deterministic one-to-one projection."""

    max_distance = _finite_positive(max_distance, "local cuboid match distance")
    consistency_distance = _finite_positive(
        consistency_distance, "local cuboid consistency distance"
    )
    ordered = sorted(
        observations,
        key=lambda item: (item.timestamp_ns, item.observation_index),
    )
    if not ordered:
        return LocalTrackResult(accepted=(), rejected=())
    robot_ids = {item.robot_id for item in ordered}
    if len(robot_ids) != 1:
        raise ExtractionError("local tracking input must contain exactly one robot")
    robot_id = ordered[0].robot_id
    mutable_tracks: list[list[CuboidObservation]] = []

    for _, group_iterator in itertools.groupby(ordered, key=lambda item: item.timestamp_ns):
        detections = list(group_iterator)
        assigned_detections: set[int] = set()
        if mutable_tracks:
            track_centroids = np.vstack(
                [
                    np.median(
                        np.vstack([observation.centroid_local for observation in track]),
                        axis=0,
                    )
                    for track in mutable_tracks
                ]
            )
            detection_centroids = np.vstack(
                [observation.centroid_local for observation in detections]
            )
            distances = np.linalg.norm(
                detection_centroids[:, None, :] - track_centroids[None, :, :], axis=2
            )
            rows = np.arange(distances.shape[0])[:, None]
            columns = np.arange(distances.shape[1])[None, :]
            tie_break = np.finfo(float).eps * (rows * (distances.shape[1] + 1) + columns)
            row_indices, column_indices = linear_sum_assignment(distances + tie_break)
            for detection_index, track_index in zip(row_indices, column_indices):
                if distances[detection_index, track_index] <= max_distance:
                    mutable_tracks[int(track_index)].append(
                        detections[int(detection_index)]
                    )
                    assigned_detections.add(int(detection_index))
        for detection_index, detection in enumerate(detections):
            if detection_index not in assigned_detections:
                mutable_tracks.append([detection])

    accepted: list[LocalCuboidTrack] = []
    rejected: list[RejectedLocalTrack] = []
    for index, observations_for_track in enumerate(mutable_tracks):
        track = _representative_track(
            f"r{robot_id}_t{index:06d}", observations_for_track
        )
        deviations = [
            float(np.linalg.norm(item.centroid_local - track.centroid_local))
            for item in track.observations
        ]
        maximum_deviation = max(deviations)
        if maximum_deviation > consistency_distance:
            rejected.append(
                RejectedLocalTrack(
                    track=track,
                    reason="centroid_consistency_exceeded",
                    maximum_centroid_deviation_m=maximum_deviation,
                )
            )
        else:
            accepted.append(track)
    return LocalTrackResult(tuple(accepted), tuple(rejected))


def extract_robot(
    samples: Sequence[DecodedSample],
    thresholds: KeyframeThresholds,
    *,
    odometry_kappa: float,
    odometry_tau: float,
    local_match_distance: float,
    consistency_distance: float,
) -> RobotExtraction:
    keyframes = select_keyframes(samples, thresholds)
    odometry_factors = make_odometry_factors(
        keyframes, kappa=odometry_kappa, tau=odometry_tau
    )
    observations = tuple(
        cuboid for keyframe in keyframes for cuboid in keyframe.cuboids
    )
    tracks = track_local_cuboids(
        observations,
        max_distance=local_match_distance,
        consistency_distance=consistency_distance,
    )
    return RobotExtraction(
        robot_id=keyframes[0].robot_id,
        keyframes=keyframes,
        odometry_factors=odometry_factors,
        tracks=tracks,
        source_sample_count=len(samples),
        semantic_sample_count=sum(sample.is_semantic for sample in samples),
        cuboid_observation_count=len(observations),
    )


def _message_type(name: str) -> str:
    parts = name.split("/")
    return f"{parts[0]}/msg/{parts[1]}" if len(parts) == 2 else name


def _stamp_ns(header: object, fallback_ns: int) -> int:
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", 0))
    value = seconds * 1_000_000_000 + nanoseconds
    return value if value > 0 else fallback_ns


def _ros_pose(value: object) -> SE3:
    position = getattr(value, "position")
    orientation = getattr(value, "orientation")
    try:
        return SE3.from_quaternion_xyzw(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            [position.x, position.y, position.z],
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ExtractionError(f"invalid ROS pose: {error}") from error


def _decode_bag(
    bag: Path,
    manifest: BenchmarkManifest,
    source: SourceSpec,
    repository_root: Path,
) -> BagReadResult:
    register_slideslam_messages(manifest, repository_root)
    from rosbags.rosbag1 import Reader
    from rosbags.serde import deserialize_ros1

    accepted_odometry = {
        _message_type(value)
        for value in manifest.required_message_types["odometry"]
    }
    accepted_semantic = {
        _message_type(value)
        for value in manifest.required_message_types["semantic"]
    }
    topic_aliases = {
        "odometry": set(manifest.topic_aliases.get("odometry", ())),
        "semantic": set(manifest.topic_aliases.get("semantic", ())),
    }
    samples: list[DecodedSample] = []
    discarded = 0
    with Reader(Path(bag)) as reader:
        cutoff_ns = int(reader.start_time + round(source.replay_offset_seconds * 1e9))
        typed_connections = [
            connection
            for connection in reader.connections
            if _message_type(connection.msgtype) in accepted_odometry | accepted_semantic
        ]
        preferred_connections = [
            connection
            for connection in typed_connections
            if (
                _message_type(connection.msgtype) in accepted_odometry
                and connection.topic in topic_aliases["odometry"]
            )
            or (
                _message_type(connection.msgtype) in accepted_semantic
                and connection.topic in topic_aliases["semantic"]
            )
        ]
        connections = preferred_connections or typed_connections
        if not connections:
            raise ExtractionError(f"no required messages in {bag}")
        for sequence, (connection, bag_timestamp_ns, rawdata) in enumerate(
            reader.messages(connections=connections)
        ):
            if bag_timestamp_ns < cutoff_ns:
                discarded += 1
                continue
            try:
                message = deserialize_ros1(rawdata, connection.msgtype)
            except Exception as error:
                raise ExtractionError(
                    f"cannot deserialize {connection.topic} at {bag_timestamp_ns}: {error}"
                ) from error
            normalized_type = _message_type(connection.msgtype)
            if normalized_type in accepted_semantic:
                timestamp_ns = _stamp_ns(message.header, int(bag_timestamp_ns))
                odometry = message.odometry
                robot_pose = _ros_pose(odometry.pose.pose)
                cuboids = tuple(
                    CuboidObservation(
                        robot_id=source.robot_id,
                        timestamp_ns=timestamp_ns,
                        observation_index=index,
                        pose_body_object=_ros_pose(cuboid.pose),
                        robot_pose_local=robot_pose,
                        dimensions=tuple(float(value) for value in cuboid.dim),
                        semantic_label=int(cuboid.semantic_label),
                        source_topic=connection.topic,
                    )
                    for index, cuboid in enumerate(message.cuboid_factors)
                )
                samples.append(
                    DecodedSample(
                        robot_id=source.robot_id,
                        timestamp_ns=timestamp_ns,
                        odometry_pose=robot_pose,
                        odometry_covariance=tuple(float(value) for value in odometry.pose.covariance),
                        cuboids=cuboids,
                        is_semantic=True,
                        source_topic=connection.topic,
                        source_sequence=sequence,
                    )
                )
            elif normalized_type in accepted_odometry:
                timestamp_ns = _stamp_ns(message.header, int(bag_timestamp_ns))
                samples.append(
                    DecodedSample(
                        robot_id=source.robot_id,
                        timestamp_ns=timestamp_ns,
                        odometry_pose=_ros_pose(message.pose.pose),
                        odometry_covariance=tuple(float(value) for value in message.pose.covariance),
                        cuboids=(),
                        is_semantic=False,
                        source_topic=connection.topic,
                        source_sequence=sequence,
                    )
                )
    if not samples:
        raise ExtractionError(f"no samples remain after replay offset for robot {source.robot_id}")
    samples.sort(key=lambda item: (item.timestamp_ns, item.source_sequence))
    return BagReadResult(tuple(samples), discarded)


def iter_synchronized_samples(
    bag: Path,
    manifest: BenchmarkManifest,
    source: SourceSpec,
    repository_root: Optional[Path] = None,
) -> Iterator[DecodedSample]:
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]
    yield from _decode_bag(bag, manifest, source, repository_root).samples


def extract_robot_bag(
    bag: Path,
    manifest: BenchmarkManifest,
    source: SourceSpec,
    repository_root: Optional[Path] = None,
) -> RobotExtraction:
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[2]
    read_result = _decode_bag(bag, manifest, source, repository_root)
    extraction = extract_robot(
        read_result.samples,
        KeyframeThresholds(
            manifest.extraction["keyframe_translation_m"],
            manifest.extraction["keyframe_rotation_deg"],
            manifest.extraction["keyframe_elapsed_s"],
        ),
        odometry_kappa=manifest.objective["odometry_kappa"],
        odometry_tau=manifest.objective["odometry_tau"],
        local_match_distance=manifest.extraction["local_cuboid_match_distance_m"],
        consistency_distance=manifest.extraction[
            "local_track_consistency_distance_m"
        ],
    )
    return dataclasses.replace(
        extraction,
        discarded_prefix_message_count=read_result.discarded_prefix_message_count,
    )


__all__ = [
    "BagReadResult",
    "CuboidObservation",
    "DecodedSample",
    "ExtractedOdometryFactor",
    "ExtractionError",
    "Keyframe",
    "KeyframeThresholds",
    "LocalCuboidTrack",
    "LocalTrackResult",
    "MessageRegistration",
    "RejectedLocalTrack",
    "RobotExtraction",
    "extract_robot",
    "extract_robot_bag",
    "iter_synchronized_samples",
    "make_odometry_factors",
    "register_slideslam_messages",
    "select_keyframes",
    "track_local_cuboids",
]
