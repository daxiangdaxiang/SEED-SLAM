"""Pinned MR.CLAM source contract and native-team eligibility audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


class MRCLAMAuditError(ValueError):
    """Raised when MR.CLAM input violates the benchmark source contract."""


@dataclass(frozen=True)
class EvaluationSpec:
    seeds: tuple[int, ...]
    fixed_checkpoints: tuple[int, ...]
    max_iterations: int
    relative_cost_tolerance: float
    gradient_tolerance: float
    consecutive_rounds: int
    object_ground_truth_label_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MRCLAMManifest:
    schema_version: str
    benchmark_id: str
    description: str
    native_multi_robot: bool
    source: Mapping[str, Any]
    physical_robot_count: int
    simultaneous_collection: bool
    generated_by_trajectory_partition: bool
    generated_by_sequential_replay: bool
    archive_filename: str
    archive_sha256: str
    dataset_directory: str
    robot_subjects: tuple[int, ...]
    static_landmark_subjects: tuple[int, ...]
    extraction: Mapping[str, Any]
    objective: Mapping[str, Any]
    initialization: Mapping[str, Any]
    evaluation: EvaluationSpec
    path: Path


@dataclass(frozen=True)
class RobotAudit:
    robot_subject: int
    odometry_rows: int
    odometry_timestamp_inversions: int
    measurement_rows: int
    ground_truth_rows: int
    static_object_observations: int
    teammate_observations: int
    unknown_barcode_observations: int
    sensor_start_s: float
    sensor_end_s: float
    odometry_sha256: str
    measurement_sha256: str
    ground_truth_sha256: str
    observed_static_subjects: tuple[int, ...]


@dataclass(frozen=True)
class DatasetAudit:
    benchmark_id: str
    dataset_directory: Path
    archive_path: Path
    archive_sha256: str
    native_multi_robot_eligible: bool
    physical_robot_count: int
    robot_subjects: tuple[int, ...]
    common_sensor_start_s: float
    common_sensor_end_s: float
    common_sensor_duration_s: float
    static_landmark_subjects: tuple[int, ...]
    robots: tuple[RobotAudit, ...]
    barcode_to_subject: Mapping[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_directory": str(self.dataset_directory),
            "archive_path": str(self.archive_path),
            "archive_sha256": self.archive_sha256,
            "native_multi_robot_eligible": self.native_multi_robot_eligible,
            "physical_robot_count": self.physical_robot_count,
            "robot_subjects": list(self.robot_subjects),
            "common_sensor_start_s": self.common_sensor_start_s,
            "common_sensor_end_s": self.common_sensor_end_s,
            "common_sensor_duration_s": self.common_sensor_duration_s,
            "static_landmark_subjects": list(self.static_landmark_subjects),
            "robots": [asdict(item) for item in self.robots],
            "barcode_to_subject": {
                str(key): value
                for key, value in sorted(self.barcode_to_subject.items())
            },
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MRCLAMAuditError(f"manifest {name} must be an object")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MRCLAMAuditError(f"manifest {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MRCLAMAuditError(f"manifest {name} must be finite and positive")
    return result


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MRCLAMAuditError(f"manifest {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise MRCLAMAuditError(f"manifest {name} must be finite and non-negative")
    return result


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise MRCLAMAuditError(f"manifest {name} must be an integer list")
    result = tuple(int(item) for item in value)
    if not result or tuple(sorted(set(result))) != result:
        raise MRCLAMAuditError(f"manifest {name} must be sorted and unique")
    return result


def _integer_permutation_map(
    value: Any, name: str
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise MRCLAMAuditError(f"manifest {name} must be an object")
    pairs = []
    for raw_source, raw_target in value.items():
        try:
            source = int(raw_source)
        except (TypeError, ValueError) as error:
            raise MRCLAMAuditError(
                f"manifest {name} keys must be integer labels"
            ) from error
        if str(source) != str(raw_source) or isinstance(raw_target, bool) or not isinstance(
            raw_target, int
        ):
            raise MRCLAMAuditError(
                f"manifest {name} must map integer labels to integer labels"
            )
        pairs.append((source, int(raw_target)))
    sources = {source for source, _ in pairs}
    targets = {target for _, target in pairs}
    if len(targets) != len(pairs) or sources != targets:
        raise MRCLAMAuditError(
            f"manifest {name} must define a closed one-to-one label permutation"
        )
    return tuple(sorted(pairs))


def load_manifest(path: Path) -> MRCLAMManifest:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MRCLAMAuditError(f"cannot load manifest {path}: {error}") from error
    if not isinstance(raw, dict):
        raise MRCLAMAuditError("manifest root must be an object")
    if raw.get("schema_version") != "mrclam_benchmark_manifest_v1":
        raise MRCLAMAuditError("unsupported MR.CLAM manifest schema version")
    if raw.get("native_multi_robot") is not True:
        raise MRCLAMAuditError("manifest must declare native_multi_robot=true")
    benchmark_id = raw.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise MRCLAMAuditError("manifest benchmark_id must be non-empty")

    provenance = _mapping(raw.get("provenance"), "provenance")
    physical_count = provenance.get("physical_robot_count")
    if isinstance(physical_count, bool) or not isinstance(physical_count, int):
        raise MRCLAMAuditError("manifest physical_robot_count must be an integer")
    if physical_count != 5:
        raise MRCLAMAuditError("MR.CLAM benchmark requires five physical robots")
    if provenance.get("simultaneous_collection") is not True:
        raise MRCLAMAuditError("manifest must declare simultaneous collection")
    if provenance.get("generated_by_trajectory_partition") is not False:
        raise MRCLAMAuditError("trajectory partition provenance is forbidden")
    if provenance.get("generated_by_sequential_replay") is not False:
        raise MRCLAMAuditError("sequential replay provenance is forbidden")

    archive = _mapping(raw.get("archive"), "archive")
    archive_filename = archive.get("filename")
    archive_sha256 = archive.get("sha256")
    dataset_directory = archive.get("dataset_directory")
    if not isinstance(archive_filename, str) or not archive_filename:
        raise MRCLAMAuditError("manifest archive filename must be non-empty")
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise MRCLAMAuditError("manifest archive SHA-256 is invalid")
    if not isinstance(dataset_directory, str) or not dataset_directory:
        raise MRCLAMAuditError("manifest dataset_directory must be non-empty")

    robots = _integer_tuple(raw.get("robot_subjects"), "robot_subjects")
    landmarks = _integer_tuple(
        raw.get("static_landmark_subjects"), "static_landmark_subjects"
    )
    if robots != (1, 2, 3, 4, 5):
        raise MRCLAMAuditError("robot subjects must be exactly 1 through 5")
    if landmarks != tuple(range(6, 21)):
        raise MRCLAMAuditError("static landmark subjects must be exactly 6 through 20")

    extraction = _mapping(raw.get("extraction"), "extraction")
    _positive(extraction.get("keyframe_period_s"), "extraction.keyframe_period_s")
    objective = _mapping(raw.get("objective"), "objective")
    for key in (
        "odometry_kappa",
        "odometry_tau",
        "observation_kappa",
        "observation_tau",
    ):
        _positive(objective.get(key), f"objective.{key}")
    factor_model = objective.get("factor_model", "unit_isotropic_v0")
    if factor_model not in ("unit_isotropic_v0", "range_bearing_median_isotropic_v1"):
        raise MRCLAMAuditError("unsupported objective.factor_model")
    if factor_model == "range_bearing_median_isotropic_v1":
        for key in (
            "odometry_translation_sigma_m",
            "odometry_yaw_sigma_rad",
            "observation_range_sigma_m",
            "observation_bearing_sigma_rad",
        ):
            _positive(objective.get(key), f"objective.{key}")
    initialization = _mapping(raw.get("initialization"), "initialization")
    if initialization.get("root_robot_subject") != 1:
        raise MRCLAMAuditError("root_robot_subject must be 1")
    if initialization.get("ground_truth_allowed", False) is not False:
        raise MRCLAMAuditError("ground truth is forbidden during initialization")
    _positive(
        initialization.get("registration_window_s"),
        "initialization.registration_window_s",
    )
    _nonnegative(
        initialization.get("translation_perturbation_m"),
        "initialization.translation_perturbation_m",
    )
    _nonnegative(
        initialization.get("yaw_perturbation_deg"),
        "initialization.yaw_perturbation_deg",
    )

    evaluation_raw = _mapping(raw.get("evaluation"), "evaluation")
    seeds = _integer_tuple(evaluation_raw.get("seeds"), "evaluation.seeds")
    checkpoints = _integer_tuple(
        evaluation_raw.get("fixed_checkpoints"), "evaluation.fixed_checkpoints"
    )
    max_iterations = evaluation_raw.get("max_iterations")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise MRCLAMAuditError("evaluation.max_iterations must be an integer")
    if seeds != (0, 1, 2):
        raise MRCLAMAuditError("evaluation seeds must be exactly 0, 1, and 2")
    if checkpoints != (1, 2, 5, 10, 20, 50):
        raise MRCLAMAuditError("fixed checkpoints must be 1, 2, 5, 10, 20, 50")
    if max_iterations < checkpoints[-1]:
        raise MRCLAMAuditError("max_iterations is below the final checkpoint")
    relative_tolerance = float(evaluation_raw.get("relative_cost_tolerance", 1e-6))
    gradient_tolerance = float(evaluation_raw.get("gradient_tolerance", 1e-4))
    consecutive = evaluation_raw.get("consecutive_rounds", 3)
    object_ground_truth_label_map = _integer_permutation_map(
        evaluation_raw.get("object_ground_truth_label_map", {}),
        "evaluation.object_ground_truth_label_map",
    )
    if any(
        source not in landmarks or target not in landmarks
        for source, target in object_ground_truth_label_map
    ):
        raise MRCLAMAuditError(
            "evaluation.object_ground_truth_label_map contains a non-landmark label"
        )
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise MRCLAMAuditError("relative_cost_tolerance must be positive")
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise MRCLAMAuditError("gradient_tolerance must be positive")
    if isinstance(consecutive, bool) or not isinstance(consecutive, int) or consecutive <= 0:
        raise MRCLAMAuditError("consecutive_rounds must be a positive integer")

    return MRCLAMManifest(
        schema_version=raw["schema_version"],
        benchmark_id=benchmark_id,
        description=str(raw.get("description", "")),
        native_multi_robot=True,
        source=MappingProxyType(dict(_mapping(raw.get("source", {}), "source"))),
        physical_robot_count=physical_count,
        simultaneous_collection=True,
        generated_by_trajectory_partition=False,
        generated_by_sequential_replay=False,
        archive_filename=archive_filename,
        archive_sha256=archive_sha256,
        dataset_directory=dataset_directory,
        robot_subjects=robots,
        static_landmark_subjects=landmarks,
        extraction=MappingProxyType(dict(extraction)),
        objective=MappingProxyType(dict(objective)),
        initialization=MappingProxyType(dict(initialization)),
        evaluation=EvaluationSpec(
            seeds=seeds,
            fixed_checkpoints=checkpoints,
            max_iterations=max_iterations,
            relative_cost_tolerance=relative_tolerance,
            gradient_tolerance=gradient_tolerance,
            consecutive_rounds=consecutive,
            object_ground_truth_label_map=object_ground_truth_label_map,
        ),
        path=path.resolve(),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MRCLAMAuditError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def read_numeric_table(path: Path, columns: int, description: str) -> np.ndarray:
    try:
        result = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise MRCLAMAuditError(f"cannot parse {description} {path}: {error}") from error
    if result.ndim != 2 or result.shape[1] != columns or result.shape[0] == 0:
        raise MRCLAMAuditError(
            f"{description} {path} must contain non-empty {columns}-column rows"
        )
    if not np.all(np.isfinite(result)):
        raise MRCLAMAuditError(f"{description} {path} contains non-finite values")
    return result


def _timestamps(
    table: np.ndarray, description: str, *, allow_unsorted: bool = False
) -> tuple[float, float]:
    values = table[:, 0]
    if np.any(np.diff(values) < 0.0):
        if not allow_unsorted:
            raise MRCLAMAuditError(f"{description} timestamps are not monotonic")
        return float(np.min(values)), float(np.max(values))
    return float(values[0]), float(values[-1])


def _read_barcode_map(path: Path) -> dict[int, int]:
    table = read_numeric_table(path, 2, "barcode map")
    subjects = table[:, 0].astype(np.int64)
    barcodes = table[:, 1].astype(np.int64)
    if not np.all(table[:, 0] == subjects) or not np.all(table[:, 1] == barcodes):
        raise MRCLAMAuditError("barcode map contains non-integer identifiers")
    if len(set(subjects.tolist())) != len(subjects) or len(set(barcodes.tolist())) != len(barcodes):
        raise MRCLAMAuditError("barcode map contains duplicate identifiers")
    return {int(barcode): int(subject) for subject, barcode in zip(subjects, barcodes)}


def audit_dataset(manifest: MRCLAMManifest, data_root: Path) -> DatasetAudit:
    data_root = Path(data_root).resolve()
    archive_path = data_root / manifest.archive_filename
    dataset_directory = data_root / manifest.dataset_directory
    if not archive_path.is_file():
        raise MRCLAMAuditError(f"MR.CLAM archive is missing: {archive_path}")
    observed_archive_hash = file_sha256(archive_path)
    if observed_archive_hash != manifest.archive_sha256:
        raise MRCLAMAuditError(
            "MR.CLAM archive SHA-256 mismatch: "
            f"expected {manifest.archive_sha256}, observed {observed_archive_hash}"
        )
    if not dataset_directory.is_dir():
        raise MRCLAMAuditError(
            f"extracted MR.CLAM directory is missing: {dataset_directory}"
        )

    barcode_to_subject = _read_barcode_map(dataset_directory / "Barcodes.dat")
    expected_subjects = set(manifest.robot_subjects) | set(
        manifest.static_landmark_subjects
    )
    if set(barcode_to_subject.values()) != expected_subjects:
        raise MRCLAMAuditError("barcode map does not cover subjects 1 through 20")
    landmark_table = read_numeric_table(
        dataset_directory / "Landmark_Groundtruth.dat", 5, "landmark ground truth"
    )
    landmark_subjects = tuple(int(value) for value in landmark_table[:, 0])
    if landmark_subjects != manifest.static_landmark_subjects:
        raise MRCLAMAuditError("landmark ground truth does not cover subjects 6 through 20")

    robots: list[RobotAudit] = []
    for subject in manifest.robot_subjects:
        odometry_path = dataset_directory / f"Robot{subject}_Odometry.dat"
        measurement_path = dataset_directory / f"Robot{subject}_Measurement.dat"
        ground_truth_path = dataset_directory / f"Robot{subject}_Groundtruth.dat"
        odometry = read_numeric_table(odometry_path, 3, f"robot {subject} odometry")
        measurements = read_numeric_table(
            measurement_path, 4, f"robot {subject} measurements"
        )
        ground_truth = read_numeric_table(
            ground_truth_path, 4, f"robot {subject} ground truth"
        )
        odometry_timestamp_inversions = int(
            np.count_nonzero(np.diff(odometry[:, 0]) < 0.0)
        )
        # Dataset 9 contains isolated official-source odometry row inversions
        # in four robot streams. Extraction sorts before integration.
        odometry_interval = _timestamps(
            odometry, f"robot {subject} odometry", allow_unsorted=True
        )
        measurement_interval = _timestamps(measurements, f"robot {subject} measurements")
        _timestamps(ground_truth, f"robot {subject} ground truth")
        sensor_start = max(odometry_interval[0], measurement_interval[0])
        sensor_end = min(odometry_interval[1], measurement_interval[1])
        if sensor_end <= sensor_start:
            raise MRCLAMAuditError(
                f"robot {subject} odometry and measurement intervals do not overlap"
            )

        static_count = 0
        teammate_count = 0
        unknown_count = 0
        observed_static: set[int] = set()
        for raw_barcode in measurements[:, 1]:
            barcode = int(raw_barcode)
            if raw_barcode != barcode:
                raise MRCLAMAuditError(
                    f"robot {subject} measurement has a non-integer barcode"
                )
            measured_subject = barcode_to_subject.get(barcode)
            if measured_subject is None:
                unknown_count += 1
            elif measured_subject in manifest.static_landmark_subjects:
                static_count += 1
                observed_static.add(measured_subject)
            elif measured_subject in manifest.robot_subjects:
                teammate_count += 1
            else:
                unknown_count += 1
        if not observed_static:
            raise MRCLAMAuditError(f"robot {subject} observes no static object")
        robots.append(
            RobotAudit(
                robot_subject=subject,
                odometry_rows=int(odometry.shape[0]),
                odometry_timestamp_inversions=odometry_timestamp_inversions,
                measurement_rows=int(measurements.shape[0]),
                ground_truth_rows=int(ground_truth.shape[0]),
                static_object_observations=static_count,
                teammate_observations=teammate_count,
                unknown_barcode_observations=unknown_count,
                sensor_start_s=sensor_start,
                sensor_end_s=sensor_end,
                odometry_sha256=file_sha256(odometry_path),
                measurement_sha256=file_sha256(measurement_path),
                ground_truth_sha256=file_sha256(ground_truth_path),
                observed_static_subjects=tuple(sorted(observed_static)),
            )
        )

    for attribute in ("odometry_sha256", "measurement_sha256", "ground_truth_sha256"):
        hashes = {getattr(robot, attribute) for robot in robots}
        if len(hashes) != len(robots):
            raise MRCLAMAuditError(
                f"native robot streams must have distinct {attribute.replace('_sha256', '')} files"
            )
    common_start = max(item.sensor_start_s for item in robots)
    common_end = min(item.sensor_end_s for item in robots)
    if common_end <= common_start:
        raise MRCLAMAuditError(
            "original five-robot sensor intervals have no common overlap"
        )

    # Shared static identities must connect the physical robot team.
    adjacency = {subject: set() for subject in manifest.robot_subjects}
    observed = {
        robot.robot_subject: set(robot.observed_static_subjects) for robot in robots
    }
    for index, subject_a in enumerate(manifest.robot_subjects):
        for subject_b in manifest.robot_subjects[index + 1 :]:
            if observed[subject_a] & observed[subject_b]:
                adjacency[subject_a].add(subject_b)
                adjacency[subject_b].add(subject_a)
    visited = {manifest.robot_subjects[0]}
    frontier = list(visited)
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if visited != set(manifest.robot_subjects):
        raise MRCLAMAuditError("static-object incidence graph is disconnected")

    return DatasetAudit(
        benchmark_id=manifest.benchmark_id,
        dataset_directory=dataset_directory,
        archive_path=archive_path,
        archive_sha256=observed_archive_hash,
        native_multi_robot_eligible=True,
        physical_robot_count=manifest.physical_robot_count,
        robot_subjects=manifest.robot_subjects,
        common_sensor_start_s=common_start,
        common_sensor_end_s=common_end,
        common_sensor_duration_s=common_end - common_start,
        static_landmark_subjects=manifest.static_landmark_subjects,
        robots=tuple(robots),
        barcode_to_subject=MappingProxyType(barcode_to_subject),
    )


__all__ = [
    "DatasetAudit",
    "EvaluationSpec",
    "MRCLAMAuditError",
    "MRCLAMManifest",
    "RobotAudit",
    "audit_dataset",
    "file_sha256",
    "load_manifest",
    "read_numeric_table",
]
