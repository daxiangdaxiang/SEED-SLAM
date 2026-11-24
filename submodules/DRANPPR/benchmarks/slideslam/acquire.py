"""Source manifest loading and native multi-robot ROS bag auditing."""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class AuditError(ValueError):
    """Raised when source provenance cannot satisfy the benchmark contract."""


@dataclass(frozen=True)
class SourceSpec:
    robot_id: int
    filename_patterns: tuple[str, ...]
    replay_offset_seconds: float


@dataclass(frozen=True)
class EvaluationSpec:
    seeds: tuple[int, ...]
    fixed_checkpoints: tuple[int, ...]
    max_iterations: int
    relative_cost_tolerance: float
    gradient_tolerance: float
    consecutive_rounds: int


@dataclass(frozen=True)
class BenchmarkManifest:
    schema_version: str
    benchmark_id: str
    description: str
    native_multi_robot: bool
    require_temporal_overlap: bool
    source: Mapping[str, Any]
    sources: tuple[SourceSpec, ...]
    required_message_types: Mapping[str, tuple[str, ...]]
    topic_aliases: Mapping[str, tuple[str, ...]]
    extraction: Mapping[str, Any]
    association: Mapping[str, Any]
    objective: Mapping[str, Any]
    initialization: Mapping[str, Any]
    evaluation: EvaluationSpec
    path: Path


@dataclass(frozen=True)
class TopicInventory:
    name: str
    message_type: str
    message_count: int


@dataclass(frozen=True)
class BagInventory:
    robot_id: int
    path: Path
    sha256: str
    size_bytes: int
    bag_version: str
    start_ns: int
    end_ns: int
    topics: Mapping[str, TopicInventory]
    generated_by_trajectory_partition: bool = False


@dataclass(frozen=True)
class AuditReport:
    benchmark_id: str
    native_multi_robot: bool
    require_temporal_overlap: bool
    temporal_overlap_verified: bool
    passed: bool
    robot_ids: tuple[int, ...]
    effective_overlap_start_ns: int
    effective_overlap_end_ns: int
    discarded_prefix_seconds: Mapping[int, float]
    inventories: tuple[BagInventory, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "native_multi_robot": self.native_multi_robot,
            "require_temporal_overlap": self.require_temporal_overlap,
            "temporal_overlap_verified": self.temporal_overlap_verified,
            "passed": self.passed,
            "robot_ids": list(self.robot_ids),
            "effective_overlap_start_ns": self.effective_overlap_start_ns,
            "effective_overlap_end_ns": self.effective_overlap_end_ns,
            "discarded_prefix_seconds": {
                str(key): value
                for key, value in sorted(self.discarded_prefix_seconds.items())
            },
            "inventories": [
                {
                    "robot_id": item.robot_id,
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "bag_version": item.bag_version,
                    "start_ns": item.start_ns,
                    "end_ns": item.end_ns,
                    "generated_by_trajectory_partition": (
                        item.generated_by_trajectory_partition
                    ),
                    "topics": [
                        dataclasses.asdict(topic)
                        for _, topic in sorted(item.topics.items())
                    ],
                }
                for item in self.inventories
            ],
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"manifest {name} must be an object")
    return MappingProxyType(dict(value))


def _finite_positive(value: Any, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"manifest {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise AuditError(f"manifest {name} must be finite and {qualifier}")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AuditError(f"manifest {name} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise AuditError(f"manifest {name} must contain non-empty strings")
    return tuple(value)


def load_manifest(path: Path) -> BenchmarkManifest:
    """Load and validate the tracked benchmark manifest."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot load manifest {path}: {error}") from error
    if not isinstance(raw, dict):
        raise AuditError("manifest root must be an object")
    if raw.get("schema_version") != "slideslam_benchmark_manifest_v1":
        raise AuditError("unsupported SlideSLAM manifest schema version")
    benchmark_id = raw.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise AuditError("manifest benchmark_id must be a non-empty string")
    native_multi_robot = raw.get("native_multi_robot")
    if not isinstance(native_multi_robot, bool):
        raise AuditError("manifest native_multi_robot must be boolean")
    require_temporal_overlap = raw.get("require_temporal_overlap", True)
    if not isinstance(require_temporal_overlap, bool):
        raise AuditError("manifest require_temporal_overlap must be boolean")

    source_records = raw.get("sources")
    if not isinstance(source_records, list) or len(source_records) != 3:
        raise AuditError("manifest must describe exactly 3 native source robots")
    sources: list[SourceSpec] = []
    for index, record in enumerate(source_records):
        if not isinstance(record, dict):
            raise AuditError(f"manifest source {index} must be an object")
        robot_id = record.get("robot_id")
        if isinstance(robot_id, bool) or not isinstance(robot_id, int):
            raise AuditError("manifest robot IDs must be integers")
        sources.append(
            SourceSpec(
                robot_id=robot_id,
                filename_patterns=_string_tuple(
                    record.get("filename_patterns"),
                    f"source {robot_id} filename_patterns",
                ),
                replay_offset_seconds=_finite_positive(
                    record.get("replay_offset_seconds"),
                    f"source {robot_id} replay_offset_seconds",
                    allow_zero=True,
                ),
            )
        )
    robot_ids = [source.robot_id for source in sources]
    if sorted(robot_ids) != [0, 1, 2] or len(set(robot_ids)) != len(robot_ids):
        raise AuditError("manifest robot IDs must be unique and equal to 0, 1, 2")
    sources.sort(key=lambda item: item.robot_id)

    required_raw = _mapping(raw.get("required_message_types"), "required_message_types")
    required = {
        name: _string_tuple(value, f"required_message_types.{name}")
        for name, value in required_raw.items()
    }
    if set(required) != {"odometry", "semantic"}:
        raise AuditError("manifest must require odometry and semantic message types")
    aliases_raw = _mapping(raw.get("topic_aliases"), "topic_aliases")
    aliases = {
        name: _string_tuple(value, f"topic_aliases.{name}")
        for name, value in aliases_raw.items()
    }

    evaluation_raw = _mapping(raw.get("evaluation"), "evaluation")
    seeds_raw = evaluation_raw.get("seeds")
    checkpoints_raw = evaluation_raw.get("fixed_checkpoints")
    if not isinstance(seeds_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in seeds_raw
    ):
        raise AuditError("manifest evaluation seeds must be integers")
    if tuple(seeds_raw) != (0, 1, 2):
        raise AuditError("manifest evaluation seeds must be exactly 0, 1, 2")
    if not isinstance(checkpoints_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in checkpoints_raw
    ):
        raise AuditError("manifest fixed checkpoints must be positive integers")
    max_iterations = evaluation_raw.get("max_iterations")
    consecutive_rounds = evaluation_raw.get("consecutive_rounds")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise AuditError("manifest max_iterations must be an integer")
    if isinstance(consecutive_rounds, bool) or not isinstance(consecutive_rounds, int):
        raise AuditError("manifest consecutive_rounds must be an integer")
    if max_iterations <= 0 or max(checkpoints_raw, default=0) > max_iterations:
        raise AuditError("manifest fixed checkpoints exceed max_iterations")
    if consecutive_rounds <= 0:
        raise AuditError("manifest consecutive_rounds must be positive")
    evaluation = EvaluationSpec(
        seeds=tuple(seeds_raw),
        fixed_checkpoints=tuple(checkpoints_raw),
        max_iterations=max_iterations,
        relative_cost_tolerance=_finite_positive(
            evaluation_raw.get("relative_cost_tolerance"),
            "evaluation.relative_cost_tolerance",
        ),
        gradient_tolerance=_finite_positive(
            evaluation_raw.get("gradient_tolerance"),
            "evaluation.gradient_tolerance",
        ),
        consecutive_rounds=consecutive_rounds,
    )

    return BenchmarkManifest(
        schema_version=raw["schema_version"],
        benchmark_id=benchmark_id,
        description=str(raw.get("description", "")),
        native_multi_robot=native_multi_robot,
        require_temporal_overlap=require_temporal_overlap,
        source=_mapping(raw.get("source"), "source"),
        sources=tuple(sources),
        required_message_types=MappingProxyType(required),
        topic_aliases=MappingProxyType(aliases),
        extraction=_mapping(raw.get("extraction"), "extraction"),
        association=_mapping(raw.get("association"), "association"),
        objective=_mapping(raw.get("objective"), "objective"),
        initialization=_mapping(raw.get("initialization"), "initialization"),
        evaluation=evaluation,
        path=path.resolve(),
    )


def _normalize_message_type(name: str) -> str:
    parts = name.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return name


def _has_required_type(
    inventory: BagInventory, accepted_types: Sequence[str]
) -> bool:
    accepted = {_normalize_message_type(value) for value in accepted_types}
    return any(
        topic.message_count > 0
        and _normalize_message_type(topic.message_type) in accepted
        for topic in inventory.topics.values()
    )


def audit_inventories(
    manifest: BenchmarkManifest, inventories: Sequence[BagInventory]
) -> AuditReport:
    """Prove that inventories represent the required native team experiment."""

    if len(inventories) != len(manifest.sources):
        raise AuditError(
            f"audit requires exactly {len(manifest.sources)} distinct source bags"
        )
    ordered = sorted(inventories, key=lambda item: item.robot_id)
    expected_ids = tuple(source.robot_id for source in manifest.sources)
    actual_ids = tuple(item.robot_id for item in ordered)
    if actual_ids != expected_ids:
        raise AuditError(
            f"source robot IDs {actual_ids} do not match manifest {expected_ids}"
        )
    paths = [str(item.path.resolve()) for item in ordered]
    hashes = [item.sha256.lower() for item in ordered]
    if len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
        raise AuditError("native source bags must have distinct paths and hashes")

    effective_starts: list[int] = []
    effective_ends: list[int] = []
    discarded: dict[int, float] = {}
    sources = {source.robot_id: source for source in manifest.sources}
    for inventory in ordered:
        if inventory.generated_by_trajectory_partition:
            raise AuditError(
                f"robot {inventory.robot_id} inventory was generated by trajectory partition"
            )
        if inventory.size_bytes <= 0:
            raise AuditError(f"robot {inventory.robot_id} bag is empty")
        if not inventory.sha256 or len(inventory.sha256) != 64:
            raise AuditError(f"robot {inventory.robot_id} has an invalid SHA-256")
        if inventory.start_ns < 0 or inventory.end_ns <= inventory.start_ns:
            raise AuditError(f"robot {inventory.robot_id} has an invalid time interval")
        for kind, accepted in manifest.required_message_types.items():
            if not _has_required_type(inventory, accepted):
                raise AuditError(
                    f"robot {inventory.robot_id} has no required {kind} messages"
                )
        offset_seconds = sources[inventory.robot_id].replay_offset_seconds
        offset_ns = int(round(offset_seconds * 1e9))
        effective_start = inventory.start_ns + offset_ns
        if effective_start >= inventory.end_ns:
            raise AuditError(
                f"robot {inventory.robot_id} replay offset removes the whole bag"
            )
        effective_starts.append(effective_start)
        effective_ends.append(inventory.end_ns)
        discarded[inventory.robot_id] = offset_seconds

    overlap_start = max(effective_starts)
    overlap_end = min(effective_ends)
    has_temporal_overlap = overlap_start < overlap_end
    if manifest.require_temporal_overlap and not has_temporal_overlap:
        raise AuditError("effective native team replay windows have no overlap")
    return AuditReport(
        benchmark_id=manifest.benchmark_id,
        native_multi_robot=manifest.native_multi_robot,
        require_temporal_overlap=manifest.require_temporal_overlap,
        temporal_overlap_verified=has_temporal_overlap,
        passed=True,
        robot_ids=actual_ids,
        effective_overlap_start_ns=overlap_start,
        effective_overlap_end_ns=overlap_end,
        discarded_prefix_seconds=MappingProxyType(discarded),
        inventories=tuple(ordered),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_rosbag(path: Path, robot_id: int) -> BagInventory:
    """Read a ROS1 bag inventory without deserializing message payloads."""

    from rosbags.rosbag1 import Reader

    path = Path(path).resolve()
    if not path.is_file():
        raise AuditError(f"ROS bag does not exist: {path}")
    with path.open("rb") as stream:
        version_line = stream.readline(64).decode("ascii", errors="replace").strip()
    if not version_line.startswith("#ROSBAG V"):
        raise AuditError(f"not a ROS1 bag: {path}")
    with Reader(path) as reader:
        topics = {
            name: TopicInventory(
                name=name,
                message_type=info.msgtype or "",
                message_count=int(info.msgcount),
            )
            for name, info in reader.topics.items()
        }
        start_ns = int(reader.start_time)
        end_ns = int(reader.end_time)
    return BagInventory(
        robot_id=robot_id,
        path=path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        bag_version=version_line.removeprefix("#ROSBAG V"),
        start_ns=start_ns,
        end_ns=end_ns,
        topics=MappingProxyType(topics),
    )


def _match_source_files(
    manifest: BenchmarkManifest, raw_dir: Path
) -> tuple[tuple[SourceSpec, Path], ...]:
    candidates = sorted(path for path in raw_dir.rglob("*.bag") if path.is_file())
    matches: list[tuple[SourceSpec, Path]] = []
    for source in manifest.sources:
        source_matches = {
            path.resolve()
            for path in candidates
            if any(
                fnmatch.fnmatch(path.name.lower(), pattern.lower())
                for pattern in source.filename_patterns
            )
        }
        if len(source_matches) != 1:
            raise AuditError(
                f"robot {source.robot_id} matched {len(source_matches)} bag files "
                f"under {raw_dir}; expected exactly one"
            )
        matches.append((source, next(iter(source_matches))))
    paths = [path for _, path in matches]
    if len(set(paths)) != len(paths):
        raise AuditError("one bag file matched more than one source robot")
    return tuple(matches)


def _download_official_folder(url: str, raw_dir: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise AuditError(
            "official bags are absent and Python package 'gdown' is not installed"
        ) from error
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="slideslam-download-", dir=raw_dir.parent))
    try:
        result = gdown.download_folder(
            url=url, output=str(temp_dir), quiet=False, use_cookies=False
        )
        if not result:
            raise AuditError("official Google Drive folder download returned no files")
        for source in sorted(temp_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(temp_dir)
            target = raw_dir / relative
            if target.exists():
                raise AuditError(f"download would replace existing file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def acquire_sources(
    manifest_path: Path, data_root: Path, download: bool = False
) -> AuditReport:
    """Locate, optionally download, inspect, and audit the three source bags."""

    manifest = load_manifest(manifest_path)
    data_root = Path(data_root)
    source_subdirectory = manifest.source.get(
        "data_subdirectory", "raw/parking-lot"
    )
    if not isinstance(source_subdirectory, str) or not source_subdirectory:
        raise AuditError("manifest source data_subdirectory must be non-empty")
    source_path = Path(source_subdirectory)
    raw_dir = source_path if source_path.is_absolute() else data_root / source_path
    try:
        matches = _match_source_files(manifest, raw_dir)
    except AuditError:
        if not download:
            raise
        url = manifest.source.get("download_url")
        if not isinstance(url, str) or not url:
            raise AuditError("manifest source download_url is missing")
        _download_official_folder(url, raw_dir)
        matches = _match_source_files(manifest, raw_dir)
    inventories = [inspect_rosbag(path, source.robot_id) for source, path in matches]
    report = audit_inventories(manifest, inventories)
    _write_json_atomic(
        data_root / "audit" / f"{manifest.benchmark_id}.json", report.to_dict()
    )
    return report


__all__ = [
    "AuditError",
    "AuditReport",
    "BagInventory",
    "BenchmarkManifest",
    "EvaluationSpec",
    "SourceSpec",
    "TopicInventory",
    "acquire_sources",
    "audit_inventories",
    "inspect_rosbag",
    "load_manifest",
]
