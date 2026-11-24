"""Canonical graph records and deterministic serialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Optional, Sequence, Type, TypeVar, Union

import numpy as np

from .se3 import SE3


class SchemaError(ValueError):
    """Raised when canonical graph data violates the benchmark schema."""


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SchemaError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise SchemaError(f"{name} must be at least {minimum}")
    return result


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise SchemaError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise SchemaError(f"{name} must be finite and positive")
    return result


def _finite_tuple(
    value: Sequence[Any], size: int, name: str, *, positive: bool = False
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise SchemaError(f"{name} must contain {size} values")
    return tuple(_finite(item, name, positive=positive) for item in value)


def _pose(value: Any, name: str) -> SE3:
    if not isinstance(value, SE3):
        raise SchemaError(f"{name} must be an SE3 value")
    return value


def _pose_from_dict(value: Any, name: str) -> SE3:
    try:
        return SE3.from_dict(value)
    except (TypeError, ValueError) as error:
        raise SchemaError(f"invalid {name}: {error}") from error


def _covariance(value: Any) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) not in (36, 144):
        raise SchemaError("source covariance must contain 36 or 144 finite values")
    return tuple(_finite(item, "source covariance") for item in value)


@dataclass(frozen=True)
class PoseVertex:
    record_type: ClassVar[str] = "pose_vertex"

    global_id: int
    robot_id: int
    local_index: int
    timestamp_ns: int
    initial_pose: SE3
    fixed: bool = False
    reference_metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_id", _integer(self.global_id, "pose global_id"))
        object.__setattr__(self, "robot_id", _integer(self.robot_id, "pose robot_id"))
        object.__setattr__(self, "local_index", _integer(self.local_index, "pose local_index"))
        object.__setattr__(self, "timestamp_ns", _integer(self.timestamp_ns, "pose timestamp_ns"))
        object.__setattr__(self, "initial_pose", _pose(self.initial_pose, "initial_pose"))
        if not isinstance(self.fixed, bool):
            raise SchemaError("pose fixed flag must be boolean")
        if self.reference_metadata is not None and not isinstance(
            self.reference_metadata, Mapping
        ):
            raise SchemaError("pose reference_metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "record_type": self.record_type,
            "global_id": self.global_id,
            "robot_id": self.robot_id,
            "local_index": self.local_index,
            "timestamp_ns": self.timestamp_ns,
            "initial_pose": self.initial_pose.to_dict(),
            "fixed": self.fixed,
        }
        if self.reference_metadata is not None:
            result["reference_metadata"] = dict(self.reference_metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoseVertex":
        return cls(
            global_id=value.get("global_id"),
            robot_id=value.get("robot_id"),
            local_index=value.get("local_index"),
            timestamp_ns=value.get("timestamp_ns"),
            initial_pose=_pose_from_dict(value.get("initial_pose"), "initial_pose"),
            fixed=value.get("fixed", False),
            reference_metadata=value.get("reference_metadata"),
        )


@dataclass(frozen=True)
class ObjectVertex:
    record_type: ClassVar[str] = "object_vertex"

    global_id: int
    semantic_label: str
    dimensions: tuple[float, float, float]
    observing_robots: tuple[int, ...]
    source_track_ids: tuple[str, ...]
    initial_pose: SE3

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_id", _integer(self.global_id, "object global_id"))
        if not isinstance(self.semantic_label, str) or not self.semantic_label:
            raise SchemaError("object semantic_label must be a non-empty string")
        object.__setattr__(
            self,
            "dimensions",
            _finite_tuple(self.dimensions, 3, "object dimensions", positive=True),
        )
        robots = tuple(_integer(item, "observing robot") for item in self.observing_robots)
        if len(set(robots)) != len(robots) or tuple(sorted(robots)) != robots:
            raise SchemaError("object observing_robots must be sorted and unique")
        tracks = tuple(self.source_track_ids)
        if len(tracks) != len(robots) or any(
            not isinstance(item, str) or not item for item in tracks
        ):
            raise SchemaError("object source_track_ids must match observing_robots")
        object.__setattr__(self, "observing_robots", robots)
        object.__setattr__(self, "source_track_ids", tracks)
        object.__setattr__(self, "initial_pose", _pose(self.initial_pose, "initial_pose"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "global_id": self.global_id,
            "semantic_label": self.semantic_label,
            "dimensions": list(self.dimensions),
            "observing_robots": list(self.observing_robots),
            "source_track_ids": list(self.source_track_ids),
            "initial_pose": self.initial_pose.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectVertex":
        return cls(
            global_id=value.get("global_id"),
            semantic_label=value.get("semantic_label"),
            dimensions=tuple(value.get("dimensions", ())),
            observing_robots=tuple(value.get("observing_robots", ())),
            source_track_ids=tuple(value.get("source_track_ids", ())),
            initial_pose=_pose_from_dict(value.get("initial_pose"), "initial_pose"),
        )


@dataclass(frozen=True)
class OdometryFactor:
    record_type: ClassVar[str] = "factor"
    kind: ClassVar[str] = "odometry"

    factor_id: str
    from_id: int
    to_id: int
    measurement: SE3
    kappa: float
    tau: float
    source_covariance: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _validate_factor(self)

    def to_dict(self) -> dict[str, Any]:
        return _factor_to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OdometryFactor":
        return cls(
            factor_id=value.get("factor_id"),
            from_id=value.get("from_id"),
            to_id=value.get("to_id"),
            measurement=_pose_from_dict(value.get("measurement"), "measurement"),
            kappa=value.get("kappa"),
            tau=value.get("tau"),
            source_covariance=_covariance(value.get("source_covariance")),
        )


@dataclass(frozen=True)
class PoseObjectFactor:
    record_type: ClassVar[str] = "factor"
    kind: ClassVar[str] = "pose_object_observation"

    factor_id: str
    from_id: int
    to_id: int
    measurement: SE3
    kappa: float
    tau: float
    source_covariance: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _validate_factor(self)

    def to_dict(self) -> dict[str, Any]:
        return _factor_to_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoseObjectFactor":
        return cls(
            factor_id=value.get("factor_id"),
            from_id=value.get("from_id"),
            to_id=value.get("to_id"),
            measurement=_pose_from_dict(value.get("measurement"), "measurement"),
            kappa=value.get("kappa"),
            tau=value.get("tau"),
            source_covariance=_covariance(value.get("source_covariance")),
        )


Factor = Union[OdometryFactor, PoseObjectFactor]
Vertex = Union[PoseVertex, ObjectVertex]


def _validate_factor(factor: Factor) -> None:
    if not isinstance(factor.factor_id, str) or not factor.factor_id:
        raise SchemaError("factor_id must be a non-empty string")
    object.__setattr__(factor, "from_id", _integer(factor.from_id, "factor from_id"))
    object.__setattr__(factor, "to_id", _integer(factor.to_id, "factor to_id"))
    if factor.from_id == factor.to_id:
        raise SchemaError("factor endpoints must be different")
    object.__setattr__(factor, "measurement", _pose(factor.measurement, "measurement"))
    object.__setattr__(factor, "kappa", _finite(factor.kappa, "factor kappa", positive=True))
    object.__setattr__(factor, "tau", _finite(factor.tau, "factor tau", positive=True))
    object.__setattr__(factor, "source_covariance", _covariance(
        list(factor.source_covariance) if factor.source_covariance is not None else None
    ))


def _factor_to_dict(factor: Factor) -> dict[str, Any]:
    result = {
        "record_type": factor.record_type,
        "kind": factor.kind,
        "factor_id": factor.factor_id,
        "from_id": factor.from_id,
        "to_id": factor.to_id,
        "measurement": factor.measurement.to_dict(),
        "kappa": factor.kappa,
        "tau": factor.tau,
    }
    if factor.source_covariance is not None:
        result["source_covariance"] = list(factor.source_covariance)
    return result


@dataclass(frozen=True)
class AssociationRecord:
    record_type: ClassVar[str] = "association"

    global_object_id: int
    robot_id: int
    local_track_id: str
    aligned_centroid: tuple[float, float, float]
    assignment_distance: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "global_object_id", _integer(self.global_object_id, "global_object_id")
        )
        object.__setattr__(self, "robot_id", _integer(self.robot_id, "robot_id"))
        if not isinstance(self.local_track_id, str) or not self.local_track_id:
            raise SchemaError("local_track_id must be a non-empty string")
        object.__setattr__(
            self,
            "aligned_centroid",
            _finite_tuple(self.aligned_centroid, 3, "aligned centroid"),
        )
        distance = _finite(self.assignment_distance, "assignment distance")
        if distance < 0.0:
            raise SchemaError("assignment distance must be finite and non-negative")
        object.__setattr__(self, "assignment_distance", distance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "global_object_id": self.global_object_id,
            "robot_id": self.robot_id,
            "local_track_id": self.local_track_id,
            "aligned_centroid": list(self.aligned_centroid),
            "assignment_distance": self.assignment_distance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssociationRecord":
        return cls(
            global_object_id=value.get("global_object_id"),
            robot_id=value.get("robot_id"),
            local_track_id=value.get("local_track_id"),
            aligned_centroid=tuple(value.get("aligned_centroid", ())),
            assignment_distance=value.get("assignment_distance"),
        )


@dataclass(frozen=True)
class InitialStateRecord:
    record_type: ClassVar[str] = "initial_state"

    variable_id: int
    variable_kind: str
    seed: int
    pose: SE3

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", _integer(self.variable_id, "variable_id"))
        if self.variable_kind not in ("pose", "object"):
            raise SchemaError("variable_kind must be pose or object")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(self, "pose", _pose(self.pose, "state pose"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "variable_id": self.variable_id,
            "variable_kind": self.variable_kind,
            "seed": self.seed,
            "pose": self.pose.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InitialStateRecord":
        return cls(
            variable_id=value.get("variable_id"),
            variable_kind=value.get("variable_kind"),
            seed=value.get("seed"),
            pose=_pose_from_dict(value.get("pose"), "state pose"),
        )


CanonicalRecord = Union[
    PoseVertex,
    ObjectVertex,
    OdometryFactor,
    PoseObjectFactor,
    AssociationRecord,
    InitialStateRecord,
]
T = TypeVar("T", bound=CanonicalRecord)


def write_jsonl_atomic(path: Path, records: Iterable[CanonicalRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record.to_dict(), sort_keys=True, separators=(",", ":")
                    )
                )
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl(path: Path, record_type: Type[T]) -> list[T]:
    path = Path(path)
    result: list[T] = []
    expected_record_type = record_type.record_type
    expected_kind = getattr(record_type, "kind", None)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SchemaError(f"cannot read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SchemaError("record must be an object")
            if value.get("record_type") != expected_record_type:
                continue
            if expected_kind is not None and value.get("kind") != expected_kind:
                continue
            result.append(record_type.from_dict(value))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SchemaError(f"invalid {path} line {line_number}: {error}") from error
    return result


def validate_canonical_graph(
    vertices: Sequence[Vertex], factors: Sequence[Factor]
) -> None:
    if not vertices:
        raise SchemaError("canonical graph has no vertices")
    ids = [vertex.global_id for vertex in vertices]
    if len(ids) != len(set(ids)):
        raise SchemaError("canonical graph has a duplicate vertex ID")
    by_id = {vertex.global_id: vertex for vertex in vertices}
    poses = [vertex for vertex in vertices if isinstance(vertex, PoseVertex)]
    objects = [vertex for vertex in vertices if isinstance(vertex, ObjectVertex)]
    anchors = [pose for pose in poses if pose.fixed]
    if len(anchors) != 1:
        raise SchemaError("canonical graph must have exactly one fixed pose")
    if not objects:
        raise SchemaError("canonical graph has no shared object vertices")
    factor_ids = [factor.factor_id for factor in factors]
    if len(factor_ids) != len(set(factor_ids)):
        raise SchemaError("canonical graph has a duplicate factor ID")

    observation_robots: dict[int, set[int]] = {
        object_vertex.global_id: set() for object_vertex in objects
    }
    robot_ids = {pose.robot_id for pose in poses}
    robot_adjacency: dict[int, set[int]] = {robot_id: set() for robot_id in robot_ids}
    for factor in factors:
        if factor.from_id not in by_id or factor.to_id not in by_id:
            raise SchemaError(
                f"factor {factor.factor_id} endpoint is not a canonical vertex"
            )
        from_vertex = by_id[factor.from_id]
        to_vertex = by_id[factor.to_id]
        if isinstance(factor, OdometryFactor):
            if not isinstance(from_vertex, PoseVertex) or not isinstance(
                to_vertex, PoseVertex
            ):
                raise SchemaError("odometry factor endpoints must both be poses")
            if from_vertex.robot_id != to_vertex.robot_id:
                raise SchemaError("odometry factor cannot cross source robots")
        elif isinstance(factor, PoseObjectFactor):
            if not isinstance(from_vertex, PoseVertex) or not isinstance(
                to_vertex, ObjectVertex
            ):
                raise SchemaError(
                    "pose-object factor must point from a pose to an object"
                )
            observation_robots[to_vertex.global_id].add(from_vertex.robot_id)
        else:
            raise SchemaError(f"unsupported factor type {type(factor).__name__}")

    for object_vertex in objects:
        actual = observation_robots[object_vertex.global_id]
        expected = set(object_vertex.observing_robots)
        if actual != expected:
            raise SchemaError(
                f"robot-object incidence metadata for object "
                f"{object_vertex.global_id} does not match its factors"
            )
        if len(actual) < 2:
            raise SchemaError(
                f"object {object_vertex.global_id} is not observed by multiple robots"
            )
        ordered = sorted(actual)
        for index, robot_a in enumerate(ordered):
            for robot_b in ordered[index + 1 :]:
                robot_adjacency[robot_a].add(robot_b)
                robot_adjacency[robot_b].add(robot_a)

    root = min(robot_ids)
    visited = {root}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for neighbor in sorted(robot_adjacency[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if visited != robot_ids:
        raise SchemaError("robot-object incidence graph is disconnected")


def canonical_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    normalized = sorted((Path(path).name, Path(path)) for path in paths)
    names = [name for name, _ in normalized]
    if len(names) != len(set(names)):
        raise SchemaError("canonical fingerprint paths must have unique file names")
    for name, path in normalized:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise SchemaError(f"cannot fingerprint {path}: {error}") from error
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "AssociationRecord",
    "CanonicalRecord",
    "Factor",
    "InitialStateRecord",
    "ObjectVertex",
    "OdometryFactor",
    "PoseObjectFactor",
    "PoseVertex",
    "SchemaError",
    "Vertex",
    "canonical_fingerprint",
    "read_jsonl",
    "validate_canonical_graph",
    "write_jsonl_atomic",
]
