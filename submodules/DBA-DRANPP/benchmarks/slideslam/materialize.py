"""Canonical SlideSLAM graph construction and lossless solver adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .acquire import BenchmarkManifest
from .associate import AssociationResult
from .extract import LocalCuboidTrack, RobotExtraction
from .schema import (
    AssociationRecord,
    Factor,
    InitialStateRecord,
    ObjectVertex,
    OdometryFactor,
    PoseObjectFactor,
    PoseVertex,
    SchemaError,
    Vertex,
    canonical_fingerprint,
    validate_canonical_graph,
    write_jsonl_atomic,
)
from .se3 import SE3, g2o_information, perturb_pose, rotation_angle


class MaterializationError(ValueError):
    """Raised when a canonical graph or solver adapter loses information."""


@dataclass(frozen=True)
class CanonicalGraph:
    vertices: tuple[Vertex, ...]
    factors: tuple[Factor, ...]
    associations: tuple[AssociationRecord, ...]
    initial_states: tuple[InitialStateRecord, ...]
    audit: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        try:
            validate_canonical_graph(self.vertices, self.factors)
        except SchemaError as error:
            raise MaterializationError(str(error)) from error
        vertex_ids = {vertex.global_id for vertex in self.vertices}
        object_ids = {
            vertex.global_id
            for vertex in self.vertices
            if isinstance(vertex, ObjectVertex)
        }
        if any(record.global_object_id not in object_ids for record in self.associations):
            raise MaterializationError("association references a non-object vertex")
        seeds = sorted({record.seed for record in self.initial_states})
        if not seeds or seeds[0] != 0:
            raise MaterializationError("initial states must include seed zero")
        for seed in seeds:
            state_ids = {
                record.variable_id
                for record in self.initial_states
                if record.seed == seed
            }
            if state_ids != vertex_ids:
                raise MaterializationError(
                    f"seed {seed} initial state does not cover every canonical vertex"
                )


@dataclass(frozen=True)
class MaterializedPackage:
    root: Path
    larpg_dir: Path
    dran_dir: Path
    distributed_mapper_dir: Path
    canonical_factor_hash: str
    canonical_graph_sha256: str
    association_sha256: str


@dataclass(frozen=True)
class NormalizedFactor:
    factor_id: str
    kind: str
    from_id: int
    to_id: int
    measurement: SE3
    kappa: float
    tau: float


@dataclass(frozen=True)
class AdapterInspection:
    vertex_count: int
    factor_count: int
    factor_hash: str
    states: Mapping[int, SE3]
    factors: tuple[NormalizedFactor, ...]


@dataclass(frozen=True)
class G2OInspection:
    states: Mapping[int, SE3]
    factors: tuple[NormalizedFactor, ...]


@dataclass(frozen=True)
class DistributedMapperPreflight:
    checkout: Path
    binary: Path
    commit: str
    dirty: bool
    dirty_fingerprint: str


def distributed_mapper_preflight(
    checkout: Path, *, binary: Optional[Path] = None
) -> DistributedMapperPreflight:
    checkout = Path(checkout).resolve()
    if not checkout.is_dir():
        raise MaterializationError(
            f"Distributed Mapper checkout is unavailable: {checkout}"
        )
    if binary is None:
        candidates = (
            checkout / "distributed_mapper_core/cpp/build-local/runDistributedMapper",
            checkout / "distributed_mapper_core/cpp/build/runDistributedMapper",
        )
        binary = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    binary = Path(binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise MaterializationError(
            f"Distributed Mapper binary is unavailable or not executable: {binary}"
        )
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise MaterializationError(
            f"failed to inspect Distributed Mapper checkout: {checkout}"
        ) from error
    commit = commit_result.stdout.decode("ascii").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise MaterializationError(
            f"Distributed Mapper checkout returned an invalid commit: {commit!r}"
        )
    status = status_result.stdout
    return DistributedMapperPreflight(
        checkout=checkout,
        binary=binary,
        commit=commit,
        dirty=bool(status),
        dirty_fingerprint=hashlib.sha256(status).hexdigest(),
    )


def _json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_from_factor(factor: Factor) -> NormalizedFactor:
    return NormalizedFactor(
        factor.factor_id,
        factor.kind,
        factor.from_id,
        factor.to_id,
        factor.measurement,
        factor.kappa,
        factor.tau,
    )


def _quantized_factor(factor: NormalizedFactor) -> dict[str, Any]:
    def quantize(value: float) -> float:
        # Ten decimal places remain far below solver tolerances while keeping
        # the wire fingerprint invariant to matrix/quaternion round trips.
        # Direct adapter inspection still enforces 1e-12 numeric equality.
        result = round(float(value), 10)
        return 0.0 if result == 0.0 else result

    return {
        "factor_id": factor.factor_id,
        "kind": factor.kind,
        "from_id": factor.from_id,
        "to_id": factor.to_id,
        "measurement": [
            quantize(value)
            for value in (
                *factor.measurement.translation,
                *factor.measurement.quaternion_xyzw(),
            )
        ],
        "kappa": quantize(factor.kappa),
        "tau": quantize(factor.tau),
    }


def _manifest_factor(factor: NormalizedFactor) -> dict[str, Any]:
    """Serialize enough precision for the adapter's 1e-12 numeric audit."""

    return {
        "factor_id": factor.factor_id,
        "kind": factor.kind,
        "from_id": factor.from_id,
        "to_id": factor.to_id,
        "measurement": [
            float(value) for value in factor.measurement.as_matrix().reshape(-1)
        ],
        "kappa": float(factor.kappa),
        "tau": float(factor.tau),
    }


def _normalized_factor_hash(factors: Iterable[NormalizedFactor]) -> str:
    records = sorted(
        (_quantized_factor(factor) for factor in factors),
        key=lambda item: item["factor_id"],
    )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_factor_hash(factors: Sequence[Factor]) -> str:
    return _normalized_factor_hash(_normalized_from_factor(factor) for factor in factors)


def legacy_canonical_factor_hash(factors: Sequence[Factor]) -> str:
    """Return the matrix/12-decimal fingerprint used by v1 frozen packages."""

    def quantize(value: float) -> float:
        result = round(float(value), 12)
        return 0.0 if result == 0.0 else result

    records = []
    for item in factors:
        factor = _normalized_from_factor(item)
        records.append(
            {
                "factor_id": factor.factor_id,
                "kind": factor.kind,
                "from_id": factor.from_id,
                "to_id": factor.to_id,
                "measurement": [
                    quantize(value)
                    for value in factor.measurement.as_matrix().reshape(-1)
                ],
                "kappa": quantize(factor.kappa),
                "tau": quantize(factor.tau),
            }
        )
    encoded = json.dumps(
        sorted(records, key=lambda item: item["factor_id"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_initial_states(
    vertices: Sequence[Vertex],
    *,
    seeds: Sequence[int],
    translation_bound_m: float,
    rotation_bound_deg: float,
) -> tuple[InitialStateRecord, ...]:
    if tuple(seeds) != tuple(sorted(set(seeds))) or not seeds or seeds[0] != 0:
        raise MaterializationError("seeds must be sorted, unique, and start at zero")
    states = []
    for seed in seeds:
        for vertex in sorted(vertices, key=lambda item: item.global_id):
            fixed = isinstance(vertex, PoseVertex) and vertex.fixed
            pose = (
                vertex.initial_pose
                if fixed
                else perturb_pose(
                    vertex.initial_pose,
                    variable_id=vertex.global_id,
                    seed=seed,
                    translation_bound_m=translation_bound_m,
                    rotation_bound_deg=rotation_bound_deg,
                )
            )
            states.append(
                InitialStateRecord(
                    variable_id=vertex.global_id,
                    variable_kind=("pose" if isinstance(vertex, PoseVertex) else "object"),
                    seed=seed,
                    pose=pose,
                )
            )
    return tuple(states)


def _state_map(graph: CanonicalGraph, seed: int) -> dict[int, SE3]:
    result = {
        record.variable_id: record.pose
        for record in graph.initial_states
        if record.seed == seed
    }
    if len(result) != len(graph.vertices):
        raise MaterializationError(f"seed {seed} is missing canonical initial states")
    return result


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _average_cuboid_poses(poses: Sequence[SE3]) -> SE3:
    if not poses:
        raise MaterializationError("cannot initialize an object without poses")
    reference = poses[0].rotation
    half_turn = np.diag([-1.0, -1.0, 1.0])
    rotations = []
    for pose in poses:
        candidates = (pose.rotation, pose.rotation @ half_turn)
        rotations.append(
            min(candidates, key=lambda value: rotation_angle(reference.T @ value))
        )
    return SE3(
        _project_rotation(np.sum(rotations, axis=0)),
        np.median(np.vstack([pose.translation for pose in poses]), axis=0),
    )


def build_canonical_graph(
    extractions: Sequence[RobotExtraction],
    association: AssociationResult,
    robot_alignments: Mapping[int, SE3],
    manifest: BenchmarkManifest,
    *,
    audit: Optional[Mapping[str, Any]] = None,
) -> CanonicalGraph:
    """Convert frozen extraction/association output into one common graph."""

    ordered_extractions = sorted(extractions, key=lambda item: item.robot_id)
    robot_ids = tuple(item.robot_id for item in ordered_extractions)
    if robot_ids != tuple(source.robot_id for source in manifest.sources):
        raise MaterializationError("extractions do not match manifest source robots")
    if set(robot_alignments) != set(robot_ids):
        raise MaterializationError("one root-frame alignment is required per robot")

    vertices: list[Vertex] = []
    pose_id_by_local: dict[tuple[int, int], int] = {}
    next_id = 0
    fixed_robot = int(manifest.initialization["fixed_robot_id"])
    fixed_local = int(manifest.initialization["fixed_pose_local_index"])
    for extraction in ordered_extractions:
        alignment = robot_alignments[extraction.robot_id]
        for keyframe in extraction.keyframes:
            pose_id_by_local[(extraction.robot_id, keyframe.local_index)] = next_id
            vertices.append(
                PoseVertex(
                    global_id=next_id,
                    robot_id=extraction.robot_id,
                    local_index=keyframe.local_index,
                    timestamp_ns=keyframe.timestamp_ns,
                    initial_pose=alignment.compose(keyframe.odometry_pose),
                    fixed=(
                        extraction.robot_id == fixed_robot
                        and keyframe.local_index == fixed_local
                    ),
                )
            )
            next_id += 1

    track_by_id: dict[str, LocalCuboidTrack] = {}
    for extraction in ordered_extractions:
        for track in extraction.tracks.accepted:
            if track.track_id in track_by_id:
                raise MaterializationError(f"duplicate accepted track {track.track_id}")
            track_by_id[track.track_id] = track
    object_id_by_track: dict[str, int] = {}
    canonical_object_id_by_association: dict[int, int] = {}
    for item in association.global_objects:
        canonical_id = next_id
        next_id += 1
        canonical_object_id_by_association[item.global_object_id] = canonical_id
        aligned_poses = []
        for track_id in item.member_track_ids:
            if track_id not in track_by_id:
                raise MaterializationError(
                    f"association references rejected or missing track {track_id}"
                )
            track = track_by_id[track_id]
            object_id_by_track[track_id] = canonical_id
            aligned_poses.append(
                robot_alignments[track.robot_id].compose(track.representative_pose)
            )
        vertices.append(
            ObjectVertex(
                global_id=canonical_id,
                semantic_label=str(item.semantic_label),
                dimensions=item.dimensions,
                observing_robots=item.observing_robots,
                source_track_ids=item.member_track_ids,
                initial_pose=_average_cuboid_poses(aligned_poses),
            )
        )

    factors: list[Factor] = []
    for extraction in ordered_extractions:
        for factor in extraction.odometry_factors:
            from_id = pose_id_by_local[
                (extraction.robot_id, factor.from_local_index)
            ]
            to_id = pose_id_by_local[(extraction.robot_id, factor.to_local_index)]
            factors.append(
                OdometryFactor(
                    factor_id=(
                        f"odom_r{extraction.robot_id}_"
                        f"{factor.from_local_index}_{factor.to_local_index}"
                    ),
                    from_id=from_id,
                    to_id=to_id,
                    measurement=factor.measurement,
                    kappa=factor.kappa,
                    tau=factor.tau,
                    source_covariance=factor.source_covariance,
                )
            )
        keyframe_by_timestamp = {
            keyframe.timestamp_ns: keyframe for keyframe in extraction.keyframes
        }
        for track in extraction.tracks.accepted:
            if track.track_id not in object_id_by_track:
                continue
            object_id = object_id_by_track[track.track_id]
            for observation in track.observations:
                keyframe = keyframe_by_timestamp.get(observation.timestamp_ns)
                if keyframe is None:
                    raise MaterializationError(
                        "semantic observation timestamp is not a retained keyframe"
                    )
                pose_id = pose_id_by_local[
                    (extraction.robot_id, keyframe.local_index)
                ]
                factors.append(
                    PoseObjectFactor(
                        factor_id=(
                            f"obs_r{extraction.robot_id}_k{keyframe.local_index}_"
                            f"d{observation.observation_index}_o{object_id}"
                        ),
                        from_id=pose_id,
                        to_id=object_id,
                        measurement=observation.pose_body_object,
                        kappa=float(manifest.objective["observation_kappa"]),
                        tau=float(manifest.objective["observation_tau"]),
                    )
                )

    incident_distances: dict[str, list[float]] = {}
    for match in association.matches:
        incident_distances.setdefault(match.reference_track_id, []).append(
            match.distance_m
        )
        incident_distances.setdefault(match.query_track_id, []).append(match.distance_m)
    association_records = []
    for item in association.global_objects:
        canonical_id = canonical_object_id_by_association[item.global_object_id]
        for track_id in item.member_track_ids:
            track = track_by_id[track_id]
            distances = incident_distances.get(track_id)
            if not distances:
                raise MaterializationError(
                    f"collaborative track {track_id} has no frozen pair match"
                )
            centroid = robot_alignments[track.robot_id].compose(
                track.representative_pose
            ).translation
            association_records.append(
                AssociationRecord(
                    global_object_id=canonical_id,
                    robot_id=track.robot_id,
                    local_track_id=track_id,
                    aligned_centroid=tuple(float(value) for value in centroid),
                    assignment_distance=min(distances),
                )
            )

    vertices.sort(key=lambda item: item.global_id)
    factors.sort(key=lambda item: item.factor_id)
    initial_states = make_initial_states(
        vertices,
        seeds=manifest.evaluation.seeds,
        translation_bound_m=float(
            manifest.initialization["translation_perturbation_m"]
        ),
        rotation_bound_deg=float(
            manifest.initialization["rotation_perturbation_deg"]
        ),
    )
    graph = CanonicalGraph(
        vertices=tuple(vertices),
        factors=tuple(factors),
        associations=tuple(
            sorted(
                association_records,
                key=lambda item: (item.global_object_id, item.robot_id),
            )
        ),
        initial_states=initial_states,
        audit=dict(audit or {}),
        metadata={
            "benchmark_id": manifest.benchmark_id,
            "official_commit": manifest.source["official_commit"],
            "association_policy": "official_transform_deterministic_hard_projection",
        },
    )
    graph.validate()
    return graph


def _g2o_vertex_line(vertex_id: int, pose: SE3) -> str:
    quaternion = pose.quaternion_xyzw()
    values = [*pose.translation, *quaternion]
    return "VERTEX_SE3:QUAT {} {}".format(
        vertex_id, " ".join(format(float(value), ".17g") for value in values)
    )


def _g2o_upper_information(kappa: float, tau: float) -> list[float]:
    information = g2o_information(kappa, tau)
    return [float(information[row, column]) for row in range(6) for column in range(row, 6)]


def _g2o_edge_line(
    from_id: int, to_id: int, measurement: SE3, kappa: float, tau: float
) -> str:
    quaternion = measurement.quaternion_xyzw()
    values = [
        *measurement.translation,
        *quaternion,
        *_g2o_upper_information(kappa, tau),
    ]
    return "EDGE_SE3:QUAT {} {} {}".format(
        from_id,
        to_id,
        " ".join(format(float(value), ".17g") for value in values),
    )


def _write_g2o(
    path: Path,
    states: Mapping[int, SE3],
    factors: Sequence[NormalizedFactor],
    endpoint_map: Optional[Mapping[int, int]] = None,
    factor_endpoint_map: Optional[Mapping[str, tuple[int, int]]] = None,
) -> None:
    endpoint_map = endpoint_map or {key: key for key in states}
    factor_endpoint_map = factor_endpoint_map or {}
    lines = []
    for global_id, pose in sorted(states.items()):
        lines.append(_g2o_vertex_line(endpoint_map[global_id], pose))
    for factor in sorted(factors, key=lambda item: item.factor_id):
        lines.append(f"# factor {factor.factor_id} {factor.kind}")
        endpoints = factor_endpoint_map.get(
            factor.factor_id,
            (endpoint_map[factor.from_id], endpoint_map[factor.to_id]),
        )
        lines.append(
            _g2o_edge_line(
                endpoints[0],
                endpoints[1],
                factor.measurement,
                factor.kappa,
                factor.tau,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    os.replace(temporary, path)


def _factor_manifest(factors: Sequence[Factor]) -> list[dict[str, Any]]:
    return [_manifest_factor(_normalized_from_factor(factor)) for factor in factors]


def _emit_larpg(graph: CanonicalGraph, root: Path, seeds: Sequence[int]) -> None:
    poses = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, PoseVertex)),
        key=lambda item: item.global_id,
    )
    objects = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, ObjectVertex)),
        key=lambda item: item.global_id,
    )
    private_index = {vertex.global_id: index for index, vertex in enumerate(poses)}
    object_index = {vertex.global_id: index for index, vertex in enumerate(objects)}
    fixed = [private_index[vertex.global_id] for vertex in poses if vertex.fixed]
    if len(fixed) != 1:
        raise MaterializationError("LARPG adapter requires one fixed private pose")
    for seed in seeds:
        states = _state_map(graph, seed)
        problem = {
            "schema_version": "slideslam_object_problem_v1",
            "seed": seed,
            "communication": "full",
            "factor_hash": canonical_factor_hash(graph.factors),
            "fixed_private_pose_index": fixed[0],
            "private_poses": [
                {
                    "index": private_index[vertex.global_id],
                    "id": vertex.global_id,
                    "agent_id": vertex.robot_id,
                    "pose": states[vertex.global_id].to_dict(),
                }
                for vertex in poses
            ],
            "objects": [
                {
                    "index": object_index[vertex.global_id],
                    "id": vertex.global_id,
                    "pose": states[vertex.global_id].to_dict(),
                }
                for vertex in objects
            ],
            "odometry_factors": [
                {
                    "factor_id": factor.factor_id,
                    "from_private_pose_index": private_index[factor.from_id],
                    "to_private_pose_index": private_index[factor.to_id],
                    "measurement": factor.measurement.to_dict(),
                    "kappa": factor.kappa,
                    "tau": factor.tau,
                    "precision_diagonal": [factor.kappa] * 9 + [factor.tau] * 3,
                }
                for factor in graph.factors
                if isinstance(factor, OdometryFactor)
            ],
            "observation_factors": [
                {
                    "factor_id": factor.factor_id,
                    "private_pose_index": private_index[factor.from_id],
                    "object_index": object_index[factor.to_id],
                    "measurement": factor.measurement.to_dict(),
                    "kappa": factor.kappa,
                    "tau": factor.tau,
                    "precision_diagonal": [factor.kappa] * 9 + [factor.tau] * 3,
                }
                for factor in graph.factors
                if isinstance(factor, PoseObjectFactor)
            ],
            "canonical_factors": _factor_manifest(graph.factors),
        }
        _json_atomic(root / f"problem_seed_{seed}.json", problem)


def _emit_dran(graph: CanonicalGraph, root: Path, seeds: Sequence[int]) -> None:
    poses_by_robot: dict[int, list[PoseVertex]] = {}
    vertices_by_id = {vertex.global_id: vertex for vertex in graph.vertices}
    objects = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, ObjectVertex)),
        key=lambda item: item.global_id,
    )
    for vertex in graph.vertices:
        if isinstance(vertex, PoseVertex):
            poses_by_robot.setdefault(vertex.robot_id, []).append(vertex)
    normalized = [_normalized_from_factor(factor) for factor in graph.factors]
    factors_by_robot: dict[int, list[NormalizedFactor]] = {
        robot_id: [] for robot_id in poses_by_robot
    }
    for factor in normalized:
        source_vertex = vertices_by_id[factor.from_id]
        if not isinstance(source_vertex, PoseVertex):
            raise MaterializationError("DRAN factor source must be a pose vertex")
        factors_by_robot[source_vertex.robot_id].append(factor)
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        state_map = _state_map(graph, seed)
        robot_manifests = []
        for robot_id in sorted(poses_by_robot):
            poses = sorted(poses_by_robot[robot_id], key=lambda item: item.local_index)
            if [item.local_index for item in poses] != list(range(len(poses))):
                raise MaterializationError("DRAN requires consecutive local trajectory IDs")
            object_start = len(poses)
            endpoint_map = {
                vertex.global_id: vertex.local_index for vertex in poses
            }
            endpoint_map.update(
                {
                    vertex.global_id: object_start + index
                    for index, vertex in enumerate(objects)
                }
            )
            robot_states = {
                vertex.global_id: state_map[vertex.global_id]
                for vertex in [*poses, *objects]
            }
            robot_factors = factors_by_robot[robot_id]
            _write_g2o(
                seed_root / f"rob_{robot_id}.g2o",
                robot_states,
                robot_factors,
                endpoint_map,
            )
            robot_manifests.append(
                {
                    "robot_id": robot_id,
                    "object_start": object_start,
                    "trajectory_local_to_global": {
                        str(vertex.local_index): vertex.global_id for vertex in poses
                    },
                    "object_local_to_global": {
                        str(object_start + index): vertex.global_id
                        for index, vertex in enumerate(objects)
                    },
                    "factor_ids": [factor.factor_id for factor in sorted(robot_factors, key=lambda item: item.factor_id)],
                }
            )
        _json_atomic(
            seed_root / "adapter_manifest.json",
            {
                "schema_version": "slideslam_dran_adapter_v1",
                "seed": seed,
                "num_robots": len(poses_by_robot),
                "num_objects": len(objects),
                "factor_hash": canonical_factor_hash(graph.factors),
                "robots": robot_manifests,
                "canonical_factors": _factor_manifest(graph.factors),
            },
        )


def _symbol_key(symbol: str, index: int) -> int:
    if len(symbol) != 1 or not (0 <= index < (1 << 56)):
        raise MaterializationError("invalid GTSAM symbol key")
    return (ord(symbol) << 56) | index


def _emit_distributed_mapper(
    graph: CanonicalGraph, root: Path, seeds: Sequence[int]
) -> None:
    robot_ids = sorted(
        {vertex.robot_id for vertex in graph.vertices if isinstance(vertex, PoseVertex)}
    )
    symbols = {robot_id: chr(ord("a") + robot_id) for robot_id in robot_ids}
    if robot_ids != list(range(len(robot_ids))):
        raise MaterializationError(
            "Distributed Mapper requires consecutive robot IDs starting at zero"
        )
    if len(robot_ids) > 26:
        raise MaterializationError("Distributed Mapper supports at most 26 robots")

    poses = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, PoseVertex)),
        key=lambda item: item.global_id,
    )
    objects = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, ObjectVertex)),
        key=lambda item: item.global_id,
    )
    object_index = {vertex.global_id: index for index, vertex in enumerate(objects)}
    states_by_seed = {seed: _state_map(graph, seed) for seed in seeds}
    variables = []
    key_by_vertex_robot: dict[tuple[int, int], int] = {}
    representative_robot_by_object = {
        vertex.global_id: min(vertex.observing_robots) for vertex in objects
    }

    for vertex in poses:
        robot = vertex.robot_id
        key = _symbol_key(symbols[robot], vertex.local_index)
        key_by_vertex_robot[(vertex.global_id, robot)] = key
        variables.append(
            {
                "global_id": vertex.global_id,
                "kind": "pose",
                "robot_id": robot,
                "owner_robot": robot,
                "local_index": vertex.local_index,
                "symbol": symbols[robot],
                "key": key,
                "is_copy": False,
            }
        )
    for vertex in objects:
        local_index = object_index[vertex.global_id]
        for robot in sorted(vertex.observing_robots):
            symbol = symbols[robot].upper()
            key = _symbol_key(symbol, local_index)
            key_by_vertex_robot[(vertex.global_id, robot)] = key
            variables.append(
                {
                    "global_id": vertex.global_id,
                    "kind": "object",
                    "robot_id": robot,
                    "owner_robot": robot,
                    "local_index": local_index,
                    "symbol": symbol,
                    "key": key,
                    "is_copy": True,
                }
            )

    normalized = [_normalized_from_factor(factor) for factor in graph.factors]
    vertices_by_id = {vertex.global_id: vertex for vertex in graph.vertices}
    factor_partitions: dict[str, list[int]] = {}
    factor_endpoints: dict[tuple[int, str], tuple[int, int]] = {}
    for factor in normalized:
        source = vertices_by_id[factor.from_id]
        if not isinstance(source, PoseVertex):
            raise MaterializationError(
                f"Distributed Mapper factor {factor.factor_id} does not start at a pose"
            )
        robot = source.robot_id
        factor_partitions[factor.factor_id] = [robot]
        factor_endpoints[(robot, factor.factor_id)] = (
            key_by_vertex_robot[(factor.from_id, robot)],
            key_by_vertex_robot[(factor.to_id, robot)],
        )

    # The paper's object model represents one physical object by one local
    # uppercase copy per observing robot and connects copies with zero relative
    # pose factors. Point-object benchmarks keep the historical nuisance
    # rotation defaults. Full ObjectPose3 derivatives explicitly override both
    # precisions in canonical metadata so all six object-copy coordinates agree.
    consensus_semantics = str(
        graph.metadata.get(
            "distributed_mapper_object_consensus_semantics", "point2_nuisance_yaw"
        )
    )
    consensus_kappa = float(
        graph.metadata.get("distributed_mapper_object_consensus_kappa", 1e-6)
    )
    consensus_tau = float(
        graph.metadata.get("distributed_mapper_object_consensus_tau", 1.0)
    )
    if consensus_kappa <= 0.0 or consensus_tau <= 0.0:
        raise MaterializationError(
            "Distributed Mapper object consensus precisions must be positive"
        )
    consensus_factors: list[NormalizedFactor] = []
    for vertex in objects:
        observers = sorted(vertex.observing_robots)
        for left_index, left in enumerate(observers):
            for right in observers[left_index + 1 :]:
                factor_id = (
                    f"object_consensus_o{vertex.global_id}_r{left}_r{right}"
                )
                factor = NormalizedFactor(
                    factor_id,
                    "object_copy_consensus",
                    vertex.global_id,
                    vertex.global_id,
                    SE3.identity(),
                    consensus_kappa,
                    consensus_tau,
                )
                consensus_factors.append(factor)
                factor_partitions[factor_id] = [left, right]
                endpoints = (
                    key_by_vertex_robot[(vertex.global_id, left)],
                    key_by_vertex_robot[(vertex.global_id, right)],
                )
                factor_endpoints[(left, factor_id)] = endpoints
                factor_endpoints[(right, factor_id)] = endpoints

    adapter_factors = [*normalized, *consensus_factors]
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for robot_id in robot_ids:
            partition_factors = sorted(
                (
                    factor
                    for factor in adapter_factors
                    if robot_id in factor_partitions[factor.factor_id]
                ),
                key=lambda item: item.factor_id,
            )
            partition_vertex_ids = {
                variable["global_id"]
                for variable in variables
                if variable["robot_id"] == robot_id
            }
            partition_states = {
                global_id: states_by_seed[seed][global_id]
                for global_id in sorted(partition_vertex_ids)
            }
            endpoint_map = {
                variable["global_id"]: variable["key"]
                for variable in variables
                if variable["robot_id"] == robot_id
            }
            endpoint_overrides = {
                factor.factor_id: factor_endpoints[(robot_id, factor.factor_id)]
                for factor in partition_factors
            }
            _write_g2o(
                seed_root / f"{robot_id}.g2o",
                partition_states,
                partition_factors,
                endpoint_map,
                endpoint_overrides,
            )
        # This is deliberately the canonical physical graph, not the augmented
        # copy-consensus graph. It is used only for the native cost reference;
        # the public evaluator reads each robot's optimized file and averages
        # the object copies instead of selecting this representative.
        _write_g2o(
            seed_root / "fullGraph.g2o",
            states_by_seed[seed],
            normalized,
        )
        mapping_lines = ["globalId\trobotId\tlocalId\tkey"]
        for variable in variables:
            if variable["is_copy"] and variable["robot_id"] != representative_robot_by_object[variable["global_id"]]:
                continue
            mapping_lines.append(
                "{global_id}\t{robot_id}\t{local_index}\t{key}".format(**variable)
            )
        (seed_root / "mapping.tsv").write_text(
            "\n".join(mapping_lines) + "\n", encoding="ascii"
        )
        copy_mapping_lines = ["globalId\trobotId\tlocalId\tkey\tsymbol"]
        for variable in variables:
            copy_mapping_lines.append(
                "{global_id}\t{robot_id}\t{local_index}\t{key}\t{symbol}".format(
                    **variable
                )
            )
        (seed_root / "object_copies.tsv").write_text(
            "\n".join(copy_mapping_lines) + "\n", encoding="ascii"
        )
        _json_atomic(
            seed_root / "adapter_manifest.json",
            {
                "schema_version": "slideslam_distributed_mapper_object_adapter_v2",
                "seed": seed,
                "num_robots": len(robot_ids),
                "variables": variables,
                "factor_partitions": factor_partitions,
                "factor_hash": canonical_factor_hash(graph.factors),
                "canonical_factors": _factor_manifest(graph.factors),
                "consensus_factors": [_manifest_factor(factor) for factor in consensus_factors],
                "object_consensus_information": {
                    "kappa": consensus_kappa,
                    "tau": consensus_tau,
                    "semantics": consensus_semantics,
                    "description": (
                        "zero relative Pose3 equality between per-robot object copies"
                    ),
                },
                "object_copy_count": sum(len(vertex.observing_robots) for vertex in objects),
                "use_landmarks": True,
            },
        )


def _write_canonical_g2o(path: Path, graph: CanonicalGraph, seed: int = 0) -> None:
    _write_g2o(
        path,
        _state_map(graph, seed),
        [_normalized_from_factor(factor) for factor in graph.factors],
    )


def materialize_graph(graph: CanonicalGraph, target: Path) -> MaterializedPackage:
    graph.validate()
    target = Path(target)
    if target.exists():
        raise MaterializationError(f"canonical target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        write_jsonl_atomic(temporary / "vertices.jsonl", graph.vertices)
        write_jsonl_atomic(temporary / "factors.jsonl", graph.factors)
        write_jsonl_atomic(temporary / "associations.jsonl", graph.associations)
        write_jsonl_atomic(temporary / "initial_state.jsonl", graph.initial_states)
        _write_canonical_g2o(temporary / "global_graph.g2o", graph)
        _json_atomic(temporary / "audit.json", dict(graph.audit))
        seeds = sorted({record.seed for record in graph.initial_states})
        adapters = temporary / "adapters"
        _emit_larpg(graph, adapters / "larpg", seeds)
        _emit_dran(graph, adapters / "dran", seeds)
        _emit_distributed_mapper(graph, adapters / "distributed_mapper", seeds)

        association_sha256 = _sha256(temporary / "associations.jsonl")
        graph_sha256 = canonical_fingerprint(
            [
                temporary / "vertices.jsonl",
                temporary / "factors.jsonl",
                temporary / "associations.jsonl",
                temporary / "initial_state.jsonl",
                temporary / "global_graph.g2o",
            ]
        )
        factor_hash = canonical_factor_hash(graph.factors)
        _json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": "slideslam_canonical_graph_v1",
                **dict(graph.metadata),
                "vertex_count": len(graph.vertices),
                "factor_count": len(graph.factors),
                "association_count": len(graph.associations),
                "seeds": seeds,
                "canonical_factor_hash": factor_hash,
                "canonical_graph_sha256": graph_sha256,
                "association_sha256": association_sha256,
                "g2o_information_rule": {
                    "translation_diagonal": "tau",
                    "rotation_diagonal": "2*kappa",
                },
            },
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MaterializedPackage(
        root=target,
        larpg_dir=target / "adapters" / "larpg",
        dran_dir=target / "adapters" / "dran",
        distributed_mapper_dir=target / "adapters" / "distributed_mapper",
        canonical_factor_hash=factor_hash,
        canonical_graph_sha256=graph_sha256,
        association_sha256=association_sha256,
    )


def inspect_g2o(path: Path) -> G2OInspection:
    states: dict[int, SE3] = {}
    factors: list[NormalizedFactor] = []
    pending_factor_id: Optional[str] = None
    pending_kind: Optional[str] = None
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# factor "):
            fields = line.split(maxsplit=3)
            if len(fields) != 4:
                raise MaterializationError(f"invalid factor comment at {path}:{line_number}")
            pending_factor_id, pending_kind = fields[2], fields[3]
            continue
        if line.startswith("#") or line.startswith("%"):
            continue
        fields = line.split()
        if fields[0] == "VERTEX_SE3:QUAT":
            if len(fields) != 9:
                raise MaterializationError(f"invalid g2o vertex at {path}:{line_number}")
            vertex_id = int(fields[1])
            if vertex_id in states:
                raise MaterializationError(f"duplicate g2o vertex {vertex_id}")
            states[vertex_id] = SE3.from_quaternion_xyzw(
                [float(value) for value in fields[5:9]],
                [float(value) for value in fields[2:5]],
            )
        elif fields[0] == "EDGE_SE3:QUAT":
            if len(fields) != 31:
                raise MaterializationError(f"invalid g2o edge at {path}:{line_number}")
            from_id, to_id = int(fields[1]), int(fields[2])
            measurement = SE3.from_quaternion_xyzw(
                [float(value) for value in fields[6:10]],
                [float(value) for value in fields[3:6]],
            )
            upper = [float(value) for value in fields[10:31]]
            information = np.zeros((6, 6))
            index = 0
            for row in range(6):
                for column in range(row, 6):
                    information[row, column] = information[column, row] = upper[index]
                    index += 1
            if not np.all(np.isfinite(information)):
                raise MaterializationError("g2o information contains non-finite values")
            off_diagonal = information - np.diag(np.diag(information))
            if np.max(np.abs(off_diagonal)) > 1e-12:
                raise MaterializationError("g2o adapter information must be diagonal")
            if not np.allclose(
                np.diag(information)[:3], information[0, 0], rtol=0.0, atol=1e-12
            ) or not np.allclose(
                np.diag(information)[3:], information[3, 3], rtol=0.0, atol=1e-12
            ):
                raise MaterializationError("g2o adapter precision is not isotropic")
            factor_id = pending_factor_id or f"edge_{len(factors)}"
            kind = pending_kind or "unknown"
            factors.append(
                NormalizedFactor(
                    factor_id,
                    kind,
                    from_id,
                    to_id,
                    measurement,
                    information[3, 3] / 2.0,
                    information[0, 0],
                )
            )
            pending_factor_id = None
            pending_kind = None
        else:
            raise MaterializationError(
                f"unsupported g2o record {fields[0]} at {path}:{line_number}"
            )
    if pending_factor_id is not None:
        raise MaterializationError("factor comment is not followed by an edge")
    return G2OInspection(states, tuple(factors))


def _factor_from_manifest(value: Mapping[str, Any]) -> NormalizedFactor:
    matrix = np.asarray(value["measurement"], dtype=float).reshape(4, 4)
    return NormalizedFactor(
        str(value["factor_id"]),
        str(value["kind"]),
        int(value["from_id"]),
        int(value["to_id"]),
        SE3.from_matrix(matrix),
        float(value["kappa"]),
        float(value["tau"]),
    )


def _assert_factor_equal(actual: NormalizedFactor, expected: NormalizedFactor) -> None:
    if (
        actual.factor_id != expected.factor_id
        or actual.kind != expected.kind
        or actual.from_id != expected.from_id
        or actual.to_id != expected.to_id
        or abs(actual.kappa - expected.kappa) > 1e-12
        or abs(actual.tau - expected.tau) > 1e-12
        or not np.allclose(
            actual.measurement.as_matrix(),
            expected.measurement.as_matrix(),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise MaterializationError(
            f"adapter factor {actual.factor_id} does not match canonical data"
        )


def _verified_factor_hash(
    actual: Sequence[NormalizedFactor],
    expected: Mapping[str, NormalizedFactor],
    declared_hash: str,
) -> str:
    """Verify factors numerically before trusting the canonical wire hash.

    Matrix-to-quaternion-to-matrix round trips can move an entry across a
    decimal quantization boundary even when the matrices differ by less than
    machine precision. Equality is therefore checked directly at 1e-12; the
    fingerprint is recomputed from the canonical factor manifest, which is the
    representation shared by every adapter.
    """

    actual_by_id = {factor.factor_id: factor for factor in actual}
    if len(actual_by_id) != len(actual) or set(actual_by_id) != set(expected):
        raise MaterializationError("adapter factor IDs do not match canonical data")
    for factor_id, factor in actual_by_id.items():
        _assert_factor_equal(factor, expected[factor_id])
    expected_hash = _normalized_factor_hash(expected.values())
    if expected_hash != declared_hash:
        raise MaterializationError("canonical factor manifest fingerprint mismatch")
    return declared_hash


def inspect_larpg_adapter(root: Path, seed: int) -> AdapterInspection:
    value = json.loads((Path(root) / f"problem_seed_{seed}.json").read_text())
    if value.get("schema_version") != "slideslam_object_problem_v1":
        raise MaterializationError("unsupported LARPG adapter schema")
    private_id = {
        int(item["index"]): int(item["id"]) for item in value["private_poses"]
    }
    object_id = {int(item["index"]): int(item["id"]) for item in value["objects"]}
    states = {
        int(item["id"]): SE3.from_dict(item["pose"])
        for item in [*value["private_poses"], *value["objects"]]
    }
    factors = []
    for item in value["odometry_factors"]:
        if item["precision_diagonal"] != [item["kappa"]] * 9 + [item["tau"]] * 3:
            raise MaterializationError("LARPG odometry precision diagonal is invalid")
        factors.append(
            NormalizedFactor(
                item["factor_id"],
                "odometry",
                private_id[int(item["from_private_pose_index"])],
                private_id[int(item["to_private_pose_index"])],
                SE3.from_dict(item["measurement"]),
                float(item["kappa"]),
                float(item["tau"]),
            )
        )
    for item in value["observation_factors"]:
        if item["precision_diagonal"] != [item["kappa"]] * 9 + [item["tau"]] * 3:
            raise MaterializationError("LARPG observation precision diagonal is invalid")
        factors.append(
            NormalizedFactor(
                item["factor_id"],
                "pose_object_observation",
                private_id[int(item["private_pose_index"])],
                object_id[int(item["object_index"])],
                SE3.from_dict(item["measurement"]),
                float(item["kappa"]),
                float(item["tau"]),
            )
        )
    expected = {
        item.factor_id: item
        for item in (_factor_from_manifest(record) for record in value["canonical_factors"])
    }
    factor_hash = _verified_factor_hash(factors, expected, value["factor_hash"])
    return AdapterInspection(len(states), len(factors), factor_hash, states, tuple(factors))


def inspect_dran_adapter(root: Path, seed: int) -> AdapterInspection:
    seed_root = Path(root) / f"seed_{seed}"
    manifest = json.loads((seed_root / "adapter_manifest.json").read_text())
    states: dict[int, SE3] = {}
    factors: list[NormalizedFactor] = []
    expected = {
        item.factor_id: item
        for item in (
            _factor_from_manifest(record) for record in manifest["canonical_factors"]
        )
    }
    for robot in manifest["robots"]:
        parsed = inspect_g2o(seed_root / f"rob_{robot['robot_id']}.g2o")
        local_to_global = {
            **{
                int(local): int(global_id)
                for local, global_id in robot["trajectory_local_to_global"].items()
            },
            **{
                int(local): int(global_id)
                for local, global_id in robot["object_local_to_global"].items()
            },
        }
        for local_id, pose in parsed.states.items():
            global_id = local_to_global[local_id]
            if global_id in states and not np.allclose(
                states[global_id].as_matrix(), pose.as_matrix(), rtol=0.0, atol=1e-12
            ):
                raise MaterializationError("DRAN object copies have different initial states")
            states[global_id] = pose
        for factor in parsed.factors:
            normalized = NormalizedFactor(
                factor.factor_id,
                factor.kind,
                local_to_global[factor.from_id],
                local_to_global[factor.to_id],
                factor.measurement,
                factor.kappa,
                factor.tau,
            )
            _assert_factor_equal(normalized, expected[normalized.factor_id])
            factors.append(normalized)
    factor_hash = _verified_factor_hash(
        factors, expected, manifest["factor_hash"]
    )
    return AdapterInspection(len(states), len(factors), factor_hash, states, tuple(factors))


def inspect_distributed_mapper_adapter(root: Path, seed: int) -> AdapterInspection:
    seed_root = Path(root) / f"seed_{seed}"
    manifest = json.loads((seed_root / "adapter_manifest.json").read_text())
    if manifest.get("schema_version") != "slideslam_distributed_mapper_object_adapter_v2":
        raise MaterializationError("unsupported Distributed Mapper adapter schema")
    variables = {int(item["key"]): item for item in manifest["variables"]}
    key_to_global = {
        key: int(item["global_id"]) for key, item in variables.items()
    }
    expected = {
        item.factor_id: item
        for item in (
            _factor_from_manifest(record) for record in manifest["canonical_factors"]
        )
    }
    expected_consensus = {
        item.factor_id: item
        for item in (
            _factor_from_manifest(record)
            for record in manifest.get("consensus_factors", [])
        )
    }
    expected_all = {**expected, **expected_consensus}
    states: dict[int, SE3] = {}
    copy_states: dict[tuple[int, int], SE3] = {}
    factors_by_id: dict[str, NormalizedFactor] = {}
    observed_partitions: dict[str, list[int]] = {
        factor_id: [] for factor_id in expected_all
    }
    for robot_id in range(int(manifest["num_robots"])):
        parsed = inspect_g2o(seed_root / f"{robot_id}.g2o")
        for key, pose in parsed.states.items():
            if key not in variables:
                raise MaterializationError(
                    f"Distributed Mapper file contains unknown key {key}"
                )
            variable = variables[key]
            global_id = int(variable["global_id"])
            variable_robot = int(variable["robot_id"])
            if variable_robot != robot_id:
                raise MaterializationError(
                    f"Distributed Mapper robot {robot_id} contains key owned by robot {variable_robot}"
                )
            if variable["kind"] == "object":
                copy_key = (global_id, variable_robot)
                if copy_key in copy_states:
                    raise MaterializationError(
                        f"Distributed Mapper duplicate object copy {copy_key}"
                    )
                copy_states[copy_key] = pose
            else:
                if global_id in states and not np.allclose(
                    states[global_id].as_matrix(),
                    pose.as_matrix(),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise MaterializationError(
                        "Distributed Mapper trajectory copies have different initial states"
                    )
                states[global_id] = pose
        for factor in parsed.factors:
            normalized = NormalizedFactor(
                factor.factor_id,
                factor.kind,
                key_to_global[factor.from_id],
                key_to_global[factor.to_id],
                factor.measurement,
                factor.kappa,
                factor.tau,
            )
            if normalized.factor_id not in expected_all:
                raise MaterializationError(
                    f"Distributed Mapper contains unknown factor {normalized.factor_id}"
                )
            _assert_factor_equal(normalized, expected_all[normalized.factor_id])
            if normalized.factor_id in factors_by_id:
                _assert_factor_equal(normalized, factors_by_id[normalized.factor_id])
            factors_by_id[normalized.factor_id] = normalized
            observed_partitions[normalized.factor_id].append(robot_id)
    for factor_id, partitions in manifest["factor_partitions"].items():
        if sorted(observed_partitions[factor_id]) != sorted(partitions):
            raise MaterializationError(
                f"Distributed Mapper factor {factor_id} partition multiplicity mismatch"
            )
    full_graph = inspect_g2o(seed_root / "fullGraph.g2o")
    canonical_ids = set(int(item["global_id"]) for item in variables.values())
    for factor in full_graph.factors:
        if factor.factor_id not in expected:
            raise MaterializationError(
                f"Distributed Mapper full graph contains non-canonical factor {factor.factor_id}"
            )
        try:
            # ``fullGraph.g2o`` is the evaluator input, so its endpoints are
            # canonical global IDs. The robot files use full GTSAM Symbol keys.
            from_global = (
                key_to_global[factor.from_id]
                if factor.from_id in key_to_global
                else factor.from_id
            )
            to_global = (
                key_to_global[factor.to_id]
                if factor.to_id in key_to_global
                else factor.to_id
            )
            if from_global not in canonical_ids or to_global not in canonical_ids:
                raise KeyError
            normalized = NormalizedFactor(
                factor.factor_id,
                factor.kind,
                from_global,
                to_global,
                factor.measurement,
                factor.kappa,
                factor.tau,
            )
        except KeyError as error:
            raise MaterializationError(
                "Distributed Mapper full graph contains an unknown endpoint"
            ) from error
        _assert_factor_equal(normalized, expected[factor.factor_id])
    if set(factors_by_id) != set(expected_all):
        missing = sorted(set(expected_all) - set(factors_by_id))
        raise MaterializationError(
            f"Distributed Mapper adapter is missing factors: {missing}"
        )
    for variable in variables.values():
        if variable["kind"] == "object":
            copy_key = (int(variable["global_id"]), int(variable["robot_id"]))
            if copy_key not in copy_states:
                raise MaterializationError(
                    f"Distributed Mapper is missing object copy {copy_key}"
                )
            if int(variable["global_id"]) in states and not np.allclose(
                states[int(variable["global_id"])].as_matrix(),
                copy_states[copy_key].as_matrix(),
                rtol=0.0,
                atol=1e-12,
            ):
                raise MaterializationError(
                    "Distributed Mapper object copies have different initial states"
                )
            states[int(variable["global_id"])] = copy_states[copy_key]
    factors = tuple(
        sorted(
            (factors_by_id[factor_id] for factor_id in expected),
            key=lambda item: item.factor_id,
        )
    )
    factor_hash = _verified_factor_hash(
        factors, expected, manifest["factor_hash"]
    )
    return AdapterInspection(len(states), len(factors), factor_hash, states, factors)


__all__ = [
    "AdapterInspection",
    "CanonicalGraph",
    "DistributedMapperPreflight",
    "G2OInspection",
    "MaterializationError",
    "MaterializedPackage",
    "NormalizedFactor",
    "build_canonical_graph",
    "canonical_factor_hash",
    "distributed_mapper_preflight",
    "inspect_distributed_mapper_adapter",
    "inspect_dran_adapter",
    "inspect_g2o",
    "inspect_larpg_adapter",
    "legacy_canonical_factor_hash",
    "make_initial_states",
    "materialize_graph",
]
