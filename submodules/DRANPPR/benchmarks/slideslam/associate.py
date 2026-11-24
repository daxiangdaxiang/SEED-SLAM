"""Official SlideSLAM pair registration and frozen object-ID projection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .extract import LocalCuboidTrack
from .se3 import SE3


class AssociationError(ValueError):
    """Raised when pair registration cannot produce a valid frozen association."""


@dataclass(frozen=True)
class TrackMatch:
    reference_robot: int
    query_robot: int
    reference_track_id: str
    query_track_id: str
    distance_m: float

    def __post_init__(self) -> None:
        if self.reference_robot == self.query_robot:
            raise AssociationError("a cross-robot track match must use two robots")
        if not self.reference_track_id or not self.query_track_id:
            raise AssociationError("track match IDs must be non-empty")
        if not math.isfinite(self.distance_m) or self.distance_m < 0.0:
            raise AssociationError("track match distance must be finite and non-negative")


@dataclass(frozen=True)
class PairRegistration:
    reference_robot: int
    query_robot: int
    accepted: bool
    support: int
    transform_reference_from_query: SE3
    reference_count: int
    query_count: int
    projected_matches: tuple[TrackMatch, ...] = ()
    command: tuple[str, ...] = ()
    binary_sha256: str = ""
    official_commit: str = ""
    solve_seconds: float = 0.0
    wall_seconds: float = 0.0
    pair_directory: str = ""

    def __post_init__(self) -> None:
        if self.reference_robot == self.query_robot:
            raise AssociationError("pair registration requires two different robots")
        if self.support < 0 or self.reference_count < 0 or self.query_count < 0:
            raise AssociationError("pair registration counts must be non-negative")
        if not isinstance(self.transform_reference_from_query, SE3):
            raise AssociationError("pair registration transform must be SE3")
        if not math.isfinite(self.solve_seconds) or self.solve_seconds < 0.0:
            raise AssociationError("pair solve time must be finite and non-negative")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0.0:
            raise AssociationError("pair wall time must be finite and non-negative")


@dataclass(frozen=True)
class PairTree:
    root_robot: int
    robot_ids: tuple[int, ...]
    edges: tuple[PairRegistration, ...]


@dataclass(frozen=True)
class GlobalObjectAssociation:
    global_object_id: int
    member_track_ids: tuple[str, ...]
    observing_robots: tuple[int, ...]
    semantic_label: int
    dimensions: tuple[float, float, float]


@dataclass(frozen=True)
class AssociationResult:
    global_objects: tuple[GlobalObjectAssociation, ...]
    local_only_track_ids: tuple[str, ...]
    matches: tuple[TrackMatch, ...]


class _DisjointSet:
    def __init__(self, values: Sequence[Any]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: Any) -> Any:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: Any, right: Any) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def maximum_support_tree(
    pairs: Sequence[PairRegistration], root_robot: int
) -> PairTree:
    """Build a deterministic maximum-support spanning tree."""

    robot_ids = sorted(
        {robot for pair in pairs for robot in (pair.reference_robot, pair.query_robot)}
    )
    if root_robot not in robot_ids:
        raise AssociationError("pair graph does not contain the requested root robot")
    if len(robot_ids) < 2:
        raise AssociationError("pair graph must contain at least two robots")
    accepted = [pair for pair in pairs if pair.accepted and pair.support > 0]
    accepted.sort(
        key=lambda pair: (
            -pair.support,
            min(pair.reference_robot, pair.query_robot),
            max(pair.reference_robot, pair.query_robot),
            pair.reference_robot,
            pair.query_robot,
        )
    )
    disjoint = _DisjointSet(robot_ids)
    selected = []
    for pair in accepted:
        if disjoint.union(pair.reference_robot, pair.query_robot):
            selected.append(pair)
            if len(selected) == len(robot_ids) - 1:
                break
    if len(selected) != len(robot_ids) - 1:
        raise AssociationError("accepted SlideSLAM pair graph is disconnected")
    return PairTree(root_robot, tuple(robot_ids), tuple(selected))


def compose_robot_alignments(tree: PairTree) -> dict[int, SE3]:
    """Return root-from-robot alignments using the frozen pair tree."""

    alignments = {tree.root_robot: SE3.identity()}
    remaining = list(tree.edges)
    while remaining:
        progress = False
        next_remaining = []
        for edge in remaining:
            reference = edge.reference_robot
            query = edge.query_robot
            reference_from_query = edge.transform_reference_from_query
            if reference in alignments and query not in alignments:
                alignments[query] = alignments[reference].compose(reference_from_query)
                progress = True
            elif query in alignments and reference not in alignments:
                alignments[reference] = alignments[query].compose(
                    reference_from_query.inverse()
                )
                progress = True
            elif reference in alignments and query in alignments:
                predicted = alignments[reference].compose(reference_from_query)
                if not np.allclose(
                    predicted.as_matrix(), alignments[query].as_matrix(), rtol=0.0, atol=1e-9
                ):
                    raise AssociationError("pair tree contains inconsistent transforms")
            else:
                next_remaining.append(edge)
        if not progress and next_remaining:
            raise AssociationError("cannot compose disconnected pair-tree alignments")
        remaining = next_remaining
    if set(alignments) != set(tree.robot_ids):
        raise AssociationError("pair tree did not align every source robot")
    return {robot_id: alignments[robot_id] for robot_id in sorted(alignments)}


def assign_aligned_tracks(
    reference: Sequence[LocalCuboidTrack],
    query: Sequence[LocalCuboidTrack],
    transform_reference_from_query: SE3,
    max_distance: float,
) -> tuple[TrackMatch, ...]:
    """Project a pair transform into deterministic one-to-one hard matches."""

    max_distance = float(max_distance)
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise AssociationError("hard projection distance must be finite and positive")
    if not reference or not query:
        return ()
    reference_sorted = sorted(reference, key=lambda item: (item.robot_id, item.track_id))
    query_sorted = sorted(query, key=lambda item: (item.robot_id, item.track_id))
    reference_robots = {item.robot_id for item in reference_sorted}
    query_robots = {item.robot_id for item in query_sorted}
    if len(reference_robots) != 1 or len(query_robots) != 1:
        raise AssociationError("each pair side must contain one robot's tracks")
    reference_robot = next(iter(reference_robots))
    query_robot = next(iter(query_robots))
    if reference_robot == query_robot:
        raise AssociationError("hard projection requires tracks from different robots")

    # The official core estimates an XY/yaw loop transform. Match the existing
    # SlideSLAM adapter protocol by projecting centroids onto that same plane.
    reference_centroids = np.vstack(
        [item.centroid_local[:2] for item in reference_sorted]
    )
    query_centroids = np.vstack(
        [
            transform_reference_from_query.compose(item.representative_pose).translation[:2]
            for item in query_sorted
        ]
    )
    distances = np.linalg.norm(
        reference_centroids[:, None, :] - query_centroids[None, :, :], axis=2
    )
    rows = np.arange(distances.shape[0])[:, None]
    columns = np.arange(distances.shape[1])[None, :]
    tie_break = np.finfo(float).eps * (rows * (distances.shape[1] + 1) + columns)
    row_indices, column_indices = linear_sum_assignment(distances + tie_break)
    matches = []
    for reference_index, query_index in zip(row_indices, column_indices):
        distance = float(distances[reference_index, query_index])
        if distance <= max_distance:
            matches.append(
                TrackMatch(
                    reference_robot=reference_robot,
                    query_robot=query_robot,
                    reference_track_id=reference_sorted[int(reference_index)].track_id,
                    query_track_id=query_sorted[int(query_index)].track_id,
                    distance_m=distance,
                )
            )
    return tuple(
        sorted(
            matches,
            key=lambda item: (item.reference_track_id, item.query_track_id),
        )
    )


def _validate_incidence(global_objects: Sequence[GlobalObjectAssociation], robot_ids: set[int]) -> None:
    adjacency = {robot_id: set() for robot_id in robot_ids}
    for item in global_objects:
        for index, robot_a in enumerate(item.observing_robots):
            for robot_b in item.observing_robots[index + 1 :]:
                adjacency[robot_a].add(robot_b)
                adjacency[robot_b].add(robot_a)
    root = min(robot_ids)
    visited = {root}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if visited != robot_ids:
        raise AssociationError("retained robot-object incidence graph is disconnected")


def build_global_objects(
    matches: Sequence[TrackMatch],
    tracks: Sequence[LocalCuboidTrack],
    minimum_observing_robots: int,
) -> AssociationResult:
    """Turn pairwise matches into immutable, deterministic global object IDs."""

    if minimum_observing_robots < 2:
        raise AssociationError("global objects must require at least two robots")
    track_by_id = {track.track_id: track for track in tracks}
    if len(track_by_id) != len(tracks):
        raise AssociationError("local track IDs must be globally unique")
    if not track_by_id:
        raise AssociationError("cannot associate an empty local-track set")
    disjoint = _DisjointSet(sorted(track_by_id))
    ordered_matches = sorted(
        matches,
        key=lambda item: (
            item.reference_robot,
            item.query_robot,
            item.reference_track_id,
            item.query_track_id,
        ),
    )
    for match in ordered_matches:
        if match.reference_track_id not in track_by_id or match.query_track_id not in track_by_id:
            raise AssociationError("track match references an unknown local track")
        reference_track = track_by_id[match.reference_track_id]
        query_track = track_by_id[match.query_track_id]
        if (
            reference_track.robot_id != match.reference_robot
            or query_track.robot_id != match.query_robot
        ):
            raise AssociationError("track match robot metadata is inconsistent")
        disjoint.union(match.reference_track_id, match.query_track_id)

    components: dict[str, list[LocalCuboidTrack]] = {}
    for track_id, track in track_by_id.items():
        components.setdefault(disjoint.find(track_id), []).append(track)
    retained_components = []
    local_only = []
    for component in components.values():
        component.sort(key=lambda item: (item.robot_id, item.track_id))
        robots = [item.robot_id for item in component]
        if len(set(robots)) != len(robots):
            raise AssociationError(
                "a global object component contains more than one track from one robot"
            )
        if len(robots) >= minimum_observing_robots:
            retained_components.append(component)
        else:
            local_only.extend(item.track_id for item in component)
    retained_components.sort(key=lambda component: tuple(item.track_id for item in component))
    global_objects = []
    for global_object_id, component in enumerate(retained_components):
        labels, counts = np.unique(
            np.asarray([item.semantic_label for item in component], dtype=np.int64),
            return_counts=True,
        )
        semantic_label = int(np.min(labels[counts == np.max(counts)]))
        dimensions = tuple(
            float(value)
            for value in np.median(
                np.asarray([item.representative_dimensions for item in component]), axis=0
            )
        )
        global_objects.append(
            GlobalObjectAssociation(
                global_object_id=global_object_id,
                member_track_ids=tuple(item.track_id for item in component),
                observing_robots=tuple(item.robot_id for item in component),
                semantic_label=semantic_label,
                dimensions=dimensions,
            )
        )
    if not global_objects:
        raise AssociationError("hard projection retained no collaborative objects")
    _validate_incidence(
        global_objects, {track.robot_id for track in track_by_id.values()}
    )
    return AssociationResult(
        global_objects=tuple(global_objects),
        local_only_track_ids=tuple(sorted(local_only)),
        matches=tuple(ordered_matches),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_track_csv(path: Path, tracks: Sequence[LocalCuboidTrack]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "local_id",
                "category",
                "x",
                "y",
                "z",
                "shape_x",
                "shape_y",
                "shape_z",
            ],
        )
        writer.writeheader()
        for track in sorted(tracks, key=lambda item: item.track_id):
            writer.writerow(
                {
                    "local_id": track.track_id,
                    "category": track.semantic_label,
                    "x": format(float(track.centroid_local[0]), ".17g"),
                    "y": format(float(track.centroid_local[1]), ".17g"),
                    "z": format(float(track.centroid_local[2]), ".17g"),
                    "shape_x": format(track.representative_dimensions[0], ".17g"),
                    "shape_y": format(track.representative_dimensions[1], ".17g"),
                    "shape_z": format(track.representative_dimensions[2], ".17g"),
                }
            )
    os.replace(temporary, path)


def _planar_transform(summary: Mapping[str, Any]) -> SE3:
    rotation_values = summary.get("rotation2d")
    translation_values = summary.get("translation2d")
    if not isinstance(rotation_values, list) or len(rotation_values) != 4:
        raise AssociationError("official adapter returned invalid rotation2d")
    if not isinstance(translation_values, list) or len(translation_values) != 2:
        raise AssociationError("official adapter returned invalid translation2d")
    rotation2d = np.asarray(rotation_values, dtype=float).reshape(2, 2)
    translation2d = np.asarray(translation_values, dtype=float)
    if not np.all(np.isfinite(rotation2d)) or not np.all(np.isfinite(translation2d)):
        raise AssociationError("official adapter returned a non-finite transform")
    if not np.allclose(rotation2d.T @ rotation2d, np.eye(2), rtol=0.0, atol=1e-8):
        raise AssociationError("official adapter rotation2d is not orthonormal")
    if abs(float(np.linalg.det(rotation2d)) - 1.0) > 1e-8:
        raise AssociationError("official adapter rotation2d determinant is not one")
    rotation3d = np.eye(3)
    rotation3d[:2, :2] = rotation2d
    translation3d = np.array([translation2d[0], translation2d[1], 0.0])
    return SE3(rotation3d, translation3d)


def run_pair_registration(
    *,
    reference_tracks: Sequence[LocalCuboidTrack],
    query_tracks: Sequence[LocalCuboidTrack],
    pair_dir: Path,
    binary: Path,
    sigma: float,
    epsilon: float,
    min_pairs: int,
    min_map_objects: int,
    matching_threshold: float,
    projection_distance: float,
    timeout_seconds: float,
    official_commit: str,
) -> PairRegistration:
    """Invoke the compiled official core and record immutable pair provenance."""

    if not reference_tracks or not query_tracks:
        raise AssociationError("official pair registration requires two non-empty maps")
    reference_robots = {track.robot_id for track in reference_tracks}
    query_robots = {track.robot_id for track in query_tracks}
    if len(reference_robots) != 1 or len(query_robots) != 1:
        raise AssociationError("each official pair map must contain exactly one robot")
    reference_robot = next(iter(reference_robots))
    query_robot = next(iter(query_robots))
    if reference_robot == query_robot:
        raise AssociationError("official pair registration requires different robots")
    binary = Path(binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise AssociationError(f"official SlideSLAM adapter is not executable: {binary}")
    pair_dir = Path(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    reference_path = pair_dir / "reference_map.csv"
    query_path = pair_dir / "query_map.csv"
    summary_path = pair_dir / "solver_summary.json"
    _write_track_csv(reference_path, reference_tracks)
    _write_track_csv(query_path, query_tracks)
    command = [
        str(binary),
        "--reference",
        str(reference_path),
        "--query",
        str(query_path),
        "--summary",
        str(summary_path),
        "--sigma",
        str(float(sigma)),
        "--epsilon",
        str(float(epsilon)),
        "--min-pairs",
        str(int(min_pairs)),
        "--min-map-objects",
        str(int(min_map_objects)),
        "--matching-threshold",
        str(float(matching_threshold)),
    ]
    (pair_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AssociationError(f"official SlideSLAM adapter execution failed: {error}") from error
    wall_seconds = time.monotonic() - started
    (pair_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (pair_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise AssociationError(
            f"official SlideSLAM adapter exited {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssociationError(f"cannot read official adapter summary: {error}") from error
    if int(summary.get("reference_count", -1)) != len(reference_tracks) or int(
        summary.get("query_count", -1)
    ) != len(query_tracks):
        raise AssociationError("official adapter summary object counts do not match inputs")
    solver_accepted = summary.get("accepted") is True
    transform = _planar_transform(summary) if solver_accepted else SE3.identity()
    matches = (
        assign_aligned_tracks(
            reference_tracks,
            query_tracks,
            transform,
            max_distance=projection_distance,
        )
        if solver_accepted
        else ()
    )
    accepted = solver_accepted and len(matches) >= int(min_pairs)
    solve_seconds = float(summary.get("solve_seconds", 0.0))
    if not math.isfinite(solve_seconds) or solve_seconds < 0.0:
        raise AssociationError("official adapter returned invalid solve_seconds")
    registration = PairRegistration(
        reference_robot=reference_robot,
        query_robot=query_robot,
        accepted=accepted,
        support=len(matches),
        transform_reference_from_query=transform,
        reference_count=len(reference_tracks),
        query_count=len(query_tracks),
        projected_matches=matches,
        command=tuple(command),
        binary_sha256=_sha256(binary),
        official_commit=official_commit,
        solve_seconds=solve_seconds,
        wall_seconds=wall_seconds,
        pair_directory=str(pair_dir.resolve()),
    )
    _write_json_atomic(
        pair_dir / "provenance.json",
        {
            "reference_robot": reference_robot,
            "query_robot": query_robot,
            "reference_count": len(reference_tracks),
            "query_count": len(query_tracks),
            "solver_accepted": solver_accepted,
            "accepted": accepted,
            "support": len(matches),
            "transform_reference_from_query": transform.to_dict(),
            "projected_matches": [
                {
                    "reference_track_id": match.reference_track_id,
                    "query_track_id": match.query_track_id,
                    "distance_m": match.distance_m,
                }
                for match in matches
            ],
            "command": command,
            "binary_sha256": registration.binary_sha256,
            "official_commit": official_commit,
            "parameters": {
                "sigma": float(sigma),
                "epsilon": float(epsilon),
                "min_pairs": int(min_pairs),
                "min_map_objects": int(min_map_objects),
                "matching_threshold": float(matching_threshold),
                "projection_distance": float(projection_distance),
            },
            "solve_seconds": solve_seconds,
            "wall_seconds": wall_seconds,
            "official_summary": summary,
        },
    )
    return registration


__all__ = [
    "AssociationError",
    "AssociationResult",
    "GlobalObjectAssociation",
    "PairRegistration",
    "PairTree",
    "TrackMatch",
    "assign_aligned_tracks",
    "build_global_objects",
    "compose_robot_alignments",
    "maximum_support_tree",
    "run_pair_registration",
]
