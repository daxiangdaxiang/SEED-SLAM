"""Ground-truth-free MR.CLAM initialization and canonical graph materialization."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.slideslam.materialize import (
    CanonicalGraph,
    MaterializedPackage,
    materialize_graph,
)
from benchmarks.slideslam.schema import (
    AssociationRecord,
    InitialStateRecord,
    ObjectVertex,
    OdometryFactor,
    PoseObjectFactor,
    PoseVertex,
    Vertex,
)
from benchmarks.slideslam.se3 import SE3

from .extract import PlanarPose, RobotExtraction
from .source import MRCLAMManifest


class MRCLAMMaterializationError(ValueError):
    """Raised when the planar MR.CLAM graph cannot be materialized fairly."""


@dataclass(frozen=True)
class PairRegistration:
    target_robot_subject: int
    source_robot_subject: int
    target_from_source: PlanarPose
    common_landmarks: tuple[int, ...]
    residual_rms_m: float


@dataclass(frozen=True)
class CommonInitialization:
    robot_alignments: Mapping[int, PlanarPose]
    object_positions: Mapping[int, np.ndarray]
    local_object_maps: Mapping[int, Mapping[int, np.ndarray]]
    pair_registrations: tuple[PairRegistration, ...]
    spanning_tree: tuple[tuple[int, int], ...]


def planar_pose_to_se3(pose: PlanarPose) -> SE3:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation[np.abs(rotation) < 1e-15] = 0.0
    translation = np.array([pose.x, pose.y, 0.0])
    translation[np.abs(translation) < 1e-15] = 0.0
    return SE3(rotation, translation)


def se3_to_planar_pose(pose: SE3) -> PlanarPose:
    return PlanarPose(
        float(pose.translation[0]),
        float(pose.translation[1]),
        math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])),
    )


def estimate_se2_registration(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
) -> tuple[PlanarPose, float]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise MRCLAMMaterializationError(
            "SE(2) registration requires matching N by 2 point arrays"
        )
    if source.shape[0] < 2 or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise MRCLAMMaterializationError(
            "SE(2) registration requires at least two finite correspondences"
        )
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    if np.linalg.norm(source_zero) <= 1e-12 or np.linalg.norm(target_zero) <= 1e-12:
        raise MRCLAMMaterializationError(
            "SE(2) registration correspondences have no spatial extent"
        )
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transformed = (rotation @ source.T).T + translation
    residual = float(
        np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1)))
    )
    return (
        PlanarPose(
            float(translation[0]),
            float(translation[1]),
            math.atan2(float(rotation[1, 0]), float(rotation[0, 0])),
        ),
        residual,
    )


def _local_object_map(
    extraction: RobotExtraction, *, maximum_timestamp_s: float | None = None
) -> dict[int, np.ndarray]:
    keyframes = {item.local_index: item for item in extraction.keyframes}
    grouped: dict[int, list[np.ndarray]] = {}
    for observation in extraction.object_observations:
        try:
            keyframe = keyframes[observation.keyframe_index]
        except KeyError as error:
            raise MRCLAMMaterializationError(
                "object observation references a missing keyframe"
            ) from error
        if (
            maximum_timestamp_s is not None
            and keyframe.timestamp_s > maximum_timestamp_s
        ):
            continue
        grouped.setdefault(observation.landmark_subject, []).append(
            keyframe.pose.transform_point(observation.translation_body)
        )
    return {
        landmark: np.median(np.vstack(points), axis=0)
        for landmark, points in sorted(grouped.items())
    }


def build_common_initialization(
    extractions: Sequence[RobotExtraction],
    *,
    root_robot_subject: int,
    registration_window_s: float = 30.0,
) -> CommonInitialization:
    ordered = tuple(sorted(extractions, key=lambda item: item.robot_subject))
    subjects = tuple(item.robot_subject for item in ordered)
    if not subjects or len(set(subjects)) != len(subjects):
        raise MRCLAMMaterializationError("robot extractions must be non-empty and unique")
    if root_robot_subject not in subjects:
        raise MRCLAMMaterializationError("root robot extraction is missing")
    if not math.isfinite(registration_window_s) or registration_window_s <= 0.0:
        raise MRCLAMMaterializationError(
            "registration window must be finite and positive"
        )
    common_start = max(item.keyframes[0].timestamp_s for item in ordered)
    registration_end = common_start + registration_window_s
    local_maps = {item.robot_subject: _local_object_map(item) for item in ordered}
    registration_maps = {
        item.robot_subject: _local_object_map(
            item, maximum_timestamp_s=registration_end
        )
        for item in ordered
    }
    if any(len(registration_maps[subject]) < 2 for subject in subjects):
        raise MRCLAMMaterializationError(
            "each robot needs at least two static objects for common initialization"
        )

    registrations: list[PairRegistration] = []
    adjacency: dict[int, list[tuple[int, PlanarPose, int, float]]] = {
        subject: [] for subject in subjects
    }
    for index, target_subject in enumerate(subjects):
        for source_subject in subjects[index + 1 :]:
            common = tuple(
                sorted(
                    set(registration_maps[target_subject])
                    & set(registration_maps[source_subject])
                )
            )
            if len(common) < 2:
                continue
            target_from_source, residual = estimate_se2_registration(
                [registration_maps[source_subject][landmark] for landmark in common],
                [registration_maps[target_subject][landmark] for landmark in common],
            )
            registration = PairRegistration(
                target_robot_subject=target_subject,
                source_robot_subject=source_subject,
                target_from_source=target_from_source,
                common_landmarks=common,
                residual_rms_m=residual,
            )
            registrations.append(registration)
            adjacency[target_subject].append(
                (source_subject, target_from_source, len(common), residual)
            )
            adjacency[source_subject].append(
                (target_subject, target_from_source.inverse(), len(common), residual)
            )

    alignments: dict[int, PlanarPose] = {root_robot_subject: PlanarPose.identity()}
    tree: list[tuple[int, int]] = []
    while len(alignments) < len(subjects):
        candidates = []
        for parent in sorted(alignments):
            for child, parent_from_child, support, residual in adjacency[parent]:
                if child in alignments:
                    continue
                candidates.append(
                    (-support, residual, parent, child, parent_from_child)
                )
        if not candidates:
            raise MRCLAMMaterializationError(
                "shared-object registration graph does not connect all robots"
            )
        _, _, parent, child, parent_from_child = min(
            candidates, key=lambda item: item[:4]
        )
        alignments[child] = alignments[parent].compose(parent_from_child)
        tree.append((parent, child))

    object_samples: dict[int, list[np.ndarray]] = {}
    for subject in subjects:
        for landmark, local_point in registration_maps[subject].items():
            object_samples.setdefault(landmark, []).append(
                alignments[subject].transform_point(local_point)
            )
    for subject in subjects:
        for landmark, local_point in local_maps[subject].items():
            if landmark not in object_samples:
                object_samples.setdefault(landmark, []).append(
                    alignments[subject].transform_point(local_point)
                )
    object_positions = {
        landmark: np.median(np.vstack(points), axis=0)
        for landmark, points in sorted(object_samples.items())
    }
    return CommonInitialization(
        robot_alignments=dict(sorted(alignments.items())),
        object_positions=object_positions,
        local_object_maps=local_maps,
        pair_registrations=tuple(
            sorted(
                registrations,
                key=lambda item: (
                    item.target_robot_subject,
                    item.source_robot_subject,
                ),
            )
        ),
        spanning_tree=tuple(tree),
    )


def _planar_initial_states(
    vertices: Sequence[Vertex],
    *,
    seeds: Sequence[int],
    translation_bound_m: float,
    yaw_bound_deg: float,
) -> tuple[InitialStateRecord, ...]:
    records = []
    yaw_bound = math.radians(float(yaw_bound_deg))
    perturbation_enabled = translation_bound_m > 0.0 or yaw_bound > 0.0
    for seed in seeds:
        for vertex in sorted(vertices, key=lambda item: item.global_id):
            fixed = isinstance(vertex, PoseVertex) and vertex.fixed
            pose = vertex.initial_pose
            if perturbation_enabled and seed != 0 and not fixed:
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(vertex.global_id)])
                )
                radius = float(translation_bound_m) * math.sqrt(float(rng.random()))
                angle = 2.0 * math.pi * float(rng.random())
                planar = se3_to_planar_pose(pose)
                pose = planar_pose_to_se3(
                    PlanarPose(
                        planar.x + radius * math.cos(angle),
                        planar.y + radius * math.sin(angle),
                        planar.yaw + float(rng.uniform(-yaw_bound, yaw_bound)),
                    )
                )
            records.append(
                InitialStateRecord(
                    variable_id=vertex.global_id,
                    variable_kind=("pose" if isinstance(vertex, PoseVertex) else "object"),
                    seed=int(seed),
                    pose=pose,
                )
            )
    return tuple(records)


def build_canonical_graph(
    extractions: Sequence[RobotExtraction],
    manifest: MRCLAMManifest,
    *,
    audit: Mapping[str, Any] | None = None,
) -> CanonicalGraph:
    ordered = tuple(sorted(extractions, key=lambda item: item.robot_subject))
    subjects = tuple(item.robot_subject for item in ordered)
    if subjects != manifest.robot_subjects:
        raise MRCLAMMaterializationError(
            "extractions do not cover the five manifest robot subjects"
        )
    initialization = build_common_initialization(
        ordered,
        root_robot_subject=int(manifest.initialization["root_robot_subject"]),
        registration_window_s=float(
            manifest.initialization["registration_window_s"]
        ),
    )
    robot_id_by_subject = {
        subject: index for index, subject in enumerate(manifest.robot_subjects)
    }
    factor_model = str(manifest.objective.get("factor_model", "unit_isotropic_v0"))
    if factor_model == "range_bearing_median_isotropic_v1":
        odometry_translation_sigma = float(
            manifest.objective["odometry_translation_sigma_m"]
        )
        odometry_yaw_sigma = float(manifest.objective["odometry_yaw_sigma_rad"])
        odometry_tau = 1.0 / odometry_translation_sigma**2
        odometry_kappa = 1.0 / (2.0 * odometry_yaw_sigma**2)
    else:
        odometry_tau = float(manifest.objective["odometry_tau"])
        odometry_kappa = float(manifest.objective["odometry_kappa"])

    vertices: list[Vertex] = []
    pose_id_by_local: dict[tuple[int, int], int] = {}
    next_id = 0
    for extraction in ordered:
        expected_indices = list(range(len(extraction.keyframes)))
        if [item.local_index for item in extraction.keyframes] != expected_indices:
            raise MRCLAMMaterializationError("keyframe indices must be consecutive")
        alignment = initialization.robot_alignments[extraction.robot_subject]
        for keyframe in extraction.keyframes:
            pose_id_by_local[
                (extraction.robot_subject, keyframe.local_index)
            ] = next_id
            vertices.append(
                PoseVertex(
                    global_id=next_id,
                    robot_id=robot_id_by_subject[extraction.robot_subject],
                    local_index=keyframe.local_index,
                    timestamp_ns=int(round(keyframe.timestamp_s * 1e9)),
                    initial_pose=planar_pose_to_se3(
                        alignment.compose(keyframe.pose)
                    ),
                    fixed=(
                        extraction.robot_subject
                        == int(manifest.initialization["root_robot_subject"])
                        and keyframe.local_index == 0
                    ),
                    reference_metadata={
                        "mrclam_robot_subject": extraction.robot_subject,
                        "timestamp_s": keyframe.timestamp_s,
                    },
                )
            )
            next_id += 1

    observations_by_landmark: dict[int, list[tuple[int, Any]]] = {}
    for extraction in ordered:
        for observation in extraction.object_observations:
            observations_by_landmark.setdefault(
                observation.landmark_subject, []
            ).append((extraction.robot_subject, observation))
    eligible_landmarks = tuple(
        landmark
        for landmark in manifest.static_landmark_subjects
        if len({item[0] for item in observations_by_landmark.get(landmark, [])}) >= 2
    )
    if not eligible_landmarks:
        raise MRCLAMMaterializationError("no shared static objects survive extraction")
    object_id_by_subject = {}
    for landmark in eligible_landmarks:
        object_id_by_subject[landmark] = next_id
        observed_subjects = tuple(
            sorted({item[0] for item in observations_by_landmark[landmark]})
        )
        observing_robot_ids = tuple(
            robot_id_by_subject[subject] for subject in observed_subjects
        )
        position = initialization.object_positions[landmark]
        vertices.append(
            ObjectVertex(
                global_id=next_id,
                semantic_label=f"mrclam_cylinder_{landmark}",
                dimensions=(1.0, 1.0, 1.0),
                observing_robots=observing_robot_ids,
                source_track_ids=tuple(
                    f"robot{subject}_landmark{landmark}"
                    for subject in observed_subjects
                ),
                initial_pose=planar_pose_to_se3(
                    PlanarPose(float(position[0]), float(position[1]), 0.0)
                ),
            )
        )
        next_id += 1

    factors = []
    association_records = []
    for extraction in ordered:
        subject = extraction.robot_subject
        robot_id = robot_id_by_subject[subject]
        for factor in extraction.odometry_factors:
            factors.append(
                OdometryFactor(
                    factor_id=(
                        f"odom_s{subject}_"
                        f"{factor.from_keyframe_index}_{factor.to_keyframe_index}"
                    ),
                    from_id=pose_id_by_local[(subject, factor.from_keyframe_index)],
                    to_id=pose_id_by_local[(subject, factor.to_keyframe_index)],
                    measurement=planar_pose_to_se3(factor.measurement),
                    kappa=odometry_kappa,
                    tau=odometry_tau,
                )
            )
        observed_landmarks: set[int] = set()
        for observation in extraction.object_observations:
            if observation.landmark_subject not in object_id_by_subject:
                continue
            object_id = object_id_by_subject[observation.landmark_subject]
            observation_tau = float(manifest.objective["observation_tau"])
            if factor_model == "range_bearing_median_isotropic_v1":
                measurement_range = float(np.linalg.norm(observation.translation_body))
                range_sigma = float(
                    manifest.objective["observation_range_sigma_m"]
                )
                bearing_sigma = float(
                    manifest.objective["observation_bearing_sigma_rad"]
                )
                raw_per_coordinate_variance = 0.5 * (
                    range_sigma**2 + (measurement_range * bearing_sigma) ** 2
                )
                raw_per_coordinate_variance += observation.translation_dispersion_m**2
                median_per_coordinate_variance = (
                    math.pi
                    * raw_per_coordinate_variance
                    / (2.0 * observation.source_count)
                )
                observation_tau = 1.0 / median_per_coordinate_variance
            factors.append(
                PoseObjectFactor(
                    factor_id=(
                        f"obs_s{subject}_k{observation.keyframe_index}_"
                        f"l{observation.landmark_subject}"
                    ),
                    from_id=pose_id_by_local[(subject, observation.keyframe_index)],
                    to_id=object_id,
                    measurement=planar_pose_to_se3(
                        PlanarPose(
                            float(observation.translation_body[0]),
                            float(observation.translation_body[1]),
                            0.0,
                        )
                    ),
                    kappa=float(manifest.objective["observation_kappa"]),
                    tau=observation_tau,
                )
            )
            observed_landmarks.add(observation.landmark_subject)
        alignment = initialization.robot_alignments[subject]
        for landmark in sorted(observed_landmarks):
            local_point = initialization.local_object_maps[subject][landmark]
            aligned = alignment.transform_point(local_point)
            global_point = initialization.object_positions[landmark]
            association_records.append(
                AssociationRecord(
                    global_object_id=object_id_by_subject[landmark],
                    robot_id=robot_id,
                    local_track_id=f"robot{subject}_landmark{landmark}",
                    aligned_centroid=(float(aligned[0]), float(aligned[1]), 0.0),
                    assignment_distance=float(np.linalg.norm(aligned - global_point)),
                )
            )

    vertices.sort(key=lambda item: item.global_id)
    factors.sort(key=lambda item: item.factor_id)
    initial_states = _planar_initial_states(
        vertices,
        seeds=manifest.evaluation.seeds,
        translation_bound_m=float(
            manifest.initialization["translation_perturbation_m"]
        ),
        yaw_bound_deg=float(manifest.initialization["yaw_perturbation_deg"]),
    )
    pair_report = [
        {
            "target_robot_subject": item.target_robot_subject,
            "source_robot_subject": item.source_robot_subject,
            "common_landmark_count": len(item.common_landmarks),
            "residual_rms_m": item.residual_rms_m,
        }
        for item in initialization.pair_registrations
    ]
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
            "dataset_family": "UTIAS MR.CLAM",
            "dataset_number": int(manifest.benchmark_id.split("dataset", 1)[1].split("_", 1)[0]),
            "native_physical_robot_count": len(manifest.robot_subjects),
            "robot_subject_to_canonical_robot_id": {
                str(subject): robot_id_by_subject[subject]
                for subject in manifest.robot_subjects
            },
            "planar_lift": "SE2 odometry and Cartesian point observations lifted to SE3",
            "object_state_semantics": "position_only",
            "observation_rotation_role": "nuisance_orientation_regularizer",
            "observation_rotation_kappa": float(
                manifest.objective["observation_kappa"]
            ),
            "factor_model": factor_model,
            "factor_model_uses_robust_loss": False,
            "distributed_mapper_object_model_compatibility": (
                "diagnostic_only_requires_full_se3_object_observations"
            ),
            "ground_truth_used_for_initialization": False,
            "evaluation_object_ground_truth_label_map": {
                str(source): target
                for source, target in manifest.evaluation.object_ground_truth_label_map
            },
            "initialization_translation_perturbation_m": float(
                manifest.initialization["translation_perturbation_m"]
            ),
            "initialization_yaw_perturbation_deg": float(
                manifest.initialization["yaw_perturbation_deg"]
            ),
            "repeat_initial_states_identical": (
                float(manifest.initialization["translation_perturbation_m"])
                == 0.0
                and float(manifest.initialization["yaw_perturbation_deg"])
                == 0.0
            ),
            "pair_registrations": pair_report,
            "initialization_spanning_tree": [
                list(edge) for edge in initialization.spanning_tree
            ],
        },
    )
    graph.validate()
    return graph


def materialize_mrclam_graph(
    graph: CanonicalGraph, target: str | Path
) -> MaterializedPackage:
    package = materialize_graph(graph, target)
    write_fixed_identity_associations(
        graph, package.dran_dir / "fixed_associations.csv"
    )
    return package


def write_fixed_identity_associations(
    graph: CanonicalGraph, path: str | Path
) -> int:
    """Write the frozen barcode-ID association consumed by DRAN++(Object)."""

    robot_ids = sorted(
        {
            vertex.robot_id
            for vertex in graph.vertices
            if isinstance(vertex, PoseVertex)
        }
    )
    objects = sorted(
        (vertex for vertex in graph.vertices if isinstance(vertex, ObjectVertex)),
        key=lambda item: item.global_id,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "receiver_robot",
                "sender_robot",
                "receiver_object",
                "sender_object",
                "weight",
                "true_object_id",
            ],
        )
        writer.writeheader()
        for receiver in robot_ids:
            for sender in robot_ids:
                if receiver == sender:
                    continue
                for object_index, vertex in enumerate(objects):
                    observers = set(vertex.observing_robots)
                    if receiver not in observers or sender not in observers:
                        continue
                    writer.writerow(
                        {
                            "receiver_robot": receiver,
                            "sender_robot": sender,
                            "receiver_object": object_index,
                            "sender_object": object_index,
                            "weight": "1",
                            "true_object_id": vertex.global_id,
                        }
                    )
                    count += 1
    return count


__all__ = [
    "CommonInitialization",
    "MRCLAMMaterializationError",
    "PairRegistration",
    "build_canonical_graph",
    "build_common_initialization",
    "estimate_se2_registration",
    "materialize_mrclam_graph",
    "planar_pose_to_se3",
    "se3_to_planar_pose",
    "write_fixed_identity_associations",
]
