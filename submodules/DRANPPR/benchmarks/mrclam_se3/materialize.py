"""Canonical assembly for the MR.CLAM-derived synthetic Pose3 benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.mrclam.evaluate import load_ground_truth
from benchmarks.mrclam.extract import extract_dataset
from benchmarks.mrclam.materialize import (
  build_canonical_graph as build_source_graph,
  materialize_mrclam_graph,
)
from benchmarks.mrclam.source import audit_dataset, load_manifest
from benchmarks.slideslam.materialize import (
  CanonicalGraph,
  MaterializedPackage,
  canonical_factor_hash,
  inspect_distributed_mapper_adapter,
  inspect_dran_adapter,
  inspect_larpg_adapter,
)
from benchmarks.slideslam.schema import (
  AssociationRecord,
  InitialStateRecord,
  ObjectVertex,
  OdometryFactor,
  PoseObjectFactor,
  PoseVertex,
  read_jsonl,
)
from benchmarks.slideslam.se3 import SE3, rotation_angle

from .config import (
  GenerationConfig,
  GenerationManifest,
  normalize_object_identity_corrections,
)
from .generate import (
  lift_odometry_factor,
  lift_pose_object_factor,
  planted_object_poses,
  planted_robot_pose,
)
from .initialize import InitializationResult, initialize_from_factors


class MRCLAMSE3MaterializationError(ValueError):
  """Raised when a derived Pose3 graph violates the frozen benchmark contract."""


@dataclass(frozen=True)
class DerivedGraph:
  graph: CanonicalGraph
  planted_reference: Mapping[int, SE3]
  generation_config: GenerationConfig
  source_factor_hash: str
  initialization: InitializationResult


def _object_subject(vertex: ObjectVertex) -> int:
  match = re.fullmatch(r"mrclam_cylinder_(\d+)", vertex.semantic_label)
  if match is None:
    raise MRCLAMSE3MaterializationError(
      f"object {vertex.global_id} lacks an MR.CLAM landmark subject"
    )
  return int(match.group(1))


def _pose_subject(vertex: PoseVertex) -> int:
  metadata = vertex.reference_metadata
  if not isinstance(metadata, Mapping):
    raise MRCLAMSE3MaterializationError(
      f"pose {vertex.global_id} lacks MR.CLAM provenance"
    )
  subject = metadata.get("mrclam_robot_subject")
  if isinstance(subject, bool) or not isinstance(subject, int):
    raise MRCLAMSE3MaterializationError(
      f"pose {vertex.global_id} has an invalid MR.CLAM robot subject"
    )
  return subject


def _yaw(pose: SE3) -> float:
  return math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0]))


def _reference_hash(reference: Mapping[int, SE3]) -> str:
  records = [
    {
      "variable_id": variable_id,
      "pose": reference[variable_id].to_dict(),
    }
    for variable_id in sorted(reference)
  ]
  encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()


def build_derived_graph(
  source_graph: CanonicalGraph,
  source_pose_truth: Mapping[int, SE3],
  source_object_positions: Mapping[int, Sequence[float]],
  config: GenerationConfig,
  *,
  benchmark_id: str,
  repeats: Sequence[int] = (0, 1, 2),
  object_identity_corrections: Mapping[int, int] | None = None,
  object_identity_correction_basis: str = "none",
) -> DerivedGraph:
  """Lift one frozen planar graph while keeping its incidence unchanged."""

  source_graph.validate()
  repeats = tuple(int(item) for item in repeats)
  if repeats != tuple(sorted(set(repeats))) or not repeats or repeats[0] != 0:
    raise MRCLAMSE3MaterializationError(
      "repeats must be sorted, unique, and start at zero"
    )
  pose_vertices = {
    item.global_id: item
    for item in source_graph.vertices
    if isinstance(item, PoseVertex)
  }
  object_vertices = {
    item.global_id: item
    for item in source_graph.vertices
    if isinstance(item, ObjectVertex)
  }
  if set(source_pose_truth) != set(pose_vertices):
    raise MRCLAMSE3MaterializationError(
      "source pose truth must cover every trajectory vertex"
    )
  timestamps = [item.timestamp_ns for item in pose_vertices.values()]
  minimum_timestamp = min(timestamps)
  duration = max(timestamps) - minimum_timestamp
  if duration <= 0:
    # Tiny unit fixtures may contain one timestamp per robot. Their relative
    # timestamps still provide a deterministic profile coordinate.
    duration = 1

  planted: dict[int, SE3] = {}
  for variable_id, vertex in sorted(pose_vertices.items()):
    source_pose = source_pose_truth[variable_id]
    normalized_time = (vertex.timestamp_ns - minimum_timestamp) / duration
    planted[variable_id] = planted_robot_pose(
      float(source_pose.translation[0]),
      float(source_pose.translation[1]),
      _yaw(source_pose),
      float(normalized_time),
      _pose_subject(vertex),
      config,
    )

  reported_subjects_by_object = {
    variable_id: _object_subject(vertex)
    for variable_id, vertex in object_vertices.items()
  }
  identity_corrections = normalize_object_identity_corrections(
    {} if object_identity_corrections is None else object_identity_corrections
  )
  if not isinstance(object_identity_correction_basis, str):
    raise MRCLAMSE3MaterializationError(
      "object identity correction basis must be a string"
    )
  if bool(identity_corrections) == (
    object_identity_correction_basis == "none"
  ):
    raise MRCLAMSE3MaterializationError(
      "object identity correction basis must agree with the correction map"
    )
  unused_corrections = set(identity_corrections) - set(
    reported_subjects_by_object.values()
  )
  if unused_corrections:
    raise MRCLAMSE3MaterializationError(
      "object identity corrections reference reported subjects not present "
      f"in the graph: {sorted(unused_corrections)}"
    )
  physical_subjects_by_object = {
    variable_id: identity_corrections.get(subject, subject)
    for variable_id, subject in reported_subjects_by_object.items()
  }
  required_subjects = set(physical_subjects_by_object.values())
  if not required_subjects <= set(source_object_positions):
    missing = sorted(required_subjects - set(source_object_positions))
    raise MRCLAMSE3MaterializationError(
      f"source object truth is missing landmark subjects {missing}"
    )
  planted_by_subject = planted_object_poses(
    {
      subject: source_object_positions[subject]
      for subject in sorted(required_subjects)
    },
    config,
  )
  planted.update(
    {
      variable_id: planted_by_subject[subject]
      for variable_id, subject in physical_subjects_by_object.items()
    }
  )

  provisional_vertices = []
  for vertex in sorted(source_graph.vertices, key=lambda item: item.global_id):
    if isinstance(vertex, PoseVertex):
      provisional_vertices.append(
        PoseVertex(
          vertex.global_id,
          vertex.robot_id,
          vertex.local_index,
          vertex.timestamp_ns,
          SE3.identity(),
          vertex.fixed,
          vertex.reference_metadata,
        )
      )
    else:
      provisional_vertices.append(
        ObjectVertex(
          vertex.global_id,
          f"mrclam_synthetic_rigid_{physical_subjects_by_object[vertex.global_id]}",
          vertex.dimensions,
          vertex.observing_robots,
          vertex.source_track_ids,
          SE3.identity(),
        )
      )

  lifted_factors = []
  for factor in sorted(source_graph.factors, key=lambda item: item.factor_id):
    if isinstance(factor, OdometryFactor):
      lifted_factors.append(
        lift_odometry_factor(
          factor,
          source_pose_truth[factor.from_id],
          source_pose_truth[factor.to_id],
          planted[factor.from_id],
          planted[factor.to_id],
          config,
        )
      )
    elif isinstance(factor, PoseObjectFactor):
      subject = physical_subjects_by_object[factor.to_id]
      lifted_factors.append(
        lift_pose_object_factor(
          factor,
          source_pose_truth[factor.from_id],
          source_object_positions[subject],
          planted[factor.from_id],
          planted[factor.to_id],
          config,
        )
      )
    else:
      raise MRCLAMSE3MaterializationError(
        f"unsupported source factor type: {type(factor).__name__}"
      )

  initialization = initialize_from_factors(
    provisional_vertices,
    lifted_factors,
    source_graph.associations,
    root_robot_id=next(
      item.robot_id
      for item in provisional_vertices
      if isinstance(item, PoseVertex) and item.fixed
    ),
  )
  vertices = []
  for vertex in provisional_vertices:
    initial_pose = initialization.states[vertex.global_id]
    if isinstance(vertex, PoseVertex):
      vertices.append(
        PoseVertex(
          vertex.global_id,
          vertex.robot_id,
          vertex.local_index,
          vertex.timestamp_ns,
          initial_pose,
          vertex.fixed,
          vertex.reference_metadata,
        )
      )
    else:
      vertices.append(
        ObjectVertex(
          vertex.global_id,
          vertex.semantic_label,
          vertex.dimensions,
          vertex.observing_robots,
          vertex.source_track_ids,
          initial_pose,
        )
      )

  initial_states = tuple(
    InitialStateRecord(
      vertex.global_id,
      "pose" if isinstance(vertex, PoseVertex) else "object",
      repeat,
      initialization.states[vertex.global_id],
    )
    for repeat in repeats
    for vertex in vertices
  )
  observation_factors = [
    factor for factor in lifted_factors if isinstance(factor, PoseObjectFactor)
  ]
  graph = CanonicalGraph(
    vertices=tuple(vertices),
    factors=tuple(lifted_factors),
    associations=source_graph.associations,
    initial_states=initial_states,
    audit=dict(source_graph.audit),
    metadata={
      "benchmark_id": benchmark_id,
      "dataset_family": "MR.CLAM-derived Synthetic SE(3)-ObjectPose3",
      "source_benchmark_id": source_graph.metadata.get("benchmark_id"),
      "native_physical_robot_count": len(
        {item.robot_id for item in vertices if isinstance(item, PoseVertex)}
      ),
      "native_multi_robot_source": True,
      "synthetic_nonplanar_extension": True,
      "object_state_semantics": "full_object_pose3",
      "factor_incidence_preserved": True,
      "ground_truth_used_for_initialization": False,
      "runtime_perturbation": False,
      "repeat_initial_states_identical": True,
      "generation_config": config.to_dict(),
      "object_identity_corrections": {
        str(reported): physical
        for reported, physical in sorted(identity_corrections.items())
      },
      "object_identity_correction_basis": (
        object_identity_correction_basis
      ),
      "initialization_spanning_tree": [
        list(edge) for edge in initialization.spanning_tree
      ],
      "distributed_mapper_object_consensus_semantics": "full_pose3",
      "distributed_mapper_object_consensus_kappa": max(
        factor.kappa for factor in observation_factors
      ),
      "distributed_mapper_object_consensus_tau": max(
        factor.tau for factor in observation_factors
      ),
    },
  )
  graph.validate()
  source_incidence = {
    (item.factor_id, item.from_id, item.to_id) for item in source_graph.factors
  }
  derived_incidence = {
    (item.factor_id, item.from_id, item.to_id) for item in graph.factors
  }
  if source_incidence != derived_incidence:
    raise MRCLAMSE3MaterializationError("factor incidence changed during lifting")
  return DerivedGraph(
    graph,
    dict(sorted(planted.items())),
    config,
    canonical_factor_hash(source_graph.factors),
    initialization,
  )


def build_from_generation_manifest(
  manifest: GenerationManifest, data_root: str | Path
) -> DerivedGraph:
  """Audit/extract the native source and build one derived benchmark graph."""

  source_manifest = load_manifest(manifest.source_manifest)
  audit = audit_dataset(source_manifest, Path(data_root))
  extractions = extract_dataset(source_manifest, audit)
  source_graph = build_source_graph(extractions, source_manifest, audit=audit.to_dict())
  ground_truth = load_ground_truth(source_manifest, audit.dataset_directory)
  source_pose_truth, extrapolated_counts = build_source_pose_truth(
    source_graph, ground_truth
  )
  derived = build_derived_graph(
    source_graph,
    source_pose_truth,
    ground_truth.landmarks,
    manifest.config,
    benchmark_id=manifest.benchmark_id,
    repeats=manifest.repeats,
    object_identity_corrections=manifest.object_identity_corrections,
    object_identity_correction_basis=manifest.object_identity_correction_basis,
  )
  graph = replace(
    derived.graph,
    metadata={
      **dict(derived.graph.metadata),
      "vicon_supported_pose_count": len(source_pose_truth)
      - sum(extrapolated_counts.values()),
      "odometry_extrapolated_pose_count": sum(extrapolated_counts.values()),
      "odometry_extrapolated_pose_count_by_robot_subject": {
        str(subject): count
        for subject, count in sorted(extrapolated_counts.items())
      },
      "reference_boundary_policy": (
        "Vicon interpolation inside support; endpoint-anchored source odometry "
        "relative motion outside support"
      ),
    },
  )
  return replace(derived, graph=graph)


def build_source_pose_truth(
  source_graph: CanonicalGraph, ground_truth: Any
) -> tuple[dict[int, SE3], dict[int, int]]:
  """Interpolate Vicon and explicitly extrapolate sensor-only boundary tails."""

  vertices_by_subject: dict[int, list[PoseVertex]] = {}
  for vertex in source_graph.vertices:
    if isinstance(vertex, PoseVertex):
      vertices_by_subject.setdefault(_pose_subject(vertex), []).append(vertex)
  result = {}
  extrapolated_counts = {}
  for subject, vertices in sorted(vertices_by_subject.items()):
    if subject not in ground_truth.robots:
      raise MRCLAMSE3MaterializationError(
        f"Vicon ground truth is missing robot subject {subject}"
      )
    series = ground_truth.robots[subject]
    ordered = sorted(vertices, key=lambda item: (item.timestamp_ns, item.global_id))
    supported = [
      vertex
      for vertex in ordered
      if series.supports(float(vertex.reference_metadata["timestamp_s"]))
    ]
    if not supported:
      raise MRCLAMSE3MaterializationError(
        f"robot subject {subject} has no Vicon-supported trajectory node"
      )
    for vertex in supported:
      position, yaw = series.interpolate(
        float(vertex.reference_metadata["timestamp_s"])
      )
      result[vertex.global_id] = SE3.from_rotvec(
        [0.0, 0.0, yaw], [position[0], position[1], 0.0]
      )
    first_supported = supported[0]
    last_supported = supported[-1]
    extrapolated = 0
    for vertex in ordered:
      if vertex.global_id in result:
        continue
      anchor = (
        first_supported
        if vertex.timestamp_ns < first_supported.timestamp_ns
        else last_supported
      )
      source_relative = anchor.initial_pose.between(vertex.initial_pose)
      result[vertex.global_id] = result[anchor.global_id].compose(source_relative)
      extrapolated += 1
    extrapolated_counts[subject] = extrapolated
  return dict(sorted(result.items())), extrapolated_counts


def _write_json(path: Path, value: Any) -> None:
  temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
  temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  os.replace(temporary, path)


def _write_reference(path: Path, graph: CanonicalGraph, reference: Mapping[int, SE3]) -> None:
  kinds = {
    vertex.global_id: "pose" if isinstance(vertex, PoseVertex) else "object"
    for vertex in graph.vertices
  }
  lines = [
    json.dumps(
      {
        "record_type": "planted_reference",
        "variable_id": variable_id,
        "variable_kind": kinds[variable_id],
        "pose": reference[variable_id].to_dict(),
      },
      sort_keys=True,
      separators=(",", ":"),
    )
    for variable_id in sorted(reference)
  ]
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_derived_graph(
  derived: DerivedGraph, target: str | Path
) -> MaterializedPackage:
  package = materialize_mrclam_graph(derived.graph, Path(target))
  _write_reference(
    package.root / "planted_reference.jsonl",
    derived.graph,
    derived.planted_reference,
  )
  pose_z = [
    pose.translation[2]
    for variable_id, pose in derived.planted_reference.items()
    if any(
      isinstance(vertex, PoseVertex) and vertex.global_id == variable_id
      for vertex in derived.graph.vertices
    )
  ]
  _write_json(
    package.root / "generation_audit.json",
    {
      "schema_version": "mrclam_synthetic_se3_generation_audit_v1",
      "source_factor_hash": derived.source_factor_hash,
      "derived_factor_hash": package.canonical_factor_hash,
      "planted_reference_sha256": _reference_hash(derived.planted_reference),
      "factor_incidence_preserved": True,
      "ground_truth_used_for_initialization": False,
      "runtime_perturbation": False,
      "repeat_initial_states_identical": True,
      "trajectory_z_range_m": max(pose_z) - min(pose_z),
      "initialization_spanning_tree": [
        list(edge) for edge in derived.initialization.spanning_tree
      ],
      "vicon_supported_pose_count": derived.graph.metadata.get(
        "vicon_supported_pose_count", len(pose_z)
      ),
      "odometry_extrapolated_pose_count": derived.graph.metadata.get(
        "odometry_extrapolated_pose_count", 0
      ),
      "odometry_extrapolated_pose_count_by_robot_subject": (
        derived.graph.metadata.get(
          "odometry_extrapolated_pose_count_by_robot_subject", {}
        )
      ),
      "object_identity_corrections": derived.graph.metadata.get(
        "object_identity_corrections", {}
      ),
      "object_identity_correction_basis": derived.graph.metadata.get(
        "object_identity_correction_basis", "none"
      ),
    },
  )
  _write_json(
    package.root / "derived_manifest.json",
    {
      "schema_version": "mrclam_synthetic_se3_materialized_v1",
      "benchmark_id": derived.graph.metadata["benchmark_id"],
      "claim": (
        "MR.CLAM-derived Synthetic SE(3)-ObjectPose3; non-planar motion "
        "and object orientations are synthetic"
      ),
      "generation": derived.generation_config.to_dict(),
      "object_identity_corrections": derived.graph.metadata.get(
        "object_identity_corrections", {}
      ),
      "object_identity_correction_basis": derived.graph.metadata.get(
        "object_identity_correction_basis", "none"
      ),
      "planted_reference_location": "planted_reference.jsonl",
      "reference_excluded_from_adapters": True,
    },
  )
  return package


def validate_materialized_graph(root: str | Path) -> dict[str, Any]:
  root = Path(root)
  manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
  expected_hash = manifest["canonical_factor_hash"]
  seeds = tuple(int(item) for item in manifest["seeds"])
  if seeds != (0, 1, 2):
    raise MRCLAMSE3MaterializationError("materialized repeats must be 0, 1, and 2")
  inspections = {
    "larpg": inspect_larpg_adapter(root / "adapters/larpg", 0),
    "dran": inspect_dran_adapter(root / "adapters/dran", 0),
    "distributed_mapper": inspect_distributed_mapper_adapter(
      root / "adapters/distributed_mapper", 0
    ),
  }
  if any(item.factor_hash != expected_hash for item in inspections.values()):
    raise MRCLAMSE3MaterializationError("adapter factor hashes do not agree")
  initial_states = read_jsonl(root / "initial_state.jsonl", InitialStateRecord)
  by_seed = {
    seed: [
      record.pose.to_dict()
      for record in initial_states
      if record.seed == seed
    ]
    for seed in seeds
  }
  if not (by_seed[0] == by_seed[1] == by_seed[2]):
    raise MRCLAMSE3MaterializationError("repeat initial states are not identical")
  reference_path = root / "planted_reference.jsonl"
  if not reference_path.is_file():
    raise MRCLAMSE3MaterializationError("planted reference is missing")
  if list((root / "adapters").rglob("*reference*")):
    raise MRCLAMSE3MaterializationError("an adapter contains planted reference data")
  audit = json.loads((root / "generation_audit.json").read_text(encoding="utf-8"))
  if audit["trajectory_z_range_m"] <= 0.0:
    raise MRCLAMSE3MaterializationError("derived graph has no vertical excitation")
  return {
    "valid": True,
    "benchmark_id": manifest["benchmark_id"],
    "vertex_count": manifest["vertex_count"],
    "factor_count": manifest["factor_count"],
    "canonical_factor_hash": expected_hash,
    "adapter_factor_hashes": {
      name: inspection.factor_hash for name, inspection in inspections.items()
    },
    "trajectory_z_range_m": audit["trajectory_z_range_m"],
    "reference_excluded_from_adapters": True,
  }


__all__ = [
  "DerivedGraph",
  "MRCLAMSE3MaterializationError",
  "build_derived_graph",
  "build_from_generation_manifest",
  "build_source_pose_truth",
  "materialize_derived_graph",
  "validate_materialized_graph",
]
