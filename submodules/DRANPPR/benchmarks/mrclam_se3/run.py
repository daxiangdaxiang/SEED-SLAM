"""Complete-topology runner for the derived full ObjectPose3 benchmark."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.mrclam.run import (
  accumulate_distributed_mapper_communication,
  accumulate_dran_communication,
  accumulate_larpg_communication,
  build_edgewise_dran_command,
  default_mrclam_executables,
)
from benchmarks.slideslam.evaluate import (
  evaluate_state,
  read_distributed_mapper_state,
  read_dran_state,
  read_larpg_state,
  read_sesync_state,
)
from benchmarks.slideslam.materialize import CanonicalGraph
from benchmarks.slideslam.run import (
  BenchmarkRunConfig,
  CommandSpec,
  ExecutableSet,
  build_command_specs,
  execute_command,
  load_canonical_package,
)
from benchmarks.slideslam.schema import ObjectVertex, PoseObjectFactor, PoseVertex

from .config import (
  GenerationManifest,
  load_generation_manifest,
  normalize_object_identity_corrections,
)
from .evaluate import evaluate_benchmark_state, read_planted_reference


class MRCLAMSE3RunnerError(RuntimeError):
  """Raised when a full-Pose3 solver violates the comparison protocol."""


PRIMARY_METHODS = (
  "dranpp_objectpose3",
  "larpg_objectpose3",
  "object_dgs_pose3",
)


def validate_generation_provenance(
  canonical_manifest: Mapping[str, Any],
  generation_manifest: GenerationManifest,
) -> None:
  """Reject a same-named canonical package built with different semantics."""

  if canonical_manifest.get("benchmark_id") != generation_manifest.benchmark_id:
    raise MRCLAMSE3RunnerError("generation and canonical benchmark IDs differ")
  if canonical_manifest.get("generation_config") != generation_manifest.config.to_dict():
    raise MRCLAMSE3RunnerError("generation and canonical parameters differ")
  try:
    corrections = normalize_object_identity_corrections(
      canonical_manifest.get("object_identity_corrections")
    )
  except ValueError as error:
    raise MRCLAMSE3RunnerError(
      f"canonical object identity corrections are invalid: {error}"
    ) from error
  if corrections != dict(generation_manifest.object_identity_corrections):
    raise MRCLAMSE3RunnerError(
      "generation and canonical object identity corrections differ"
    )
  if canonical_manifest.get(
    "object_identity_correction_basis"
  ) != generation_manifest.object_identity_correction_basis:
    raise MRCLAMSE3RunnerError(
      "generation and canonical object identity correction bases differ"
    )


@dataclass(frozen=True)
class MRCLAMSE3RunConfig:
  repository_root: Path
  manifest_path: Path
  canonical_dir: Path
  output_dir: Path
  repeats: tuple[int, ...] = (0, 1, 2)
  checkpoints: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
  timeout_seconds: float = 3600.0
  dry_run: bool = False
  dran_beta: float = 0.1
  dran_local_max_iterations: int = 3
  dran_exact_proximal_curvature: bool = False
  dran_proximal_curvature_scale: float = 1.0
  dran_dual_step_size: float = 1.0

  def __post_init__(self) -> None:
    for name in ("repository_root", "manifest_path", "canonical_dir", "output_dir"):
      object.__setattr__(self, name, Path(getattr(self, name)).resolve())
    if self.repeats != tuple(sorted(set(self.repeats))) or not self.repeats:
      raise MRCLAMSE3RunnerError("repeats must be sorted and unique")
    if self.checkpoints != tuple(sorted(set(self.checkpoints))) or not self.checkpoints:
      raise MRCLAMSE3RunnerError("checkpoints must be sorted and unique")
    if self.checkpoints[0] < 1:
      raise MRCLAMSE3RunnerError("checkpoints must be positive")
    if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
      raise MRCLAMSE3RunnerError("timeout must be finite and positive")
    if not math.isfinite(self.dran_beta) or self.dran_beta <= 0.0:
      raise MRCLAMSE3RunnerError("DRAN beta must be finite and positive")
    if self.dran_local_max_iterations < 1:
      raise MRCLAMSE3RunnerError("DRAN local iterations must be positive")
    if (
      not math.isfinite(self.dran_proximal_curvature_scale)
      or self.dran_proximal_curvature_scale < 0.5
      or self.dran_proximal_curvature_scale > 1.0
    ):
      raise MRCLAMSE3RunnerError(
        "DRAN proximal curvature scale must be in [0.5, 1]"
      )
    if (
      not math.isfinite(self.dran_dual_step_size)
      or not 0.0 < self.dran_dual_step_size <= 1.6
    ):
      raise MRCLAMSE3RunnerError("DRAN dual step size must be in (0, 1.6]")


def _replace_option(argv: Sequence[str], option: str, value: str) -> tuple[str, ...]:
  result = list(argv)
  try:
    index = result.index(option)
  except ValueError as error:
    raise MRCLAMSE3RunnerError(f"DRAN command lacks required option {option}") from error
  if index + 1 >= len(result):
    raise MRCLAMSE3RunnerError(f"DRAN command has no value for {option}")
  result[index + 1] = value
  return tuple(result)


def build_dran_pose3_command(
  *,
  binary: Path,
  data_dir: Path,
  association_file: Path,
  output_dir: Path,
  num_robots: int,
  num_objects: int,
  repeat: int,
  iterations: int,
  beta: float = 0.1,
  local_max_iterations: int = 3,
  exact_proximal_curvature: bool = False,
  proximal_curvature_scale: float = 1.0,
  dual_step_size: float = 1.0,
  dual_relaxation_residual_threshold: float = 0.0,
  spectral_dual_step: bool = False,
  global_spectral_dual_step: bool = False,
  spectral_dual_step_min: float = 0.75,
  spectral_dual_step_max: float = 1.5,
  spectral_dual_step_ema_decay: float = 0.5,
  object_interface_mode: str = "copy_baseline",
  object_interface_response_period: int = 1,
  object_interface_pcg_iterations: int = 1,
  object_interface_pcg_preconditioner: str = "private_block_jacobi",
  object_interface_pcg_private_solver: str = "sparse_ldlt",
  object_interface_pcg_post_activation_local_iterations: int = 1,
  object_interface_pcg_terminal_local_polish_iterations: int = 0,
  object_interface_pcg_replace_final_round_with_terminal_polish: bool = False,
  object_interface_pcg_min_translation_spread: float = 0.06,
  object_interface_pcg_min_rotation_spread_deg: float = 1.0,
  object_interface_pcg_min_initial_relative_decrease: float = 0.01,
  object_interface_pcg_min_continuation_relative_decrease: float = 0.0005,
  object_interface_step_gain: float = 1.0,
) -> CommandSpec:
  """Build the DRAN++ lower backend with full six-DoF shared objects."""

  spec = build_edgewise_dran_command(
    binary=Path(binary),
    data_dir=Path(data_dir),
    association_file=Path(association_file),
    output_dir=Path(output_dir),
    num_robots=num_robots,
    num_objects=num_objects,
    seed=repeat,
    iterations=iterations,
    local_max_iterations=local_max_iterations,
    reduced_rotation_preconditioner="cholesky",
    topology="complete",
    beta=beta,
    exact_proximal_curvature=exact_proximal_curvature,
    proximal_curvature_scale=proximal_curvature_scale,
    dual_step_size=dual_step_size,
    dual_relaxation_residual_threshold=dual_relaxation_residual_threshold,
    spectral_dual_step=spectral_dual_step,
    global_spectral_dual_step=global_spectral_dual_step,
    spectral_dual_step_min=spectral_dual_step_min,
    spectral_dual_step_max=spectral_dual_step_max,
    spectral_dual_step_ema_decay=spectral_dual_step_ema_decay,
    point2_consensus_projection=False,
    save_iteration_estimate_snapshots=False,
  )
  argv = _replace_option(
    spec.argv, "--object_orientation_observed", "true"
  )
  argv = _replace_option(
    argv, "--object_communication_representation", "lifted_pose"
  )
  if object_interface_mode not in {
    "copy_baseline",
    "reduced_schur_response",
    "complete_graph_coarse_response",
    "matrix_free_objectpose3_pcg",
  }:
    raise MRCLAMSE3RunnerError("unknown object interface mode")
  if object_interface_pcg_preconditioner not in {
    "interface_hessian",
    "exact_schur",
    "private_block_jacobi",
  }:
    raise MRCLAMSE3RunnerError("unknown object interface PCG preconditioner")
  if object_interface_pcg_private_solver not in {
    "sparse_ldlt",
    "block_tridiagonal_ldlt",
  }:
    raise MRCLAMSE3RunnerError("unknown object interface PCG private solver")
  if (
    object_interface_pcg_replace_final_round_with_terminal_polish
    and object_interface_pcg_terminal_local_polish_iterations <= 0
  ):
    raise MRCLAMSE3RunnerError(
      "final-round replacement requires a positive terminal polish budget"
    )
  argv = _replace_option(argv, "--object_interface_mode", object_interface_mode)
  argv = argv + (
    "--object_interface_response_period",
    str(object_interface_response_period),
    "--object_interface_pcg_iterations",
    str(object_interface_pcg_iterations),
    "--object_interface_pcg_preconditioner",
    object_interface_pcg_preconditioner,
    "--object_interface_pcg_private_solver",
    object_interface_pcg_private_solver,
    "--object_interface_pcg_post_activation_local_iterations",
    str(object_interface_pcg_post_activation_local_iterations),
    "--object_interface_pcg_terminal_local_polish_iterations",
    str(object_interface_pcg_terminal_local_polish_iterations),
    "--object_interface_pcg_replace_final_round_with_terminal_polish",
    (
      "true"
      if object_interface_pcg_replace_final_round_with_terminal_polish
      else "false"
    ),
    "--object_interface_pcg_min_translation_spread",
    str(object_interface_pcg_min_translation_spread),
    "--object_interface_pcg_min_rotation_spread_deg",
    str(object_interface_pcg_min_rotation_spread_deg),
    "--object_interface_pcg_min_initial_relative_decrease",
    str(object_interface_pcg_min_initial_relative_decrease),
    "--object_interface_pcg_min_continuation_relative_decrease",
    str(object_interface_pcg_min_continuation_relative_decrease),
    "--object_interface_step_gain",
    str(object_interface_step_gain),
  )
  curvature_method = "dranpp_objectpose3"
  if exact_proximal_curvature:
    curvature_method += "_exact_curvature"
  elif proximal_curvature_scale < 1.0:
    curvature_method += (
      f"_curvature_{proximal_curvature_scale:g}".replace(".", "p")
    )
  if dual_step_size != 1.0:
    curvature_method += f"_dual_{dual_step_size:g}".replace(".", "p")
  if dual_relaxation_residual_threshold > 0.0:
    curvature_method += (
      f"_settle_{dual_relaxation_residual_threshold:g}".replace(".", "p")
    )
  if spectral_dual_step:
    curvature_method += "_spectral"
  if global_spectral_dual_step:
    curvature_method += "_global_spectral"
  if object_interface_mode == "reduced_schur_response":
    curvature_method += "_schur"
  elif object_interface_mode == "complete_graph_coarse_response":
    curvature_method += "_coarse"
  elif object_interface_mode == "matrix_free_objectpose3_pcg":
    curvature_method += f"_dpcg{object_interface_pcg_iterations}"
    curvature_method += f"_{object_interface_pcg_preconditioner}"
    curvature_method += f"_p{object_interface_response_period}"
    if object_interface_pcg_private_solver != "sparse_ldlt":
      curvature_method += f"_{object_interface_pcg_private_solver}"
    if object_interface_pcg_terminal_local_polish_iterations > 0:
      curvature_method += (
        f"_polish{object_interface_pcg_terminal_local_polish_iterations}"
      )
    if object_interface_pcg_replace_final_round_with_terminal_polish:
      curvature_method += "_replace_final"
  return replace(
    spec,
    name=curvature_method + f"_repeat_{repeat}_round_{iterations}",
    method=curvature_method,
    argv=argv,
  )


def initial_rows(
  graph: CanonicalGraph,
  *,
  repeats: Sequence[int],
  methods: Sequence[str] = PRIMARY_METHODS,
) -> tuple[dict[str, Any], ...]:
  state_by_repeat = {
    repeat: {
      record.variable_id: record.pose
      for record in graph.initial_states
      if record.seed == repeat
    }
    for repeat in repeats
  }
  expected_ids = {vertex.global_id for vertex in graph.vertices}
  if any(set(state) != expected_ids for state in state_by_repeat.values()):
    raise MRCLAMSE3RunnerError("a repeat is missing common initial states")
  signatures = {
    json.dumps(
      {
        variable_id: pose.to_dict()
        for variable_id, pose in sorted(state.items())
      },
      sort_keys=True,
      separators=(",", ":"),
    )
    for state in state_by_repeat.values()
  }
  if len(signatures) != 1:
    raise MRCLAMSE3RunnerError("repeat initial states are not byte-identical")
  objective = evaluate_state(graph, next(iter(state_by_repeat.values())))
  return tuple(
    {
      "method": method,
      "repeat": int(repeat),
      "iteration": 0,
      "total_cost": objective.cost,
      "mean_factor_cost": objective.cost / len(graph.factors),
      "logical_bytes": 0,
      "setup_logical_bytes": 0,
      "iterative_logical_bytes": 0,
      "payload_blocks": 0,
      "messages": 0,
      "solver_wall_time_s": 0.0,
    }
    for repeat in repeats
    for method in methods
  )


def _legacy_config(config: MRCLAMSE3RunConfig) -> BenchmarkRunConfig:
  return BenchmarkRunConfig(
    repository_root=config.repository_root,
    manifest_path=config.manifest_path,
    data_root=config.repository_root / "datasets/MRCLAM/raw",
    canonical_dir=config.canonical_dir,
    output_dir=config.output_dir,
    seeds=config.repeats,
    checkpoints=config.checkpoints,
    max_iterations=config.checkpoints[-1],
    timeout_seconds=config.timeout_seconds,
    dry_run=config.dry_run,
  )


def _graph_sizes(graph: CanonicalGraph) -> tuple[int, int, int]:
  robot_ids = sorted(
    {item.robot_id for item in graph.vertices if isinstance(item, PoseVertex)}
  )
  objects = [item for item in graph.vertices if isinstance(item, ObjectVertex)]
  if robot_ids != list(range(len(robot_ids))) or not objects:
    raise MRCLAMSE3RunnerError("canonical graph is not solver-adapter compatible")
  active_blocks = len(
    {
      (next(
        pose.robot_id
        for pose in graph.vertices
        if isinstance(pose, PoseVertex) and pose.global_id == factor.from_id
      ), factor.to_id)
      for factor in graph.factors
      if isinstance(factor, PoseObjectFactor)
    }
  )
  return len(robot_ids), len(objects), active_blocks


def _fixed_association_counts(path: Path) -> tuple[int, int]:
  with Path(path).open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  entries = {
    (
      int(row["receiver_robot"]),
      int(row["sender_robot"]),
      int(row["receiver_object"]),
      int(row["sender_object"]),
    )
    for row in rows
  }
  if len(entries) != len(rows) or any(float(row["weight"]) != 1.0 for row in rows):
    raise MRCLAMSE3RunnerError("fixed associations are not unique unit matches")
  if len(entries) % 2:
    raise MRCLAMSE3RunnerError("fixed associations are not reciprocal")
  return len(entries), len(entries) // 2


def _prepare_sesync_input(source: Path, target: Path) -> None:
  lines = [
    line
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith(("#", "%"))
  ]
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _command_manifest(spec: CommandSpec) -> dict[str, Any]:
  return {
    "name": spec.name,
    "method": spec.method,
    "repeat": spec.seed,
    "iteration": spec.iteration,
    "argv": list(spec.argv),
    "environment": dict(spec.environment),
  }


def communication_fields(
  communication: Any, *, setup_logical_bytes: int = 0
) -> dict[str, Any]:
  if (
    isinstance(setup_logical_bytes, bool)
    or not isinstance(setup_logical_bytes, int)
    or setup_logical_bytes < 0
    or setup_logical_bytes > communication.logical_bytes
  ):
    raise MRCLAMSE3RunnerError(
      "setup communication must be an integer within total logical bytes"
    )
  return {
    "logical_bytes": communication.logical_bytes,
    "setup_logical_bytes": setup_logical_bytes,
    "iterative_logical_bytes": communication.logical_bytes - setup_logical_bytes,
    "payload_blocks": communication.payload_blocks,
    "messages": communication.messages,
    "solver_wall_time_s": communication.solver_wall_time_s,
  }


def _record(
  *,
  method: str,
  repeat: int,
  iteration: int,
  graph: CanonicalGraph,
  state: Mapping[int, Any],
  reference: Mapping[int, Any],
  communication: Any,
  setup_logical_bytes: int = 0,
) -> dict[str, Any]:
  evaluation = evaluate_benchmark_state(graph, state, reference)
  return {
    "method": method,
    "repeat": repeat,
    "iteration": iteration,
    "total_cost": evaluation.total_cost,
    "mean_factor_cost": evaluation.mean_factor_cost,
    "robot_translation_rmse_m": evaluation.physical.robot_translation_rmse_m,
    "robot_rotation_rmse_deg": evaluation.physical.robot_rotation_rmse_deg,
    "object_translation_rmse_m": evaluation.physical.object_translation_rmse_m,
    "object_rotation_rmse_deg": evaluation.physical.object_rotation_rmse_deg,
    **communication_fields(
      communication, setup_logical_bytes=setup_logical_bytes
    ),
  }


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
  if not records:
    raise MRCLAMSE3RunnerError("benchmark produced no records")
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)


def validate_checkpoint_records(
  records: Sequence[Mapping[str, Any]],
  *,
  repeats: Sequence[int],
  checkpoints: Sequence[int],
  methods: Sequence[str] = PRIMARY_METHODS,
) -> None:
  """Enforce finite metrics and monotonic cumulative resource coordinates."""

  grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
  for row in records:
    grouped.setdefault((str(row["method"]), int(row["repeat"])), []).append(row)
  expected_groups = {(method, repeat) for method in methods for repeat in repeats}
  if set(grouped) != expected_groups:
    raise MRCLAMSE3RunnerError("checkpoint result method/repeat coverage differs")
  finite_fields = (
    "mean_factor_cost",
    "robot_translation_rmse_m",
    "robot_rotation_rmse_deg",
    "object_translation_rmse_m",
    "object_rotation_rmse_deg",
    "solver_wall_time_s",
  )
  cumulative_fields = (
    "logical_bytes",
    "iterative_logical_bytes",
    "payload_blocks",
    "messages",
    "solver_wall_time_s",
  )
  for key, group in grouped.items():
    ordered = sorted(group, key=lambda row: int(row["iteration"]))
    if tuple(int(row["iteration"]) for row in ordered) != tuple(checkpoints):
      raise MRCLAMSE3RunnerError(
        f"checkpoint coverage differs for {key[0]} repeat {key[1]}"
      )
    setup_values = {int(row["setup_logical_bytes"]) for row in ordered}
    if len(setup_values) != 1:
      raise MRCLAMSE3RunnerError(
        f"setup communication changes for {key[0]} repeat {key[1]}"
      )
    for row in ordered:
      if any(not math.isfinite(float(row[field])) for field in finite_fields):
        raise MRCLAMSE3RunnerError(
          f"non-finite metric for {key[0]} repeat {key[1]}"
        )
      if int(row["logical_bytes"]) != int(row["setup_logical_bytes"]) + int(
        row["iterative_logical_bytes"]
      ):
        raise MRCLAMSE3RunnerError(
          f"communication decomposition differs for {key[0]} repeat {key[1]}"
        )
    for field in cumulative_fields:
      values = [float(row[field]) for row in ordered]
      if any(current < previous for previous, current in zip(values, values[1:])):
        raise MRCLAMSE3RunnerError(
          f"{field} is not monotonic for {key[0]} repeat {key[1]}"
        )


def run_benchmark(
  config: MRCLAMSE3RunConfig,
  *,
  executables: ExecutableSet | None = None,
) -> Mapping[str, Any]:
  if config.output_dir.exists():
    raise MRCLAMSE3RunnerError(f"output already exists: {config.output_dir}")
  generation_manifest = load_generation_manifest(config.manifest_path)
  if any(item not in generation_manifest.repeats for item in config.repeats):
    raise MRCLAMSE3RunnerError("requested repeat is not pinned by the manifest")
  if any(item not in generation_manifest.checkpoints for item in config.checkpoints):
    raise MRCLAMSE3RunnerError("requested checkpoint is not pinned by the manifest")
  package = load_canonical_package(config.canonical_dir)
  validate_generation_provenance(package.manifest, generation_manifest)
  reference = read_planted_reference(config.canonical_dir / "planted_reference.jsonl")
  if executables is None:
    executables = default_mrclam_executables(config.repository_root)
  executables = replace(
    executables, dran=config.repository_root / "build/bin/manual-object-dran-example"
  )
  for name in ("larpg", "dran", "distributed_mapper", "sesync"):
    executable = Path(getattr(executables, name))
    if not executable.is_file() or not os.access(executable, os.X_OK):
      raise MRCLAMSE3RunnerError(f"required executable is unavailable: {executable}")

  num_robots, num_objects, active_blocks = _graph_sizes(package.graph)
  association_file = config.canonical_dir / "adapters/dran/fixed_associations.csv"
  dran_blocks, edge_hypotheses = _fixed_association_counts(association_file)
  legacy = _legacy_config(config)
  commands = []
  for repeat in config.repeats:
    base = build_command_specs(
      legacy,
      executables,
      seed=repeat,
      iterations=config.checkpoints[-1],
      work_root=config.output_dir,
      package=package,
    )
    commands.append(_command_manifest(base["larpg"]))
    for checkpoint in config.checkpoints:
      dran = build_dran_pose3_command(
        binary=executables.dran,
        data_dir=config.canonical_dir / f"adapters/dran/seed_{repeat}",
        association_file=association_file,
        output_dir=(
          config.output_dir
          / "native"
          / (("dranpp_objectpose3_exact_curvature"
              if config.dran_exact_proximal_curvature
              else "dranpp_objectpose3")
             + f"_repeat_{repeat}_round_{checkpoint}")
        ),
        num_robots=num_robots,
        num_objects=num_objects,
        repeat=repeat,
        iterations=checkpoint,
        beta=config.dran_beta,
        local_max_iterations=config.dran_local_max_iterations,
        exact_proximal_curvature=config.dran_exact_proximal_curvature,
        proximal_curvature_scale=config.dran_proximal_curvature_scale,
        dual_step_size=config.dran_dual_step_size,
      )
      commands.append(_command_manifest(dran))
      dm = build_command_specs(
        legacy,
        executables,
        seed=repeat,
        iterations=checkpoint,
        work_root=config.output_dir,
        package=package,
      )["distributed_mapper"]
      commands.append(_command_manifest(dm))
  reference_spec = build_command_specs(
    legacy,
    executables,
    seed=config.repeats[0],
    iterations=config.checkpoints[-1],
    work_root=config.output_dir,
    package=package,
  )["sesync"]
  commands.append(_command_manifest(reference_spec))

  manifest: dict[str, Any] = {
    "schema_version": "mrclam_synthetic_se3_objectpose3_run_v1",
    "benchmark_id": generation_manifest.benchmark_id,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dry_run": config.dry_run,
    "methods": list(PRIMARY_METHODS),
    "centralized_reference": "SE-Sync",
    "repeats": list(config.repeats),
    "checkpoints": list(config.checkpoints),
    "topology": "complete",
    "communication_policy": "full",
    "object_state_semantics": "full_object_pose3",
    "canonical_factor_hash": package.manifest["canonical_factor_hash"],
    "generation_config": generation_manifest.config.to_dict(),
    "object_identity_corrections": {
      str(reported): physical
      for reported, physical in generation_manifest.object_identity_corrections.items()
    },
    "object_identity_correction_basis": (
      generation_manifest.object_identity_correction_basis
    ),
    "round_zero_shared_across_methods": True,
    "dran_beta": config.dran_beta,
    "dran_local_max_iterations": config.dran_local_max_iterations,
    "dran_exact_proximal_curvature": config.dran_exact_proximal_curvature,
    "dran_proximal_curvature_scale": config.dran_proximal_curvature_scale,
    "dran_dual_step_size": config.dran_dual_step_size,
    "dran_shared_backend": "direct_edge_dual_shared_only",
    "dran_object_communication_representation": "lifted_pose",
    "communication_accounting": {
      "logical_bytes": "end-to-end payload including method-specific setup",
      "setup_logical_bytes": "pre-iteration neighbor-cache bootstrap payload",
      "iterative_logical_bytes": "logical_bytes minus setup_logical_bytes",
    },
    "distributed_mapper_name": "original Object-DGS (Pose3 copy consensus)",
    "commands": commands,
  }
  config.output_dir.mkdir(parents=True)
  for name in ("commands", "logs", "native", "reference", "inputs"):
    (config.output_dir / name).mkdir()
  zero_rows = initial_rows(package.graph, repeats=config.repeats)
  _write_records(config.output_dir / "round_zero.csv", zero_rows)
  if config.dry_run:
    (config.output_dir / "run_manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

  reference_input = Path(
    reference_spec.argv[reference_spec.argv.index("--input") + 1]
  )
  _prepare_sesync_input(config.canonical_dir / "global_graph.g2o", reference_input)
  execute_command(
    reference_spec,
    command_dir=config.output_dir / "commands",
    log_dir=config.output_dir / "logs",
    timeout_seconds=config.timeout_seconds,
  )
  reference_output = Path(
    reference_spec.argv[reference_spec.argv.index("--output") + 1]
  )
  central_state = read_sesync_state(reference_output, package.graph)
  central_evaluation = evaluate_benchmark_state(package.graph, central_state, reference)
  manifest["centralized_reference_metrics"] = {
    "total_cost": central_evaluation.total_cost,
    "mean_factor_cost": central_evaluation.mean_factor_cost,
    **{
      key: value
      for key, value in asdict(central_evaluation.physical).items()
      if key != "truth_from_estimate"
    },
  }

  records = []
  for repeat in config.repeats:
    max_checkpoint = config.checkpoints[-1]
    base = build_command_specs(
      legacy,
      executables,
      seed=repeat,
      iterations=max_checkpoint,
      work_root=config.output_dir,
      package=package,
    )
    larpg = base["larpg"]
    execute_command(
      larpg,
      command_dir=config.output_dir / "commands",
      log_dir=config.output_dir / "logs",
      timeout_seconds=config.timeout_seconds,
    )
    larpg_output = Path(larpg.argv[larpg.argv.index("--output-dir") + 1])
    for checkpoint in config.checkpoints:
      state = read_larpg_state(
        larpg_output / f"states/iteration_{checkpoint:03d}.jsonl", package.graph
      )
      communication = accumulate_larpg_communication(
        larpg_output / "iterations.csv",
        checkpoint,
        expected_active_blocks=active_blocks,
      )
      records.append(
        _record(
          method="larpg_objectpose3",
          repeat=repeat,
          iteration=checkpoint,
          graph=package.graph,
          state=state,
          reference=reference,
          communication=communication,
        )
      )

    for checkpoint in config.checkpoints:
      dran_output = (
        config.output_dir
        / "native"
        / (("dranpp_objectpose3_exact_curvature"
            if config.dran_exact_proximal_curvature
            else "dranpp_objectpose3")
           + f"_repeat_{repeat}_round_{checkpoint}")
      )
      dran = build_dran_pose3_command(
        binary=executables.dran,
        data_dir=config.canonical_dir / f"adapters/dran/seed_{repeat}",
        association_file=association_file,
        output_dir=dran_output,
        num_robots=num_robots,
        num_objects=num_objects,
        repeat=repeat,
        iterations=checkpoint,
        beta=config.dran_beta,
        local_max_iterations=config.dran_local_max_iterations,
        exact_proximal_curvature=config.dran_exact_proximal_curvature,
        proximal_curvature_scale=config.dran_proximal_curvature_scale,
        dual_step_size=config.dran_dual_step_size,
      )
      execute_command(
        dran,
        command_dir=config.output_dir / "commands",
        log_dir=config.output_dir / "logs",
        timeout_seconds=config.timeout_seconds,
      )
      state = read_dran_state(
        dran_output / "object_poses.txt",
        config.canonical_dir
        / f"adapters/dran/seed_{repeat}/adapter_manifest.json",
        package.graph,
        transform_path=dran_output / "final_association_transforms.csv",
        common_frame_robot=0,
        point_objects=False,
      )
      communication = accumulate_dran_communication(
        dran_output / "iteration_summary.csv",
        checkpoint,
        expected_blocks_per_round=dran_blocks,
        expected_edge_hypotheses=edge_hypotheses,
        object_block_bytes=3 * 4 * 8,
        require_shared_only=True,
      )
      records.append(
        _record(
          method="dranpp_objectpose3",
          repeat=repeat,
          iteration=checkpoint,
          graph=package.graph,
          state=state,
          reference=reference,
          communication=communication,
          setup_logical_bytes=dran_blocks * 3 * 4 * 8,
        )
      )

      dm_input = config.output_dir / f"inputs/dgs_repeat_{repeat}_round_{checkpoint}"
      shutil.copytree(
        config.canonical_dir / f"adapters/distributed_mapper/seed_{repeat}",
        dm_input,
      )
      dm = build_command_specs(
        legacy,
        executables,
        seed=repeat,
        iterations=checkpoint,
        work_root=config.output_dir,
        package=package,
      )["distributed_mapper"]
      argv = list(dm.argv)
      argv[argv.index("--dataDir") + 1] = f"{dm_input}{os.sep}"
      dm = replace(
        dm,
        name=f"object_dgs_pose3_repeat_{repeat}_round_{checkpoint}",
        method="object_dgs_pose3",
        argv=tuple(argv),
      )
      execute_command(
        dm,
        command_dir=config.output_dir / "commands",
        log_dir=config.output_dir / "logs",
        timeout_seconds=config.timeout_seconds,
      )
      state = read_distributed_mapper_state(
        dm_input, dm_input / "adapter_manifest.json", package.graph
      )
      communication = accumulate_distributed_mapper_communication(
        Path(dm.argv[dm.argv.index("--commCsv") + 1]),
        Path(dm.argv[dm.argv.index("--lossCsv") + 1]),
        checkpoint=checkpoint,
      )
      records.append(
        _record(
          method="object_dgs_pose3",
          repeat=repeat,
          iteration=checkpoint,
          graph=package.graph,
          state=state,
          reference=reference,
          communication=communication,
        )
      )
  validate_checkpoint_records(
    records,
    repeats=config.repeats,
    checkpoints=config.checkpoints,
  )
  _write_records(config.output_dir / "checkpoint_results.csv", records)
  manifest["record_count"] = len(records)
  manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
  (config.output_dir / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
  )
  return manifest


__all__ = [
  "MRCLAMSE3RunConfig",
  "MRCLAMSE3RunnerError",
  "PRIMARY_METHODS",
  "build_dran_pose3_command",
  "communication_fields",
  "initial_rows",
  "run_benchmark",
  "validate_checkpoint_records",
  "validate_generation_provenance",
]
