#!/usr/bin/env python3
"""Run only DRAN checkpoints for controlled ObjectPose3 curvature gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
  sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.mrclam.run import CommunicationTotals, accumulate_dran_communication
from benchmarks.mrclam_se3.config import load_generation_manifest
from benchmarks.mrclam_se3.evaluate import read_planted_reference
from benchmarks.mrclam_se3.run import (
  _fixed_association_counts,
  _graph_sizes,
  _record,
  _write_records,
  build_dran_pose3_command,
  validate_generation_provenance,
)
from benchmarks.slideslam.evaluate import read_dran_state
from benchmarks.slideslam.run import execute_command, load_canonical_package


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--canonical-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--binary", type=Path,
                      default=Path("build/bin/manual-object-dran-example"))
  parser.add_argument("--repeats", nargs="+", type=int, default=[0])
  parser.add_argument("--checkpoints", nargs="+", type=int,
                      default=[1, 2, 5, 10, 20])
  parser.add_argument("--beta", type=float, default=0.1)
  parser.add_argument("--local-max-iterations", type=int, default=3)
  parser.add_argument("--exact-proximal-curvature", action="store_true")
  parser.add_argument("--proximal-curvature-scale", type=float, default=1.0)
  parser.add_argument("--dual-step-size", type=float, default=1.0)
  parser.add_argument(
    "--dual-relaxation-residual-threshold", type=float, default=0.0)
  parser.add_argument("--spectral-dual-step", action="store_true")
  parser.add_argument("--global-spectral-dual-step", action="store_true")
  parser.add_argument("--spectral-dual-step-min", type=float, default=0.75)
  parser.add_argument("--spectral-dual-step-max", type=float, default=1.5)
  parser.add_argument(
    "--spectral-dual-step-ema-decay", type=float, default=0.5)
  parser.add_argument(
    "--object-interface-mode",
    choices=("copy_baseline", "reduced_schur_response",
             "complete_graph_coarse_response",
             "matrix_free_objectpose3_pcg"),
    default="copy_baseline")
  parser.add_argument("--object-interface-response-period", type=int,
                      default=1)
  parser.add_argument("--object-interface-pcg-iterations", type=int,
                      default=1)
  parser.add_argument(
    "--object-interface-pcg-preconditioner",
    choices=("interface_hessian", "exact_schur", "private_block_jacobi"),
    default="private_block_jacobi")
  parser.add_argument(
    "--object-interface-pcg-private-solver",
    choices=("sparse_ldlt", "block_tridiagonal_ldlt"),
    default="sparse_ldlt")
  parser.add_argument(
    "--object-interface-pcg-post-activation-local-iterations",
    type=int, default=1)
  parser.add_argument(
    "--object-interface-pcg-terminal-local-polish-iterations",
    type=int, default=0)
  parser.add_argument(
    "--object-interface-pcg-replace-final-round-with-terminal-polish",
    action="store_true")
  parser.add_argument("--object-interface-pcg-min-translation-spread",
                      type=float, default=0.06)
  parser.add_argument("--object-interface-pcg-min-rotation-spread-deg",
                      type=float, default=1.0)
  parser.add_argument(
    "--object-interface-pcg-min-initial-relative-decrease",
    type=float, default=0.01)
  parser.add_argument(
    "--object-interface-pcg-min-continuation-relative-decrease",
    type=float, default=0.0005)
  parser.add_argument("--object-interface-step-gain", type=float, default=1.0)
  parser.add_argument("--timeout", type=float, default=3600.0)
  args = parser.parse_args()

  output_dir = args.output_dir.resolve()
  if output_dir.exists():
    raise RuntimeError(f"output already exists: {output_dir}")
  manifest = load_generation_manifest(args.manifest)
  package = load_canonical_package(args.canonical_dir)
  validate_generation_provenance(package.manifest, manifest)
  reference = read_planted_reference(
    args.canonical_dir / "planted_reference.jsonl")
  num_robots, num_objects, _ = _graph_sizes(package.graph)
  association_file = args.canonical_dir / "adapters/dran/fixed_associations.csv"
  blocks_per_round, edge_hypotheses = _fixed_association_counts(
    association_file)

  for directory in ("commands", "logs", "native"):
    (output_dir / directory).mkdir(parents=True, exist_ok=True)
  records = []
  commands = []
  for repeat in args.repeats:
    for checkpoint in args.checkpoints:
      native = output_dir / "native" / f"repeat_{repeat}_round_{checkpoint}"
      command = build_dran_pose3_command(
        binary=args.binary.resolve(),
        data_dir=(args.canonical_dir
                  / f"adapters/dran/seed_{repeat}"),
        association_file=association_file,
        output_dir=native,
        num_robots=num_robots,
        num_objects=num_objects,
        repeat=repeat,
        iterations=checkpoint,
        beta=args.beta,
        local_max_iterations=args.local_max_iterations,
        exact_proximal_curvature=args.exact_proximal_curvature,
        proximal_curvature_scale=args.proximal_curvature_scale,
        dual_step_size=args.dual_step_size,
        dual_relaxation_residual_threshold=(
          args.dual_relaxation_residual_threshold
        ),
        spectral_dual_step=args.spectral_dual_step,
        global_spectral_dual_step=args.global_spectral_dual_step,
        spectral_dual_step_min=args.spectral_dual_step_min,
        spectral_dual_step_max=args.spectral_dual_step_max,
        spectral_dual_step_ema_decay=args.spectral_dual_step_ema_decay,
        object_interface_mode=args.object_interface_mode,
        object_interface_response_period=(
          args.object_interface_response_period
        ),
        object_interface_pcg_iterations=(
          args.object_interface_pcg_iterations
        ),
        object_interface_pcg_preconditioner=(
          args.object_interface_pcg_preconditioner
        ),
        object_interface_pcg_private_solver=(
          args.object_interface_pcg_private_solver
        ),
        object_interface_pcg_post_activation_local_iterations=(
          args.object_interface_pcg_post_activation_local_iterations
        ),
        object_interface_pcg_terminal_local_polish_iterations=(
          args.object_interface_pcg_terminal_local_polish_iterations
        ),
        object_interface_pcg_replace_final_round_with_terminal_polish=(
          args
          .object_interface_pcg_replace_final_round_with_terminal_polish
        ),
        object_interface_pcg_min_translation_spread=(
          args.object_interface_pcg_min_translation_spread
        ),
        object_interface_pcg_min_rotation_spread_deg=(
          args.object_interface_pcg_min_rotation_spread_deg
        ),
        object_interface_pcg_min_initial_relative_decrease=(
          args.object_interface_pcg_min_initial_relative_decrease
        ),
        object_interface_pcg_min_continuation_relative_decrease=(
          args.object_interface_pcg_min_continuation_relative_decrease
        ),
        object_interface_step_gain=args.object_interface_step_gain,
      )
      execute_command(
        command,
        command_dir=output_dir / "commands",
        log_dir=output_dir / "logs",
        timeout_seconds=args.timeout,
      )
      state = read_dran_state(
        native / "object_poses.txt",
        args.canonical_dir
        / f"adapters/dran/seed_{repeat}/adapter_manifest.json",
        package.graph,
        transform_path=native / "final_association_transforms.csv",
        common_frame_robot=0,
        point_objects=False,
      )
      response_rows = []
      if args.object_interface_mode == "matrix_free_objectpose3_pcg":
        with (native / "object_interface_response_summary.csv").open(
          encoding="utf-8", newline=""
        ) as stream:
          response_rows = list(csv.DictReader(stream))
      replacement_applied = (
        args
        .object_interface_pcg_replace_final_round_with_terminal_polish
        and any(
          int(row["iter"]) == checkpoint
          and row["status"] == "terminal_local_polish_replaced_round"
          for row in response_rows
        )
      )
      expected_blocks = [blocks_per_round] * (checkpoint + 1)
      if replacement_applied:
        expected_blocks[-1] = 0
      communication = accumulate_dran_communication(
        native / "iteration_summary.csv",
        checkpoint,
        expected_blocks_per_round=expected_blocks,
        expected_edge_hypotheses=edge_hypotheses,
        object_block_bytes=3 * 4 * 8,
        require_shared_only=True,
      )
      if args.object_interface_mode == "matrix_free_objectpose3_pcg":
        dpcg_bytes = sum(
          int(row["communication_payload_bytes"])
          for row in response_rows
          if int(row["iter"]) <= checkpoint
        )
        dpcg_messages = sum(
          int(row["communication_messages"])
          for row in response_rows
          if int(row["iter"]) <= checkpoint
        )
        communication = CommunicationTotals(
          logical_bytes=communication.logical_bytes + dpcg_bytes,
          payload_blocks=(
            communication.payload_blocks + dpcg_bytes // (6 * 8)
          ),
          messages=communication.messages + dpcg_messages,
          solver_wall_time_s=communication.solver_wall_time_s,
        )
      records.append(_record(
        method=command.method,
        repeat=repeat,
        iteration=checkpoint,
        graph=package.graph,
        state=state,
        reference=reference,
        communication=communication,
        setup_logical_bytes=blocks_per_round * 3 * 4 * 8,
      ))
      commands.append({
        "name": command.name,
        "method": command.method,
        "repeat": repeat,
        "iteration": checkpoint,
        "argv": list(command.argv),
      })
  _write_records(output_dir / "checkpoint_results.csv", records)
  run_manifest = {
    "schema_version": "mrclam_se3_dran_curvature_gate_v1",
    "benchmark_id": manifest.benchmark_id,
    "canonical_factor_hash": package.manifest["canonical_factor_hash"],
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repeats": args.repeats,
    "checkpoints": args.checkpoints,
    "beta": args.beta,
    "local_max_iterations": args.local_max_iterations,
    "exact_proximal_curvature": args.exact_proximal_curvature,
    "proximal_curvature_scale": args.proximal_curvature_scale,
    "dual_step_size": args.dual_step_size,
    "dual_relaxation_residual_threshold": (
      args.dual_relaxation_residual_threshold
    ),
    "spectral_dual_step": args.spectral_dual_step,
    "global_spectral_dual_step": args.global_spectral_dual_step,
    "spectral_dual_step_min": args.spectral_dual_step_min,
    "spectral_dual_step_max": args.spectral_dual_step_max,
    "spectral_dual_step_ema_decay": args.spectral_dual_step_ema_decay,
    "object_interface_mode": args.object_interface_mode,
    "object_interface_response_period": args.object_interface_response_period,
    "object_interface_pcg_iterations": args.object_interface_pcg_iterations,
    "object_interface_pcg_preconditioner": (
      args.object_interface_pcg_preconditioner
    ),
    "object_interface_pcg_private_solver": (
      args.object_interface_pcg_private_solver
    ),
    "object_interface_pcg_post_activation_local_iterations": (
      args.object_interface_pcg_post_activation_local_iterations
    ),
    "object_interface_pcg_terminal_local_polish_iterations": (
      args.object_interface_pcg_terminal_local_polish_iterations
    ),
    "object_interface_pcg_replace_final_round_with_terminal_polish": (
      args.object_interface_pcg_replace_final_round_with_terminal_polish
    ),
    "object_interface_pcg_min_translation_spread": (
      args.object_interface_pcg_min_translation_spread
    ),
    "object_interface_pcg_min_rotation_spread_deg": (
      args.object_interface_pcg_min_rotation_spread_deg
    ),
    "object_interface_pcg_min_initial_relative_decrease": (
      args.object_interface_pcg_min_initial_relative_decrease
    ),
    "object_interface_pcg_min_continuation_relative_decrease": (
      args.object_interface_pcg_min_continuation_relative_decrease
    ),
    "object_interface_step_gain": args.object_interface_step_gain,
    "topology": "complete",
    "communication_policy": "full",
    "commands": commands,
  }
  (output_dir / "run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")


if __name__ == "__main__":
  main()
