"""CLI for the MR.CLAM-derived synthetic SE(3)-ObjectPose3 benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .config import load_generation_manifest
from .materialize import (
  build_from_generation_manifest,
  materialize_derived_graph,
  validate_materialized_graph,
)
from .run import MRCLAMSE3RunConfig, run_benchmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
  REPOSITORY_ROOT / "benchmarks/mrclam_se3/manifests/dataset1.json"
)
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "datasets/MRCLAM/raw"


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="MR.CLAM-derived Synthetic SE(3)-ObjectPose3 benchmark"
  )
  commands = parser.add_subparsers(dest="command", required=True)
  materialize = commands.add_parser(
    "materialize", help="build the derived graph and solver adapters"
  )
  materialize.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  materialize.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
  materialize.add_argument("--output-dir", type=Path)
  validate = commands.add_parser(
    "validate", help="verify provenance, repeats, and adapter hashes"
  )
  validate.add_argument("--canonical-dir", type=Path, required=True)
  run = commands.add_parser(
    "run", help="run DRAN++, LARPG, Object-DGS, and centralized SE-Sync"
  )
  run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  run.add_argument("--canonical-dir", type=Path)
  run.add_argument("--output-dir", type=Path)
  run.add_argument("--repeats", nargs="+", type=int)
  run.add_argument("--checkpoints", nargs="+", type=int)
  run.add_argument("--timeout", type=float, default=3600.0)
  run.add_argument("--dry-run", action="store_true")
  run.add_argument("--dran-beta", type=float, default=0.1)
  run.add_argument("--dran-local-max-iterations", type=int, default=3)
  run.add_argument("--dran-exact-proximal-curvature", action="store_true")
  run.add_argument("--dran-proximal-curvature-scale", type=float, default=1.0)
  run.add_argument("--dran-dual-step-size", type=float, default=1.0)
  return parser


def _materialize(arguments: argparse.Namespace) -> dict[str, object]:
  manifest = load_generation_manifest(arguments.manifest)
  derived = build_from_generation_manifest(manifest, arguments.data_root)
  target = arguments.output_dir
  if target is None:
    target = arguments.data_root.parent / "canonical" / manifest.benchmark_id
  package = materialize_derived_graph(derived, target)
  result = validate_materialized_graph(package.root)
  result["root"] = str(package.root)
  return result


def _run(arguments: argparse.Namespace) -> dict[str, object]:
  manifest = load_generation_manifest(arguments.manifest)
  canonical_dir = arguments.canonical_dir
  if canonical_dir is None:
    canonical_dir = DEFAULT_DATA_ROOT.parent / "canonical" / manifest.benchmark_id
  output_dir = arguments.output_dir
  if output_dir is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPOSITORY_ROOT / "results" / f"mrclam_se3_{timestamp}"
  return dict(
    run_benchmark(
      MRCLAMSE3RunConfig(
        repository_root=REPOSITORY_ROOT,
        manifest_path=arguments.manifest,
        canonical_dir=canonical_dir,
        output_dir=output_dir,
        repeats=tuple(arguments.repeats or manifest.repeats),
        checkpoints=tuple(arguments.checkpoints or manifest.checkpoints),
        timeout_seconds=arguments.timeout,
        dry_run=arguments.dry_run,
        dran_beta=arguments.dran_beta,
        dran_local_max_iterations=arguments.dran_local_max_iterations,
        dran_exact_proximal_curvature=(
          arguments.dran_exact_proximal_curvature
        ),
        dran_proximal_curvature_scale=(
          arguments.dran_proximal_curvature_scale
        ),
        dran_dual_step_size=arguments.dran_dual_step_size,
      )
    )
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  arguments = _parser().parse_args(argv)
  try:
    if arguments.command == "materialize":
      result = _materialize(arguments)
    elif arguments.command == "validate":
      result = validate_materialized_graph(arguments.canonical_dir)
    elif arguments.command == "run":
      result = _run(arguments)
    else:
      raise AssertionError("unreachable command")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
  except (OSError, ValueError, RuntimeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
