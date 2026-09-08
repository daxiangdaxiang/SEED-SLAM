"""Command-line interface for the strict native-team MR.CLAM benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .extract import extract_dataset
from .materialize import build_canonical_graph, materialize_mrclam_graph
from .run import MRCLAMRunConfig, run_benchmark
from .source import audit_dataset, load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/mrclam/manifests/dataset1.json"
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "datasets/MRCLAM/raw"
DEFAULT_CANONICAL_NAME = "mrclam_dataset1_native5_no_perturb_v1"


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UTIAS MR.CLAM native five-robot object-backend benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="verify native collection provenance")
    _source_arguments(audit)

    materialize = commands.add_parser(
        "materialize", help="freeze the common graph and three solver adapters"
    )
    _source_arguments(materialize)
    materialize.add_argument("--output-dir", type=Path)

    run = commands.add_parser(
        "run", help="run the MR.CLAM object backends with full communication"
    )
    _source_arguments(run)
    run.add_argument("--canonical-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--checkpoints", nargs="+", type=int)
    run.add_argument("--timeout", type=float, default=3600.0)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--dran-local-max-iterations", type=int, default=3)
    run.add_argument("--dran-beta", type=float, default=1.0)
    run.add_argument(
        "--dran-reduced-rotation-preconditioner",
        choices=("none", "jacobi", "schur_jacobi", "cholesky", "portfolio"),
        default="cholesky",
    )
    run.add_argument("--dran-initial-local-max-iterations", type=int, default=0)
    run.add_argument("--dran-initial-local-rounds", type=int, default=0)
    run.add_argument("--dgs-point2-inner-sweeps", type=int, default=1)
    run.add_argument(
        "--allow-uncertified-reference",
        action="store_true",
        help="continue when SE-Sync returns a feasible but uncertified reference",
    )

    summarize = commands.add_parser("summarize", help="print a completed summary")
    summarize.add_argument("--run-dir", type=Path, required=True)
    return parser


def _materialize(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(arguments.manifest)
    audit = audit_dataset(manifest, arguments.data_root)
    extractions = extract_dataset(manifest, audit)
    graph = build_canonical_graph(extractions, manifest, audit=audit.to_dict())
    target = arguments.output_dir
    if target is None:
        target = arguments.data_root.parent / "canonical" / DEFAULT_CANONICAL_NAME
    package = materialize_mrclam_graph(graph, target)
    return {
        "root": str(package.root),
        "canonical_factor_hash": package.canonical_factor_hash,
        "canonical_graph_sha256": package.canonical_graph_sha256,
        "association_sha256": package.association_sha256,
        "native_multi_robot_eligible": audit.native_multi_robot_eligible,
        "physical_robot_count": audit.physical_robot_count,
    }


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(arguments.manifest)
    canonical_dir = arguments.canonical_dir
    if canonical_dir is None:
        canonical_dir = arguments.data_root.parent / "canonical" / DEFAULT_CANONICAL_NAME
    output_dir = arguments.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPOSITORY_ROOT / "results" / f"mrclam_object_backend_{timestamp}"
    config = MRCLAMRunConfig(
        repository_root=REPOSITORY_ROOT,
        manifest_path=arguments.manifest,
        data_root=arguments.data_root,
        canonical_dir=canonical_dir,
        output_dir=output_dir,
        seeds=tuple(arguments.seeds or manifest.evaluation.seeds),
        checkpoints=tuple(
            arguments.checkpoints or manifest.evaluation.fixed_checkpoints
        ),
        timeout_seconds=arguments.timeout,
        dry_run=arguments.dry_run,
        dran_local_max_iterations=arguments.dran_local_max_iterations,
        dran_beta=arguments.dran_beta,
        dran_reduced_rotation_preconditioner=(
            arguments.dran_reduced_rotation_preconditioner
        ),
        dran_initial_local_max_iterations=(
            arguments.dran_initial_local_max_iterations
        ),
        dran_initial_local_rounds=arguments.dran_initial_local_rounds,
        require_certified_reference=not arguments.allow_uncertified_reference,
        dgs_point2_inner_sweeps=arguments.dgs_point2_inner_sweeps,
    )
    return dict(run_benchmark(config))


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "audit":
            manifest = load_manifest(arguments.manifest)
            result = audit_dataset(manifest, arguments.data_root).to_dict()
        elif arguments.command == "materialize":
            result = _materialize(arguments)
        elif arguments.command == "run":
            result = _run(arguments)
        elif arguments.command == "summarize":
            summary = arguments.run_dir / "summary.md"
            if not summary.is_file():
                raise ValueError(f"summary does not exist: {summary}")
            print(summary.read_text(encoding="utf-8"), end="")
            return 0
        else:
            raise AssertionError("unreachable command")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
