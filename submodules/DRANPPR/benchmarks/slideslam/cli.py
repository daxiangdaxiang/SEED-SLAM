"""Command-line interface for the SlideSLAM backend benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .acquire import acquire_sources, load_manifest
from .run import (
    BenchmarkRunConfig,
    RunnerError,
    materialize_native_sources,
    run_benchmark,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "benchmarks/slideslam/manifests/parking_lot_v1.json"
)


def _common_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/SlideSLAM",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native multi-robot SlideSLAM object-backend benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit the three official ROS bags")
    _common_source_arguments(audit)
    audit.add_argument("--download", action="store_true")

    materialize = subparsers.add_parser(
        "materialize", help="freeze the canonical graph and solver adapters"
    )
    _common_source_arguments(materialize)
    materialize.add_argument("--output-dir", type=Path)
    materialize.add_argument("--download", action="store_true")
    materialize.add_argument("--association-timeout", type=float, default=120.0)

    run = subparsers.add_parser("run", help="run the three full-communication backends")
    _common_source_arguments(run)
    run.add_argument("--canonical-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--checkpoints", nargs="+", type=int)
    run.add_argument("--max-iters", type=int)
    run.add_argument("--timeout", type=float, default=3600.0)
    run.add_argument("--dry-run", action="store_true")

    summarize = subparsers.add_parser(
        "summarize", help="print an existing benchmark summary"
    )
    summarize.add_argument("--run-dir", type=Path, required=True)
    return parser


def _run_command(arguments: argparse.Namespace) -> int:
    manifest = load_manifest(arguments.manifest)
    canonical_dir = arguments.canonical_dir
    if canonical_dir is None:
        canonical_dir = (
            arguments.data_root / "canonical" / manifest.benchmark_id
        )
    output_dir = arguments.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            REPOSITORY_ROOT
            / "results"
            / f"slideslam_object_backend_{timestamp}"
        )
    seeds = tuple(arguments.seeds or manifest.evaluation.seeds)
    checkpoints = tuple(
        arguments.checkpoints or manifest.evaluation.fixed_checkpoints
    )
    max_iterations = arguments.max_iters or manifest.evaluation.max_iterations
    config = BenchmarkRunConfig(
        repository_root=REPOSITORY_ROOT,
        manifest_path=arguments.manifest,
        data_root=arguments.data_root,
        canonical_dir=canonical_dir,
        output_dir=output_dir,
        seeds=seeds,
        checkpoints=checkpoints,
        max_iterations=max_iterations,
        timeout_seconds=arguments.timeout,
        dry_run=arguments.dry_run,
    )
    result = run_benchmark(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "audit":
            report = acquire_sources(
                arguments.manifest,
                arguments.data_root,
                download=arguments.download,
            )
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            return 0
        if arguments.command == "materialize":
            package = materialize_native_sources(
                arguments.manifest,
                arguments.data_root,
                output_dir=arguments.output_dir,
                download=arguments.download,
                association_timeout_seconds=arguments.association_timeout,
            )
            print(
                json.dumps(
                    {
                        "root": str(package.root),
                        "canonical_factor_hash": package.canonical_factor_hash,
                        "canonical_graph_sha256": package.canonical_graph_sha256,
                        "association_sha256": package.association_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "run":
            return _run_command(arguments)
        if arguments.command == "summarize":
            summary = arguments.run_dir / "summary.md"
            if not summary.is_file():
                raise RunnerError(f"summary does not exist: {summary}")
            print(summary.read_text(encoding="utf-8"), end="")
            return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
