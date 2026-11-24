import json
from pathlib import Path
import sys

from scripts.run_dba_feedback_safe_ablation import (
    VARIANTS,
    build_case_records,
    execute_case_records,
    plan_cases,
)


def test_plan_contains_all_variants_for_five_feasible_starts():
    cases = plan_cases(
        {
            "05": ("teaserpp", "clipper", "slideslam_core"),
            "07": ("clipper", "slideslam_core"),
        }
    )

    assert len(cases) == 5 * len(VARIANTS)
    assert {case.variant for case in cases} == set(VARIANTS)
    assert not any(
        case.dataset == "07" and case.initializer == "teaserpp"
        for case in cases
    )


def test_dry_run_emits_every_command_without_execution(tmp_path: Path):
    marker = tmp_path / "executed"
    records = [
        {
            "case_id": f"case_{index}",
            "case_hash": f"hash_{index}",
            "output_dir": str(tmp_path / f"case_{index}"),
            "command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('x')",
            ],
        }
        for index in range(4)
    ]

    outcomes = execute_case_records(
        records, dry_run=True, resume=False
    )

    assert [value["status"] for value in outcomes] == ["dry_run"] * 4
    assert not marker.exists()


def test_generated_manifest_uses_absolute_candidate_support(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    required = {}
    for name in (
        "metadata",
        "lower_topology",
        "upper_topology",
        "ground_truth",
        "initial_association",
    ):
        path = inputs / f"{name}.csv"
        path.write_text("first_robot,first_object,second_robot,second_object\n", encoding="utf-8")
        required[name] = str(path)
    config = {
        "binary": str(tmp_path / "solver"),
        "lower_iterations": 1,
        "safety_max_outer_iterations": 1,
        "datasets": [
            {
                "name": "toy",
                "data_dir": str(inputs),
                "metadata": required["metadata"],
                "lower_topology": required["lower_topology"],
                "upper_topology": required["upper_topology"],
                "ground_truth": required["ground_truth"],
                "num_robots": 2,
                "num_objects": 1,
                "plane_axes": "xy",
                "initializers": {"seed": required["initial_association"]},
            }
        ],
    }

    records = build_case_records(
        config,
        output_dir=tmp_path / "relative-looking-output",
        repo_root=Path(__file__).parents[1],
    )

    manifest_path = Path(records[0]["command"][3])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert Path(manifest["candidate_support"]).is_absolute()


def test_resume_requires_terminal_status_and_matching_input_hash(
    tmp_path: Path,
):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    summary = case_dir / "summary.json"
    summary.write_text(
        json.dumps({"status": "partition_fixed_point"}),
        encoding="utf-8",
    )
    status_path = case_dir / "case_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "case_hash": "matching",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "reran"
    base = {
        "case_id": "case",
        "output_dir": str(case_dir),
        "command": [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('x')",
        ],
    }

    skipped = execute_case_records(
        [{**base, "case_hash": "matching"}],
        dry_run=False,
        resume=True,
    )
    assert skipped[0]["status"] == "skipped_complete"
    assert not marker.exists()

    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "case_hash": "stale",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    rerun = execute_case_records(
        [{**base, "case_hash": "matching"}],
        dry_run=False,
        resume=True,
    )
    assert rerun[0]["status"] == "failed"
    assert marker.exists()


def test_runner_bookkeeping_does_not_pollute_solver_output(tmp_path: Path):
    case_dir = tmp_path / "case"
    control_dir = tmp_path / "_runner" / "case"
    script = (
        "import json, pathlib, sys; "
        "out = pathlib.Path(sys.argv[1]); "
        "assert not any(out.iterdir()); "
        "(out / 'summary.json').write_text("
        "json.dumps({'status': 'partition_fixed_point'}))"
    )
    outcomes = execute_case_records(
        [
            {
                "case_id": "case",
                "case_hash": "hash",
                "output_dir": str(case_dir),
                "control_dir": str(control_dir),
                "command": [sys.executable, "-c", script, str(case_dir)],
            }
        ],
        dry_run=False,
        resume=False,
    )

    assert outcomes[0]["status"] == "complete"
    assert set(path.name for path in case_dir.iterdir()) == {"summary.json"}
    assert (control_dir / "command.json").is_file()
    assert (control_dir / "case_status.json").is_file()
