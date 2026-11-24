import csv
import json
from pathlib import Path
import stat
import sys

import pytest

from scripts.audit_dba_leaveout_fidelity import compare_means, main


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_mean_fidelity_aligns_each_disconnected_robot_component(
    tmp_path: Path,
) -> None:
    fields = ("robot", "valid", "object_id", "x", "y", "z")
    reference = tmp_path / "reference.csv"
    estimate = tmp_path / "estimate.csv"
    reference_rows = (
        {"robot": 0, "valid": 1, "object_id": 0, "x": 0, "y": 0, "z": 0},
        {"robot": 0, "valid": 1, "object_id": 1, "x": 1, "y": 0, "z": 0},
        {"robot": 1, "valid": 1, "object_id": 0, "x": 0, "y": 2, "z": 0},
        {"robot": 1, "valid": 1, "object_id": 1, "x": 1, "y": 2, "z": 0},
    )
    estimate_rows = tuple(
        {
            **row,
            "x": row["x"] + (10 if row["robot"] == 0 else -20),
            "y": row["y"] + (3 if row["robot"] == 0 else 7),
        }
        for row in reference_rows
    )
    _write_csv(reference, fields, reference_rows)
    _write_csv(estimate, fields, estimate_rows)

    mean, maximum, count = compare_means(
        estimate,
        reference,
        robot_components=((0,), (1,)),
    )

    assert mean == pytest.approx(0.0, abs=1e-12)
    assert maximum == pytest.approx(0.0, abs=1e-12)
    assert count == 4


def test_leaveout_audit_removes_one_pair_and_compares_stub_lower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    association = tmp_path / "association.csv"
    fields = (
        "receiver_robot",
        "sender_robot",
        "receiver_object",
        "sender_object",
        "weight",
    )
    _write_csv(
        association,
        fields,
        (
            {
                "receiver_robot": 0,
                "sender_robot": 1,
                "receiver_object": 4,
                "sender_object": 8,
                "weight": 1,
            },
            {
                "receiver_robot": 1,
                "sender_robot": 0,
                "receiver_object": 8,
                "sender_object": 4,
                "weight": 1,
            },
            {
                "receiver_robot": 1,
                "sender_robot": 2,
                "receiver_object": 9,
                "sender_object": 3,
                "weight": 1,
            },
        ),
    )
    topology = tmp_path / "topology.csv"
    _write_csv(
        topology,
        ("first_robot", "second_robot"),
        ({"first_robot": 0, "second_robot": 1},),
    )
    metadata = tmp_path / "metadata.csv"
    _write_csv(
        metadata,
        ("robot_id", "local_slot_id"),
        (
            {"robot_id": 0, "local_slot_id": 4},
            {"robot_id": 1, "local_slot_id": 8},
        ),
    )
    support = tmp_path / "support.csv"
    _write_csv(support, fields, ())
    heldout = tmp_path / "heldout/leave_pair_cavity_0_1"
    mean_fields = (
        "robot",
        "valid",
        "status",
        "object_id",
        "x",
        "y",
        "z",
        "response_norm",
    )
    covariance_fields = (
        "robot",
        "valid",
        "status",
        "row_object",
        "column_object",
        "covariance",
    )
    _write_csv(
        heldout / "final_pair_cavity_means.csv",
        mean_fields,
        (
            {
                "robot": 0,
                "valid": 1,
                "status": "ok",
                "object_id": 4,
                "x": 1,
                "y": 2,
                "z": 0,
                "response_norm": 0,
            },
            {
                "robot": 1,
                "valid": 1,
                "status": "ok",
                "object_id": 8,
                "x": 3,
                "y": 2,
                "z": 0,
                "response_norm": 0,
            },
        ),
    )
    _write_csv(
        heldout / "final_pair_cavity_covariance.csv",
        covariance_fields,
        (
            {
                "robot": 0,
                "valid": 1,
                "status": "ok",
                "row_object": 4,
                "column_object": 4,
                "covariance": 2,
            },
            {
                "robot": 1,
                "valid": 1,
                "status": "ok",
                "row_object": 8,
                "column_object": 8,
                "covariance": 3,
            },
        ),
    )
    _write_csv(
        heldout / "candidate_edit_scores.csv",
        ("edit_id", "delta"),
        (
            {"edit_id": "merge:0:4:1:8", "delta": -1},
            {"edit_id": "noop", "delta": 1},
        ),
    )
    lower = tmp_path / "stub_lower.py"
    lower.write_text(
        """#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--association_file", required=True)
parser.add_argument("--output_dir", required=True)
args, _ = parser.parse_known_args()
with open(args.association_file, newline="") as handle:
    rows = list(csv.DictReader(handle))
assert not any(
    {int(row["receiver_robot"]), int(row["sender_robot"])} == {0, 1}
    for row in rows
)
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
with (output / "final_pair_cavity_means.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("robot","valid","status","object_id","x","y","z","response_norm"))
    writer.writeheader()
    writer.writerows((
        {"robot":0,"valid":1,"status":"ok","object_id":4,"x":11,"y":-3,"z":1,"response_norm":0},
        {"robot":1,"valid":1,"status":"ok","object_id":8,"x":13,"y":-3,"z":1,"response_norm":0},
    ))
with (output / "final_pair_cavity_covariance.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("robot","valid","status","row_object","column_object","covariance"))
    writer.writeheader()
    writer.writerows((
        {"robot":0,"valid":1,"status":"ok","row_object":4,"column_object":4,"covariance":2},
        {"robot":1,"valid":1,"status":"ok","row_object":8,"column_object":8,"covariance":3},
    ))
with (output / "candidate_edit_scores.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=("edit_id","delta"))
    writer.writeheader()
    writer.writerows(({"edit_id":"merge:0:4:1:8","delta":-2},{"edit_id":"noop","delta":3}))
""",
        encoding="utf-8",
    )
    lower.chmod(lower.stat().st_mode | stat.S_IEXEC)
    common_args = tmp_path / "common_args.json"
    common_args.write_text(json.dumps(["--iters", "2"]), encoding="utf-8")
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_dba_leaveout_fidelity.py",
            "--incumbent-association",
            str(association),
            "--metadata",
            str(metadata),
            "--topology",
            str(topology),
            "--lower-executable",
            str(lower),
            "--lower-common-args",
            str(common_args),
            "--heldout-bundle",
            str(tmp_path / "heldout"),
            "--candidate-support",
            str(support),
            "--output-dir",
            str(output),
        ],
    )

    main()

    rows = list(csv.DictReader((output / "summary.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["mean_object_displacement"]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(rows[0]["covariance_relative_error"]) == pytest.approx(0.0)
    assert float(rows[0]["edit_sign_agreement"]) == pytest.approx(1.0)
    removed = list(
        csv.DictReader(
            (output / "edge_0_1/association_without_edge.csv").open()
        )
    )
    assert len(removed) == 1
    assert {removed[0]["receiver_robot"], removed[0]["sender_robot"]} == {
        "1",
        "2",
    }
    report = (output / "summary.md").read_text()
    assert "local quadratic fidelity audit" in report
    assert "not a proof of exact nonlinear independence" in report
