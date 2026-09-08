import csv
from pathlib import Path
import sys

import pytest

from scripts.audit_dba_topology_and_scaling import (
    chain_topology,
    main,
    star_topology,
    validate_tree_backbone,
)


def test_chain_and_star_cover_same_robots_and_are_trees():
    robots = tuple(range(6))
    chain = chain_topology(robots)
    star = star_topology(robots, root_robot=0)

    assert len(chain) == len(star) == 5
    assert validate_tree_backbone(robots, chain, root_robot=0) == 5
    assert validate_tree_backbone(robots, star, root_robot=0) == 5
    assert {
        robot for edge in chain for robot in edge
    } == {robot for edge in star for robot in edge} == set(robots)


@pytest.mark.parametrize(
    "topology",
    (
        ((0, 1), (2, 3)),
        ((0, 1), (1, 2), (2, 0)),
    ),
)
def test_backbone_rejects_disconnected_or_cyclic_graph(topology):
    with pytest.raises(ValueError, match="tree"):
        validate_tree_backbone((0, 1, 2, 3), topology, root_robot=0)


def test_audit_writes_topology_initialization_and_scaling_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_dba_topology_and_scaling.py",
            "--output-dir",
            str(output),
        ],
    )

    main()

    for name in (
        "topology_results.csv",
        "initialization_results.csv",
        "scaling_results.csv",
        "summary.md",
    ):
        assert (output / name).is_file()
    scaling = list(
        csv.DictReader((output / "scaling_results.csv").open())
    )
    assert [int(row["robot_count"]) for row in scaling] == [2, 3, 6, 10]
    assert all(float(row["relative_objective_error"]) < 1e-10 for row in scaling)
