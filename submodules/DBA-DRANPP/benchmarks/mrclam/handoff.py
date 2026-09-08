"""State handoff utilities for staged MR.CLAM DRAN experiments."""

from __future__ import annotations

import math
import shutil
from pathlib import Path


class MRCLAMHandoffError(ValueError):
    """Raised when a staged DRAN state cannot initialize an adapter."""


def _read_dran_vertices(
    path: Path, *, fuse_point_objects: bool
) -> dict[tuple[int, int], tuple[str, ...]]:
    expected_header = (
        "type robot_id local_vertex_id object_id x y z qx qy qz qw".split()
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split() != expected_header:
        raise MRCLAMHandoffError(f"invalid DRAN state header: {path}")
    vertices: dict[tuple[int, int], tuple[str, ...]] = {}
    object_copies: dict[int, list[tuple[int, int]]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(expected_header):
            raise MRCLAMHandoffError(
                f"invalid DRAN state row at {path}:{line_number}"
            )
        try:
            robot = int(fields[1])
            local_id = int(fields[2])
            object_id = int(fields[3])
            values = tuple(fields[4:11])
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("non-finite pose")
        except ValueError as error:
            raise MRCLAMHandoffError(
                f"invalid DRAN state row at {path}:{line_number}"
            ) from error
        key = (robot, local_id)
        if key in vertices:
            raise MRCLAMHandoffError(
                f"duplicate DRAN state vertex robot={robot} local_id={local_id}"
            )
        vertices[key] = values
        if fields[0] == "object":
            if object_id < 0:
                raise MRCLAMHandoffError(
                    f"invalid object ID at {path}:{line_number}"
                )
            object_copies.setdefault(object_id, []).append(key)
    if fuse_point_objects:
        for keys in object_copies.values():
            mean = [
                sum(float(vertices[key][axis]) for key in keys) / len(keys)
                for axis in range(3)
            ]
            for key in keys:
                vertices[key] = (
                    *(format(value, ".17g") for value in mean),
                    *vertices[key][3:],
                )
    return vertices


def materialize_dran_state_adapter(
    source_adapter: Path,
    state_path: Path,
    destination_adapter: Path,
    *,
    fuse_point_objects: bool = False,
) -> dict[int, int]:
    """Copy an adapter and replace every g2o vertex with a DRAN checkpoint."""

    source_adapter = Path(source_adapter)
    destination_adapter = Path(destination_adapter)
    if destination_adapter.exists():
        raise FileExistsError(
            f"handoff adapter already exists: {destination_adapter}"
        )
    vertices = _read_dran_vertices(
        Path(state_path), fuse_point_objects=fuse_point_objects
    )
    shutil.copytree(source_adapter, destination_adapter)
    counts: dict[int, int] = {}
    consumed: set[tuple[int, int]] = set()
    robot_files = sorted(destination_adapter.glob("rob_*.g2o"))
    if not robot_files:
        raise MRCLAMHandoffError(
            f"source adapter has no robot g2o files: {source_adapter}"
        )
    for path in robot_files:
        try:
            robot = int(path.stem.removeprefix("rob_"))
        except ValueError as error:
            raise MRCLAMHandoffError(f"invalid robot g2o filename: {path}") from error
        output = []
        count = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if fields and fields[0] == "VERTEX_SE3:QUAT":
                if len(fields) != 9:
                    raise MRCLAMHandoffError(
                        f"invalid g2o vertex at {path}:{line_number}"
                    )
                try:
                    local_id = int(fields[1])
                except ValueError as error:
                    raise MRCLAMHandoffError(
                        f"invalid g2o vertex ID at {path}:{line_number}"
                    ) from error
                key = (robot, local_id)
                if key not in vertices:
                    raise MRCLAMHandoffError(
                        f"checkpoint is missing robot={robot} local_id={local_id}"
                    )
                output.append(
                    " ".join(("VERTEX_SE3:QUAT", str(local_id), *vertices[key]))
                )
                consumed.add(key)
                count += 1
            else:
                output.append(line)
        path.write_text("\n".join(output) + "\n", encoding="utf-8")
        counts[robot] = count
    unused = sorted(set(vertices) - consumed)
    if unused:
        raise MRCLAMHandoffError(
            f"checkpoint has vertices absent from adapter: {unused[:10]}"
        )
    return counts


__all__ = [
    "MRCLAMHandoffError",
    "materialize_dran_state_adapter",
]
