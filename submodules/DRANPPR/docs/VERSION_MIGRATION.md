# DRAN++ version selection and DBA migration

## Compared snapshots

The previous DBA repository was `DRAN-SEEDSLAM`, commit
`35d79e9803b7fbbfbdf06e027949a118cc701c7f` (2026-07-27), plus local working
tree edits. Its `manual-object-dran-example` binary was built on 2026-07-21.

The selected DRAN++ source came from the parent repository at commit
`a105e82ca1c3cae7b4d53d0e28560a186f9cec24` (2026-07-31), plus the newer
working-tree source. A commit alone is not sufficient provenance because both
snapshots contained uncommitted research changes; the hashes below identify
the packaged source exactly.

## Why the packaged backend is newer

Compared with the previous DBA backend, the packaged implementation adds or
extends:

- exact Point2 owner, Schur-oracle and owner-PCG lower modes;
- matrix-free ObjectPose3 interface PCG;
- complete-graph coarse interface response;
- private block-Jacobi and block-tridiagonal private solvers;
- compact Point2/Pose3 communication representations;
- event-scheduled message handling and expanded communication diagnostics.

The three principal DRAN++ files differ from the old DBA copy by:

| File | Added lines | Removed lines |
| --- | ---: | ---: |
| `src/ManualObjectDran.cpp` | 6358 | 733 |
| `include/DPGO/ManualObjectDran.h` | 466 | 3 |
| `examples/ManualObjectDranExample.cpp` | 737 | 24 |

## Compatibility restoration

The newer backend had removed two outputs required by the final DBA loop. They
were restored without reintroducing deprecated component-profiled lower modes:

- `--association_cavity_export_leave_pair_states`;
- `--association_object_observation_track_empirical_information`.

The first exports one association-excluded state prediction per communication
edge. The second calibrates repeated observations within a track and records
the resulting information summary.

## Packaged source hashes

```text
48e5e23469b7c4ae26532eb795fc2e952bc99fd7f4b061b5dae93ac103cc7f43  src/ManualObjectDran.cpp
af1b12b5a309b6960933657f65eafa670e521cb11e002ad72585651fbfc2ac46  include/DPGO/ManualObjectDran.h
b8e53d4e338d6911bee90f97f996c16b258844d4d423a8417373cbd83067cc2f  examples/ManualObjectDranExample.cpp
0452ab1024d3ecfaa43b10dff3a655067c14ca40aca1c9e67da2ac636ca072db  src/bilevel/BilevelAssociation.cpp
f6d5064a2b8a6ce3fe8df71a0f089222122d14c142614a30085597cbd85faaea  include/DPGO/BilevelAssociation.h
```

## Reproduction check

On the bundled 05/07 inputs, the packaged backend reproduced every final
partition hash and association metric of the previous backend. Recorded total
closed-loop communication changed from `5.578486` to `4.664699` MB on 05 and
from `2.030701` to `1.809326` MB on 07. The comparison is stored in
`results/latest/backend_comparison.json`.

