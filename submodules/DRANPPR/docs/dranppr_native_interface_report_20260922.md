# Routed Native Interface: Correctness and Initialization Evidence

Date: 2026-09-22. Research-only, single-process message simulation. The packaged
DBA-DRANPP tree is unchanged. This is progress toward native design gate 7,
not a completed distributed online solver or a robustness improvement.

## Implemented Boundary

`NativeInterface` connects immutable robot-local `NativeSchurModel` responses.
Each Point2 block has a least-ID observing owner. Owners retain the block's CG
state; observers receive only their own direction blocks, apply their local
Schur response, and return product blocks. Scalar inner products are reduced
and broadcast on a BFS spanning tree of the supplied communication topology.
Private factorizations are reused during the complete interface solve.

The new implementation does not assemble a global Hessian or a global gradient
vector. The old `ManualObjectDran` owner-PCG diagnostic wrappers do assemble
global vectors and estimate route-hop costs; they remain unchanged and are not
called by the new solver. Reusing their PCG mathematics is not a novelty claim.
The optional SSE-compatible GTSAM target remains separate from the legacy AVX
library to preserve the previously audited Eigen allocation ABI.

This is deliberately a deterministic simulator, not a socket/MPI deployment.
It has a supplied identity roster, common Point2 charts, and connected topology.
Metadata discovery, chart agreement and bootstrap communication are not yet
implemented or counted. Each explicit forwarded numerical packet records its
origin, destination, hop, epoch, originating model revision and payload. The
application-byte model uses a 40-byte logical header plus 8 bytes per double;
it is not the measured size of a middleware serialization. TCP/IP framing,
latency, retransmission, packet loss, and asynchronous revisions are absent.

The ledger includes local-gradient/block-diagonal collection, each direction
scatter and product return, every scalar reduce/broadcast, final state scatter,
and a separate final product/residual verification. Relays can have no objects
of their own. A local Schur block is probed one Point2 diagonal at a time; there
is no full reduced-matrix assembly in the production solver. This costs one
two-column application per local interface object, which still needs profiling
on large physical prefixes. The simulator retains all packet traces in memory;
that is diagnostic behavior, not bounded-memory online execution.

## Linear Verification

Physical fixtures use exact native Pose2 odometry and anisotropic bearing/range
factors. Every robot has two private poses and four object observations sets;
neighboring robots share two objects along a chain. Only robot 0's first pose
is fixed. Other root poses remain free and move in the computed step. Both
private increments and shared increments are checked against a monolithic full
tangent LDLT solve constructed only in the test. No damping or extra priors
are added. These are deterministic synthetic correctness fixtures, not benchmark
accuracy or timing results.

| Robots | PCG iterations | Application bytes | Relative step error | Relative full residual | Relative interface residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 12 | 5,792 | 1.75e-15 | 2.95e-16 | 6.14e-16 |
| 3 | 16 | 15,040 | 2.01e-15 | 2.53e-16 | 6.67e-16 |
| 5 | 23 | 42,176 | 8.50e-15 | 3.31e-16 | 1.15e-14 |
| 10 | 39 | 157,104 | 3.47e-14 | 8.71e-16 | 1.01e-13 |

The requested PCG tolerance is 1e-12, with a 1000-iteration test cap. Residuals
use `norm(S d + b) / max(1, norm(b))`. A separate three-robot sparse-observer
fixture takes 6 iterations / 6400 application bytes, with residual 1.98e-16.
Adding a direct topology edge reduces bytes without changing the step. The
first iteration independently matches dense block-Jacobi-preconditioned CG.

Additional tests cover zero RHS, disconnected/duplicate links, inconsistent
charts, relays without interface variables, negative curvature, invalid
preconditioners, duplicate endpoints and absent charts. A new regression caught
and fixed an actual validation bug: checking only total dimension accepted a
1-dimensional plus 3-dimensional boundary as two Point2 blocks. The immutable
Schur response now exports per-key dimensions and the interface validates each.

Importantly, a gauge-singular model with zero RHS can have a zero linear
residual. An explicit regression preserves this distinction: the returned
`linear_residual_small` status does NOT certify observability or a unique solution.
Likewise, block-Jacobi SPD and the absence of encountered negative curvature do
not prove global SPD. Gauge/rank validity remains the caller's physical contract.

## Real-Data Support Audit

The new `audit_native_interface_support.py` reads only the causal event stream
and a matching saved checkpoint. It checks prefix/state key equality, pose
timestamps, duplicated inputs and causal support. It records when each supplied
neighbor pair first has two declared shared object identities. At the FINAL
checkpoint only, it applies the existing SE(2) registration routine to the two
independently estimated object maps. No GT is opened and no network state is
initialized. Input/script/registration-source hashes are archived.

The diagnostic topology is the explicitly assumed chain `1-2-3-4-5`; it is not
a recorded MRCLAM communication trace. Snapshots are the unchanged `native_cached`
runs from the chart-reuse study. Labels are fixed declared barcodes.

| Prefix | Time when every chain edge has two shared labels, s | Shared counts on edges 12/23/34/45 | Best rigid-fit RMS on those edges, m |
| --- | ---: | --- | --- |
| D1 120 s | 27.153 | 10/15/10/6 | .364/.349/.659/1.168 |
| D2 120 s | 21.877 | 8/11/7/9 | .301/.589/.258/.319 |
| D3 120 s | 67.764 | 10/12/11/12 | .244/.404/.514/.395 |
| D4 120 s | 15.534 | 8/10/15/15 | .677/1.010/.267/.511 |
| D9 120 s | not reached | 0/0/3/1 | n/a/n/a/.165/n/a |
| D9 300 s | 131.723 | 14/15/15/14 | 2.102/.304/1.441/1.075 |

D9's last robot first appears at 38.344 s, so its much later pairwise support is
not explained only by staggered robot starts. Nevertheless, these times are NOT
network observability bounds. Two labels need spatial extent; two correspondences
can be geometrically ill-conditioned; distributed cycle constraints or allowed
relayed evidence can offer information before every pair has a full SE(2) fit.
This audit neither estimates covariance nor proves that a transform is accurate.
It rejects the assumption that every chosen neighbor pair has a usable rigid
registration from the first frame. It does not justify inventing a fixed warmup
duration or using future measurements for initialization.

The large rigid-fit residual also argues against freezing separately estimated
maps after a one-time frame transform. As a follow-up using the prior D9 300 s
completion controls, replacing ONLY robot 1 by its completed fixed-likelihood
local solution changes edge 12 RMS from 2.102067 to 2.053426 m for the native
start, and 2.125928 to 2.053426 m for the iSAM2-exact start. Other edges are
unchanged. Local optimization lag explains a large physical-loss decrease in
the prior report, but removes only a small part of this map-shape discrepancy.
Joint pose/object adjustment is therefore a necessary experiment, not an already
demonstrated solution. Model mismatch, noise assumptions and geometric weakness
remain possible explanations. No wrong-association verdict follows from this
RMS, and no noise covariance should be inflated solely because it is large.

Completion comparison inputs are
`/tmp/dranppr_native_causal_300_20260922/dataset9_{native,isam2_exact}/final_estimates.csv`
and `/tmp/dranppr_native_causal_d9r1_completion_20260922/{native,isam2_exact}/exact_schur_estimates.csv`.
Call `analyze` with the former rows, then replace robot-1 rows with the latter;
keep the stream, other robot rows, duration 300 and four chain links unchanged.
Their estimator provenance and physical-loss controls remain in the previous
native-causal report and manifests; this is not a new solver run.

## Remaining Work and Design Decision

Recommended representation: one common chart per connected component, one exact
component gauge, with all other roots and physical states free. Use only arrived
shared objects for an initial frame guess; the joint likelihood continues to
refine that guess. Do not introduce permanently fixed inter-robot transforms.
An explicit-transform formulation is an alternative, not a required extra layer.
The asynchronous representation question is still pending, so this turn does
not implement the nonlinear bootstrap or claim that its communication is free.

The planned frozen REAL joint-system solve is deferred until that gauge/bootstrap
contract is selected. This turn instead completes the independent linear kernel,
physical synthetic parity and real support audit. Next validate common-frame
transformation invariance, gauge release on component merging, native joint
physical descent, and only then causal arrival/revision updates. A kernel residual
is not a replacement for those tests. Global PCG scalar synchronization may also
be costly; it must be evaluated, not assumed suitable for online deployment.

No full MRCLAM distributed replay, equal-wall-time/communication comparison to
riMESA, incremental factorization, or cause-aware covariance/identity adaptation
has been completed here. The overall DRANPPR research goal remains active.

## Commands and Artifacts

```sh
cmake --build /tmp/dranppr_backend_build --target testIncrementalLocalModel native-local-audit native-causal-replay -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel --gtest_output=xml:/tmp/dranppr_native_interface_20260922/native_tests.xml
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_LOCAL_BINARY=/tmp/dranppr_backend_build/bin/native-local-audit python3 -m pytest -q tests/test_native_local_audit.py tests/test_audit_native_local.py tests/test_native_causal_replay.py tests/test_native_replay_profile.py tests/test_native_interface_support.py --junitxml=/tmp/dranppr_native_interface_20260922/python_tests.xml
/tmp/dranppr_backend_build/bin/testDPGO --gtest_filter=testDPGO.ManualObjectDranPoint2SchurMatchesDirectElimination:testDPGO.ManualObjectDranPoint2MatrixFreeProductMatchesDenseSchur:testDPGO.ManualObjectDranBlockTridiagonalBlocksMatchDenseLdlt:testDPGO.ManualObjectDranSparseCouplingAggregatesAndMatchesDense:testDPGO.ManualObjectDranPoint2OwnerPcgMatchesDenseFirstStepAndLedger:testDPGO.ManualObjectDranRoutedOwnerPcgMatchesDirectNumerically --gtest_output=xml:/tmp/dranppr_native_interface_20260922/legacy_tests.xml
python3 -m experiments.audit_native_interface_support --stream /tmp/dranppr_causal_streams_v1/dataset9 --snapshot /tmp/dranppr_chart_reuse_300_20260922/dataset9_native_cached/final_estimates.csv --duration 300 --links 1:2 2:3 3:4 4:5 --output /tmp/dranppr_native_interface_20260922/dataset9_300_support.json
```

All 34 native C++ tests, 21 Python tests and six legacy regressions pass (61 total).
The seven new native tests include the four physical sizes in one test. Five
new Python contracts cover the support audit. These counts are checks, not
independent experiments. MPI: 0 ranks. Centralized CCI and D-CCI are unchanged.
Optional native targets use the pinned GTSAM dependency already described in
the native-local report. `git diff --check` passes; packaged-tree status is clean.
No commit, push, deletion of prior results or default solver promotion occurred.

Full numerical results are in `/tmp/dranppr_native_interface_20260922`; small
JSON/XML artifacts and current source snapshots are copied to
`results/dranppr_native_interface_20260922/` in the research worktree. The output
path is exclusive: choose a new filename when reproducing an existing audit.
