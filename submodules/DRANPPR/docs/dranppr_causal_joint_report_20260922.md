# Causal Joint Replay: Accuracy Pass, Online Efficiency Failure

## Main Decision

DRANPPR now has an actual chronological joint replay, not just a frozen graph or
an exchange of final local-map initializations. Native factors arrive frame by
frame, local models persist, components merge from already-arrived neighbor
observations, and the affected component receives one joint GN step per frame.

The real-data results establish useful joint state tracking, but **do not pass
the online efficiency gate**. D9 ends extremely close to its frozen physical
solution while accumulating 53.2 s of measured serial queue delay. Continuing
to solve each interface to nearly machine precision is not a viable final
online policy. No robust/noise adaptation is introduced or validated here.

## What Was Implemented

- `NativeInterfaceEndpoint::interfaceKeys` separates stable local factor keys
  from current network identity keys. Renaming or splitting a network identity
  does not rename nonlinear local factors. Empty maps preserve existing behavior.
  An explicit new-epoch split test reuses immutable local responses and matches
  the newly assembled dense physical system. Missing or duplicate aliases fail.
- `experiments/native_local/joint_step.h` contains the shared frozen/online step
  implementation. Both old real frozen routed state CSVs are byte-identical after
  extraction and introduction of optional identity maps.
- `native-joint-replay` appends raw native Pose2/Point2 factors to persistent
  local models. The physical likelihood and fixed sigmas are unchanged from
  the previous Gaussian controls. One GN step is attempted per arrived frame.
- Initially each disconnected component has its own gauge. A neighbor pair needs
  two distinct shared points to initialize a component merge. This is a geometric
  bootstrap requirement, not a tuned statistical acceptance threshold or a
  necessary condition for network observability. Only one root remains fixed
  after merging; every other robot root stays free.
- Rigidly transforming a merging component preserves its old physical factor
  costs. Enforcing newly shared object identities is a separate model change,
  which may increase the initial constrained cost. The true descent check applies
  to the subsequent fixed-graph step, not across different feasible sets.
- Object sharing is restricted to connected components of the observer-induced
  communication graph. Equal labels separated by a nonobserving robot do not
  become one object. A three-robot synthetic test checks this explicitly.
- Peer maps, merge transforms/membership, arrived-object rosters and identity
  chart broadcasts have explicit per-hop numerical control packets. Their bytes
  are included in addition to all routed PCG/model/cost-reduction bytes.

The first replay intentionally uses synchronous control and fresh local response
factorization/relinearization. It is a single-process numerical communication
simulator, not socket/MPI deployment, asynchronous revision recovery, or completed
incremental elimination. Repeated roster broadcasts are a conservative explicit
control-plane implementation, not a claimed efficient new protocol.

## Causal Merge Evidence

The fixed topology is `1-2-3-4-5`. No GT, final checkpoint, future measurement or
future relative transform enters initialization or any estimator step.

| Prefix | Arrived frames | Full-component time s | Component merge times s |
| --- | ---: | ---: | --- |
| D4 120 s | 1710 | 15.534 | 3.185, 8.925, 14.494, 15.534 |
| D9 300 s | 4541 | 131.723 | 101.179, 121.662, 127.695, 131.723 |

The D9 120 s boundary now has an operational interpretation: it remains multiple
components rather than receiving a fabricated frame constraint. The former
frozen bootstrap failure is not bypassed by a hand-selected warmup time.
All real gauge-transform cost changes are below `1.02e-13` absolute. Final
shared-copy discrepancy is exactly zero. Causal CSVs retain their component root
and a fully-connected flag; old coordinates are not silently treated as sharing
the final component frame.

## State Quality

Main final metrics use one global SE(2) alignment after estimation. D4 has 1710
GT-supported poses and 68 object copies. D9 has 4104 supported poses, 437 omitted
timestamps and 73 object copies. Labels are supplied barcodes, not estimated
associations. Object metrics are copy-weighted, as in the earlier reports.

| Prefix | Final global trajectory RMSE m | Object-copy RMSE m | Terminal physical cost | Frozen completion cost |
| --- | ---: | ---: | ---: | ---: |
| D4 120 s | .285676 | .184899 | 1230.388544407 | 1230.388541743 |
| D9 300 s | .401063 | .228986 | 3280.284013642 | 3280.283915983 |

The separate LM completion needs 3 and 4 iterations, changing physical loss by
only `2.66e-6` and `9.77e-5`. This is evidence of good terminal convergence, not
global optimality or a statement that early predictions were equally accurate.

D9 also reaches a better candidate than the previous frozen experiment's
`3635.515115` / `.482310 m` trajectory RMSE. The physical package, factor records,
topology, identities and root are structurally identical after stripping initial
values. Therefore this difference is attributable to a different estimation
trajectory/initialization basin, not a changed objective or extra data. It does
not show that chronological processing always finds a better basin.

For a controlled information diagnostic, give both independent and joint states
the SAME per-robot alignment, removing relative-frame error:

| Prefix | Arm | Final trajectory RMSE m | Final object-copy RMSE m | Causal tail trajectory RMSE m |
| --- | --- | ---: | ---: | ---: |
| D4 | Independent, one step/frame | .289545 | .600412 | .259256 |
| D4 | Joint, one step/frame | .216583 | .480487 | .248215 |
| D9 | Independent, one step/frame | .809533 | 1.262388 | 1.217126 |
| D9 | Joint, one step/frame | .370984 | .299034 | .531402 |

Causal tails use identical timestamps in both arms, starting at the joint
full-component time. They contain 1563 D4 poses and 2515 D9 poses. Joint causal
global-alignment RMSE is `.277125/.559460 m`. Before full connection, a common
global causal gauge is unavailable and no such metric is asserted.

Independent states match the old native replay to maximum positional differences
`4.07e-14 m` (D4) and `3.49e-12 m` (D9), confirming unchanged local controls.
However, independent models have fewer equality constraints and different
information. This is NOT a matched distributed riMESA/iMESA comparison, and
the two arms are not equal communication or equal compute budgets.

## Online Failure And Its Cause

| Quantity | D4 joint | D4 independent | D9 joint | D9 independent |
| --- | ---: | ---: | ---: | ---: |
| Total measured service s | 37.312 | 6.900 | 224.163 | 47.235 |
| Maximum serial queue delay s | .180 | .018 | 53.215 | .065 |
| Joint frame p99 service s | .0445 | n/a | .1265 | n/a |
| Joint PCG iterations | 59881 | 0 | 101959 | 0 |
| Joint maximum verified PCG relative residual | 9.96e-11 | n/a | 1.00e-10 | n/a |
| Total application bytes | 588372896 | 0 | 1028680816 | 0 |
| Control-plane bytes, included | 8287976 | 0 | 14544456 | 0 |

Joint optimizer bytes alone are `580084920/1014136360`. All fixed-graph updates
pass physical descent checks; update totals match the packet/summary ledgers.
PCG iteration ranges are 10--41 and 13--50 for joint component steps. Numerical
controls remain the same fixed high-accuracy controls as the frozen reference,
not a per-dataset sweep. One outer step is emphatically not bounded work.

Service decomposition (non-overlapping top-level timers):

| Prefix | Factor insertion s | Coordination s | Solve/publication s |
| --- | ---: | ---: | ---: |
| D4 | .858 | .386 | 36.067 |
| D9 | 6.781 | .982 | 216.399 |

The native update/solve work, not input parsing or component bookkeeping, dominates.
The 224 s total being less than the 300 s prefix does not preclude a queue: early
idle time cannot be saved for later frames. After D9 connects, 213.140 s of
service is demanded over about 168.3 s of sensor time, with additional bursts.
The measured virtual serial queue follows `finish_k=max(t_k,finish_{k-1})+work_k`.

A posthoc state-age diagnostic holds the same causal-tail alignment fixed and
compares the emitted, non-extrapolated pose with GT at its measurement timestamp
and at virtual delivery time:

| Prefix | Measurement-time RMSE m | Stale-pose delivery-time RMSE m | Mean / p99 age s |
| --- | ---: | ---: | --- |
| D4 | .277125 | .277723 | .0336 / .1322 |
| D9 | .559460 | 2.030059 | 13.001 / 51.396 |

This is an illustrative serial-service model with zero network delay and no
motion extrapolation, NOT a measured deployment error or a fair baseline ranking.
Nevertheless, it demonstrates why accurate historical estimates cannot establish
current-time tracking performance. The delay is measured independently of the
residual/noise model; fitting a larger sensor variance would not remove this
computational queue. A deployed system also needs a defined prediction-to-query-
time interface rather than silently presenting a timestamped old pose as current.

## Research Implications And Remaining Work

The goal should be tracking a changing joint estimation problem at a useful
state age, not repeatedly completing each frozen interface to `1e-10` residual.
The next algorithmic work is inexact interface tracking with reusable local
responses and explicit invalidation. Inner accuracy should be related to model
change/linearization error and arrivals, not chosen by a per-sequence sweep.
The current accurate replay is the oracle for that comparison, not the final
algorithm. The present timing breakdown does not yet isolate which solve substage
will give the largest gain, so do not assert that loosening PCG alone is sufficient.

Available evidence now separates three effects more clearly: local fixed-graph
completion measures state-reducible optimization error; joint constraints test
missing cross-robot information; service timestamps expose state age independently
of measurement noise. None uniquely classifies persistent raw measurement bias
versus identity error. The earlier same-frame geometric discrepancies remain.
No covariance adaptation or outlier classification is enabled in this prototype.

Matched native-factor riMESA/iMESA controls, incremental elimination/response reuse,
asynchronous communication, held-out sequence transfer, cause-aware statistical
adaptation and end-to-end robust gains remain open. The existing iMESA MRCLAM
adapter uses batch canonical chordal/Cartesian factors, so it cannot be presented
as the matching native online baseline without a new adapter.

## Verification And Provenance

36 native tests, 50 Python tests and 6 focused legacy ManualObjectDran tests pass
(92 total). New tests cover causal delayed merge, future mutation invariance,
missing support, same-timestamp packet interleaving, topology-restricted labels,
byte ledgers and stale-state evaluation. The two real frozen routed CSVs are
byte-identical after helper extraction; all six frozen parity controls exit 0.
CCI/D-CCI unchanged; MPI rank count 0 (not run). No default, commit or push.

Four chronological estimator runs and two separate terminal completions exit 0.
The initial Python postprocessor failed on a duplicate `mode` keyword AFTER all
six native processes completed. It was corrected and rerun with `--evaluate-only`;
no estimator was restarted. Original source snapshot plus postprocessing recovery
sources are retained. The CMake reconfigure encountered a ROPTLIB network update
warning; the cached dependency remained usable and native targets built normally
against the existing pinned GTSAM package.

Replay binary SHA256:
`f4019b1aa004035fd5b0744a05b8db59c632b0aa854a29c54074dd1ea43a2c5a`.
Frozen oracle SHA256:
`194f7e1d938c7f6910f5143299d40c646202ea929d096b610fa374ae7ff6c579`.
Full artifacts: `/tmp/dranppr_causal_joint_20260922/`; archive at root
`results/dranppr_causal_joint_20260922/artifacts.tar.gz`.

```sh
cmake --build /tmp/dranppr_backend_build --target native-joint-replay native-joint-audit testIncrementalLocalModel -j2
python3 -m experiments.audit_native_joint_replay --binary /tmp/dranppr_backend_build/bin/native-joint-replay --oracle /tmp/dranppr_backend_build/bin/native-joint-audit --stream-root /tmp/dranppr_causal_streams_v1 --case 4:120 --case 9:300 --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw --output /tmp/dranppr_causal_joint_repeat
python3 -m experiments.analyze_joint_delivery --results /tmp/dranppr_causal_joint_repeat --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw --output /tmp/dranppr_causal_joint_repeat/delivery.json
```
