# Nonlinear Conditional Replay Audit, 2026-10-02

## Decision

The persistent conditional response is now integrated into the chronological
nonlinear replay, not merely exercised on frozen linearizations. It preserves
the reference trajectory numerically, but is slower in all four short paired
checks. Do not promote it to the default and do not claim online acceleration.

The important distinction is now experimentally explicit: reusable elimination
under frozen linearization does not imply reusable elimination when nonlinear
updates repeatedly invalidate historical factors.

## Implementation

- `IncrementalLocalModel(true)` stores immutable whitened Jacobian factors and
  their nonlinear chart bases. It does not maintain the aggregate normal blocks
  or assemble a full normal matrix. The default constructor keeps the original
  normal-model implementation.
- Unchanged factors preserve pointer identity. A response update compares the
  last consumed snapshot with the current one, so multiple intervening model
  updates are handled without dropping an edit. Only changed factors are sent
  to `IncrementalConditionalModel`.
- The fixed root's columns are removed exactly. No gauge prior, damping, noise
  floor, covariance fitting, robust kernel, or acceptance threshold was added.
- The optimizer still uses the existing `1e-4` chart-refresh tolerance. Shared
  chart coordination, topology, PCG policy, and true physical-cost backtracking
  are unchanged. Current estimates remain separate from cached chart bases.
- The new modes are `joint_conditional_cached` and
  `joint_conditional_cached_one`. They explicitly require a connected common-past
  warm start. Cold start, component merging, Pose3 deployment, and asynchronous
  revision recovery are not implemented by this prototype.

For the whitened cached model, the line-search quantities are evaluated directly:

\[
q(d)=\tfrac12\sum_f\|A_fd-b_f\|^2,\qquad
q'(d;s)=\sum_f(A_fd-b_f)^\top A_fs,\qquad
q''(d;s)=\sum_f\|A_fs\|^2.
\]

These are the same quadratic slope and curvature as the reference normal model,
not a new objective. Fixed increments are zero; private increments are recovered
from the conditional response before testing the true nonlinear objective.

## Controlled Experiment

Inputs are the existing native MRCLAM Pose2/Point2 streams and common-past states.
No truth enters the solver. The experiment holds known identities and physical
noise fixed; it is not a new robustness or association benchmark.

| Case | Common-past cutoff | End cutoff | Suffix frames | Physical factors, start to end |
| --- | ---: | ---: | ---: | ---: |
| D4 | 15.5340001583 s | 20 s | 84 | 352 to 549 |
| D9 | 131.7230000496 s | 136 s | 75 | 4222 to 4404 |

For each dataset and PCG policy, run order was reference, candidate, candidate,
reference, single process, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`. These are two
timing repeats on the same data, not independent statistical seeds. Tight PCG
retains its existing stopping rule and cap 500; the one-step policy keeps cap 1.

Service includes insertion, in-memory coordination and the nonlinear solve;
JSON input/output is excluded. Initial setup is separately recorded and also
included in an auxiliary ratio. There is no simulated network latency.

## Results

| Case | Reference service, repeats (s) | Candidate service, repeats (s) | Candidate slowdown | Conditional reuse |
| --- | --- | --- | --- | ---: |
| D4, one PCG | 0.365810 / 0.365117 | 0.461849 / 0.461527 | 26.25% / 26.41% | 0.0251% |
| D4, tight PCG | 0.425446 / 0.429577 | 0.495055 / 0.500870 | 16.36% / 16.60% | 6.6102% |
| D9, one PCG | 3.262229 / 3.295047 | 4.503490 / 4.460154 | 38.05% / 35.36% | 0% |
| D9, tight PCG | 3.383469 / 3.412387 | 4.343984 / 4.298105 | 28.39% / 25.96% | 1.8527% |

Reuse is `conditionals_reused / (conditionals_reused + conditionals_rebuilt)`
among response-update calls, including the first build. Entire-response reuse
is counted separately: D4 tight 17, D9 tight 3, both one-step cases 0.
Setup-inclusive ratios are also negative in every repeat.

### Numerical and Communication Parity

All 16 runs completed. Every paired run used the same frames and final factor
count. Accepted/rejected steps, chart-refresh counts, and optimizer-stage factor
linearizations matched at every frame. Every fixed-graph step was nonincreasing
in true physical cost; control and optimizer byte ledgers were checked.

| Case | Max final coordinate difference | Max causal coordinate difference | Reference/candidate PCG total | Reference/candidate application bytes |
| --- | ---: | ---: | ---: | ---: |
| D4, one | 2.45e-15 | 3.15e-14 | 84 / 84 | 1,335,528 / 1,335,528 |
| D4, tight | 2.31e-14 | 1.72e-13 | 2988 / 2988 | 13,790,216 / 13,790,216 |
| D9, one | 1.12e-14 | 2.89e-14 | 75 / 75 | 1,030,392 / 1,030,392 |
| D9, tight | 1.47e-12 | 9.21e-12 | 2417 / 2226 | 9,672,504 / 8,966,136 |

Coordinate differences are componentwise comparisons of stored x/y/yaw values,
not a new ATE metric. Across the four cases, final physical-cost differences
were at most 1.43e-13; per-frame cost-after differences were at most 5.22e-10.

The D9 tight iteration/byte difference is real: 74 of 75 frames stopped at a
different PCG iteration. The QR-derived local boundary matrix and the reference
sparse Schur operator are algebraically equivalent but not bitwise identical.
Their finite-precision behavior near the stopping tolerance is a plausible
explanation, not a demonstrated new communication algorithm. The one-step cases
are the strict equal-iteration/equal-byte comparisons, and both are slower.

Tight maximum verified relative residuals were below 9.91e-11. One-step maximum
residuals were 1.21859 on D4 and 2.08483 on D9, so those updates must not be called
converged linear solves. MPI ranks: 0; CCI/D-CCI were not changed or benchmarked.

### Where the Time Went

D9 one-step, first paired run:

| Stage | Reference (s) | Candidate (s) |
| --- | ---: | ---: |
| Model stage, total | 2.582063 | 3.721711 |
| Interface PCG | 0.085791 | 0.010084 |
| Evaluation and collectives, total | 0.270087 | 0.501353 |
| Nonlinear update snapshots, nested | 0.136048 | 0.053468 |
| Nonlinear factor preparation/linearization, nested | 0.494377 | 0.524379 |
| Model contribution maintenance, nested | 0.764276 | 0.214135 |
| Conditional response update, nested | n/a | 2.256048 |
| Complete private-state recovery, nested | 0.007315 | 0.114618 |
| Quadratic directional evaluation, nested | 0.007924 | 0.113886 |

Nested rows must not be added to their parent stages. Insertion and coordination
also contribute to service but are not isolated in the optimizer's nested
nonlinear-update counters.

The intended removal of normal-block maintenance did work. The compact local
boundary operator also made interface applications cheaper. However, all
154,800 requested private conditionals were rebuilt in the D9 one-step slice,
with zero numerical conditional reuse. QR reconstruction, map-based recovery,
and factor-by-factor directional evaluation outweighed those savings.

The candidate therefore increased D9 one-step serial maximum backlog from
0.107/0.132 s to 0.549/0.518 s. This is a single-process scheduling diagnostic,
not a prediction of robot-parallel deployment latency.

## Interpretation and Next Decision

Previously, frozen-chart component tests reused about 89.8%/99.2% of private
conditionals on D4/D9. That assumption fails in these nonlinear suffixes: chart
changes touch old factors, and their effect propagates through the elimination
chain. An append-only graph does not imply append-only numerical work.

This experiment rejects promoting this implementation as an online speedup.
It does not prove that all incremental elimination methods fail, nor that later
steady windows have identical reuse. These are short slices near the common-past
restart. Because both datasets failed this preliminary gate, no long full-stream
matrix was launched. Memory and multi-process deployment were not measured.

The next investigation should distinguish two causes of historical invalidation:
true changes in observable local geometry versus coordinated changes of local
coordinate charts. Then test whether physical model error, rather than raw
state displacement alone, can determine which old factors must be refreshed.
Do not simply loosen `1e-4`, freeze history indefinitely, or add a portfolio of
heuristic gates. Any new update policy must separately pass fixed-objective
descent, equal-budget state quality, and delayed-observation recovery tests.

This is a proposed next diagnostic, not an implemented policy or established
convergence result. Noise adaptation, outlier robustness and a full online
asynchronous algorithm remain open.

## Verification and Reproduction

Build commands:

```sh
cmake --build /tmp/dranppr_backend_build --target testIncrementalLocalModel -j2
cmake --build /tmp/dranppr_backend_build --target native-joint-replay native-joint-audit native-conditional-audit -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_native_matched_online.py \
  tests/test_native_conditional_replay.py tests/test_audit_conditional_replay.py \
  tests/test_native_conditional_audit.py tests/test_native_joint_audit.py \
  tests/test_native_joint_replay.py tests/test_native_interface_support.py \
  tests/test_audit_native_online.py
```

Results: 53 C++ tests and 70 Python tests passed. The new replay tests first
failed because the modes did not exist. Coverage includes square-root/normal
quadratic parity, fixed-column handling, factor snapshot identity, transaction
rollback, causal suffixes, new shared labels, split observer identity, and
explicit refusal of unsupported cold start. The C++ square-root directional
parity tolerance is 1e-10; matrix/gradient relative tolerance is 1e-12.

An additional D9 one-step replay using the pre-edit executable produced exactly
the same state CSV values, physical costs, per-frame decisions, refresh counts,
and bytes as the new executable's normal-model mode. Timing was not identical.

Example experiment command:

```sh
python3 -m experiments.audit_conditional_replay \
  --binary /tmp/dranppr_backend_build/bin/native-joint-replay \
  --events /tmp/dranppr_causal_streams_v1/dataset9/events.jsonl \
  --warm-start /tmp/dranppr_matched_bootstrap_d9_20260923/frozen.json \
  --duration 136 --budget one --output /tmp/new_conditional_d9_one
```

The duration is the absolute elapsed cutoff from the first stream timestamp,
not the suffix length. Run `--budget tight` for the other solver policy; D4 uses
the D4 input files and `--duration 20`.

Archive: `results/dranppr_nonlinear_conditional_20261002/`. Each case contains
the executed binary, source snapshots, warm start, commands, both timing pairs,
logs, summary JSON, state CSVs, control ledger, and comparison JSON. The parent
archive contains both full input streams, tests/XML, CMake cache, the pre-edit
binary and its regression output, aggregate audit, and a SHA256 manifest.

Executed binary SHA256:
`7bc552df6ad21b64a5ee477444a4ce9e44b582db65a6563efcfad452b1fbabb5`.
GTSAM library SHA256:
`280ba9f45a49464f3ad1c4385526fc355300644147cc0a798d2f0a2185bdd28e`.

All changes remain in the research worktree. Production `DBA-DRANPP` is clean;
no commits, pushes, baseline replacement, or production default changes.
