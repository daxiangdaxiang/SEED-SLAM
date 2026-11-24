# Asynchronous Interface Study: Algorithmic Progress And Remaining Gap

Date: 2026-11-23. Research DRANPPR only; no production default, commit or push.

## Decision

We now have a barrier-free fixed-interface experiment and a matrix-valued
delay-control candidate. This changes the optimization update, not merely its
cache or transport implementation. At identical message schedules, every-prefix
bytes and local-update counts, the directional candidate improves the final
quadratic objective gap over a scalar bounded-delay reference in 102/135 cases,
ties 24, and loses 9. It is NOT a completed asynchronous nonlinear SLAM backend
or a new comparison win against production DRAN/iMESA/riMESA.

The simpler stale-cache iteration often makes more progress per simulated time
than a synchronous block iteration, but a positive-definite counterexample
shows it can diverge. Conservative delay control repairs that counterexample
but can destroy speed. Directional control partially reduces this conservatism;
it does not close the stability/speed tradeoff and is not promoted as a default.

## Fixed Experiment Contract

- Five MRCLAM physical Jacobian snapshots: D1/D2/D3/D4/D9. Nominal physical
  factors, identities and exact root exclusion are unchanged.
- Three genuine six-DoF SE3 pose/object Jacobians, from the archived nominal
  seeds 31/32/33. They are synthetic fixed tangent models, NOT online 3D runs.
- One five-block SPD coupling control. Nine models total.
- Ideal links, delayed links, heterogeneous compute periods, bounded source-
  link burst loss, and a finite outage. Three network-calendar seeds, 2000
  synthetic ticks, four methods: 540 final runs. These are correlated numerical
  conditions, not 540 independent SLAM datasets or significance trials.
- Common initial perturbation y[k]=sin(k+1) in information-scaled coordinates.
  This prevents already-converged archived gradients from making the task
  trivial. Neither H nor g nor sensor weights change. No GT is loaded.
- Every actor owns only a reduced row and received neighbor caches. Updates
  do not access global cost/residual, another actor's live state, or the future.
- Absolute states carry revisions and timestamps; old/duplicate deliveries
  cannot overwrite newer states. Messages travel only over declared links.
- The network has per-hop delay and loss accounting but NO contention or real
  link throughput model. A tick is a compute opportunity, not milliseconds.
- Immutable rows and common initial charts are pre-distributed OFFLINE. Their
  storage is reported, but setup communication is excluded. This is a major
  deployment boundary, not a free distributed Schur-assembly claim.

The scalar reference is a fixed-SPD specialization of the local delay bounds
in [Ubl and Hale](https://arxiv.org/html/2203.11307). Its D_ij are supplied valid
communication contracts, not data-adaptive parameters or hindsight-fitted ages.
No step-size sweep was performed. The synchronous control has an optimistic
free barrier and uses the matched block iteration, not the existing native PCG.

## Algorithmic Change

After private elimination, normalize by owner-block curvature. Let B be the
resulting positive-definite Hessian, with B_ii=I up to roundoff. The scalar rule
uses one inverse step size per robot:

`m_i = sum_j ||B_ij||_2 (1 + D_ij + D_ji)`.

The candidate retains coupling directions:

`C_ij = sqrt(B_ij B_ij^T)`

`M_i = B_ii + sum_(j!=i) (1 + D_ij + D_ji) C_ij`

`delta_i = -M_i^-1 cached_gradient_i`.

Thus a direction with no cross-robot coupling is not slowed merely because
another direction depends on stale neighbors. No robust kernel, variance floor,
sensor downweighting, extra acceptance gate or tunable damping gain is added.
The finite bound D is still required. Extra local SVD/factorization work and
matrix-vector work are NOT equal CPU cost to the scalar method.

The [derivation note](dranppr_async_directional_model_20261123.md) constructs an
objective-plus-update-history energy with a descent bound for a fixed SPD model.
The test suite and all 135 candidate traces check the corresponding inequality
within 1e-10 normalized numerical tolerance. This is not a proof of nonlinear
online convergence or evidence that matrix preconditioning itself is novel.

## Results

Each row aggregates 5 network scenarios x 3 calendar seeds. The reduction is
the median percentage reduction in the final RELATIVE QUADRATIC GAP versus the
scalar bounded-delay reference, not ATE, pose RMSE, or physical-cost reduction.
Machine-precision endpoints are treated as ties, not accuracy gains.

| Model | Better / tied / worse | Median gap reduction |
| --- | ---: | ---: |
| MRCLAM D1 | 12 / 0 / 3 | 15.77% |
| MRCLAM D2 | 15 / 0 / 0 | 51.33% |
| MRCLAM D3 | 12 / 0 / 3 | 38.45% |
| MRCLAM D4 | 15 / 0 / 0 | 77.85% |
| MRCLAM D9 | 12 / 0 / 3 | 42.35% |
| SE3 seed 31 | 12 / 3 / 0 | 6.39% |
| SE3 seed 32 | 12 / 3 / 0 | 4.83% |
| SE3 seed 33 | 12 / 3 / 0 | 5.25% |
| Scalar-block SPD control | 0 / 15 / 0 | 0% |

All 135 directional/scalar pairs have identical bytes AND update counts at
EVERY prefix. The candidate has zero observed delay-bound violations. This
isolates an update-rule effect, rather than spending extra iterations or bytes.
It does not establish equal processor time or end-to-end network latency.

The failures are substantive: D1 heterogeneous periods worsens the gap by
23.4--26.0%; D3 outage worsens it by 41.7--42.4%; D9 outage by 103.8--105.9%.
All three network seeds show each failure. A smaller local matrix majorizer
does not order the full asynchronous finite-horizon trajectories.

At seed 31 the candidate initially progresses faster in all three failing
conditions, then retains more error in weak network directions. At the final
time, the three smallest eigenvectors of B account for 99.96%, 92.89%, and
98.26% of its error energy in D1/D3/D9 respectively. This is a diagnostic of
the remaining slow modes, not a causal proof or a reason to hide the failures.

An additional 6000-tick seed-31 diagnostic reverses those three objective-gap
rankings: D1 scalar/directional 2.5391e-5/2.4502e-5, D3 4.6597e-5/3.6906e-5,
D9 6.0060e-5/4.5541e-5. However, the candidate's gradient residual is still
larger in all three. These six runs diagnose a transient/metric tradeoff; they
do NOT replace the fixed primary 2000-tick endpoint or erase its nine failures.

Against the synchronous control, the directional candidate wins/ties/loses
26/12/97 of 135 conditions. Against the direct stale-cache iteration it is
15/12/108. Therefore it is NOT generally faster than all alternatives.
The stale-cache iteration's apparent advantage also shrinks at matched update
counts: of 108 nonideal-network comparisons, it beats the synchronous control
72 times at matched simulated time, but only 14 times at matched update count.
Most of that advantage comes from not waiting, not a better individual step.

## A Necessary Failure Control

For B=0.2 I+0.8 11^T (smallest eigenvalue 0.2), five scalar blocks on a complete
communication graph, and a uniform four-tick delay, after 1000 ticks:

| Method | Relative quadratic gap |
| --- | ---: |
| Synchronous block control | 7.10e-12 |
| Direct stale-cache update | 6.94e61 |
| Scalar bounded-delay update | 3.79e-5 |
| Directional update | 3.79e-5 |

The two bounded variants coincide for scalar blocks. This is a fixed algebraic
counterexample, not a real-data failure or a new SLAM benchmark. It demonstrates
why merely dropping barriers is not a complete algorithm. Positive definite
physical information alone does not protect an arbitrary stale iteration.

## Evidence And Reproduction

Canonical final archives:

- `results/dranppr_async_primary_20261123`: 405 final reference runs.
- `results/dranppr_async_directional_20261123`: 135 matched candidate runs.
- `results/dranppr_async_final_review_20261123`: independent metrics, byte/update
  parity, history-energy checks, and the delayed-SPD counterexample.
- `results/dranppr_async_tail_d1_20261123` and `async_tail_d39_20261123`: six
  longer-horizon diagnostic runs on the retained negative cases.

The earlier `async_fixed` archive is the near-stationary zero-increment control;
`async_unit` precedes correction of an unnecessarily conservative outage bound.
Only routes crossing the failed edge receive the outage-duration term in the
final primary and candidate archives. Earlier archives remain diagnostic, not
the final comparison. No sealed archive was overwritten.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_async_interface \
  --output results/NEW_async_primary --ticks 2000 --initial-state unit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_async_interface \
  --output results/NEW_async_directional --ticks 2000 --initial-state unit --modes directional
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.review_async_interface \
  --roots results/NEW_async_primary --candidate-root results/NEW_async_directional \
  --output results/NEW_async_review
```

Verification: 52 focused Python tests pass, including 26 new asynchronous/model/
review tests and existing native regression coverage; all 67 existing native
C++ tests pass. No production C++ change, CCI change or new PCG implementation.
MPI ranks: zero, not exercised. New paths use local block-matrix updates.

## What Is Still Missing

1. Nonlinear state/chart/model revisions: stale matrices from different epochs
   cannot be substituted into the fixed-model argument.
2. Local nonlinear acceptance without the current all-robot backtracking sum.
3. Distributed row/model setup and its actual communication/computation cost.
4. Unknown or unbounded delays, general multi-hop loss, queue contention,
   partitions and recovery without a predeclared maximum downtime.
5. End-to-end Pose3 online accuracy, common-stream baseline reruns, and actual
   delivered-state age/accuracy. The current results do not establish these.
6. The remaining weak network modes and the repeated finite-horizon regressions.

The next algorithmic question is how to preserve fast progress in weak global
directions while controlling stale cross-robot response, not how to add another
age threshold or inflate sensor noise. Keep the directional metric as a bounded,
audited research candidate; do not merge it into the online default yet.
