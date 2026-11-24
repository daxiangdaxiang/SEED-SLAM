# Asynchronous Fixed-Interface Diagnostic Plan

> **For agentic workers:** Execute inline using executing-plans. Continue in the
> existing isolated DRANPPR research checkout; do not modify production defaults.

**Goal:** Determine whether a barrier-free, curvature-scaled interface iteration
can solve the SAME fixed physical quadratic under delayed/lost messages, before
attempting a nonlinear online replacement.

**Architecture:** Eliminate each robot's private pose increments from archived
physical Jacobians, retaining shared object increments. Partition the reduced
rows among robot owners. Actors store only their row and received neighbor
states. A deterministic event simulator routes absolute versioned states over
the existing chain. No global cost, residual, reference solution or future
packet outcome enters an asynchronous update. Immutable, pre-distributed rows
are an explicit prerequisite of this first diagnostic, not a deployment claim.

**Tech Stack:** Existing NumPy/SciPy sparse factorization and pytest. No new
dependencies, variance model, rejection cost or production C++ change.

## Decision And Scope

Compare (1) synchronous block-preconditioned gradient with barriers,
(2) the same update with stale caches and no barriers, (3) a conservative
locally bounded-delay rule. For H with owner-block Cholesky factors L_i, use
x_i=L_i^{-T} y_i, B_ij=L_i^{-1} H_ij L_j^{-T}, c_i=L_i^{-1} g_i.
Each actor updates y_i -= gamma_i (c_i + sum_j B_ij cached_y_j).
Curvature-only gamma_i=1/sum_j ||B_ij||_2. The bounded-delay reference uses
gamma_i=1/sum_j ||B_ij||_2 (1+D_ij+D_ji), the midpoint of the strict admissible
interval in Ubl/Hale Theorem 1: https://arxiv.org/html/2203.11307 .
This is a reference method, NOT a new theorem or a newly invented algorithm.
The bounds are supplied communication contracts, not inferred from data.

Known alternatives: edge consensus ADMM is a strong later baseline; arbitrary
asynchronous CG is invalid because stale products need not share one operator.
Do not replace a fixed delay bound by observed message age and inherit the
theorem without a separate argument. Full private elimination is held fixed.

## Tasks

- [x] Add failing tests in tests/test_async_interface.py: exact elimination,
  singular rejection, curvature normalization, isolated actor state, stale and
  duplicate delivery, every-hop bytes, disconnected topology rejection,
  synchronous/async zero-delay parity, fixed delay-bound compliance, recovery.
- [x] Implement experiments/async_interface_model.py using scipy sparse LU for
  private normal blocks; compare assembled Schur solve with a full sparse solve.
  Parse the existing physical-factor schema, without truth access or new priors.
- [x] Implement experiments/async_interface.py with actor-local rows and a
  heap-based routed event queue. Use deterministic per-link delay/loss calendars,
  fixed heartbeat opportunities, versioned full states, heterogeneous local
  compute periods, and an explicit finite outage. Do not emulate bandwidth
  congestion or claim physical milliseconds. Setup is excluded and labeled.
- [x] Add experiments/audit_async_interface.py. Freeze five MRCLAM snapshots
  and genuine 6-DoF SE3 Jacobians from three archived seeds; include a small SPD
  coupling stress control. All modes share weights, initial state, ownership,
  network calendar and horizon. No parameter sweep. Use 3 network seeds.
- [x] Run smoke then bounded full matrix (at most 6000 ticks per run initially).
  Record residual/objective gap, updates, bytes, cache age, and bound violations.
  Evaluate reference metrics posthoc only; do not stop actors on a global norm.
- [x] Review against independent tests, preserve negative results, summarize
  limits and next gate. Seal inputs, source snapshots, traces and summaries.

## Validation Contract

The positive gate is barrier-free fixed-quadratic progress and recovery, not
better ATE or superiority over DRAN/iMESA/riMESA. The synchronous comparison is
a matched block iteration, NOT the existing PCG backend. No claim of realistic
network throughput, bounded memory, continuously changing charts, full 3D
online operation, or novelty of delay-aware coordinate descent. The assembled
reference matrix is allowed only for setup and offline evaluation. Permanent
partition and unbounded IID loss are outside the bounded-delay guarantee.

Current-code barriers to remove in a later phase: NativeInterface::reduce and
the physical-cost reduction/backtracking in native_local/joint_step.h.

## Smoke Review Before Primary Evaluation

The archived Jacobians were exported near a converged nominal solution; their
reduced gradients are only about 5e-7 to 3e-5. Keep the original zero-increment
run as a near-stationary diagnostic, but do not market it as online progress.
For the primary fixed-quadratic test, set the SAME deterministic common start
y[k]=sin(k+1) in all arms, without changing H, g, noise or observations. This is
a dimensionless information-scaled perturbation, not a truth-assisted start.
It tests linear-solver dynamics only, not validity of a nonlinear step of that
size. No amplitude sweep. Both starts and all negative cases remain recorded.

The loss control has four bad source-link ticks per twelve ticks. Relay links
are delayed but reliable, giving a finite auditable end-to-end delay bound.
This does not establish resilience to arbitrary independent multi-hop loss.

Independent review found that the first outage contract added the downtime to
every route, even routes not crossing the failed edge. It was a valid but
unnecessarily conservative bound. Add a regression test and restrict this term
to paths through that edge; rerun the primary common-perturbation matrix. Do not
use the earlier unnecessarily pessimistic outage result as the final comparison.

## One Algorithmic Follow-Up, No Parameter Sweep

Scalar delay bounds suppress every local direction equally. Test one matrix
replacement: C_ij=(B_ij B_ij^T)^(1/2), computed by SVD, and
M_i=B_ii+sum_(j!=i) (1+D_ij+D_ji) C_ij. Update by -M_i^-1 cached_gradient.
The cross inequality 2 u^T B_ij v <= u^T C_ij u + v^T C_ji v follows from
the SVD and the squared norm of the paired singular-coordinate difference.
This yields a directional analogue of the scalar cross-term bound and leaves
uncoupled directions undamped. Matrix updates and asynchronous BCD are existing
ideas; this adaptation is a candidate, not an established novelty claim.

Add tests for the cross inequality, scalar-block reduction, and an uncoupled
direction. Reuse exactly the final primary inputs, network calendars and
initial states. Run only this fourth arm and compare every-prefix bytes and
update counts to the archived scalar bounded arm. Write the fixed-quadratic
Lyapunov calculation separately, clearly excluding nonlinear changing models.

## Outcome

Completed 405 final reference and 135 directional-candidate runs, plus the
retained zero-start/pre-local-bound diagnostics and six longer-tail controls.
Directional versus scalar bounded delay: 102 wins, 24 ties, nine losses at
identical every-prefix bytes and update counts. All candidate history-energy
checks pass within numerical tolerance. The negative cases and the fact that
the candidate does not dominate the synchronous/stale controls are explicit in
docs/dranppr_async_interface_report_20261123.md. The implementation has 26 new
tests; 52 focused Python and 67 existing native C++ tests pass. This phase is a
fixed-model algorithmic feasibility result, not completion of asynchronous
nonlinear online SLAM. Keep the research branch as-is; no production promotion.
