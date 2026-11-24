# Rigid Subspace Online Response Plan

> Execute inline using executing-plans; retain the existing research branch.

**Goal:** Establish a reproducible progress milestone on online state tracking,
without changing measurements, weights, root constraints or association.

**Architecture:** Continue the approved collective-motion investigation. In
the existing reduced interface problem, minimize over the span of the original
block-Jacobi direction and three SE2 collective landmark directions. Assemble
only a 4-by-4 projected model through the existing routed scalar collective.
Private trajectories still recover through their original local elimination;
the root stays excluded. Existing physical backtracking and lazy refresh remain.
This is an opt-in research solver, not a default or a new noise model.

**Tech Stack:** C++17, Eigen, pinned GTSAM, existing Python experiment/evaluation.

## Design and Alternatives

- More PCG iterations are a control, not a new algorithm; they cost more traffic.
- Coarse initialization followed by CG may help, but does not include the original
  first-step direction in the same minimization space.
- Choose augmented one-step minimization: V=[Z, -D^-1 g],
  c=-(V' S V)^dagger V' g, delta=V c. It includes the old one-step PCG line,
  so exact quadratic minimization cannot do worse on that model. It need not
  improve nonlinear truth error. Rank handling uses machine precision, not a
  dataset-tuned threshold. All actual routed packets are counted.
- Rigid-mode coarse spaces are established numerical methods, not a novelty
  claim: https://epubs.siam.org/doi/abs/10.1137/100803651 .

## Tasks

- [x] TDD algebra: exact projected solution, duplicate/rank-deficient directions,
  current-chart displacement, shared-copy consistency, root recovery, every-hop
  accounting and unchanged baseline. Genuine 3D geometric bases get a separate
  deterministic test; this alone will not be called a 3D online backend.
- [x] Add opt-in interface path and replay flag `--rigid-subspace` for the
  common-past one-step mode. Keep PCG counters separate from subspace solves.
- [x] TDD integration: fixed root, unchanged factors, physical descent and causal
  prefix invariance. Build `testIncrementalLocalModel` and `native-joint-replay`.
- [x] Replay frozen nominal/calibrated weights on D1 and D9 failure cases, then
  D2/D3/D4 transfer controls. Freeze algorithm before comparing GT. Compare
  first-published quality, terminal quality, objective, bytes and service time.
- [x] If positive, confirm same-byte-or-lower budget and repeat determinism,
  retain all negative cases. Compare pinned iMESA/riMESA at causal byte ceilings
  where practical. Otherwise isolate the failed assumption before another change.
- [x] Record evidence and limits, seal fresh output/sources and run regressions.

Milestone criterion: measured causal tracking improvement on original failure
cases with no hidden future information/changed objective, transparent cost,
and transfer controls. Partial success must be labeled partial; no blanket
robustness, bounded-memory, full-3D-online or theoretical-novelty claims.

## Evidence-Driven Follow-Up

The coarse candidate has mixed truth-error results on five datasets; do not
promote it. Its projected execution exposes redundant final vector transport
in the existing one-step PCG path. Test an isolated algebraic protocol change:
keep the original gradient, direction, curvature reduction and alpha arithmetic;
locally reconstruct alpha times the already received direction and use the
linearity identity S(alpha p)=alpha Sp. No coarse directions or weights change.
The residual is now checked from the cached product, not by a fresh operator
application. This is a communication/computation milestone, NOT success of the
original coarse-space accuracy hypothesis. Require byte-identical causal and
terminal state CSVs on all five datasets at both weight settings before claiming
an equivalent lower-cost online update. Also audit fresh dense residuals in tests.

Outcome: coarse accuracy criterion not met; do not promote it. The fused
protocol meets strict ten-condition state and physical-trace parity with
22.78--27.13% lower traffic. No meaningful wall-time gain established. Historical
iMESA/riMESA accuracy comparisons are explicitly inherited, not rerun at the
new byte ceiling. 242 Python and 67 C++ tests pass. Keep the research branch
as-is; no commit, push or production default change. Detailed evidence is in
`docs/dranppr_subspace_milestone_report_20261123.md`.
