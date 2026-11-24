# Tracking and SE3 Evidence Diagnosis

> Continue the approved failure-diagnosis direction inline using experiment-craft,
> executing-plans and test-driven-development. Existing research worktree only;
> no agents, commits, pushes, production changes or changes to sealed archives.

**Goal:** Distinguish optimization lag from objective/model damage in D9, and
test whether weak finite variance evidence can produce the SE3 boundary fits.

**Architecture:** Reuse the existing same-arrived-graph checkpoint audit with
explicit dataset/checkpoint validation; keep solver budgets/likelihood unchanged.
Then generate exact linear Gaussian samples on an archived physical SE3 design,
varying only independent repetitions of the same measurement geometry. This is
an information diagnostic, not a new online solver or another robustness gate.

**Tech Stack:** Existing C++ native binaries, Python/SciPy/SuiteSparseQR REML,
pytest, SHA256 archives and independent GT evaluation after estimation.

## Task 1: D9 Same-Graph Completion

- [x] Add failing tests in `tests/test_checkpoint_tracking.py` for source-declared
  datasets, finite/increasing checkpoint times after warm-up, dataset provenance
  and a streaming horizon ceiling. Preserve D1 defaults [30,60,90].
- [x] Extend `experiments/audit_checkpoint_tracking.py` with optional dataset and
  checkpoint CLI arguments. Do not fork a second solver/runner. Snapshot source,
  verify replay binary and causal-prefix equality, and retain producer failures.
- [x] Run D9 checkpoints [150,210,270], fixed/calibrated weights, frozen-graph
  LM100/GN100. Same physical factors, no future events or refitting. Existing
  180 s per-command cap; no timeout extension or alternative starting point.
- [x] Evaluate whole graph and five latest poses with one fixed prefix frame.
  Preserve LM/GN disagreement if observed. Completion gains show reachable
  state changes, not that a distributed online budget can achieve those gains.

## Task 2: Exact-Linear SE3 Evidence Control

- [x] Add failing tests for independent repeated noise with identical state
  columns/group labels, seed reproducibility, and nominal/fitted risks measured
  on a separately drawn test vector. Duplicate noise must not count as new data.
- [x] Add `experiments/audit_pose3_evidence.py`, using the archived seed-35 full
  SE3 physical Jacobian only. Fixed seeds 401--440; nominal [1,1,1,1] and
  heterogeneous [.1,4,9,.25]; 1x and 4x independent observations, same A columns.
  Initial log scales zero, original bounds/tolerances; all 160 fits counted.
- [x] Estimate Fisher log-scale information at the known generating variances
  for each design, and compare boundary counts, scale estimation error and
  independent-test weighted-state estimation risk. Truth is supplied only to
  this explicitly oracle-generative audit, not a deployed calibration method.
  Use exact linear y=A*0+epsilon so nonlinear mean error is absent by design.
- [x] If boundaries occur in this exact model, weak finite-sample evidence is
  sufficient to explain such behavior in principle, not proven the sole cause
  of the previous nonlinear cases. 4x observations are a diagnostic intervention,
  not a recommendation to wait for a tuned number of frames.

## Task 3: Verification and Report

- [x] Run focused and existing regression selections; rebuild/run 62 native tests.
- [x] Check archived source hashes and production status, preserve all declared
  failures and negative controls, and write the report and next decision.
- [x] Keep full 3D online migration, continuous distributed calibration and
  guaranteed no-regression performance explicitly open.

## Self-Review

Increasing online iterations directly would mix budget and information timing;
refitting noise would change the objective. The fixed-graph intervention isolates
completion of the existing objective first. Clamping 3D variances could hide weak
evidence; the exact-linear repeated-observation control changes only information.
No result-dependent seed removal or new parameter selection is allowed.

## Outcomes

All 18 D9 processes and 160 SE3 exact-linear fits complete. Independent dense
QR checks pass for all 160. D9 completion helps latest fixed-frame states, but
does not remove the t150 calibrated regression. A posthoc single-rigid-fit
description (added after seeing the state/registration discrepancy) finds
mostly collective completion displacement; it is tested, never applied to
estimator output. This is not a spectral proof or a new solver.

Heterogeneous boundary counts change 7/40 -> 0/40 with independent 4x training
evidence and identical test data. Correct-nominal risk penalty changes 2.216%
-> .160%. No estimator/default change. 237 Python and 62 C++ tests pass.
