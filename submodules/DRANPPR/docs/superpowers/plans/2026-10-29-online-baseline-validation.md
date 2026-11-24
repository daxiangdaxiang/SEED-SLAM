# Online Calibration and Baseline Validation

> Execute inline in the existing research worktree. No agents, commits, pushes
> or production changes. Continuation approved by the user, now requiring our
> fixed-noise baseline and riMESA as explicit references.

**Goal:** Test transfer of the already fitted Q/R scales to persistent causal
updates, against fixed-noise native DRAN and pinned riMESA, without comparing
centralized batch completion directly against an online baseline.

**Architecture:** Add an opt-in external five-vector to the research replay's
suffix factor construction. Keep its nominal prefix, solver, refresh policy and
default path unchanged. Fit nothing during this benchmark. Restore pinned
baseline dependencies under research build/. Run D1/D4 to 120 seconds, then
evaluate GT only after producer completion. Parameter-uncertainty diagnostics
remain separate from estimator-comparison claims.

**Tech stack:** existing C++17/GTSAM native replay, riMESA/riSAM pinned sources,
Python runner/evaluator, pytest/GoogleTest.

## Fixed Comparison Contract

- Source: checksummed `dranppr_network_noise_transfer_qr_20261003` D1/D4
  events, common-past states and frozen five-component scale fits.
- Our baseline: native Pose2/Point2 DRAN `joint_cached_one --lazy-refresh`,
  fixed nominal physical noise. This is NOT the packaged 3D DRAN++ executable.
- Candidate: same replay and same initialization, only suffix variances changed
  to the old fitted multipliers. No new sweep, mean states, rejection gate,
  covariance floor, robust kernel or additional optimizer budget.
- riMESA: pinned commit ffddc21dad530ba0b0e57c42495146b410581fc4, pinned riSAM
  c6fa05eaece103efd7e44e640bda8df6eff30e8d with the already disclosed endpoint
  and merit repairs; never label repaired code as unchanged official upstream.
- Include paper-iMESA Gaussian control to separate robustness from consensus.
- Baselines use one full sweep/frame and a causal cumulative byte ceiling equal
  to the smaller of the two native arms' current cumulative communication. Never
  compare only an artificially expensive byte-filling schedule's runtime.
- All modes share event support, root gauge, neighbor topology and prefix
  physical graph. Baseline internal dual/kernel state starts fresh at common
  primal warm start, as in the previous audited adapter.
- Report final trajectory/object-copy RMSE, causal RMSE, copy disagreement,
  actual communication, service time and failures. Costs under different noise
  weights or inconsistent copies must NOT be ranked as equivalent objectives.
- Keep historical checkpoint NLL and uncertainty analysis explicitly separate:
  robust riMESA has no audited calibrated predictive covariance export here.
- Sequential one-thread execution, 600 s per estimator, preserve failures.
  No claims of equal wall time, exact byte equality, real network latency,
  independent sequences, high-noise robustness or universal superiority.

## Implementation and Verification

- [x] Restore riMESA dependencies and test the unchanged baseline adapter;
  verify source snapshots/pins and no system-GTSAM fallback.
- [x] Add tests first for opt-in five-vector: physical insertion loss formula,
  unchanged prefix, explicit-one bitwise parity, causality, invalid input and
  duplicate/no-warm-start rejection. Then implement suffix construction.
- [x] Build research replay and run native/noise/online regression suites.
- [x] Implement matched run/evaluate runner using existing provenance, ledger
  and GT-alignment helpers. Test min-credit schedule before implementation.
- [x] Execute both datasets and all declared schedules, retain every outcome,
  check arrived-graph parity and per-frame communication ceilings.
- [x] Check a fresh nominal D4 endpoint against the archived lazy-refresh arm;
  preserve differences and distinguish timing repeats from independent trials.
- [x] Report results, status of parameter-uncertainty validation, exact commands,
  hashes and current limitations. No promotion to production default.

## Separate Bounded Parameter-Uncertainty Diagnostic

Freeze six discovery cases before looking at uncertainty results: D1/D4 at
30, 50 and 85 s from the completed rolling archive. They include an old control,
the newly exposed D1 failure, and late-time cases. They are deliberately selected
diagnostics, NOT a replacement for full-suffix or untouched-sequence evaluation.

Use the inverse expected REML Fisher information for log scales on the ORIGINAL
training contrasts. Approximate that local asymptotic Gaussian with ten equal,
positive-weight cubature points (plus/minus sqrt(5) times Cholesky columns).
For each point, re-solve the past-only nonlinear checkpoint and recompute its
future Gaussian forecast. Average DENSITIES, not NLLs or just covariance scalars.
This includes mean, conditional covariance and between-mean variation without
an arbitrary inflation factor. No GT or future observation is used to fit weights.

This is an asymptotic sampling-uncertainty sensitivity diagnostic, NOT an exact
Bayesian posterior or a misspecification-robust sandwich covariance. Cubature
accuracy, calibration/state dependence and misspecification remain limitations.
Report mixture NLL and range mean/variance, but do not call its moment-matched
ellipse a calibrated 95% mixture region. Do not impute such a distribution to
riMESA or mix these batch predictions into the online state comparison.

- [x] Test Fisher scaling, cubature moments and mixture-density/moment algebra.
- [x] Complete six cases, ten nonlinear solves each, 240 s per worker and a
  1800 s total watchdog. Preserve all points; do not select a favorable one.
- [x] Report whether this specific uncertainty approximation repairs the
  observed failures, without extrapolating to all parameter-uncertainty methods.

Outcome: all 12 online runs and 60 separate uncertainty solves finish. Final
state gains transfer, but D1 first-published trajectory degrades. Scale
uncertainty softens, not repairs, selected predictive failures. Details in
`docs/dranppr_online_calibration_baselines_report_20261029.md`. No production
promotion. Next diagnostic separates finite-work tracking from insufficient
past-only physical evidence on identical checkpoints.
