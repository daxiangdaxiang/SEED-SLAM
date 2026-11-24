# History-Based Neighbor Prediction Implementation Plan

> **For agentic workers:** Use executing-plans inline. Do not delegate or change production defaults.

**Goal:** Determine whether received history can reduce stale coupling-gradient error and improve the existing update rule at equal communication, without replacing DRAN with CORD.

**Architecture:** Keep the fixed physical Schur model, owner rows, directional metric, update opportunities, absolute message schedule and sensor weights unchanged. An optional receiver-local predictor reads only accepted, timestamped neighbor states; predicted states never overwrite the received cache. Source timestamps refer to post-update states, so the target at the beginning of tick t is t-1, not t.

**Tech Stack:** Existing Python/NumPy/SciPy research simulator and pytest. Separate sealed result folders. Existing research branch; no commit/push of the preexisting dirty tree.

## Fixed Contract

- User approved prediction research, including history-based second/higher order estimation, not conversion to CORD.
- Start with hold, two-sample secant, and anchored least-squares polynomials of degrees 1/2/3 using eight received samples. Eight is an explicit memory/design parameter, not claimed parameter-free. No per-dataset tuning.
- Fit irregular source timestamps with normalized time; do not apply equispaced Savitzky-Golay coefficients to delayed irregular arrivals. Latest received state stays exact.
- Compare the current directional solver with only predictor changed. No momentum, noise adaptation, ground truth, global acceptance or future packets.
- Development: D1/D4/SE3_31, seeds 31, delay/heterogeneous/outage, 600 ticks. Freeze a candidate before primary validation on the remaining models, network seeds and 2000 ticks. Report all explored arms and failures.
- Primary metrics: quadratic objective gap, true-gradient residual, squared coupling-gradient error, byte/update prefix parity, completion. Simulation runtime is not deployed robot CPU/network latency.
- Delay-bound theory from the previous iteration does not automatically survive extrapolation.
- At most three mechanism revisions before a checkpoint. A negative result is not permission to tune indefinitely. Positive conclusions must state their scope.

## Task 1: Predictor And Simulator Contract

- [x] Add tests for irregular linear/quadratic/cubic reproduction, exact latest state, no future/out-of-order mutation, copied history, reset, noisy derivative variance, and zero-delay parity.
- [x] Observe missing-feature failures using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_neighbor_prediction.py`.
- [x] Implement `experiments/neighbor_prediction.py`: bounded received history, anchored polynomial least squares with `np.linalg.lstsq`, explicit secant control.
- [x] Extend `Actor(..., prediction='hold', history_size=8)` and `simulate(..., prediction='hold', history_size=8)`. Keep hold's legacy path bit-identical. Build all eligible gradients before any updates.
- [x] Record estimated and held-cache gradients; compute their errors against actual pre-update states offline, never inside actor decisions.
- [x] Test zero-delay state equality, every-prefix byte/update equality, and causal prefix equality.

## Task 2: Reproducible Experiments

- [x] Add `experiments/audit_neighbor_prediction.py` using archived physical inputs and existing parsers. Snapshot source, exact input hashes, software versions, command, predictions and window size.
- [x] Reproduce current directional traces before candidate comparisons. Save states, counters and per-tick gradient-error diagnostics.
- [x] Run development controls. Diagnose high-order long-horizon/noise failures before proposing one mechanistic correction.
- [x] Freeze one candidate, run full primary comparison and held-out data/calendar checks. Never overwrite sealed archives.

## Task 3: Independent Review And Scope

- [x] Independently recompute final objective/residual and compare every-prefix bytes/updates and zero-delay traces. Review units, timestamps and information access.
- [x] Test history-size transfer (4/8/16) without selecting a best size per dataset, and selected failure controls with abrupt changes and noisy history.
- [x] Run focused Python and existing native C++ regressions. Check ledgers and `git diff --check`.
- [x] Write a report with hypothesis, settings, all outcomes, failure explanations, theoretical intuition and the remaining nonlinear/online/SE3 implementation gap. Do not call a receiver-history predictor CORD or claim an end-to-end baseline win.

## Sources And Prior Evidence

- CORD, Shin/Ghaffari/Tian, RSS 2026: https://arxiv.org/html/2605.11210v1 . State/velocity prediction is prior art; its stated theorem is synchronous and constant-mass.
- Local polynomial derivative estimation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html . Our irregular-time anchored regression is explicit rather than assuming uniform samples.
- Existing report: `docs/dranppr_async_interface_report_20261123.md`. Scalar safety is conservative; directional updates still have nine primary regressions. Offline pre-distribution, bounded delays and synthetic time remain limitations.

## Development Checkpoint 1

Archive `results/dranppr_history_development_20261123`, 45 runs, all complete.
Hold traces match the sealed previous directional archive exactly. At 600 ticks,
secant wins 7/9 objective comparisons; history-linear 5/9; quadratic 8/9; cubic
8/9. Every predictor only wins 5/9 gradient-residual comparisons. Quadratic and
cubic sometimes INCREASE cumulative gradient error despite a better endpoint.
In SE3 outage cubic reduces forecast error dramatically but does not materially
improve the endpoint over quadratic. Do not conflate prediction error and speed.

Freeze `poly2`, eight received samples, as the first candidate before primary
validation. Retain hold and secant controls on all nine models, all five
communication schedules, seeds 31/32/33, 2000 ticks. No parameter change for
D2/D3/D9/SE3_32/SE3_33 or unseen network seeds. Cubic remains a development
ablation, not an unreported unsuccessful attempt or a selected best-per-case.

## Mechanism Revision 2: Noise-Supported History Terms

A seeded noisy quadratic-trajectory diagnostic confirms that higher order is
not inherently robust: with noise standard deviation 0.01 and horizon 16,
mean squared forecast error is about 0.157 hold, 0.114 secant, 0.011 linear
history, 0.025 quadratic and 1.034 cubic. These are synthetic prediction errors,
not physical SLAM errors. Do not tune each polynomial order on each dataset.

Test one unified alternative `shrink3`. Fit an unanchored polynomial of maximum
degree three to all received samples using QR; forecast the fitted increment
and add it to the exact latest cache state. Estimate unexplained variance from
the residual with at least two residual degrees of freedom. In the orthogonal
history basis, shrink each nonconstant coefficient by
`max(0, 1 - residual_coupling_variance / coefficient_coupling_energy)`.
This is a positive-part moment/empirical-Bayes-inspired shrinkage rule. The
receiver's existing cross block defines the coupling metric. No new mixing
gain or dataset-specific threshold is fitted. Correlated optimizer histories
violate the simplest white-noise model; this is a candidate, not calibrated
posterior uncertainty or a stability guarantee. Keep the same eight-sample
memory and development cases, then freeze before its primary run.

The shrinkage candidate fails its prespecified noisy-history regression target:
MSE 0.3005 versus raw cubic 0.9683 (target below 20% of raw cubic). Development
objective wins also decline from 8/9 to 7/9. Its 27-run sealed source archive is
retained, but the superseded candidate is removed from active code rather than
relaxing the regression target or growing a permanent optimizer portfolio.

## Mechanism Revision 3: Forecast-Horizon Risk, Not Coefficient Magnitude

The missed term is extrapolation leverage: a statistically modest high-order
coefficient can have an enormous effect far beyond the observed window. Fit
degrees p=0..3 (with at least two residual degrees of freedom) using unanchored
QR, but retain the exact latest state and forecast only the fitted increment.
If a_p(h) are the resulting absolute sample weights, estimate the coupling-space
forecast variance as residual_variance_p * ||a_p(h)||^2. Choose the degree with
minimum estimated risk; p=0 is exactly hold. This is one model-selection
criterion, not a cascade of gates or hand-tuned mixing gains. It explicitly
accounts for correlation between the retained latest sample and the fitted
increment. Its residual-variance interpretation assumes locally smooth dynamics
and roughly homogeneous unmodeled variation; future discontinuities remain
unobservable, and the selection score is not a certified risk bound.

Name the candidate `risk3`, retain eight received samples. Reuse the same
development grid and noise test; do not make a fourth mechanism revision in
this stage. Freeze it before primary/held-out verification if it passes.

Revision 3 passes the unchanged noisy-history reduction target. Its development
grid has 8/9 objective wins and 5/9 residual wins, with the same repeated D1
heterogeneous limitation. Freeze `risk3` now; evaluate it alongside hold,
secant and poly2 over the full 2000-tick matrix. Do not silently replace the
original poly2 candidate or tune the adaptive selector after held-out results.

The first primary runner stopped on the final scalar SPD model because its
independent residual check compared 6.1e-17 with 1.5e-16 using only relative
tolerance. This is a GEMV/GEMM roundoff audit bug, not solver divergence. Add an
explicit regression test and use a dimension/data-scaled floating-point
forward-error bound. Rerun a complete primary archive; retain the interrupted
folder only as an audit diagnostic, not final evidence.

## Outcome

Canonical primary: 540 complete runs; adaptive history versus hold has 93 wins,
27 ideal-network ties and 15 losses. Independent review checks all 540 metrics,
135 old-archive state matches and 405 every-prefix budget matches. A separate
forecast audit confirms the high-order noise failure and mitigation, without
claiming universal dominance. All eight complete-graph controls are retained,
including catastrophic failure when prediction is paired with unsafe steps.
Memory 4/8/16 retains the same six-win/three-loss pattern on its limited grid.

71 focused Python tests and 67 existing C++ tests pass. Keep the current
research branch as-is; do not merge/push or promote the predictor to production.
Full findings, including the 13.39x Python service-time overhead and the lack of
dominance over simpler predictors, are in
`docs/dranppr_history_prediction_report_20261123.md`.
