# Horizon-Matched Error-Budget Implementation Plan

> Execute inline using executing-plans and TDD. The user's continuation approves the next error/step-coupling experiment. No production changes, commits, or delegation.

**Goal:** Test whether realized, horizon-matched prediction errors can support faster local updates without merely increasing extrapolation order.

**Architecture:** Keep the first-order response predictor unchanged. Record its forecast at each actual local read target (including shadow forecasts during fallback). Validate a forecast only against a received state at that source stamp, or a piecewise-constant source interval certified by consecutive revisions. Keep the last eight scalar error norms per exact age. Missing/untrained ages use the original directional hold step; trained ages use an error-penalized fresh-metric step. This is an empirical error model, not a certified future bound.

**Tech stack:** Existing Python/NumPy/SciPy fixed-quadratic simulator. No native C++ or dependency changes.

## Fixed Rule

Let P0 be the inverse of the existing zero-delay block majorizer. For each
neighbor and integer age h, epsilon_ij is the maximum of the last L=8 validated
forecast-error norms in P0 coordinates, requiring L samples of this exact age.
Zero age is exact and has zero error. Sum epsilon_ij to obtain epsilon_i.
If any nonzero age lacks support, execute the original delayed hold step.
Otherwise use s_i=-alpha_i P0_i ghat_i, with
alpha_i=max(0,1-epsilon_i/||ghat_i||_P0). A zero gradient gives a zero step.
No signal-dependent tuning constant or dataset-specific setting is introduced.

This is the closed-form minimizer of
ghat_i' s + .5||s||_M0^2 + epsilon_i||s||_M0.
If epsilon_i actually bounds the current gradient error in the dual metric,
the quadratic majorization gives a conditional descent contribution. Historical
maxima are NOT guaranteed future bounds. Mixed fallback steps do not inherit
an automatic common Lyapunov proof. Audit coverage and failures explicitly.

## Calibration Protocol

- Source states are piecewise constant between optimizer revisions. Consecutive
  received revisions certify the intervening interval; revision gaps do not.
- Do not interpolate missing revisions or score against the wrong timestamp.
- Record the bound available BEFORE the delayed label arrives. Never use the
  current label to count its own coverage. Late/obsolete packets do not alter
  the optimizer cache. Unknown labels are counted and discarded.
- Pending forecast memory is bounded by the supplied delay contract. Stats
  have at most L entries per exact age. Reset invalidates all charts/history.
- L=8 is inherited from the existing predictor, not tuned in this study.

## Arms And Attempt Budget

1. directional+hold (baseline, exact archived parity)
2. directional+response_linear (prediction only, archived parity)
3. error_budget+hold (update adaptation without prediction)
4. error_budget+response_linear (candidate)
5. fresh_unchecked+response_linear (diagnostic: remove error control)

One candidate and its ablations, no per-dataset sweep. Start D1/D4/SE3_31 x
delay/heterogeneous/outage x seed31, 600 ticks. If catastrophic regressions
occur, diagnose before expansion; negative findings are valid outcomes. Then
use the existing nine-model/five-network/three-seed matrix at 2000 ticks and
long-outage/strong-coupling controls. Compare bytes, updates, objectives,
error-bound coverage, fallback/zero steps, and service time. No ATE claim.

## Tasks

- [x] Add red tests for same-age support, consecutive-revision labels, missing
  labels, no self-coverage, bounded memory, reset, and robust-step algebra.
- [x] Implement `experiments/prediction_error_budget.py` and optional actor
  policies in `experiments/async_interface.py`. Preserve every legacy path.
- [x] Run focused tests and development; freeze candidate before wider runs.
- [x] Implement a sealed audit driver and independent reconstruction of actual
  updates, objectives, packet budgets and current-error coverage.
- [x] Run appropriate expansion/stress, same-accuracy timing where qualified,
  and report limitations, ablations and reproducible results.

## Outcome

Completed 675 main runs, 45 development runs, 40 stress runs, 36 random-start
controls and twelve memory-sensitivity runs. Both adaptive-update arms improve
108/135 main conditions and tie all 27 ideal conditions. The predictive arm
catastrophically fails three long-delay stress cases; the hold ablation improves
all eight stress cases. Size-16 resolves the two inspected predictive failures,
but size-4 worsens them: memory independence is explicitly rejected, not hidden
by changing the frozen default.

Same-accuracy timing includes 81 baseline-endpoint replays, 27 post-blackout
replays, and 24 fixed-1e-6 replays plus three censored baseline records. The
hold ablation is faster than baseline in eight of nine endpoint comparisons,
and faster than prediction in all nine. This is fixed-quadratic Python evidence,
not native online deployment. Keep the research branch and production default
unchanged. See `docs/dranppr_horizon_error_budget_report_20261124.md`.

References: local bounded-delay updates https://arxiv.org/abs/2203.11307;
relative-gradient-error analysis https://arxiv.org/abs/2506.17145. These do not
prove that a retrospective empirical envelope bounds our future errors.
