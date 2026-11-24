# Temporal Mean Response Audit Plan

> Continue the approved future-prediction validation inline, using
> experiment-craft, executing-plans and tests first. No agents, commits,
> production changes, gates, correction clipping, or new noise scales.

**Goal:** Test whether a fixed robot-wise range-mean family estimated from past
physical factors improves future observations after accounting for state/map
response and finite mean-parameter uncertainty.

**Architecture:** Reuse the archived D1/D4 nominal/joint-QR checkpoint physical
Jacobians, states and five-second forecasts. Fit the linear diagnostic
`b + A delta + B beta` using QR and SVD; B has one constant physical range
offset per robot. Feed its exact linear state response and covariance into the
existing past-only predictor. No new variables are added to the C++ backend.

**Tech Stack:** Existing Python physical-export/forecast helpers, NumPy,
SciPy QR/SVD, pytest, JSON provenance; restored pinned native dependencies.

## Protocol

- Source archive and its 360 hashes remain unchanged. D1/D4, 30/60/90 s,
  nominal and joint-QR: 12 fitted models. The previously examined 533 future
  observations per weighting arm remain the same; these are NOT blind tests.
- Estimate beta only from the arrived checkpoint graph. Keep all original
  weights, including nominal prefix weights; never refit Q/R or use future/GT.
- QR A=QR. K=-R^-1 Q'B, delta0=-R^-1 Q'b, E=B-QQ'B.
  beta=-E^+ b, V_beta=(E'E)^-1 computed by SVD. Require full physical state
  rank and full five-mean rank; otherwise record failure, no damping/prior.
- For a future physical Jacobian J and range-offset row D, use
  mean shift J*delta0 + (D+J*K)*beta. Add
  (D+J*K)*V_beta*(D+J*K)' to the original predictive covariance in the full
  uncertainty arm. This includes the state/mean cross-covariance exactly in
  the fixed linear model. The mean-only arm omits this addition diagnostically.
- Three arms: original, linear mean only, linear mean plus uncertainty.
  No arm is selected based on results. No variance inflation knob or threshold.
- Reproduce original forecasts first; fail on changed event support or source
  hash. Changing future measured values may change scores, never predictions.
- Report all windows, robots and object groups, average/joint and range NLL,
  coverage, signed innovations, correction sizes, beta uncertainty and rank.
  Report negative predicted ranges/large changes rather than hiding cases.
- This is a fixed-chart response audit, not a new nonlinear solver, real-time
  distributed algorithm, noise-parameter integration or proof of identity.
- Per worker watchdog 240 s; total producer watchdog 1200 s; one BLAS thread.
  No timeouts silently retried or dropped. No full new dataset sweep.

## Tasks

- [x] Restore pinned GTSAM 03c263fe0 with archived no-TBB/system-Eigen flags
  under ignored build/, restore native targets, and run the native suite.
- [x] Add tests for augmented least-squares/Schur covariance equality,
  rank failure, forecast response, unchanged baseline and future independence.
- [x] Add a small experiment-only response helper and optional predictor input.
- [x] Run all 12 models and forecast variants; archive commands and outputs.
- [x] Analyze every group, document useful/failed transfer and limits.
- [x] Run focused regressions (91 Python, 62 native), check four native replay
  endpoints (exact equality) and hashes; confirm
  production remains clean. No PCG/MPI dataset experiment in this stage.

## Reproduction

Post-selected approximation check: the first pass has state shifts up to
0.885 m. Re-evaluate the SAME step with the pinned Pose2 right-composition
chart and nonlinear polar geometry, keeping the original response covariance.
This is not a refitted nonlinear model or another tuned arm. Require fit and
event-support parity; retain every model. Test composition and future-data
independence first, then run one bounded 1200-second analysis process.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 1200 \
  python3 -m experiments.audit_temporal_mean_response \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output results/dranppr_temporal_mean_response_20261027
```
