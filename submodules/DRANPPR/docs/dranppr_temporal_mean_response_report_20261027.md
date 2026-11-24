# Past-Only Temporal Mean Response Audit

Date: 2026-10-27. Research-only diagnostic; production DBA-DRANPP unchanged.

## Decision

Do not add a constant range-bias state per robot to the deployed estimator on
the strength of the previous structural residual test. The tested first-order
mean response explains past residuals but does not improve aggregate future
prediction under the existing adapted Q/R weights. Accounting for finite
mean-parameter uncertainty does not reverse this result.

The native dependency/reproducibility blocker IS resolved in this turn. The
statistical/robust-online problem is NOT. This is a negative candidate result,
not a new ATE gain or a comparison against riMESA.

## Purpose and Protocol

The preceding mean-evidence study found concentrated empirical residual energy
in robot-range-offset directions. That finding neither identified physical
sensor bias nor demonstrated persistence. This audit tests the narrower question:
can a simple past-estimated mean family predict the next five seconds?

- Reuse the original, checksummed D1/D4 graphs, states and forecasts at
  30/60/90 s, for both nominal and joint-QR weights. All 360 archive hashes pass.
- Twelve past-only fits, all five robots included, no GT, future observations,
  added prior, clipping, rejection threshold, new Q/R values or parameter sweep.
- Keep the original factor weights, including nominal prefix weights. Noise
  multipliers were learned on the old past prefix and remain frozen.
- The model is a fixed linear diagnostic around each archived nonlinear state;
  no new variables or objective are introduced into the C++ backend.
- Three arms: original predictor; mean correction with original covariance;
  mean correction with its full linear parameter/state covariance contribution.
- Check identical support: 251 D1 and 282 D4 observations per weighting arm.
  Seventeen new-object observations remain unsupported, exactly as before.
- These are previously inspected discovery windows, not unseen confirmation
  data. The six windows do not cover the whole causal suffix.
- Each of 12 workers completed under a 240-second watchdog, one BLAS thread;
  cumulative worker wall time 175.68 s. Times are not an online latency benchmark.

## Linear Response, Not a Subtracted Constant

Let b be the current whitened physical residual and A the state Jacobian. B
contains the physical range derivative for one shared offset per robot:

    min_delta,beta || b + A delta + B beta ||^2.

With thin QR A=QR, compute E=B-QQ'B. SVD of E supplies beta and its covariance
without a floor or a prior. All twelve state/mean systems have full required
rank. Define

    delta0 = -R^-1 Q'b,   K = -R^-1 Q'B,
    beta_hat = -E^+ b,    V_beta = (E'E)^-1.

For a future measurement with state Jacobian J and direct mean effect D,

    predicted shift = J delta0 + (D+JK) beta_hat,
    added covariance = (D+JK) V_beta (D+JK)'.

The term JK matters: a putative sensor offset changes the inferred map and
poses. Simply adding beta to future ranges would miss that compensation and
would count uncertainty incorrectly. The full linear Gaussian block inverse
and the above formula agree in independent unit tests.

The covariance accounts for uncertainty in beta CONDITIONAL on the original
noise scales and chart. It does not integrate uncertainty in the fitted Q/R
parameters or certify the nonlinear posterior. Physical prediction still uses
the original marginal odometry process law.

## Primary Results

Mean joint predictive NLL, lower is better:

| Data / weights | Original | Mean only | Mean + uncertainty |
| --- | ---: | ---: | ---: |
| D1 nominal | -1.576927 | -1.502268 | -1.500541 |
| D1 joint-QR | -2.023980 | -1.959850 | -1.960483 |
| D4 nominal | -1.466104 | -1.466479 | -1.468358 |
| D4 joint-QR | -1.717643 | -1.688949 | -1.691993 |

The small nominal D4 score gain is not a general repair: its range prediction
RMSE increases slightly. With adapted weights, both dataset aggregate scores
worsen. All twelve augmented linear fits reduce their own training squared
residual, by 4.09 to 83.66 objective units. This is expected from an enlarged
model family, not proof of a better physical model or a decrease of the
original unaugmented nonlinear objective. Raw costs across weights are not
ranked against each other.

Joint-QR future range-prediction RMSE (NOT trajectory RMSE or ATE):

| Data | Original | Corrected mean |
| --- | ---: | ---: |
| D1 | 0.126851 m | 0.132705 m |
| D4 | 0.148934 m | 0.157052 m |

In the mean+uncertainty arm, nominal 95% joint coverage is unchanged on D1
(94.02%) and rises on D4 from 91.49% to 92.20%, still below 95%. Coverage gain
does not imply a better mean. There are no negative predicted ranges.

### Time and Robot Variation

Joint-QR mean+uncertainty NLL change versus original:

| Data | 30 s window | 60 s window | 90 s window |
| --- | ---: | ---: | ---: |
| D1 | +0.221270 | -0.004018 | +0.003904 |
| D4 | +0.005639 | +0.082648 | -0.012091 |

Six of ten aggregate robot scores improve slightly or materially, but harms
to other robots dominate the dataset means. All robots/objects/windows are
retained in `summary.json`; no subgroup is selected as the new default.

- D1/R5, the previous problem case: NLL -0.987708 -> -0.990912, but range
  RMSE 0.194817 -> 0.194876 m. This is not a meaningful mean-error repair.
- D4/R2: NLL -1.360273 -> -1.168988; range RMSE 0.180765 -> 0.194398 m.
  The attempted correction makes the previous failure worse.
- D4/R1: range RMSE 0.133362 -> 0.207541 m. Average success in several other
  robots must not hide this new regression.

The estimated coefficients also depend strongly on accumulated data. For
joint-QR, D1/R5 estimates are +0.439, +0.365 and +0.00085 m at 30/60/90 s;
D4/R2 estimates are +0.0324, -0.0474 and -0.1638 m. These overlapping-prefix
fits are correlated, so this is descriptive variation, not a change-point
test. Do not interpret the coefficients as measured hardware biases.

## Approximation Check

The largest first-order pose translation update is 0.885 m, yaw update
0.149 rad, and object update 0.725 m. The original unaugmented GN correction
is at most 6.59e-7 in its mixed state coordinates, excluding incomplete baseline
optimization as the explanation for these large new changes.

Because a large step may invalidate linear prediction, a post-selected audit
evaluates the SAME step with nonlinear polar geometry and the pinned Pose2
right-composition chart. No reoptimization or new parameter is introduced;
the original response covariance is retained to isolate mean evaluation.
All twelve beta fits reproduce within 3.89e-16 and forecast identities match.

| Data / weights | Original NLL | Linear mean + uncertainty | Nonlinear mean, same covariance |
| --- | ---: | ---: | ---: |
| D1 nominal | -1.576927 | -1.500541 | -1.489966 |
| D1 joint-QR | -2.023980 | -1.960483 | -1.959804 |
| D4 nominal | -1.466104 | -1.468358 | -1.467731 |
| D4 joint-QR | -1.717643 | -1.691993 | -1.692087 |

Maximum mean discrepancy is 0.0318 m in range and 0.0145 rad in bearing. The
aggregate conclusion does not change. This excludes simple first-order
evaluation of this SAME step as a sufficient explanation of the aggregate
negative result. It does NOT exclude a different endpoint from full nonlinear
joint mean/state optimization or prove that every mean model is unsuitable.
The extension takes 89.42 s and is archived separately, not mixed with the
predeclared primary arms.

## Native Environment Recovery

Rebuilt the original GTSAM commit
`03c263fe08b02c90a8861d36f77422765d39506d` with GCC 11.5, system Eigen 3.3.7,
Boost 1.79, Release, no TBB and no march-native. No global package installed.
Sources and builds now live in ignored, persistent research `build/`:

- `build/research-deps/gtsam-imesa`
- `build/research-deps/gtsam-imesa-build`
- `build/native/bin/{native-joint-audit,native-joint-replay,...}`

`/tmp/dranppr_backend_build` is only a compatibility symlink to `build/native`
for existing tests. Clearing /tmp no longer deletes the build itself.
Runtime `ldd` confirms the pinned local GTSAM/Metis libraries and no TBB.
Native consumers retain the existing SSE-compatible flags, no `-march=native`.

Fresh native suite: **62 passed**. Schur full-system relative residual
3.25e-16; interface PCG fixture 6 iterations, residual 1.98e-16. The physical
2/3/5/10-robot fixtures take 12/16/23/39 PCG iterations, maximum recorded
interface relative residual 1.01e-13. MPI ranks: 0. These are unit fixtures,
not a new communication/accuracy benchmark.

Four central-LM replay controls, D1/D4 at 30 s with nominal/joint-QR weights,
reproduce archived final poses AND costs exactly (reported differences zero).
Original posthoc gradients remain between 1.55e-5 and 7.83e-5; matching a prior
endpoint is not a new claim of zero nonlinear gradient or global optimality.
Binary/library hashes and commands are in `native_recovery/verification.json`.

## Verification and Reproduction

- Six new tests cover direct augmented LS/covariance parity, rank failure,
  physical units/ordering, unchanged original forecasts, no future-observation
  leakage, and retracted composition. Tests were written and failed first.
- **91 Python tests passed, zero skipped**, including restored native export
  integration and earlier noise/attribution regressions.
- All twelve original forecasts reproduce means, covariances and NLL exactly.
- Source snapshots, worker commands, outcomes and JUnit XML are archived in
  `results/dranppr_temporal_mean_response_20261027`.
- Nonlinear mean evaluation is in
  `results/dranppr_temporal_mean_retraction_20261027`.
- No CCI/D-CCI, production backend, association or baseline code was changed.
  The research predictor gains only an optional diagnostic response argument;
  its default path retains exact archived numerical outputs.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 1200 \
  python3 -m experiments.audit_temporal_mean_response \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output results/dranppr_temporal_mean_response_reproduction
```

## Next Gate

Do not continue increasing the number of bias variables based on training
residual concentration. First extend the fixed past-only evaluation to the
whole declared causal suffix, retaining unsupported objects and every robot,
before interpreting persistence or selecting a correction model. Reserve
uninspected sequences for confirmation. Model-parameter uncertainty in Q/R,
nonstationarity, conditional geometry/object effects and a full nonlinear
mean-family control remain open alternatives, not established causes.

The restored native build removes an execution blocker. The new statistical
result removes the justification for promoting this simple one-step correction;
it does not close D2 high noise, outlier/association robustness, continuous
distributed adaptation or ObjectPose3 transfer.
