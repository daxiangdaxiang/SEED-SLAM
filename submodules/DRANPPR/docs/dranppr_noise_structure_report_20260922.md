# Noise-Structure Diagnosis After the Native Feedback Failure

Date: 2026-09-22. Research worktree only; no packaged DRAN++ changes.

## Questions and Controls

The preceding chronological audit found that factor-excluded calibration could
overestimate bearing precision and produce poor future predictions. This stage
tests three explanations in order, without using GT to choose parameters:

1. Does explicit nonlinear optimization reverse the covariance-candidate ranking?
2. Does deleting an entire observation frame, rather than one factor, reverse it?
3. Is fixing the odometry covariance, or only scaling it uniformly, part of the
   observation-noise attribution problem?

All development tests target MR.CLAM Dataset 4 Robot 3, selected because of the
previous failure. Thus this is not a held-out-sequence performance claim. Training
still ends at the original global stream time +30 seconds; future validation
ends at +120 seconds. A robot-specific selection must not reset that clock.

## Nonlinear and Frame Exclusion

The four previously fitted covariance candidates are held fixed. Their range
covariates remain identical nominal factor-excluded predictions. Odometry is
retained when an observation frame is removed. Each retained graph is optimized
explicitly from the same nominal warmup solution. Covariance is recomputed from
the Gauss-Newton model at that solution, not an exact nonlinear posterior.

The warmup has 73 observation factors, of which 70 are individually predictable.
Requiring entire frames to remain observable leaves 68 matched observations in
58 frames. Three omitted frames contain source rows `[16,17]`, `[18,19]`, and
`[50]`. The same 68 observations are used for every comparison below. No damping
is inserted to turn an unobservable deleted problem into a finite score.

| Candidate | Linear factor NLL | Nonlinear factor NLL | Linear frame marginal NLL | Nonlinear frame marginal NLL |
| --- | ---: | ---: | ---: | ---: |
| Fixed | -2.46252 | -2.46283 | -2.46939 | -2.46971 |
| Fitted residual | -3.35247 | -3.35320 | -3.30578 | -3.30624 |
| Profiled diagonal | -3.28429 | -3.28458 | -3.10445 | -3.10374 |
| Profiled conditional | -3.60578 | -3.60757 | -3.40049 | -3.40180 |

The table reports per-observation marginal NLL, preserving comparability across
deletion units. Full joint frame NLL, including cross-observation covariance, is
also retained in the artifact. It gives the same best candidate: conditional
profiling, at `-3.50460` per observation, versus residual fitting `-3.32973`.

All explicit solves terminate within six LM iterations; no 100-iteration cap is
reached. For conditional profiling, mean absolute linear/nonlinear frame-NLL
difference is `0.00735`, maximum `0.08478`. This is an empirical local-accuracy
check, not an exact stationarity or global-optimality certificate.

**Decision:** these two more expensive scoring changes do not repair the ranking
that failed on future data. Keep the negative result rather than implementing
per-candidate nonlinear solves in the online loop. Whole-frame exclusion removes
same-frame direct dependence but does not remove other temporally related views.

## Whole-Model Variance Diagnosis

Consider the fixed linearized model `y = A delta + epsilon`. Choose an orthonormal
contrast basis `C` satisfying `C^T A = 0`. Then

```text
z = C^T y = C^T epsilon,
S(q) = C^T diag(v_base + D q) C,
L(q) = (z^T S(q)^(-1) z + log det S(q) + k log(2*pi))/2.
```

This is classical error-contrast / restricted likelihood (REML), not a new
DRANPPR contribution. The role here is diagnostic: test the entire covariance
model after eliminating arbitrary *linear* state changes, rather than replacing
the native failure by another residual-dependent robust weight. The construction
is described by [Harville, 1974](https://academic.oup.com/biomet/article-abstract/61/2/383/268084).
Joint state/covariance learning is also established in robotics; see
[Khosoussi and Shames](https://arxiv.org/abs/2502.04584).

The observation family stays unchanged: one bearing variance and range variance
`a+b*d^2`. Three nested odometry models are compared:

- Fixed Q: keep the declared nominal odometry diffusion covariance.
- Scalar Q: learn one multiplier shared by its three local tangent coordinates.
- Diagonal Q: learn separate longitudinal, lateral, and angular multipliers.

These are estimated statistical parameters, not constants swept against ATE.
Nevertheless the larger family has a greater estimation burden and must pass
future validation. This diagnostic does not justify unrestricted per-factor
covariances, or establish that diagonal diffusion is the correct physical model.

The reference implements the analytic log-variance score and expected Fisher
information. Tests compare scores with finite differences, check invariance to
state shifts and the scalar residual-degrees-of-freedom formula, and reject
structurally indistinguishable variance components rather than silently fitting
an arbitrary decomposition. Bounds `[-16,10]` are numerical log-coordinate bounds
and all activations are reported. Full structural rank is not strong finite-data
identifiability; the latter is not certified by these tests.

| REML family | Odometry variance multipliers (x,y,yaw) | Bearing variance (rad^2) | Contrast NLL |
| --- | --- | ---: | ---: |
| Fixed Q | 1, 1, 1 | 2.813e-10, boundary | 91.9925 |
| Scalar Q | 0.5905, 0.5905, 0.5905 | 2.813e-10, boundary | 88.3926 |
| Diagonal Q | 0.5411, 0.0210, 1.0801 | 7.248e-7 | 84.2109 |

All have 120 residual contrast degrees of freedom. The range intercept remains
at the lower numerical bound; the positive distance-dependent term supplies range
variance. The diagonal model's fitted bearing standard deviation is only
`0.000851 rad`: leaving the numerical boundary does not mean the sensor has been
correctly calibrated. Training contrast NLL cannot be compared numerically with
observation predictive NLL, since they refer to different random variables.

**Supported inference:** odometry error *direction*, not merely its total scale,
affects which part of the discrepancy is attributed to observation noise in this
prefix. A scalar-Q repair does not remove the bearing collapse. The following
chronological replay tests whether the directional family improves future state
quality, without interpreting training improvement as validation.

## Chronological Feedback Result

All five new replay arms use the same raw stream and nominal 30-second warmup.
The two original controls (`fixed`, `profiled_conditional`) reconstruct every
original Robot 3 prediction exactly: maximum checked innovation/NLL/energy
difference is `0.0`. All arms score the same 504 future known-landmark observations
and evaluate the same 353 post-warmup pose states. The learned parameters are
frozen; only future odometry and observation factors receive the new covariance.

| Method | Future NLL | Mean energy | Final trajectory RMSE (m) | Causal trajectory RMSE (m) | Object RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | -2.33577 | 1.0831 | 0.20036 | 0.24656 | 0.18108 |
| Fitted residual, previous matched control | -2.87311 | 1.7691 | 0.17394 | 0.20187 | 0.18211 |
| Profiled conditional | 3.92385 | 15.6519 | 0.33099 | 0.29475 | 0.38836 |
| REML, fixed Q | 3.73583 | 15.2361 | 0.33332 | 0.29431 | 0.38931 |
| REML, scalar Q | 3.16763 | 14.6070 | 0.28818 | 0.24891 | 0.33482 |
| REML, diagonal Q | -2.84287 | 2.4149 | 0.07851 | 0.14373 | 0.12957 |

The ordinary residual row is retained from the prior canonical run, using its
Robot 3 records in `evaluation/robots.csv` and `innovation_diagnostics.csv`.
It has the same 504/353 event/pose counts and physical time interval. Its timing
is not mixed with the new timing run. It still has slightly better NLL than the
diagonal-Q arm; the new arm does not dominate every metric.

The diagonal-Q arm reduces final trajectory error by 60.8% versus fixed and
54.9% versus ordinary residual fitting. Causal trajectory reductions are 41.7%
and 28.8%, respectively. Object error is also lower. At source row 350, the
previous `-0.5010 m / 0.2289 rad` innovation becomes `0.0733 m / -0.00836 rad`;
predictive NLL changes from `689.74` to `-4.71`. The measurement is no longer
treated as extremely surprising under this learned model.

The main supported result is a native counterexample to an observation-only
noise repair: neither more accurate deletion nor changing to REML while fixing
Q helps, whereas changing the directional Q family and learning Q/R together
improves this case. It is NOT a proof that directional odometry mismatch is the
sole source of the original failure. Both Q and the resulting fitted R change;
Q-only/R-only controls are needed to separate their contributions. Linearization,
time dependence, bias, and other covariance shapes remain possible confounders.

The new calibration takes about `0.164 s` and state updates `1.393 s`, but dense
prediction auditing takes `71.536 s` for this one robot, with `22.715 s` simulated
serial backlog. This is not an online-latency success or a runtime comparison
against SOTA. The entire five-arm diagnostic was run with a 600-second limit and
completed normally. No new rejection gate, robust-loss weight, or tuned variance
floor was introduced.

## Next Gates

1. Use fixed learned parameters to isolate Q-only, R-only and joint feedback on
   the same causal stream. Compare physical state errors and the original failed
   event, not only each model's own weighted objective.
2. Keep the model family and fitting procedure fixed for the other robots and
   separate development sequences. Do not select covariance structure per dataset
   using ATE. Report any lost accuracy, boundary estimates and unsupported cases.
3. Replace the dense prediction audit by verified sparse/cached local uncertainty
   computation before broad full-history replays. It must retain pose/landmark
   cross covariance and agree numerically with the dense reference.
4. Implement persistent factors and interface state in the actual DRAN++ backend;
   the current Python iSAM2 reference is not that implementation. Compare official
   distributed robust baselines under the same causal inputs and resource budgets.

The research goal remains open. Classical REML and covariance anisotropy alone
are not a novelty claim. The intended contribution still requires a distributed
closed loop that responds differently to optimization error, measurement-model
error and independently supported identity error, without sacrificing online
performance or fixed-model accuracy.

## Independent Reference Check

This subsection is evaluation-only. On valid GT-supported raw Robot 3
observations in the same time intervals:

| Interval | Supported observations | Bearing mean (rad) | Centered bearing SD (rad) | Bearing RMSE (rad) |
| --- | ---: | ---: | ---: | ---: |
| 0--30 s | 66 | 0.01010 | 0.01721 | 0.01995 |
| 30--120 s | 505 | -0.00527 | 0.01239 | 0.01346 |

These sets are not the 68 deleted-frame observations or the 504 known-landmark
future predictions; support and first-observation restrictions differ. They must
not be pooled as one matched experiment. Raw errors include reference/model
error, not just stochastic sensor noise. Their changing signed mean motivates
checking temporal/view-dependent bias, without proving a particular bias model.

Source: `../indoor/evaluation/dranppr_native_audit_20260921/d4_r3_raw.npz`, SHA256
`18d70f3890406d78df2d5815021a1f8438429a66bdd4804d5f27f3a6c3ac6d85`.
Filter `valid` and absolute observation time against the first causal event time
plus `[0,30]` or `(30,120]`, then compute population moments of residual column 1.

## Artifacts

`results/dranppr_noise_structure_20260922/nonlinear/` contains every deletion,
prediction, marginal/joint score and omission; `contrast/` contains all fitted
components, gradients, Fisher diagnostics and comparisons with the original
four candidates. `feedback/` and `evaluation/` contain the complete five-arm
chronological predictions, states, timing, source hashes and paired evaluation.
No previous failure is removed from the record.

The replay helper now supports a calibration callback and robot selection while
preserving the global acquisition clock. Learned Q affects future factors only;
old factors are unchanged. Calibration work starts at the actual triggering
event, rather than backdating it to the nominal warmup boundary. The previous
producer and evaluator source versions are retained under
`results/dranppr_online_noise_20260922/source/experiments/` to preserve their hashes.

Reproduction uses `PYTHONPATH=/tmp/dranppr_python`, `OPENBLAS_NUM_THREADS=1`, and
`OMP_NUM_THREADS=1` from this worktree. Output paths must be empty.

```sh
python experiments/audit_nonlinear_noise.py \
  --run results/dranppr_online_noise_20260922/run \
  --stream /tmp/dranppr_causal_streams_v1/dataset4 --robot 3 \
  --output /tmp/reproduce_nonlinear_noise
python experiments/audit_contrast_noise.py \
  --run results/dranppr_online_noise_20260922/run \
  --stream /tmp/dranppr_causal_streams_v1/dataset4 --robots 3 \
  --output /tmp/reproduce_contrast_noise
python experiments/audit_joint_noise_feedback.py \
  --reference results/dranppr_online_noise_20260922/run \
  --stream /tmp/dranppr_causal_streams_v1/dataset4 --robot 3 \
  --output /tmp/reproduce_joint_feedback
python experiments/analyze_online_noise.py \
  --run /tmp/reproduce_joint_feedback \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw \
  --native-residuals ../indoor/evaluation/dranppr_native_audit_20260921 \
  --output /tmp/reproduce_joint_feedback_evaluation
```

Final verification: 103 focused research tests passed in 17.12 seconds; Python
compilation and `git diff --check` passed. All 24 checked producer/evaluator
source hashes and the evaluation input-manifest link resolved to the current
source or its explicitly retained prior version. No experiment process remained
running. The packaged DBA-DRANPP worktree remained clean. No C++/D-CCI code was
changed or rebuilt in this diagnostic stage.
