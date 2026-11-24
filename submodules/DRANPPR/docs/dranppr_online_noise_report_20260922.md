# Native Chronological Noise Feedback Audit

Date: 2026-09-22. Status: negative research result; not a new DRAN++ default.

## Purpose

Test whether past-only, factor-excluded covariance learning improves future
prediction and physical state estimation on native observations. The prior
linear-Gaussian experiments establish a mechanism, not this native-data claim.
The second question is whether a failure comes only from incomplete incremental
optimization, or also persists after additional work on the same physical graph.

## Setting

- MR.CLAM Dataset 4, original causal packets, first 120 seconds, all five robots.
- Calibration uses only 0--30 seconds. Parameters are frozen before subsequent
  frames; evaluation uses 30--120 seconds. The run consumes 4,036 raw packets.
- Independent local Pose2/Point2 iSAM2 references, with timestamp-atomic updates.
  This is NOT the distributed DRANPPR optimizer. No inter-robot communication or
  identity edits are introduced. The packaged DBA-DRANPP tree is unchanged.
- Four arms receive identical data and nominal warmup. Calibration first refines
  the warmup graph with nonlinear LM. `fitted_residual` fits its ordinary
  residuals; `profiled_diagonal` fits separate constant bearing/range variances;
  `profiled_conditional` fits bearing variance and range variance `a+b*d^2`.
- Both profiled arms recompute the complete linear state response for each
  covariance proposal. Block PRESS predictions match explicit weighted deletion
  in deterministic tests. This remains a fixed-linearization approximation,
  whose linearization was obtained using the warmup observations.
- Calibration range covariates come from nominal factor-excluded predictions.
  Future covariates use the pre-assimilation predicted range, never the measured
  target range. Covariates during calibration are fixed, not jointly optimized.
- Previously inserted factors retain their covariance. First observations of an
  unseen landmark use nominal covariance and are not scored prequentially.
- Each known-landmark observation is scored before its entire frame is inserted.
  Predictions retain the full parent-pose/landmark cross covariance and propagate
  the incoming odometry uncertainty. All arms have the same 1,751 scored events.
- Ground truth is imported only by the post-run evaluator. Trajectories are
  aligned independently per robot/arm over the post-warmup interval (1,302 pose
  states). Object errors use those alignments. These are local accuracy metrics,
  not global multi-robot alignment accuracy.
- Odometry diffusion remains the declared nominal model, not an independently
  verified sensor calibration. Observation-noise fitting can confound its error.
- Powell uses the same log-coordinate numerical bounds `[-16,10]` for all robots.
  Bound activation is reported, not hidden by a per-dataset variance floor.
  The first L-BFGS-B attempt failed with line-search termination near a vanishing
  bearing variance; that attempt produced no complete result and is not averaged
  into the four-arm comparison. Powell is a numerical repair, not proof of a
  globally optimal covariance fit.

## Results

All NLLs include the log determinant and Gaussian normalization. Lower is better;
negative values are valid for continuous densities. The energy is the squared
whitened two-dimensional innovation, whose expectation would be 2 under the
correct independent Gaussian predictive model.

| Method | Future NLL | Mean energy | Final trajectory RMSE (m) | Final object-copy RMSE (m) | Causal trajectory RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed nominal | -2.4340 | 0.8006 | 0.22275 | 0.92346 | 0.22595 |
| Fitted residual | -3.0601 | 1.5320 | 0.22253 | 0.93068 | 0.21342 |
| Profiled diagonal | -1.1536 | 5.2451 | 0.24305 | 0.83461 | 0.22438 |
| Profiled conditional | -1.2332 | 5.4063 | 0.25280 | 0.84707 | 0.22599 |

The profiled conditional arm improves object-copy error but worsens final
trajectory RMSE by 13.5% versus fixed noise, with worse future likelihood. Its
causal trajectory RMSE is essentially unchanged. This is not an overall win.
The ordinary residual control has better predictive likelihood and causal
trajectory error here; it is not legitimate to omit it because the controlled
bad-initialization experiment previously favored predictive calibration.

The failure is concentrated, not uniform. Conditional profiling improves future
NLL for four robots but worsens Robot 3 from `-2.3358` to `3.9239`, with mean
energy `15.6519`. All observations, including this robot, remain in aggregation.
Both profiled arms put bearing variance near `2.814e-10 rad^2` for Robots 1, 2,
and 3. All five conditional fits have a near-zero range intercept; an intercept
boundary alone is not evidence of failure when the distance-dependent term is
positive. The bearing collapse is a separate and consequential observation.

### A Physical Counterexample

Robot 3, immutable source row 350, subject 14, elapsed time 67.808 seconds:

| Quantity | Range (m) | Bearing (rad) |
| --- | ---: | ---: |
| Conditional-arm innovation | -0.50101 | 0.22885 |
| Fixed-arm innovation | 0.05884 | -0.05055 |
| GT-assisted raw observation error | -0.01017 | -0.01627 |

The conditional predictor is much less accurate than the actual observation at
this event. Its NLL is `689.74`, whereas the fixed arm's is `-2.62`. Treating the
large innovation as a bad measurement or wrong identity would address the wrong
cause. The independent reference is post-hoc evidence, not information available
to an online algorithm. Reference interpolation/model error still exists; these
numbers do not establish perfect truth or a general fault classifier.

`evaluation/failure_examples.csv` retains the ten largest conditional-vs-fixed
NLL degradations and joins the GT-assisted residuals by immutable raw row ID.
This is explicitly post-selected failure analysis, not an unbiased accuracy set.

### Same-Graph Solver Interventions

First, reconstruct all four final online graphs exactly (maximum XY state
reconstruction error `0.0`), freeze every covariance, and run nonlinear LM.
Some optimization errors are substantial: Robot 1 conditional weighted cost
falls from `265067.80` to `128.28`. Nevertheless aggregate final trajectory RMSE
stays `0.25280 -> 0.25282 m`; fixed stays `0.22275 -> 0.22274 m`. Extra final
optimization therefore does not recover the accuracy gap in this intervention.
Costs under different covariance models are not compared as a common objective.

Second, return to the actual failing frame and refine only the graph available
BEFORE that frame. Neither row 350 nor any observation in its timestamp frame is
inserted. No covariance is changed, and no future data or truth is used.

| Robot 3 conditional, before row 350 | Online state | Past-only refined state |
| --- | ---: | ---: |
| Past weighted cost | 38539.36 | 1481.61 |
| Gradient norm | 1.556e7 | 0.0855 |
| Row 350 range innovation (m) | -0.50101 | -0.41260 |
| Row 350 bearing innovation (rad) | 0.22885 | 0.20068 |
| Row 350 NLL | 689.74 | 424.63 |

This took 90 LM iterations and approximately 0.67 seconds on this prefix. The
fixed arm changes negligibly with the same intervention. Finite optimization
error is real, but one much more accurate solve of the frozen learned objective
still yields a poor prediction. This does NOT prove global optimality, nor rule
out a different earlier optimization path or a nonlinear calibration repair.

A separate final-graph restart from the fixed arm's states is also preserved.
The conditional Robot 3 solve stops at cost `80727.02` with gradient `2.953e7`,
worse than the online-initialized batch solve. It is a failed numerical restart,
not evidence that all initializations lead to the same wrong optimum.

### Computation

| Method | iSAM2 updates (s) | Dense prediction audit (s) | Calibration (s) | Maximum serial backlog (s) |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 3.68 | 101.27 | 0.00 | 43.74 |
| Fitted residual | 3.38 | 100.15 | 0.19 | 42.77 |
| Profiled diagonal | 3.58 | 100.63 | 3.21 | 43.31 |
| Profiled conditional | 3.43 | 100.58 | 7.51 | 43.27 |

The dense QR covariance audit dominates. It is not a viable per-event online
backend, despite inexpensive state updates. All this work is included in serial
backlog. These are single-host diagnostic timings, not an isolated latency
benchmark or a parallel/distributed execution claim.

## Interpretation and Research Decision

1. Recomputing the lower response is necessary to avoid the earlier frozen-
   response error in the scalar pilot, but is NOT sufficient for valid native
   noise learning. The chronological test rejects that stronger inference.
2. A small excluded innovation can reflect predictable shared error rather than
   physically accurate sensing. Excluding one factor removes its direct linear
   influence but does not remove nearby correlated observations, state/model
   bias, or dependence through the calibration linearization.
3. The nominal arm's same-identity successive innovations are correlated:
   range correlations `0.468--0.724`, bearing `0.237--0.497`, at median gaps
   `0.237--0.247 s`. These are descriptive statistics, not proof that stochastic
   sensor noise is the sole source. Geometry, shared state, odometry error and
   bias remain possible explanations. No iid significance is claimed.
4. Noise feedback also changes conditioning and the amount of nonlinear work
   required. Learning weights and assuming unchanged incremental accuracy is
   unsafe. Conversely, a stationarity check alone cannot certify correct sensing
   or identity. The checkpoint intervention demonstrates both issues together.
5. Do not add a tuned minimum variance, robust gate or per-sequence threshold to
   turn this failed test positive. Keep fixed noise as the native reference and
   pause broad adaptive-feedback rollout until the model diagnosis passes.

The next bounded test should reuse the existing nominal/residual/profiled
covariance candidates on Robot 3's warmup and recompute their scores with
nonlinear exclusion of complete physical sensor frames. This changes the
linearization/exclusion approximation, not hyperparameters. Compare ranking and
predicted uncertainty against the linear audit before fitting anything new.
Frame exclusion alone is not a guarantee of temporal independence. If ranking
is unchanged, test the fixed odometry model and persistent signed errors using
temporally/multiview-separated evidence, with identifiability explicitly checked.
Only then consider a parsimonious correlated-error component. A sparse cached
prediction implementation is a separate requirement, not a statistical fix.

The hoped-for qualitative contribution remains evidence-based action selection:
more solve work for optimization error, a shared noise-model change for supported
precision changes, and DBA edits for independently supported identity errors.
This audit supplies a real counterexample and a falsified shortcut, not that
completed contribution. Covariance learning itself is already addressed by
[Joint State and Noise Covariance Estimation](https://arxiv.org/abs/2502.04584).
Robust incremental distributed estimation is already the setting of
[riMESA](https://arxiv.org/abs/2603.01178). No superiority to either is established.

## Artifacts and Reproduction

Canonical root: `results/dranppr_online_noise_20260922/`.
`run/` stores every prediction, causal pose, final estimate and calibration;
`evaluation/` stores paired metrics, robot diagnostics and raw-reference examples;
`lower/`, `lower_fixedinit/`, and `checkpoint/` store the solver interventions.
Each producer records source/event hashes and evaluation records the input
manifest and independent-reference hashes. Results are ignored by git.

Run from the DRANPPR worktree with `PYTHONPATH=/tmp/dranppr_python`,
`OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`. Output paths must be empty.

```sh
python experiments/audit_online_noise.py \
  --stream /tmp/dranppr_causal_streams_v1/dataset4 \
  --output /tmp/reproduce_online_noise --warmup 30 --duration 120
python experiments/audit_online_lower.py \
  --run /tmp/reproduce_online_noise \
  --stream /tmp/dranppr_causal_streams_v1/dataset4 \
  --output /tmp/reproduce_online_lower
python experiments/analyze_online_noise.py \
  --run /tmp/reproduce_online_noise --batch-run /tmp/reproduce_online_lower \
  --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw \
  --native-residuals ../indoor/evaluation/dranppr_native_audit_20260921 \
  --output /tmp/reproduce_online_evaluation
python experiments/audit_online_checkpoint.py \
  --run /tmp/reproduce_online_noise \
  --stream /tmp/dranppr_causal_streams_v1/dataset4 \
  --robot 3 --source-row 350 --output /tmp/reproduce_online_checkpoint
```

Tests cover exact deleted-linear likelihood, gauge/state-shift invariance,
unobservable blocks, covariance propagation, no target-range weighting,
future-data independence of calibration, matched events, immutable GT joins,
fixed-graph reconstruction, exclusion of whole target frames, and non-mutating
checkpoint prediction. No C++ solver or D-CCI behavior was changed in this stage.

Final verification: all 91 focused research tests passed in 17.12 seconds with
third-party pytest plugin autoload disabled. Python compilation passed for the
six producer/evaluator/model files changed in this stage. All 17 recorded source
and manifest-chain checks passed against the canonical artifacts. The packaged
DBA-DRANPP git worktree remained clean. These checks establish implementation
consistency, not acceptance of the statistical research hypothesis.
