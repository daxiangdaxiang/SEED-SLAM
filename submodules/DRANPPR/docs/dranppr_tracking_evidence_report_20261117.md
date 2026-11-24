# Tracking and Noise-Evidence Diagnosis

Date: 2026-11-17, follow-up to the sparse calibration/SE3 round. Research
DRANPPR only; no production solver, noise policy, gate, default, commit or push
is changed. The two interventions below diagnose different failure sources.

## Main Conclusions

1. D9 has a finite-work state-tracking component, but more iterations do not
   explain or repair every error. At 210 s, completing the same arrived graph
   reduces calibrated latest-five-pose fixed-frame RMSE from 1.079113 to
   .759587 m. At 150 s, however, completed calibrated whole-trajectory RMSE is
   .719266 m versus completed fixed-noise .502692 m. Iteration shortage alone
   is not an adequate account of the observed calibrated regression.
2. The missing D9 correction is largely collective motion, not a large change
   in map shape. At 210/270 s, a single rigid fit explains 99.886%/99.904% of
   calibrated completion displacement energy. This motivates investigating
   slow collective directions of the physical system. It does NOT establish
   a particular Hessian spectrum, authorize a free gauge shift, or implement
   a coarse solver. The root remains exactly fixed.
3. Positive variance estimates can reach the numerical lower bound even in
   a correctly specified, exact-linear Gaussian model on the physical SE3
   design: 7/40 heterogeneous-noise fits do so. With four independent training
   observations per original factor and the SAME test observations, the count
   becomes 0/40. Small-variance relative estimation RMSE falls 104.19% -> 11.52%.
   Finite evidence is a sufficient mechanism for such boundary behavior;
   it is not proven the sole cause of the previous nonlinear examples.
4. Correct nominal weights still benefit from not having to estimate them.
   Mean independent-test prediction risk rises 2.216% after 1x calibration,
   versus .160% after 4x calibration. More independent information reduces the
   estimation penalty; the experiment does not justify a tuned waiting time,
   hard variance floor or a universal no-regression guarantee.

## D9: Identical Arrived Graph, More Complete Solve

Source: sealed `results/dranppr_sparse_d9_online_20261117`. The warm prefix
ends at 131.723 s. Checkpoints 150/210/270 s were declared before this round's
results. At each checkpoint, fixed and calibrated native replays use the same
events, old weights, lazy refresh and one interface-PCG iteration per frame as
the source. Each truncated causal history matches its original prefix exactly.

Freeze the arrived graph and run existing central LM100 and GN100 from the
corresponding online endpoint. No future observations, GT, refit or changed
covariance enters these solves. All 18 processes finish under the unchanged
180 s per-process cap. This is a completion diagnostic, NOT an equal-work,
equal-communication or online competitor. GT evaluation starts after completion.

LM/GN maximum position difference is 3.55e-6 m and maximum yaw difference
1.16e-6 rad. Root displacement and shared-copy disagreement are exactly zero.
Maximum LM posthoc gradient norm is 8.70e-4. Agreement is evidence of similar
local endpoints, not a global optimality certificate or proof that all local
minima are excluded.

Each row's first pair is whole-graph trajectory RMSE after one global network
registration; the second is object-copy RMSE in that frame. The final pair is
the latest pose of each of five robots using the SAME fixed prefix registration
for online and completed states. Latest-five errors are not the previously
reported full first-published trajectory RMSE and must not replace it.

| Time / weights | Trajectory online -> LM (m) | Objects online -> LM (m) | Latest five, prefix frame, online -> LM (m) |
| --- | ---: | ---: | ---: |
| 150 / fixed | .504420 -> .502692 | 1.156900 -> 1.167044 | 1.045750 -> .956113 |
| 150 / calibrated | .698619 -> .719266 | 1.304523 -> 1.402837 | 1.059550 -> 1.007663 |
| 210 / fixed | .439652 -> .440073 | .399543 -> .398001 | 1.047848 -> .887133 |
| 210 / calibrated | .371166 -> .373643 | .358897 -> .362219 | 1.079113 -> .759587 |
| 270 / fixed | .400912 -> .401320 | .237684 -> .238153 | .688949 -> .596480 |
| 270 / calibrated | .383005 -> .383510 | .193262 -> .193431 | .754474 -> .602283 |

The calibrated physical cost drops .4441%, .06981% and .08063% at the three
checkpoints. Thus near-final objective value is not sufficient evidence that
the current state in a fixed external frame has stabilized. Conversely,
completing the objective can increase truth error, particularly at 150 s.
This leaves information/model/solution-branch limitations as well as work
limits; the experiment does not uniquely identify an erroneous sensor model.

### Collective Versus Shape Change

Posthoc, without GT, fit ONE SE(2) transform from all online pose positions to
the completed pose positions. Do not align robots separately and do not apply
the transform to any estimator output. This includes all internal poses,
including those not scored by GT. It describes displacement, not accuracy.

| Calibrated checkpoint | Raw position-change RMS (m) | Residual after rigid fit (m) | Rigid-explained energy |
| --- | ---: | ---: | ---: |
| 150 | .523751 | .096233 | 96.624% |
| 210 | .737800 | .024903 | 99.886% |
| 270 | .949494 | .029360 | 99.904% |

The fitted angles at 210/270 s are -.09745/-.13664 rad. A small nonrigid
remainder is necessary because the root is fixed; the whole graph cannot be
arbitrarily transformed while keeping that constraint. The finding suggests
slow near-rigid collective response under the current finite budget. Calling
it a proven low-eigenvalue mode would require a separate spectral/model audit.
The small changes in freely registered trajectory RMSE can hide much larger
fixed-frame corrections. Neither evaluation convention should be silently
substituted for the other.

A plausible connection to the causal/terminal gap is that these bulk offsets
change over time: published poses retain their historical errors, whereas
terminal smoothing revises the whole trajectory together. One final rigid
registration cannot generally remove time-varying offsets. This interpretation
is consistent with the measured corrections, not an isolated proof of the
complete causal-error mechanism.

## SE3: Exact-Linear Evidence Control

Source: the archived nonlinear seed-35 heterogeneous case's physical Jacobian
and four group labels, with 708 measurement coordinates and 162 free state
coordinates. This is a post-selected design based on a real six-DoF pose/object
factor construction, NOT a newly drawn random design and NOT another nonlinear
SE3 deployment test. Its original residual values are not used here.

Generate `y = A * 0 + epsilon`, with independent Gaussian epsilon and known
group variances. The zero generating increment is only a simulation reference;
the fit still eliminates all 162 unknown state coordinates. Nominal variances
are [1,1,1,1]; heterogeneous variances are [.1,4,9,.25]. Use fixed seeds
401--440 for both regimes and 1x/4x training information: 160 total fits.
Bounds, likelihood, zero initial log scales and stopping tolerances are unchanged.

In 4x, stack the same A four times but generate independent noise for every
repeat. Do NOT copy a noisy measurement four times. All repetitions share the
same state columns. The 1x training vector is the prefix of the 4x training
vector; the independent test vector and test design remain identical between
the two conditions. Training and test RNG streams are separate. Hence any
test-risk difference comes from learning weights, not extra test observations.

All 160 fits succeed. All fitted NLLs and log-scale scores are independently
checked using dense thin QR; maximum discrepancies are 1.57e-11 and 1.83e-11.
This excludes disagreement between the two numerical implementations at the
evaluated points; it is not a general floating-point or statistical guarantee.

| Regime / training repeats | Fits | Boundary fits | Odometry-rotation relative scale RMSE | Mean fitted prediction risk | Known-scale risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct nominal / 1x | 40/40 | 0 | 33.69% | .229218 | .224249 |
| Correct nominal / 4x | 40/40 | 0 | 10.26% | .224608 | .224249 |
| Heterogeneous / 1x | 40/40 | 7 | 104.19% | .231959 | .223270 |
| Heterogeneous / 4x | 40/40 | 0 | 11.52% | .223533 | .223270 |

Prediction risk is `mean((A delta_hat)^2 / V_true)` for a weighted state solve
on an independent test vector. It measures error in predicted means, not the
fitted residual, ATE or physical meters. True variances are used only for this
explicit oracle-generative evaluation. Fixed nominal risk under heterogeneous
noise is 2.010064; learned weights improve it in all 40 paired draws at both
training sizes, despite seven small-variance boundary fits at 1x.

The generating odometry rotation scale is .1. Heterogeneous estimates have
mean/median .10416/.07124 at 1x, and .10017/.09901 at 4x. A near-correct mean
does not imply reliable individual estimates. Fisher-based local log-scale
standard error at the generating law changes .98652 -> .09997. This is an
oracle design diagnostic, not a confidence interval valid at the fitted boundary.
Other variance groups have much stronger evidence already at 1x.

The experiment establishes that correct family, full rank and successful
optimization do not ensure a precise small-variance estimate. Real repeated
observations can be correlated or biased, so four times the number of rows is
not generally four times the information. No choice of repeat count is proposed
as an online tuning rule. The two original nonlinear boundary cases may also
have fixed-linearization effects; this control does not rule those out.

### Why Correct-Noise Controls Need Not Improve

For fixed full-rank A, zero-mean test noise with positive-definite covariance V,
and weights estimated independently of that noise, let
`L* = (A' V^-1 A)^-1 A' V^-1` and `Lhat A = I`. Set `D = Lhat-L*`.
Then `D A = 0`, so `L* V D' = (A' V^-1 A)^-1 A' D' = 0`. Conditioning on
training weights gives

`Cov(Lhat y | training) = Cov(L* y) + D V D'`.

The extra term is positive semidefinite; averaging over training preserves
this inequality when second moments exist. This is the standard GLS/Gauss-Markov
argument, not a new research contribution. Background:
[CMU lecture on generalized least squares](https://www.stat.cmu.edu/~cshalizi/dst/20/lectures/26/lecture-26.html).

Thus independently learning weights cannot improve expected squared state
error over already-known correct weights within this unbiased linear class.
Individual draws may improve. If learned weights induce the same L*, the
penalty is zero. This explains why our correct-nominal control is an oracle
comparison, not evidence that adaptation must fail under unknown noise.
The identity does not cover biased estimators, same-data weight fitting,
nonlinear SLAM, a changing design or arbitrary contamination.

## Decision and Next Algorithmic Target

- Do not promote the frozen calibrated path as an online default. Existing
  D1/D9 first-published regressions and clean-noise SE3 regressions remain.
- Separate two workstreams rather than adding more switches: improve physical
  state response along collective directions; treat fitted noise as finite
  evidence rather than known precision. Re-solving old data improves numerical
  optimization, not the amount of statistical evidence.
- A focused solver candidate is a small collective-motion subspace derived
  from the same physical quadratic model and exact root constraint. Its
  geometric dimension is three in SE(2), six in SE(3), rather than a chosen
  anchor count. This is a DESIGN TARGET, not an implemented or validated method.
  Audit degeneracy, descent, byte cost and interaction with private-variable
  elimination before claiming a benefit. Do not apply the posthoc fitted
  transform as an estimator correction.
- Test this solver direction first with frozen weights to avoid mixing a new
  optimizer with a new noise learner. Equal-byte/work online comparisons and
  full 3D native interfaces still have to be implemented and verified.
- Do not add a variance floor, a confidence gate, or a new Bayesian module merely
  because the point estimate reached a bound. The older Fisher/cubature study
  already failed to repair selected real-data prediction errors. A new noise
  formulation must improve independent prediction under both correct and
  mismatched conditions, with parameter uncertainty and dependence disclosed.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_checkpoint_tracking run --source results/dranppr_sparse_d9_online_20261117 --output NEW_TRACKING --dataset 9 --checkpoints 150 210 270
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_checkpoint_tracking evaluate --output NEW_TRACKING --raw /home/jackal/zyx/dpgo_ICRA/datasets/MRCLAM/raw
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_checkpoint_tracking geometry --output NEW_TRACKING
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_pose3_evidence run --source results/dranppr_pose3_calibration_20261117 --output NEW_EVIDENCE
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_pose3_evidence verify --source NEW_EVIDENCE --output NEW_REVIEW/independent_qr.json
```

Actual outputs: `results/dranppr_d9_tracking_diagnosis_20261117`,
`results/dranppr_pose3_evidence_20261117`, and
`results/dranppr_tracking_evidence_review_20261117`. The previous production,
calibration, online-baseline and nonlinear-SE3 archives are unchanged. iMESA
and riMESA were NOT rerun this round because no estimator changed; their fresh
six-arm comparison remains in the preceding report, not a new result here.

Verification: 237 Python tests across the recorded 26-file selection pass;
all four native build targets succeed and 62 C++ tests pass. Exact commands
are in the review archive's `verification_commands.txt`; both test suites
have JUnit XML. Schur full residual is 3.24647e-16 and matrix-free product error
3.53484e-17. Routed PCG uses six iterations at residual 1.97875e-16;
2/3/5/10-robot controls use 12/16/23/39 iterations. MPI ranks: zero, not run.
CCI/D-CCI and production files are unchanged. Producer SHA256 ledgers and
final review snapshots distinguish executed source from later analysis additions.
