# DRANPPR Blind-Spot Register

## Scope And Status

2026-11-23 [subspace/protocol study](dranppr_subspace_milestone_report_20261123.md):
the coarse-motion hypothesis does not yield consistent online accuracy: only
3/10 paired conditions improve the existing causal RMSE. Fixed-prefix-frame
metrics improve in 6/10, with a large calibrated D3 regression. Do not promote
it or switch evaluation conventions to hide failures. A separate exact one-step
protocol removes redundant final vector exchanges, saving 22.78--27.13% traffic
with byte-identical causal and terminal states in all ten conditions. This is
a communication milestone, not noise-robustness or meaningful wall-time progress.
3D tests cover geometry and local physical quadratic algebra, not online 3D.

2026-11-17 second follow-up:
[tracking/evidence diagnosis](dranppr_tracking_evidence_report_20261117.md)
completes the same arrived D9 graphs at 150/210/270 s. Calibrated latest-five
fixed-frame error improves 1.079113 -> .759587 m at 210 s, but completed
calibrated whole-trajectory error remains worse than fixed at 150 s. The
completion displacement is mostly collective near-rigid motion (99.886% and
99.904% of energy at 210/270 s), not proof of a particular weak eigenmode or
permission to move the fixed root. Exact-linear Gaussian controls on a physical
SE3 design produce 7/40 small-variance boundary estimates with 1x independent
training data and 0/40 with 4x, keeping test observations unchanged. All 160 fits
and independent dense-QR checks pass. Weak finite evidence is sufficient for
boundary behavior even under the correct model; full 3D online and no-regression
calibration remain open. No new estimator/default is promoted.

2026-11-17 calibration/3D follow-up: the
[sparse square-root and SE3 report](dranppr_sparse_calibration_3d_report_20261117.md)
establishes numerical equivalence on D2/D3/D9 and completes the calibration
commands in 1.873/30.367/66.929 s. The old D9 timeout occurred while exporting
dense contrasts AFTER a fit JSON was written, not proof of failed REML
convergence. The first sparse normal-equation attempt failed D3 line search;
square-root elimination fixes the reproduced numerical weakness without new
tolerances. Controlled true-SE3 graphs show large noise-mismatch gains but
2.56%/3.05% mean translation regressions when nominal noise is correct, plus
two boundary variance estimates. This tests the calibration principle in 3D,
not continuous adaptation, outlier resistance, real 3D or distributed online
ObjectPose3. Full-rank noise information is not finite-sample reliability.
Fresh D9 replay now improves final trajectory/object RMSE .401393/.230121 ->
.386044/.188604 m but worsens first-published RMSE .507303 -> .608974 m.
All six native/iMESA/riMESA arms finish. This adds a second causal regression
after D1, not closure of the finite-work or online-safety risks.

2026-10-29 baseline follow-up: the [fresh persistent-online comparison](dranppr_online_calibration_baselines_report_20261029.md)
uses our fixed-noise native baseline, frozen calibrated native candidate,
paper-iMESA and pinned numerically repaired riMESA on matched D1/D4 streams.
All 12 declared runs complete. Candidate wins final trajectory/object errors,
but D1 causal RMSE worsens 0.251488 -> 0.492764 m; D4 improves
0.271750 -> 0.201437 m. Final smoothing is not online localization. Six archived
state files reproduce exactly; 187 Python / 62 C++ tests pass. The comparison
excludes calibration overhead and does not implement continuous distributed
scale learning. A separate six-case, 60-solve expected-Fisher/cubature diagnostic
propagates fitted-scale uncertainty but does not repair any of the three selected
failures relative to nominal. This narrows, but does not close, uncertainty risk.

2026-10-28 coverage follow-up: the [complete benchmark-suffix audit](dranppr_rolling_prediction_report_20261028.md)
finishes 40 disjoint windows / 80 prediction arms over the original 120-second
D1/D4 benchmark, not the full raw missions. It scores 3282/3608 observations and
explicitly accounts for the other 326. Average NLL gains remain, but adapted
95% coverage falls to 91.65%/87.27%. D4/R2's full-suffix mean score improves;
D4/R3 is a newly exposed failure, and D1's 50--55 s window fails badly. Thus
the old three-window subgroup conclusions must not be treated as permanent
robot diagnoses. Twelve old forecasts reproduce exactly; independent input,
weight and score audits pass, with 118 Python / 62 C++ tests. This closes the
selected-window coverage gap ONLY within the declared benchmark, not statistical
reliability, full-mission coverage or robust-online deployment.

2026-10-27 temporal follow-up: the [past-only mean-response experiment](dranppr_temporal_mean_response_report_20261027.md)
does not support promoting a constant robot range-mean correction. Both adapted
dataset means worsen, including full linear mean/state uncertainty; nonlinear
evaluation of the same step leaves the aggregate result unchanged. Full
nonlinear optimization remains untested. The pinned native build has now been
restored under research `build/`: 62 C++ / 91 Python tests pass and four archived
central-LM endpoints reproduce exactly. The earlier temporary-build blocker
below is historical, not an outstanding deployment fix.

2026-10-27 follow-up: the [fixed-chart mean evidence audit](dranppr_mean_evidence_report_20261027.md)
checks mean/variance ambiguity on the same D1/D4 graph geometry under explicitly
known Gaussian laws. Robot offset rank can be full while evidence is weak;
robot-plus-object offsets introduce exact ambiguities; blind mean expansion
hurts zero-mean prediction. This is a controlled linear result, NOT closure of
the nonlinear, real-data, parameter-uncertainty or online risks below. The native
temporary build is now missing and must be restored before C++ revalidation.

This register covers currently identified risks in the adaptive-noise research
line. It does NOT assert that all possible blind spots have been eliminated.
The new evidence is detailed in the
[range attribution report](dranppr_range_prediction_attribution_report_20261005.md).
The numerical audit and frozen-weight factorial are complete; a complete robust
distributed online algorithm and its final validation are not.

Status meanings: **Checked** is a verified invariant on the audited inputs;
**Diagnosed** identifies an observed failure mechanism or limitation;
**Open** requires evidence not supplied by the present protocol. Passing a
software test does not convert an open statistical assumption into a theorem.

## Checked Invariants And Numerical Alternatives

| Risk | Evidence | Status and boundary |
| --- | --- | --- |
| Different measurements or identities between methods | Exact forecast-key sets; 533 source-event reconstructions; unchanged barcode labels and topology | Checked for known-object support, not association robustness |
| Weight intervention also changed initialization or data | Direct equality checks on all 48 saved input packages; unchanged prefix sigmas; declared suffix variance ratios agree within 6.66e-16 | Checked independently of the input-weighting helper |
| Incorrect range units or angle wrapping in scoring | Both Gaussian conditioning orders, scalar range identity, modulo-2pi reconstruction; error <=5.55e-17 | Checked in archived scoring path; not physical sensor calibration |
| Future observation leakage into predictions | Checkpoint pose timestamps, event availability/support, inherited odometry-only predictor; no GT values used | Checked protocol. The three windows were previously inspected and are not fresh final test data |
| Numerical covariance failure | SPD/PSD decomposition checks; no added damping/floor; maximum QR residual 2.23e-15 | Checked linear algebra, not correct uncertainty modelling |
| LM not converged enough to explain failure | 48 same-initialization GN controls: max position difference 1.74e-6 m; objective difference 8.42e-11 | Substantial LM/GN disagreement excluded here; common local minima remain possible |
| Stale binary/library or changed endpoints | Source hashes, binary hash, 11 library hashes; 12 original/adapted endpoint comparisons exact | Checked for these runs |
| Counting object observations as independent motions | Prior 391-frame motion audit; current 386 scored frames; frame-equal score sensitivity | Frame-count artefact is not the sole cause. Temporal dependence remains |
| Ignoring covariance coupling in score attribution | Full 2D covariance; both chain-rule orders; range marginal kept distinct | Checked. Mean/covariance swaps are not causal estimator interventions |

## Diagnosed Failure Modes

| Question | What is now established | What must NOT be inferred |
| --- | --- | --- |
| Is the largest residual channel the source of comparative failure? | Bearing improves in all 10 robot groups; range losses reverse gains in the two focal cases | Large bearing error does not justify a yaw correction as the first repair |
| Is distance sensor R alone shrinking too much? | 87.34%/95.72% of range covariance contraction is in U for D1/R5,D4/R2 | U is not Q; it includes map/state and propagation covariance |
| Is covariance contraction intrinsically harmful? | D4/R2 old mean + new covariance improves; new mean reverses the gain | Universal covariance inflation is not established |
| Which weight group can produce D4/R2's failure? | Learned Q worsens its score under all four B/R combinations in actual solves | This does not identify the true physical process covariance or a stationary bias |
| Can one global range-weight switch fix both datasets? | Nominal R under learned Q/B improves both average scores but worsens 7/10 robots | A retrospectively better average is not a universally safer default |
| Is D1/R5's problem a uniform robot-wide offset? | Its pooled mean becomes closer to zero while RMSE worsens; 5 subject-7 observations account for 56.9% of positive damage | No online subject-removal or object-bias estimator is validated |
| Can a robot bias be identified from D4/R2? | All 21 forecasts concern one subject; mean error grows while scatter decreases | Robot, object, geometry, state and time effects are confounded |
| Does a robot's aggregate sign apply at all times? | D4/R2 improves after 90 s but degrades after 60 s; D4/R4 also changes sign | Stationarity or a constant correction is unproven |
| Does full calibration rank imply trustworthy noise? | Interior rank-five fits coexist with conditional failures | Structural identifiability inside a model is not correctness of that model |

## Open Statistical And Modelling Risks

1. **Noise-parameter uncertainty.** Current covariances condition on fitted
   scales as if known. Fisher spectra and block-deletion sensitivity show
   information and sensitivity, not a propagated predictive distribution over
   those scales. A controlled calibration-resampling or justified integration
   test is needed; arbitrary covariance inflation is not a substitute. The
   October 29 local Fisher/cubature diagnostic now supplies one such sensitivity
   test: failure scores soften but remain worse than nominal. It is neither an
   exact posterior nor a dependence/misspecification-robust uncertainty estimate.
2. **Nonlinear and misspecified uncertainty.** QR exactly solves the local
   Gaussian model, not the nonlinear posterior. LM/GN agreement concerns
   optimizer endpoints only. Independent nonlinear predictive checks under
   matched synthetic generative models and real held-out data are still needed.
3. **Mean error versus random variation.** Residual contrasts remove linear
   state adjustments, not unknown biases outside the assumed model or nonlinear
   compensation. They can also make some coherent discrepancies indistinguishable
   from state changes. No GT-free estimator has yet separated these causes in
   the problematic conditional groups.
4. **Temporal, object and cross-channel dependence.** Small within-group SD
   around a nonzero mean can represent common-mode error rather than many
   independent samples. Frame weighting does not estimate effective sample size.
   A correlation-aware generative test or truly independent repeated conditions
   is required before confidence claims or new random-effect states.
5. **Shared scales across heterogeneous robots.** One five-vector is shared;
   it may redistribute error rather than improve every robot. More per-robot
   parameters could be unidentifiable, especially with single-object support.
   The required evidence dimension must be checked before expanding the family.
6. **Gauge and weak geometry.** Algebraic rank and a fixed root remove formal
   null directions but do not guarantee strong range/pose observability under
   each future geometry. Finite numerical covariance is not evidence of correct
   physical uncertainty in weakly excited directions.
7. **Dependence on nominal prefix and modelling choices.** Prefix covariances
   stay nominal, calibration is on one fixed chart, scales are learned once,
   and the process law assumes variance proportional to duration. This is not
   literally parameter-free or automatically adaptive to future noise changes.
   No new sweep was used, but model choices still require validation.

## Open Coverage And Deployment Risks

1. **Unscored new object copies.** The rolling audit leaves 326/3608 observations
   unsupported: 318 concern globally known labels without a local checkpoint
   copy, eight are globally unseen. All are counted and omission is matched,
   but admission, failure handling and their likelihood remain unverified.
   Local-copy support is a protocol restriction, not global unobservability.
2. **Bounded and repeatedly inspected evaluation.** Rolling evaluation now
   covers every interval after warm-up up to 120 s and scores 3282/3608 events,
   versus 533 previously. Full raw missions are approximately 1491/1390 s;
   D1/D4 remain discovery datasets. Untouched data and longer horizons are
   needed before claiming generalization; correlated windows are not trials.
3. **High-noise and contaminated data.** The earlier D2 Gaussian-noise negative
   remains: joint Q/R plus known added variance had mean trajectory/object RMSE
   .36303/.30465 m versus fixed weights .32648/.25889 over three fixed seeds.
   Known injection variance is an oracle, not automatic change detection.
   The current D1/D4 experiment does not repair or retest that result.
4. **Wrong associations versus noisy correct measurements.** These experiments
   use declared barcode identities. They do not show robustness to false object
   matches, coherent aliases or adversarial loop closures. Variance adaptation
   and association correctness are different inference problems.
5. **True distributed online operation.** Current mechanism tests collect a
   frozen full graph and use centralized LM/QR. They do not demonstrate causal
   incremental scale updates, bounded memory, sparse-neighbor covariance
   computation, communication cost, delayed messages, or recovery from an online
   bad update. October 29 transfers frozen scales to persistent native updates,
   but D1's first-published accuracy regresses. Those updates do not make the
   calibration step continuously distributed or close the end-to-end gap.
6. **Point2 to ObjectPose3 transfer.** MRCLAM here is a planar point-object
   problem. These results do not verify the 3D object-pose backend under its
   rotation noise, manifold geometry, association ambiguity or online arrival.
7. **Fair baseline and final utility.** October 29 adds a fresh fixed-native,
   paper-iMESA and repaired-riMESA comparison, with one-sweep and causal byte
   ceilings. No additional generic robust-kernel/GNC sweep is made. Calibration
   overhead is excluded, and final-state improvement coexists with D1 causal
   regression. Raw costs under different weights are incomparable.
   Final comparisons need common streams, information assumptions, initialization,
   communication/compute budgets and failures, with physical state errors as
   well as predictive scores. Better NLL does not guarantee better ATE.

## October 30 Extension and Remaining Risks

See [expanded online report](dranppr_expanded_online_report_20261030.md).
These observations add to, rather than replace, the preceding negative results.

| Question | New evidence | Status and boundary |
| --- | --- | --- |
| Does frozen calibration transfer beyond D1/D4? | D2/D3 final trajectory improves 48.46%/23.22%, objects 28.55%/21.22%, causal trajectory 40.94%/34.30% against fixed-native; tested riMESA/iMESA arms also lose those accuracy metrics | Checked on original, previously inspected 120 s data; not blind or injected-noise validation |
| Are gains caused only by a favorable evaluation alignment? | Common-prefix causal RMSE also improves: D2 .343323 -> .232779, D3 .313575 -> .179802; all five robot groups improve | Checked under two global-frame conventions; no per-robot realignment |
| Does calibration scale with prefix size? | D3 fit takes 544.712 s; D9 fit times out at 600 s after prefix state solve succeeds | Diagnosed deployment failure; D9 candidate unavailable, no substitute. Sparse equivalent fit remains open |
| Does D1 need future data to correct its failing state? | Same t60 arrived graph, unchanged calibrated weights: trajectory .429735 -> .229520, objects .765820 -> .239738 after completion; LM/GN positions agree within 2.03e-7 m over all checkpoints | Finite-work tracking component demonstrated. Not all causal damage explained, no fair-budget fix yet |
| Are nominal and historical baselines reproduced? | Ten final/causal CSVs identical; five physical graph checks <=6.35e-16 relative error; October 29/30 binaries and linked libraries identical | Checked audited paths, not all possible runtime environments |
| Does cleanup remove inconvenient negatives? | Only superseded aborted output, truncated download, obsolete patch and caches removed; exact abort reproducer and meaningful negatives kept | Explicit cleanup manifest; D9 failure remains in current results |

The combined attempt set is five sequences, with four successful calibration
fits and one resource failure. Final-state accuracy improves in four available
fits, first-published accuracy improves in three, and D1 still regresses. D1/D4
are reused October 29 results, not independent new repetitions. D9 excludes 437
final poses without supported GT for every method; all suffix causal poses are
supported. No generalization claim should silently omit the unavailable fit or
unsupported evaluation states.

## What This Changes In The Research Direction

In the completed frozen-graph predictive audits, tighter optimization did not
remove the conditional statistical failures. The October 30 ONLINE checkpoint
audit identifies a separate finite-work tracking problem; that must not be
confused with the earlier converged-graph diagnosis. Neither result supports
simply finding another robust threshold. A fitted relative-noise model simultaneously
changes state estimates and their claimed precision. Under conditional mean
error and weak/uneven evidence, these changes can reinforce each other.
Whether the available observations support a GT-free distinction between
random noise, structured error and sparse inconsistency is the next modelling
question. Adding arbitrary robot/object biases or gates would bypass, not
answer, that question.

Priority order:

1. Freeze evaluation coverage, including every causal time segment, unsupported
   object events, per-robot performance and numerical/communication budgets.
   Preserve D1/D4 as diagnostics and reserve uninspected data for confirmation.
2. Diagnose a weight-consistent finite-work solve on the frozen D1 checkpoints
   at the same communication/work budget. Replace the dense calibration
   bottleneck only after small-case numerical parity is established. Do not
   add model parameters to conceal an optimization or scaling problem.
3. Test the mean-error/noise-uncertainty distinction under a known generative
   model with the SAME graph, then on GT-free redundant observation evidence.
   Demand evidence that the distinction is observable before adding states.
4. Only then implement a minimal adaptive update, test the retained D2 failure
   and online deployment, and compare to matched baselines. No current toggle
   is promoted as that solution.

The narrower supported claim is: network-supported calibration can improve
average state/prediction quality under the tested protocols, while coherent
state/uncertainty errors and limited conditional evidence remain obstacles to
reliable adaptation. The full robust-online claim is still open.

## November 23 Asynchronous Interface Study

The [fixed-interface report](dranppr_async_interface_report_20261123.md) adds a
controlled delay/loss simulator and a directional matrix step, not a deployed
asynchronous online estimator. At every-prefix equal bytes and update counts,
the matrix step improves 102/135 conditions over scalar delay control, ties 24,
and worsens nine. D1 heterogeneous activation and D3/D9 outage regressions
remain; reduced local conservatism does not order global finite-horizon error.

The scalar/directional methods use supplied finite delay bounds. Current charts
and local Schur rows are immutable and distributed before the simulation;
their distribution cost is excluded. The original native online PCG and global
physical backtracking barriers are unchanged. No new iMESA/riMESA comparison,
real link congestion, changing-chart recovery, or complete SE3 online result is
established. A positive-definite delayed counterexample explicitly rules out
claiming that unguarded stale-cache iteration is universally safe.

## November 23 Received-History Prediction

The [history report](dranppr_history_prediction_report_20261123.md) records a
positive fixed-model result: adaptive forecast order improves 93/135 matched
conditions over hold, ties 27 and loses 15, without extra messages or updates.
The candidate does not dominate secant or fixed quadratic prediction. D1 is
still negative overall, and the prototype costs substantially more Python
compute. Native wall-time acceleration has NOT been established.

The forecast score combines history fit residual with horizon-dependent noise
amplification in coupling-gradient coordinates. It is not calibrated predictive
uncertainty under correlated optimizer dynamics. Higher order can amplify
noise catastrophically, and an unobserved stop/reversal cannot be inferred.
Pairing prediction with unsafe stale steps worsens the SPD divergence example.
The old bounded-delay convergence argument does not automatically apply after
prediction. Nonlinear chart/epoch changes, clock mapping, native online SE3,
unknown delays and matched production-baseline reruns remain open.

## November 24 Low-Order Error Feedback

The [response report](dranppr_response_prediction_report_20261124.md) rejects
promoting scalar retrospective gain calibration as an overall optimizer gain.
It is cheaper than multi-order fitting, but wins74/ties27/loses34 versus hold
and wins45/ties27/loses63 versus its linear ablation. MRCLAM and SE3 respond
differently; D1 remains negative in all12 nonideal conditions. Matched-accuracy
Python service time remains1.85--2.11x hold, despite fewer updates/bytes.

Packet-interval error calibration is not calibration at the consumed forecast
horizon. Synthetic controls show suppressed useful trend at long horizons,
while a newly observed stop is corrected faster. Long-outage D4 worsens16.81%.
An exact-neighbor diagnostic still loses D1 endpoints under unchanged delay
damping; fresh-information damping helps only in an explicitly nondeployable
oracle. Error-aware step/metric adaptation remains a hypothesis, not a proven
algorithm or permission to remove the stable coupling control.

## November 24 Horizon-Matched Error And Update Strength

The [error-budget report](dranppr_horizon_error_budget_report_20261124.md)
provides a positive fixed-model result, but also a decisive robustness limit.
Both empirical-error-controlled hold and first-order prediction improve all
108 nonideal main conditions, with 27 exact ideal-network ties, identical
prefix messages/bytes/update opportunities, and no physical-objective change.
The hold ablation already accounts for much of the improvement. Its measured
Python time to a common baseline-defined target improves in eight of nine
conditions; prediction is slower than hold in all nine despite often using
fewer bytes. Fixed-1e-6 and post-blackout timing controls retain that distinction.

Long-delay D1/D4/SPD predictive runs diverge to relative gaps 2.66e21,
9.79e20, and 1.61e103. Moving-step current-error coverage collapses to
22.31%, 20.77%, and 12.67%. Historical maxima lag closed-loop error growth;
conditional descent algebra is insufficient when its error-bound premise
fails. The hold ablation improves all eight stress conditions but still has
imperfect empirical coverage and no general convergence guarantee.

Memory4/8 fails and memory16 succeeds on the two inspected predictive stress
cases. Do not promote a posthoc size change as automatic adaptation. No new
scalar gates were introduced, but memory length and finite delay contracts
remain modeling assumptions. Moving/zero-step coverage, missing timestamp
labels and source revision semantics are explicit audit obligations.

Remaining deployment risks: nonlinear chart/graph changes, clock consistency,
unknown or unbounded delays, network congestion, arbitrary multihop losses,
sensor-noise robustness, native online SE3, and matched public-DRAN++/riMESA
comparisons. The bounded-delay production default is unchanged. The next
research task is a current-action-valid uncertainty budget or stability-aware
prediction/update design, not more polynomial order or another fitted gate.

## November 24 Corrected Frozen-Boundary Response

The [model-response report](dranppr_model_response_feedback_report_20261124.md)
records 140 runs. A sender's actual state, held-cache gradient and local update
policy define a saturating finite response. Incoming packets correct the model
forcing separately from its state anchor. It improves all 24 nonideal frozen-
forcing ablations and does not reproduce the three old long-delay divergences.
This is a useful mechanism result, not release acceptance: at equal bytes it
wins 0, ties 2, and loses 26 against the existing adaptive hold method.

Extra gradient traffic raises bytes to 1.501--1.841x hold. Setup is less than
1% of candidate bytes; removing setup charges leaves all comparison outcomes
unchanged. Improving a stale-model ablation therefore does not establish that
the model should be communicated. Ordinary same-opportunity progress is also
weaker than linear prediction where gaps exceed the numerical floor.

Frozen-boundary saturation is not a coupled asynchronous convergence proof.
Source forcing and gains change after the last message, empirical current-error
coverage remains imperfect, and common local charts/rows are still preloaded.
No native SE3, riMESA, changing-graph or unknown-delay claim follows. Keep the
parent GitHub release pending; retain the diagnostic and failures instead of
promoting old-baseline wins or hiding the stronger no-prediction comparator.
