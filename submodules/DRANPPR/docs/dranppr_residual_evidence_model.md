# DRANPPR Residual Evidence Model

## Question

A large physical residual is not a fault label. It may be caused by an unfinished
state solve, an inaccurate measurement model, a wrong observation relationship,
or a combination of them. No residual-only rule can distinguish causes that
produce the same observations. DRANPPR should therefore expose evidence and
ambiguity rather than rename every large residual as an outlier.

## State-Reducible and State-Orthogonal Energy

At a fixed lower-level linearization, let the already whitened physical residual
and Jacobian be `r` and `J`:

```text
r(x + delta) approximately r + J delta.
```

The least-squares response is `delta* = -J^dagger r`. This gives the orthogonal
decomposition

```text
E_state = ||J J^dagger r||^2,
E_perp  = ||(I - J J^dagger) r||^2,
||r||^2 = E_state + E_perp.
```

`E_state` measures error removable by the current state model. It should trigger
lower-level computation or communication, not automatic noise inflation.
`E_perp` is inconsistent with that fixed linearized state model. Under independent
zero-mean Gaussian errors with one unknown scale, `E_perp / (m-rank(J))` is a
degrees-of-freedom-corrected scale estimate. Outside those assumptions it is only
diagnostic evidence.

The decomposition is gauge-safe because it depends on the column space of `J`,
not on a chosen state parameterization. It has a strict boundary: at a poor
nonlinear linearization, genuine state error leaks into `E_perp`. A large leak
must produce an ambiguous result or request a trust-region response; it must not
silently change sensor covariance.

## Factor-Excluded Prediction

To test factor block `e`, estimate the state without allowing `e` to shape its own
prediction. With the remaining information, form

```text
nu_e = z_e - h_e(x^{-e}),
S_e  = R_e + H_e P^{-e} H_e^T.
```

The normalized predictive energy is `nu_e^T S_e^{-1} nu_e`. If removing `e`
makes a direction required by `H_e` unobservable, the score is undefined and the
answer is **insufficient evidence**, not accept or reject. In online operation a
new factor is already excluded before assimilation, so the same construction is
causal. Existing factors require a factor-excluded local response, which DRAN++
can approximate through its reduced local/interface model rather than a full
batch rerun.

## Evidence-to-Response Map

### Numerical Prediction Support Audit (2026-09-22)

The controlled linear reference previously used a minimum-norm state and
`pinv(A.T @ A)` covariance without checking the requested prediction. This could
give a finite, artificially confident score for a direction absent from the
independent evidence. If N spans null(A), a prediction L*x is identifiable only
when L*N=0: x and x+N*v explain the same evidence but otherwise predict different
values. Gauge-invariant relative quantities can satisfy this even when the full
state is rank deficient. A gauge-fixed minimum-norm estimator covariance is not
a proper predictive posterior for an unsupported physical quantity.

The reference now uses one SVD/rank decision for state and covariance, verifies
support of each requested prediction, and raises `UnobservablePredictionError`
when support is absent. This is insufficient evidence, not an association
rejection. Predictive group calibration applies the same check after every
factor exclusion. No covariance floor or statistical rejection threshold was
added by this repair; the pre-existing scale-fit lower bound remains and is not
claimed parameter-free.

For A=diag(1,1e-8), the old independent normal-matrix cutoff returned variance
zero for the second direction even though the state fit reported rank two.
The corrected value is 1e16. Prediction variance is computed using the covariance
square root rather than cancellation-prone dense contraction; a second exact
test gives 1.0000000000000002 instead of 1.5. Rank-aware covariance and Jacobian
SVD are established numerical principles, not a new robustness contribution.
See the [Ceres covariance documentation](https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/covariance.h).

In an explicit unsupported-candidate test, old NLLs were 1.42677 (supported) and
.91894 (unsupported), falsely favoring the latter; the new API refuses that
comparison. Eight new tests cover exclusion, weak modes, relative gauge freedom,
invertible coordinate changes, zero information and independent full-rank parity.
All 17 adaptation tests pass. The 100-seed controlled recheck retains the prior
qualitative results: nominal variance .95378x, true-9x variance estimate 8.95961x,
and correct supported relation selected in 100/100 cases. These are synthetic
mechanism checks, not native distributed adaptation or universal classification.
Artifacts: `/tmp/dranppr_prediction_rank_20260922`.

| Evidence pattern | Supported interpretation | Response |
| --- | --- | --- |
| Large `E_state`, small `E_perp` | State or distributed solve is unfinished | Spend local response work or refresh interface state |
| Coherent predictive-scale shift across independently supported factors from one sensor group | Sensor precision/model has changed | Update that group's normalized likelihood model |
| One or a few localized factor-excluded surprises plus an alternative identity explanation | Observation relationship may be wrong | Let DBA move, merge or detach the identity partition |
| Persistent signed innovation tied to range/viewpoint/time | Calibration or measurement model is incomplete | Fit the identifiable structured bias/model term |
| Unobservable exclusion or observationally equivalent explanations | Cause is not identifiable | Retain ambiguity; do not learn noise or force identity |

These are evidence patterns, not independent hard gates. The common objective is
the factor-excluded predictive likelihood of physical measurements plus the
network identity partition constraints. Solver stationarity and consensus are
certificates of how accurately that objective has been evaluated; they are not
sensor residuals.

## Parsimonious Conditional Measurement Model

The native range residuals are strongly condition dependent. A minimal model is

```text
Var(nu_range,i) = P_state,i + a_g + b_g d_i^2,
```

where `d_i` is predicted range and `g` denotes one physical sensor group.
`P_state,i` is supplied by the factor-excluded state prediction; it is not a
learned sensor term. The two nonnegative parameters `a_g` and `b_g` are fitted by
normalized predictive likelihood. This is preferable to an unconstrained
per-factor weight: the parameters have physical meaning, the number of
parameters does not grow with the graph, and no distance bins or dataset-level
thresholds are introduced.

This model must not be fitted blindly to all large innovations. Competing
explanations are compared under the same predictive likelihood:

```text
H_state:    the lower response removes the innovation;
H_sensor:   one shared conditional model explains a group-wide shift;
H_relation: an alternative feasible identity explains a localized factor;
H_unknown:  the exclusions are unobservable or explanations are equivalent.
```

Thus a wrong relation is changed only when an alternative network partition has
independent support. Without that support, the correct output is localized
inconsistency or ambiguity, not a larger sensor covariance and not a forced
identity edit.

For an existing factor `e`, cross-fitting is applied twice: `e` is removed from
the state predictor and from the sensor-group calibration used to score `e`.
Consequently a single wrong relation cannot reduce its own surprise by first
pulling the state and then inflating the covariance. A genuine group-wide scale
change is still learned from the other independently predicted factors. If no
peer factor remains, the scale is unidentifiable and no adaptation is made.
This is not immunity to coordinated contamination: several mutually consistent
wrong relations can calibrate one another. Their rejection still requires an
independent identity alternative or a larger held-out edit block. The algorithm
must report this as an identifiability boundary rather than add another residual
threshold.

## Minimal Closed Loop

1. Append a new physical factor to a persistent local graph only after evaluating
   its pre-assimilation predictive distribution.
2. Use lower stationarity/consensus and state-reducible energy to allocate solver
   and communication work.
3. Accumulate group-level predictive sufficient statistics only from identifiable,
   independently predicted factors. Score each factor with a leave-factor-out
   group model and estimate a parsimonious sensor-group model, not one covariance
   per edge.
4. Score identity edits using the calibrated factor-excluded distribution. DBA
   changes the network partition; accepted changes alter the lower graph.
5. Repeat until association edits stop and lower certificates are within the
   declared compute/communication budget. Report unresolved ambiguity explicitly.

The first implementation target is not a new stack of thresholds. It is one
shared predictive statistic, one sensor-group covariance model, and the existing
DBA partition edit space. Any temporal change detector must be justified by a
fixed statistical error rate and frozen across datasets, rather than tuned per
sequence.

## Verified So Far

The controlled linear audit shows exact invariance of `E_perp` to state error in
the column space of `J`, localized response to one corrupted factor, and failure
to detect a deliberately state-equivalent bias. Native MR.CLAM checks show that
small nonlinear perturbations are mostly separated, while large perturbations
break the fixed-linearization approximation. Persistent independent local iSAM2
on the current-source first-60-second audit has worst-dataset p99 `4.086 ms`,
maximum update `5.243 ms`, and maximum serial backlog `0.0299 s`. Full histories
expose growing reelimination instead: raw per-factor insertion accumulates
`223--926 s` maximum backlog. Treating one robot/timestamp sensor frame as one
atomic graph transaction reduces total update work by `36.8--54.8%` and backlog
by `80.0--98.2%`, with final trajectory and object RMSE changes below `0.0043 m`.
This preserves every factor but does not bound per-frame history cost; Dataset 9
still reaches `196.8 ms` p99 and `141.1 s` maximum backlog. These are independent
local-reference results, not yet distributed DRANPPR results.

A separate GT-assisted structural audit fitted `a + b d^2` on the first 30% of
each MR.CLAM robot stream and evaluated the remaining 70%. It improved held-out
range NLL over a fitted constant variance on all 25 streams, with median gain
`0.1814` nats per observation and minimum gain `0.0613`. Because truth was used
to construct the residuals, this supports the model family only; it does not
validate deployable online parameter learning.

In a 100-seed controlled cross-fit audit, a true eight-fold group covariance
change produced an `8.43x` fitted-scale ratio. One wrong relation made the
inclusive scale `9.77x` too large, whereas the scale fitted without that factor
remained `1.00x` nominal and the wrong factor ranked at the 100th predictive
energy percentile. This verifies the covariance self-validation mechanism in
the controlled model, not complete cause classification.

A matched four-arm controlled audit compares fixed covariance, residual-fitted
covariance, factor-excluded predictive calibration, and oracle covariance on the
same exact linear-Gaussian problems. With nominal noise and a deliberately poor
state, residual fitting estimated `510.47x` variance and produced `2.027x` oracle
state RMSE; predictive calibration estimated `0.954x` and produced `1.000x`.
With a true `9x` sensor-group variance, predictive calibration estimated `8.960x`
and reached `1.001x` oracle RMSE, versus `1.582x` for the fixed model. With one
wrong relation, merely inflating the inclusive scale still had `2.277x` oracle
RMSE, while an independently supported relation edit selected the true candidate
in 100/100 seeds and reached `0.998x`. Identical candidates had exactly equal
predictive NLL. Thus the evidence supports three distinct actions in the exact
model: solve the state, adapt a shared sensor model, or edit a relation. It also
preserves ambiguity when the relation is not independently identifiable. This is
a mechanism result, not a native-data or distributed-backend result.

Converged Huber IRLS and a clearly labelled GNC-style Geman--McClure control add
an important non-straw-man comparison. For the isolated gross wrong relation,
they reached `1.151x` and `1.112x` oracle RMSE by assigning median target weights
`0.0643` and `7.38e-5`. They are therefore useful for gross contamination. Under
a genuine `9x` sensor-group variance, however, they retained only `0.694` and
`0.414` median group weight and reached `1.368x` and `1.543x` oracle RMSE; the
predictive covariance arm reached `1.001x`. On nominal Gaussian data their RMSE
ratios were `1.047x` and `1.184x`. The result distinguishes responsibilities:
robust losses bound isolated influence, while a calibrated predictive likelihood
models coherent physical precision without discarding the whole group. These
are behavior controls with fixed conventional settings, not official GNC, AEROS,
riMESA or adaptive-shape implementations.

The first native factor-excluded calibration pilot uses the first 60 seconds of
MR.CLAM Dataset 4 without truth in the estimator. A deliberately cheap method
froze the state-prediction covariance at the nominal scale and fitted one shared
bearing/range multiplier. At the final checkpoint, three of five robot groups
collapsed to zero; the other two estimated `0.234` and `0.295`. This is a negative
result, not evidence of noiseless sensing. When the full fixed-linearization
state response was recomputed for every candidate scale, all five profiles were
nonzero (`0.142--0.402`). Freezing the lower response is therefore not acceptable
for a large covariance edit, even though it remains useful for a small candidate
score.

The shared multiplier is also structurally too coarse. In a separate post-hoc
truth audit of Dataset 4, range variance divided by the nominal `0.1 m` variance
spans approximately `0.964--2.933`, while bearing variance divided by the nominal
`0.05 rad` variance spans `0.063--0.874`. A common scalar must compromise between
these different components. On the factor-excluded pilot, a separate marginal
range model `a+b d^2` improved its in-sample predictive NLL for three robots by
`0.035--0.110` nats/observation; two remained at the boundary because predicted
state uncertainty already exceeded their innovations. This only motivates the
next exact/cross-fitted model. It does not validate online feedback: the pilot
uses batch-smoothed local linearizations, freezes no future-safe causal state,
and its conditional calibration is inclusive across deleted factors.

Raw traces and the current report are under
`results/dranppr_residual_evidence_20260921/`; full-history timing, accuracy and
ordering/atomic-frame diagnostics are under
`results/dranppr_fullstream_reference_20260921/`; the matched adaptation arms are
under `results/dranppr_cause_adaptation_20260921/`; the Dataset 4 native profiling
pilot is under `results/dranppr_native_predictive_scale_pilot_20260921/`.

## Native Feedback Counterexample (2026-09-22)

The chronological follow-up changes the interpretation of the positive controlled
results. On native Dataset 4, fit the first 30 seconds only, freeze parameters,
and evaluate 1,751 future observations through 120 seconds before assimilation.
Ordinary residual fitting has mean future NLL `-3.0601`, nominal noise `-2.4340`,
and full-response conditional profiling only `-1.2332`. Conditional profiling
improves object-copy RMSE but worsens final local trajectory RMSE from `0.22275`
to `0.25280 m`; causal trajectory RMSE is essentially unchanged. Three robots'
bearing variances reach the numerical lower boundary. The first scalar pilot
established that freezing the lower response can fail, not that recomputing it
makes a misspecified observation model trustworthy.

Robot 3's source row 350 provides a concrete cause-separation case: the learned
predictor has range/bearing innovations `-0.501 m / 0.229 rad`, whereas the
post-hoc independent-reference observation errors are `-0.0102 m / -0.0163 rad`.
The measurement is substantially better than the predictor. Optimizing the
unchanged past graph before that frame reduces weighted cost from `38539.36`
to `1481.61`, but still leaves a `-0.413 m / 0.201 rad` innovation. A poorly
conditioned finite solve is part of the problem, yet one stronger solve does
not repair this learned-model prediction. Neither global optimality nor a
unique causal explanation follows from that intervention.

The mechanism must distinguish short-term predictability from physical accuracy.
Factor deletion does not remove correlated retained measurements, persistent
bias, odometry-model error, or dependence through a common linearization. Native
innovation correlations support checking those alternatives, but do not identify
which is responsible. Do not introduce a fitted floor or rejection gate merely
to suppress this failure. The next bounded diagnostic is nonlinear whole-frame
exclusion with the existing covariance candidates, not a new hyperparameter scan.

These are independent-local Python GTSAM audits, not a DRANPPR implementation or
an official robust-baseline comparison. The full covariance audit is also too
expensive for the intended online loop (about 100 seconds of prediction work per
120-second arm). See `docs/dranppr_online_noise_report_20260922.md` and
`results/dranppr_online_noise_20260922/` for all four arms, physical counterexamples,
timing, frozen-graph solves, and the past-only checkpoint intervention.

## Directional Noise Follow-Up

Explicit nonlinear factor/frame deletion does not fix the previous ranking.
The next diagnostic uses classical REML contrasts `C^T y`, where `C^T A=0`, to
test the full covariance model independently of arbitrary linear state changes.
Fixed and uniformly scaled odometry covariance still yield near-zero bearing
variance. Learning separate local odometry directions removes that boundary and
improves the selected Robot 3 chronological case: final trajectory RMSE
`0.20036 -> 0.07851 m`, causal RMSE `0.24656 -> 0.14373 m`, and object RMSE
`0.18108 -> 0.12957 m` versus fixed noise. Ordinary residual calibration remains
slightly better in mean future NLL, but has worse state errors than the new arm.

This supports accounting for directional state-prediction uncertainty rather
than attributing all disagreement to observation noise. It does not establish
a new REML method, a universal cause classifier, or distributed SLAM superiority.
Q-only/R-only attribution and other-robot/sequence validation remain necessary.
Dense uncertainty work still dominates computation. Details and all controls are
in `docs/dranppr_noise_structure_report_20260922.md`.

## Finite-Data Support and Prior Boundary

The subsequent all-robot comparison in
`docs/dranppr_noise_factorial_report_20260922.md` preserves the D4 gain but fails
the unchanged D1 transfer gate. D1/R5's very small bearing estimate is not
strongly supported: after nuisance refitting, a 442,000-fold variance increase
costs only 1.00882 training nats. This motivates representing parameter
uncertainty, not merely the state covariance conditional on one noise estimate.

However, naive log-uniform variance integration is invalid for the additive
component model when the remaining covariance is positive definite: likelihood
stays positive at zero, while the 1/q prior has infinite mass there. The same
data then produce radically different conditional medians as the numerical
lower cutoff changes. The present six-component models are structurally full
rank, so this is not proof of absolute noise nonidentifiability. A conditional
Jeffreys calculation only demonstrates a finite-density alternative near zero;
it is not a validated joint posterior or an online adaptation fix.

The conditional matrix arithmetic is independently reproduced without GTSAM,
but native D4 reconstruction attempts have abnormal Python/GTSAM shutdowns and
are retained as provisional rather than clean runs. The complete boundary,
prior, numerical and process-status audit is in
`docs/dranppr_variance_boundary_report_20260922.md`.

The subsequent dependency audit in `docs/dranppr_gtsam_runtime_audit_20260922.md`
provides two normally terminated D4 reconstructions with identical numbers by
preloading one TBB allocator for the experiment process. This resolves that
bounded numerical-reconstruction caveat, not the statistical transfer failure.
The original abnormal process records remain preserved.

## Relative-Geometry Follow-Up (2026-09-22)

The native fixed-noise Schur model now exports complete object covariance for
frozen diagnostics. Comparing coordinate-invariant object-pair distances exposes
a practical approximation risk: on D9 300 s edge 1-2, full cross covariance gives
median absolute standardized discrepancy 3.736, whereas retaining only object
marginals gives 0.608. The latter inflates relative-distance variance by a median
44.27 times, conflating common map motion with shape uncertainty. Both numbers
are conditional first-order diagnostics, not calibrated fault scores.

D4 also contains large discrepancies, so these statistics cannot be a binary
gate for bad associations or unsuccessful state estimation. Neighbor-informed
object restarts and matched opposite-direction controls return to the original
local optima on D9 and D4; they do not repair the remaining discrepancy. A raw
same-frame D9 object pair changes from about 2.37--2.42 m at early distant views
to 2.94--3.03 m at later views, independent of any estimated pose. This gives
measurement/static-identity-model evidence beyond optimization convergence, but
does not identify the faulty component. Many discrepant inferred pairs lack
such co-observation support entirely.

The decision is to preserve relative-geometry information, add actual joint
physical constraints rather than another initialization-only mechanism, and
require independently supported conditional-model evidence before noise/identity
adaptation. No new variance floor, rejection threshold or learned parameter is
deployed. See `docs/dranppr_map_shape_report_20260922.md`; all 70 frozen solver
completions exit normally and 70 regression/diagnostic checks pass. These are
not distributed online results or independent statistical trials.

## Actual Joint Physical Estimation (2026-09-22)

The next frozen-data intervention adds genuine shared-object constraints rather
than another neighbor-informed initialization. A one-gauge common chart keeps
all non-root poses free; identities merge only through observing neighbors on
the supplied chain. Routed Schur/PCG and separately executed central GN/LM reach
the same physical solutions on D4 120 s and D9 300 s. Single-global-alignment
trajectory/object-copy RMSE are `.286/.185 m` and `.482/.264 m`.

Removing relative-frame effects with the SAME per-robot diagnostic in all arms,
D9 trajectory RMSE changes `.810 -> .761 -> .434 m` from the online independent
snapshot to fully completed independent maps to joint estimation. Object-copy
RMSE changes `1.262 -> 1.236 -> .380 m`. Thus joint evidence adds value beyond
the previously measured local optimization lag. This remains a frozen diagnostic,
not an equal-budget online baseline result or an innovation claim for Schur/GN.

Crucially, D9 odometry cost increases `2021.670 -> 2636.979` while the prepared
joint observation cost falls `398216.664 -> 998.536` and state error improves.
Independent fully optimized graphs have still smaller summed cost (`2935.903`)
than the constrained joint graph (`3635.515`) because their state space is larger.
Neither the huge prepared-start decrease nor individual residual growth identifies
noise or association faults. Joint state support must precede causal attribution.
The raw pose-free inconsistencies remain evidence to explain, not solved defects.

The full routed completions require 539/602 PCG iterations and 7.30/10.70 MB
application traffic, excluding offline bootstrap/rosters and network framing.
D9 120 s explicitly lacks this conservative initializer's support. Online causal
component merging, reusable responses and matched distributed robust baselines
remain the next gates. No noise/identity adaptation is enabled. Details:
`docs/dranppr_frozen_joint_report_20260922.md`; 85 focused checks pass.

## State Age Is Not Sensor Variance (2026-09-22)

The actual chronological joint replay achieves useful state quality without any
new noise model: D9 global final trajectory/object RMSE .401/.229 m. Its terminal
physical cost is only 9.77e-5 above a separate frozen completion. Yet the synchronous
serial implementation accumulates 53.215 s of service delay. With one fixed
alignment, its non-extrapolated connected-tail poses have .559 m error at their
measurement timestamps and 2.030 m against GT at virtual delivery time.

This isolates a different failure mechanism using independently measured service
timestamps: accurate historical states can be unhelpful as current-time outputs.
It is not evidence of larger sensor variance, nor proof that extrapolation would
not help. The virtual serial model excludes network latency and is not a measured
distributed deployment. It nevertheless rules out using terminal accuracy alone
as the online success criterion.

The correct next intervention is finite-response tracking and response reuse,
with state age and communication included, rather than more precise frozen solves
or covariance inflation. Persistent raw geometry conflicts and identity ambiguity
still need independent statistical evidence. No classification/adaptation is
enabled. See `docs/dranppr_causal_joint_report_20260922.md`; overall goal active.

## Solver Work and Model Freshness Are Coupled (2026-09-22)

The four-arm causal replay separates chart reuse from one-iteration interface
solving without changing any observation variance or identity rule. On D9,
tight cached solving preserves .401/.229 m terminal trajectory/object error while
reducing serial service 226.225 -> 167.477 s and maximum backlog 54.737 -> 12.352 s.
Its fixed-alignment stale-delivery diagnostic improves 2.084 -> .772 m, although
its timestamped causal error remains .559 m. This is direct evidence that useful
state-quality improvement can come from computation alone, without inflating R.

One interface iteration cuts communication over 91%, but leaves states changing
enough that cached responses are never reused. Combined cached/one-step service
is 216.694 s, worse than tight caching. It still approaches the terminal minimum,
so terminal cost alone would miss the failure. The remaining backlog prohibits
a real-time claim. We have not inferred a sensor/association fault from any of
these observations. Same-graph optimization error, model discrepancy, and data
arrival must be distinguished before deriving adaptive statistical updates.
Details: `docs/dranppr_inexact_joint_report_20260922.md`.
