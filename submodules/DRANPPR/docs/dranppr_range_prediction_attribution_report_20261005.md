# Range Prediction Attribution And Controlled Weight Interventions

## Main Findings

The two focal failures do not have one universal cause or repair.

1. D1/R5 has concentrated range-score damage. Only 17/83 observations worsen,
   while 66 improve. Five observations of subject 7 account for 56.90% of the
   positive damage. A small pooled innovation mean hides opposing object-wise
   discrepancies.
2. D4/R2 has a systematic conditional mean shift on its only forecast subject.
   Merely using the new covariance with the old mean IMPROVES its range score;
   the actual adapted mean changes enough to reverse this benefit.
3. Most predictive range variance contraction is in state/map/process
   uncertainty, not the direct range-measurement variance term: 87.34% for
   D1/R5 and 95.72% for D4/R2.
4. Actual frozen-weight interventions identify an important mechanism beyond
   synthetic score swaps: for D4/R2, activating learned Q worsens joint
   predictive NLL under ALL four bearing/range weighting combinations.
5. There is no validated universal switch. Restoring nominal range weights
   while retaining learned Q/bearing improves both dataset-average joint
   scores, but worsens 7/10 robot scores and does not repair D4/R2.

This is a mechanism and blind-spot audit, not a new adaptive method, a robust
baseline victory, or an end-to-end state improvement claim. No new estimator
parameter, gate, kernel, rejection rule, or GT correction was introduced.

## Frozen Protocol

Reuse all checksummed nominal/joint-QR forecasts in
`results/dranppr_network_noise_transfer_qr_20261003`: D1/D4, five robots each,
checkpoints 30/60/90 s and the same five-second prediction windows. Predictions
are based on past states plus future odometry, without assimilating observations
in the prediction window. The original five noise scales remain frozen.

The attribution audit verifies all 360 source files and 29 files in the
previous motion audit. It uses all 533 common forecast observations on 386
distinct robot/time frames. GT values enter neither scores, grouping, fitting
nor interventions. Existing declared barcode identities and topology remain
unchanged. No subject is deleted and no arm is selected as the deployed method.

Independent event checks reconstruct each measured bearing/range from the
stored prediction and innovation, compare against the source events, check
checkpoint state timestamps, event availability and forecast support, and
account for observations of previously unknown objects. Agreement between two
predictors alone is not the only input check.

## Mean And Covariance Are Coupled

For one range innovation e and predictive variance v,

    L(e,v) = 0.5 * [e^2/v + log(2*pi*v)].

Let e0/e1 denote nominal/adapted errors and v0/v1 their variances. Evaluate
all four combinations without refitting or changing the observation. The
interaction is

    I = L(e1,v1) - L(e1,v0) - L(e0,v1) + L(e0,v0).

The effect of changing the mean depends on the covariance at which it is
evaluated. Reporting only one substitution path can be misleading. We also
retain the full two-dimensional covariance and both Gaussian conditioning
orders, using the previous tested `split_score` implementation.

| Case | Original L00 | New mean/old covariance L10 | Old mean/new covariance L01 | Adapted L11 | Interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1/R5 | -.297960 | -.202276 | -.119034 | .169561 | .192912 |
| D4/R2 | -.525975 | -.291304 | -.686027 | -.030696 | .420660 |
| D4/R4 | -.282574 | -.302284 | -.143048 | -.197351 | -.034593 |

Lower NLL is better. Negative density NLLs are valid in fixed metre units.
D1/R5 is harmed by both single substitutions and their interaction. D4/R2's
new covariance helps the OLD mean, but amplifies the damage of the NEW mean.
D4/R4 is a counterexample to attributing all failures to worse means: its
mean improves, but its range uncertainty tightens enough to worsen range NLL.

These are valid Gaussian scoring distributions, not physically realized
estimator runs or causal effects of independently changed solver weights.
The actual interventions below address that separate question.

## Where Does Variance Contraction Come From?

The predictor stores total covariance S and observation covariance R. Define
U = S - R. U contains checkpoint pose/map uncertainty, cross covariance, and
propagated process uncertainty; it is NOT Q itself. Both full covariances and
the PSD decomposition are checked, without adding a floor or damping.

The attribution audit swaps mean, U and R separately: 2x2x2 = 8 hypothetical
distributions per observation, or 4,264 scores on the SAME 533 measurements.

| Case | Total range variance, old -> new | U contribution, old -> new | Direct R contribution, old -> new | U share of contraction |
| --- | ---: | ---: | ---: | ---: |
| D1/R5 | .040990 -> .013218 | .030990 -> .006733 | .010000 -> .006485 | 87.34% |
| D4/R2 | .041744 -> .013442 | .031744 -> .004654 | .010000 -> .008788 | 95.72% |

Units are m^2. These are averages over the same observations, not independent
variance-component estimates. They refute the simple explanation that most
predictive contraction is directly the changed range sensor variance, but by
themselves do not identify which solver weights changed U.

Range normalized error energy increases from .8240 to 2.8498 on D1/R5 and
.3561 to 2.4168 on D4/R2. Nominal 95% marginal range coverage falls from
93.98% to 81.93% and from 100% to 71.43%, respectively. These coverages concern
RANGE ONLY, not the earlier two-dimensional polar ellipse. Bias and covariance
miscalibration can both cause undercoverage; coverage alone cannot separate them.

## Object, Frame And Time Heterogeneity

D1/R5's total range NLL change is +.467521. All its declared subjects are shown:

| Subject | Observations | Range NLL change | New signed error mean (m) | New centred SD (m) |
| --- | ---: | ---: | ---: | ---: |
| 7 | 5 | +7.800764 | -.50434 | .09832 |
| 8 | 5 | +1.230937 | -.21953 | .18840 |
| 9 | 25 | -.038581 | +.06979 | .12649 |
| 10 | 15 | +.579748 | +.05284 | .21782 |
| 12 | 7 | -.377828 | -.05924 | .00989 |
| 17 | 26 | -.440043 | -.01714 | .06306 |

Subject 7 contributes 39.0038 of the total 68.5488 positive NLL-damage mass;
improving observations offset 29.7445. Use positive/negative masses separately,
not percentages of a nearly cancelling net loss. These data do NOT justify
deleting subject 7 online or claiming it is an independently identified outlier.
Its identity/geometry/time conditions are entangled.

The pooled D1/R5 signed mean moves CLOSER to zero (-.03170 -> -.02340 m),
while its RMSE rises (.17422 -> .19482 m). Thus a robot-wide mean check would
miss the conditional discrepancy. Conversely, all 21 D4/R2 forecasts concern
subject 7: its mean shifts +.08268 -> +.16756 m while centred SD decreases
.09171 -> .06781 m. One object cannot identify a robot-wide sensor bias.

Equal weighting of frames rather than observations preserves the three
robot-level range regressions: D1/R5 +.462707, D4/R2 +.495280, D4/R4 +.061514.
Both dataset-average range improvements also survive: D1 -.212474, D4 -.080382
with frame weighting. This rules out repeated objects per frame as the sole
explanation, but frame averages are still temporally dependent.

Time aggregation also hides sign changes. D4/R2 range NLL changes +1.034553
after 60 s but -.097920 after 90 s. D4/R4 changes +.338605/-.383041/+.401800
after 30/60/90 s. D1/R5 worsens after both supported checkpoints, by +1.279077
and +.258485. No constant bias/variance law is established by these few windows.

## Actual Fixed-Weight Interventions

Freeze the five multipliers from the original joint-QR fit. Toggle three groups
between nominal and their EXISTING learned values: Q (three process components),
B (bearing measurement variance), R (range measurement variance). No refit,
new values, parameter search, or GT is used. Prefix weights remain nominal.

Eight arms x two datasets x three checkpoints = 48 actual LM solves/forecasts.
Every arm starts from the same archived nominal initialization with the same
complete graph and LM cap 100. Each has a GN control from that SAME original
initialization and cap 100. No GN endpoint replaces an LM result. All commands,
return codes, library hashes and timings are recorded.

| Learned groups enabled | D1 mean joint NLL | D1/R5 joint NLL | D4 mean joint NLL | D4/R2 joint NLL |
| --- | ---: | ---: | ---: | ---: |
| None | -1.576927 | -1.130776 | -1.466104 | -1.644697 |
| R | -1.612888 | -1.098632 | -1.478820 | -1.665915 |
| B | -1.738346 | -1.280582 | -1.558311 | -1.798076 |
| B + R | -1.760008 | -1.208253 | -1.564446 | -1.819721 |
| Q | -1.872275 | -1.134352 | -1.524777 | -1.302432 |
| Q + R | -1.888571 | -.984071 | -1.514164 | -1.324216 |
| Q + B | -2.042089 | -1.254142 | -1.730179 | -1.333342 |
| Q + B + R | -2.023980 | -.987708 | -1.717643 | -1.360273 |

These are conditional plug-in marginal prediction scores, not GT state errors
and not a joint likelihood of independent future observations. All eight arms
are disclosed; no retrospectively best arm is promoted.

### D4/R2: Q Changes The Mean As Well As Confidence

Activating learned Q worsens this robot's joint NLL by .342265, .341699,
.464734 and .459448 across the four fixed B/R combinations. With Q alone,
range RMSE changes .123477 -> .177240 m and average range variance
.041744 -> .014878 m^2. B alone leaves these near .123790/.041740.

Thus in this controlled computation the Q-group intervention is sufficient
to produce much of this failure. It does not prove the actual process-noise
covariance is larger than estimated, or identify a particular physical bias.
D4's learned longitudinal/lateral/yaw variance multipliers are approximately
.0979/.5766/1.0621: different components change in different directions.

### D1/R5: Different Interactions And Concentrated Errors

B alone worsens range RMSE .174215 -> .198950 m while barely changing range
variance (.040990 -> .040047), yet improves JOINT NLL through the bearing
channel. Q alone sharply contracts range variance to .017817 but has nearly
unchanged joint NLL. Adding R to learned Q/B moves R5 joint NLL from -1.254142
to -.987708, while helping D1's other four robots. A single channel's residual
or one factor group's value therefore cannot define a universal repair.

Restoring nominal R under learned Q/B improves average joint NLL on both
datasets, but worsens D1/R1--R4 and D4/R1,R2,R5: 7/10 robots. Dataset averages
can improve while most robot-wise changes are small regressions. This is an
aggregation/utility choice, not evidence that each robot became more reliable.

## Numerical And Input Checks

- Direct comparison of all 48 saved input packages confirms identical initial
  poses, object positions, measurements, identities and topology. Prefix
  sigmas are unchanged; suffix squared-sigma ratios match only the declared
  group multipliers (maximum relative discrepancy 6.66e-16). This check does
  not call the weight-application helper used to produce the inputs.
- 48 LM predictions and 48 GN controls completed in 229.21 s producer wall
  time: summed worker time 213.22 s for LM/forecast and 14.45 s for GN.
  Dense centralized forecast QR dominates parts of this audit; these times
  are NOT deployed online or communication-efficiency results.
- Both all-nominal and all-adapted controls reproduce archived means,
  covariances and scores exactly at all 12 endpoint comparisons.
- Maximum LM/GN position difference 1.74234e-6 m, yaw 6.19740e-7 rad,
  physical objective difference 8.42419e-11 on the SAME weighted objective.
  Maximum LM iterations 16/100; maximum raw gradient .00027456.
  Agreement excludes a substantial LM/GN endpoint discrepancy here, not a
  shared local minimum or nonlinear posterior approximation error.
- Maximum covariance solve relative residual 2.23e-15. Score/group parity is
  exact; source-observation reconstruction differs by at most 5.55e-17.
- Both calibration fits are interior, rank five, with log-score maxima
  2.46e-5 and 1.39e-6. Fisher eigenvalues span 14.21--178.87 (D1) and
  2.97--84.33 (D4). These are properties WITHIN the assumed fixed-linear
  noise family, not certificates of sensor truth or model adequacy.
- Native regressions: 62 passed. Python regressions: 181 passed, zero skipped.
  Full commands, logs and XML are in the attribution archive. No experiment
  PCG iterations or MPI ranks; CCI and production `DBA-DRANPP` are unchanged.

## Validation Coverage Is Limited

Independent source-event accounting finds 550 observations in the selected
windows. Of these, 17 concern objects without a prediction at that checkpoint
(D1: 3 after 30 s and 13 after 60 s; D4: 1 after 30 s). Both methods omit the
same 17, so the known-object comparison is matched, but does not measure
new-object admission or full-stream success. Unknown observations are retained
in the diagnostic ledger, not assigned an arbitrary score.

Only 15 s per dataset are forecast. D1/D4 have 92.847/104.466 s of held-out
suffix and 1,491/2,117 held-out observations through 120 s. The 533 scored
observations are 14.77% of those 3,608 observations. Selected-window results
must not be generalized to the entire stream. D1/D4 have also been inspected
repeatedly during development; they are discovery data, not untouched final
test sets, even though no future values enter the estimator.

The detailed [blind-spot register](dranppr_blind_spot_register_20261005.md)
separates checked invariants, diagnosed failures and unresolved assumptions.
In particular, D2 high-noise failure remains unresolved; noise-parameter
uncertainty, model bias, genuine online adaptation and fresh fair robust
baseline comparisons are not established by these experiments.

## Reproduction And Artifacts

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.audit_range_prediction_attribution \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --previous results/dranppr_motion_time_structure_verified_20261004 \
  --output /tmp/dranppr_range_attribution_repeat

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m experiments.audit_frozen_weight_factorial \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output /tmp/dranppr_frozen_weights_repeat

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q tests/test_range_prediction_attribution.py \
  tests/test_frozen_weight_factorial.py
```

Attribution archive: `results/dranppr_range_prediction_attribution_verified_20261005`.
Factorial compact archive: `results/dranppr_frozen_weight_factorial_20261005`.
Large reproducible linearization exports remain in
`/tmp/dranppr_frozen_weight_factorial_20261005`; their hashes are retained in
`scratch_only_files.json`. The earlier non-verified-suffix attribution directory
is a completed pre-extension analysis, not a failed estimator run; only the
verified directory includes independent source/unknown-observation accounting.
