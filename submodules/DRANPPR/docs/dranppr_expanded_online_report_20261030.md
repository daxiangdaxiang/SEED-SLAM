# Expanded Online Validation and Tracking Diagnosis

Date: 2026-10-30. Research Pose2/Point2 backend, not production ObjectPose3
DRAN++. This round changes experiment orchestration, not the optimizer or the
statistical model. No production promotion, parameter sweep, commits or pushes.

November 17 provenance correction: inspection of the sealed D9 archive finds
a successful fit JSON followed by a truncated contrast NPZ. The old 600 s
command timed out during dense artifact export, not before a fit was computed.
Its unavailable-candidate status below remains correct under the declared
producer-success contract. Timing covers the complete command and must not be
read as isolated optimizer time. See the
[follow-up report](dranppr_sparse_calibration_3d_report_20261117.md) for the new
completed sparse workflow; the original result archive is unchanged.

## Main Findings

1. The unchanged first-prefix noise fit transfers positively to D2 and D3.
   Against fixed-native, final trajectory improves 48.46% and 23.22%, object
   copies 28.55% and 21.22%, and first-published trajectory 40.94% and 34.30%.
   Both sequences also beat the tested repaired-riMESA and paper-iMESA arms
   in these three accuracy metrics. This is evidence of transfer, not a
   universal robustness or real-time claim.
2. Calibration scalability fails the current deployment requirement: D3 fit
   takes 544.712 s; D9 does not finish within the declared 600 s watchdog.
   D9 candidate is UNAVAILABLE, not replaced by nominal or another fit.
3. D1's previously observed causal regression has a demonstrable finite-work
   component. At 60 s, completing exactly the already-arrived graph, without
   future observations or refitting, changes calibrated trajectory/object RMSE
   from 0.429735/0.765820 to 0.229520/0.239738 m. LM and GN agree closely.
   This diagnoses a mechanism; it does not fix the online algorithm.
4. Across the five attempted sequences, four calibration runs are available.
   All four improve final state quality, three improve first-published accuracy,
   and D1 regresses. The fifth is a calibration failure. Do not report only
   the four successful fits or equate better final smoothing with online safety.

## Protocol and Scope

- D1/D4 rows are immutable October 29 results, not fresh repetitions. D2/D3/D9
  are new October 30 comparisons on previously inspected sequences. These are
  all five locally available sequences, NOT all nine MRCLAM datasets or blind
  held-out tests. D1--D4 end at 120 s; D9 ends at 300 s, not full missions.
- Same five robots, known barcode identities, chain topology and common-past
  primal warm start. The candidate changes only suffix noise variances using
  five shared REML components: three motion channels, bearing and range.
  Prefix weights remain nominal. No ground truth enters fits or estimators.
- Prefix fitting is centralized, on one fixed linearization. Scales are then
  frozen. This is not continuously adaptive or decentralized noise estimation.
- Native uses `joint_cached_one --lazy-refresh`, one interface PCG iteration
  per frame. Native fixed/calibrated share the same solver and work rule.
  Baselines use nominal noise, common primal values and cold internal states.
  Their comparison does not isolate an optimizer-only or robust-kernel effect.
- `I` is paper-iMESA; `R` is pinned repaired riMESA. riMESA commit is
  `ffddc21dad530ba0b0e57c42495146b410581fc4`; riSAM is
  `c6fa05eaece103efd7e44e640bda8df6eff30e8d` with the previously disclosed
  endpoint-cancellation and merit-consistency numerical repairs. No robust
  kernel change. This is not an untouched official binary.
- `one`: one sweep per frame. `bytes`: per-frame cumulative application-byte
  ceiling equal to the smaller native-arm ledger; D9 uses nominal alone because
  calibration failed. Ceilings do not imply exactly equal bytes or equal time.
- Estimator calls run sequentially with one BLAS/OMP thread and a 600 s limit.
  All 17 launched new replays finish. GT evaluation starts after all terminate.
- Final and first-published errors each use one global SE(2) registration over
  all supported robots, never separate robot alignments. Causal states cannot
  be revised by future data; their evaluation transform is nevertheless
  posthoc. Common-prefix-aligned diagnostics are reported below.
- D9 scores 4104 final poses and records 437 without supported GT; all methods
  have identical omissions. Its 2514 first-published suffix poses are supported.
  D2/D3 have no unsupported final or causal poses. Object error counts copies,
  not one fused landmark per identity. Raw cross-weight/kernel costs are not ranked.

## Five-Sequence Comparison

All errors in meters. `Final` is trajectory translation RMSE, `Object` is
object-copy RMSE, `Causal` is first-published suffix trajectory RMSE. `Service`
is cumulative in-process suffix update time, excluding input/trace I/O,
prefix preparation and noise fitting. MB is decimal application payload plus
recorded control, excluding transport overhead and calibration/bootstrap.

| Data | Method | Final | Object | Causal | Service (s) | MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D1 | Native fixed | .266416 | .180559 | .251488 | 6.266 | 29.116 |
| D1 | Native calibrated | .157835 | .113145 | .492764 | 6.174 | 29.135 |
| D1 | I / one | .303615 | .270060 | .396938 | 3.052 | 3.714 |
| D1 | I / bytes | .302832 | .267800 | .390840 | 17.423 | 29.115 |
| D1 | R / one | .375762 | .498560 | .523592 | 19.544 | 3.714 |
| D1 | R / bytes | .366174 | .489836 | .495361 | 53.371 | 29.115 |
| D2 | Native fixed | .311673 | .244900 | .296666 | 6.979 | 24.153 |
| D2 | Native calibrated | .160632 | .174987 | .175197 | 6.652 | 24.144 |
| D2 | I / one | .295454 | .349479 | .369932 | 3.197 | 3.343 |
| D2 | I / bytes | .294966 | .351466 | .371664 | 16.831 | 24.141 |
| D2 | R / one | .222969 | .312701 | .291505 | 17.544 | 3.343 |
| D2 | R / bytes | .216134 | .326360 | .301287 | 51.264 | 24.141 |
| D3 | Native fixed | .195414 | .161719 | .212358 | 7.344 | 25.102 |
| D3 | Native calibrated | .150031 | .127401 | .139516 | 7.239 | 25.103 |
| D3 | I / one | .288240 | .266287 | .260166 | 3.123 | 2.981 |
| D3 | I / bytes | .287301 | .265933 | .259194 | 19.516 | 25.100 |
| D3 | R / one | .285732 | .250068 | .259478 | 18.417 | 2.981 |
| D3 | R / bytes | .289924 | .252920 | .258265 | 59.769 | 25.100 |
| D4 | Native fixed | .285680 | .184796 | .271750 | 9.596 | 50.555 |
| D4 | Native calibrated | .186919 | .161053 | .201437 | 9.522 | 50.564 |
| D4 | I / one | .281216 | .254721 | .290774 | 4.282 | 5.796 |
| D4 | I / bytes | .280844 | .255369 | .290196 | 29.296 | 50.555 |
| D4 | R / one | .371779 | .539241 | .408952 | 28.227 | 5.796 |
| D4 | R / bytes | .371428 | .540944 | .411905 | 88.744 | 50.555 |
| D9 | Native fixed | .401393 | .230121 | .507303 | 60.921 | 90.523 |
| D9 | Native calibrated | unavailable | unavailable | unavailable | fit timeout | -- |
| D9 | I / one | .493795 | .418158 | 1.175964 | 30.118 | 9.780 |
| D9 | I / bytes | .494341 | .417851 | 1.179365 | 198.364 | 90.519 |
| D9 | R / one | 1.579328 | 2.590361 | 3.299326 | 152.852 | 9.780 |
| D9 | R / bytes | 1.227665 | 1.642529 | 3.127440 | 510.979 | 90.519 |

The combined JSON preserves every row's originating date. Identical native and
baseline binary hashes AND all 11 linked-library hashes allow numerical
comparison across the two days. Timing remains a single realization per run,
not a statistical performance benchmark. One-sweep iMESA is cheaper and faster
than native. The deliberately byte-filling schedule is not its only baseline.
Our D2/D3 traffic is about 7.22/8.42 times the one-sweep arms; superiority at
equal traffic must be discussed against the byte-ceiling rows, with fit costs
separate. None of these rows is a complete end-to-end time comparison.

D9 is a useful baseline failure case, not a win for the absent candidate:
repaired riMESA retains maximum shared-copy disagreement of 7.819/5.460 m
(one/bytes), versus iMESA 1.126/1.123 m and native zero by construction.
The byte schedule improves riMESA but does not close that gap. This does not
identify a unique kernel failure or prove all robust consensus methods fail.

## Calibration Cost and Generalization Boundary

| Data | First connected prefix (s) | Fit time (s) | Fit dimensions (rows / columns) | Outcome |
| --- | ---: | ---: | --- | --- |
| D2 | 21.877 | 7.341 | 1436 / 778 | rank five, no numerical boundary |
| D3 | 67.764 | 544.712 | 6028 / 3130 | rank five, no numerical boundary |
| D9 | 131.723 | 600.016 | not completed | watchdog timeout |

The D9 prefix contains 2027 pose variables and 2200 object observations across
robots. Native prefix training finished in 0.217 s; the subsequent dense
calibration process did not. This localizes the observed scalability failure to
calibration, not suffix replay or prefix state optimization. No timing profile
yet proves which QR/contrast/REML sub-operation dominates. No longer timeout,
alternative fit or another dataset's scales was substituted after seeing results.

Fit time is subprocess wall time, not isolated algorithm microbenchmarking:
some offline fits overlapped checkpoint diagnostics/tests. Online comparison
arms did not overlap heavy research work. D3's nine-minute startup still rules
out the claimed immediate deployment regardless of a fast seven-second suffix.

D2/D3 causal gains persist under the SAME fixed common-prefix registration:
.343323 -> .232779 m and .313575 -> .179802 m. All five robot groups improve
on each dataset under both pooled-causal and common-prefix conventions. All
nonempty 30-second bins improve under those conventions as well. These are
posthoc correlated subgroup descriptions, not independent confirmatory trials.
D1's regression remains, and D4's common-prefix gain was previously tiny.

No injected high noise, false identities or abrupt noise changes are added this
round. The retained three-seed high-Gaussian-noise D2 negative is NOT repaired
by a positive result on original D2 measurements. Model rank does not certify
physical covariance, correct conditional means or calibrated future uncertainty.

## D1: Same Arrived Data, More Complete Optimization

At 30, 60 and 90 s, replay both native weights with the unchanged online work
budget, freeze the arrived graph, then complete LM100 and GN100 from that exact
state. No future measurements, parameter refit or GT enter these solves. All
six truncated causal histories exactly match the original replay prefixes;
reconstructed physical objectives match the online graph before completion.

| Time | Weights | Trajectory online -> LM | Objects online -> LM | Latest five poses, fixed-prefix frame, online -> LM |
| --- | --- | --- | --- | --- |
| 30 s | fixed | .260023 -> .249596 | .293491 -> .273693 | .331226 -> .315656 |
| 30 s | calibrated | .232605 -> .215241 | .274116 -> .245372 | .272068 -> .255494 |
| 60 s | fixed | .279987 -> .285568 | .329303 -> .292669 | .371820 -> .401134 |
| 60 s | calibrated | .429735 -> .229520 | .765820 -> .239738 | .697690 -> .410681 |
| 90 s | fixed | .295045 -> .294764 | .229115 -> .228388 | .353072 -> .485094 |
| 90 s | calibrated | .195542 -> .171107 | .284805 -> .122356 | 1.049023 -> .472004 |

At 60 s, calibrated R5's latest fixed-frame error changes 1.416512 -> .364842 m.
The calibrated physical cost changes 773.535186 -> 751.201679, a 2.973% gap
relative to completion. Thus a modest objective gap can coexist with a much
larger state-error gap. Do not use near-final cost alone as a tracking guarantee.

Across all six checkpoints, LM/GN maximum position difference is 2.03e-7 m,
yaw difference 4.01e-8 rad and absolute cost difference 1.26e-12 (relative
difference 4.38e-15). Maximum
posthoc gradient is 3.58e-5. Agreement supports a local completion diagnostic,
not a global-optimum certificate or nonlinear posterior validation.

Interpretation: future data are NOT necessary to correct much of the t60
calibrated state error. Finite-work tracking matters in addition to the
previously diagnosed model/uncertainty issues. It is still unproven that this
explains all D1 causal damage. Changes in conditioning are a hypothesis, not a
measured cause. Completing fixed weights can worsen GT despite reducing the
physical loss. The table's latest five poses are not the full causal trajectory;
central LM/GN are not matched-budget distributed competitors.

## Verification and Repository Hygiene

- 17/17 new launched online replays complete, plus one explicitly unavailable
  D9 calibration arm. Together with October 29: 29 successful replays, not 30.
- Ten historical final/causal state files reproduce byte-for-byte. Five native
  terminal physical-objective checks have maximum relative error 6.35e-16.
  Frame support, information, control bytes and causal ceilings pass checks.
- 18 D1 diagnostic replay/completion calls finish. Their computation and the
  five terminal audits are excluded from online accuracy/timing comparisons.
- Python regression: 198 passed, zero failures/skips, 20.728 s. Native C++:
  62 passed. Schur residual 3.25e-16, matrix-free product error 3.54e-17;
  unit PCG six iterations, residual 1.98e-16. Multi-robot fixtures use
  12/16/23/39 iterations for 2/3/5/10 robots. MPI ranks: zero; no CCI/D-CCI edit.
  These tight-fixture numbers do NOT describe the one-PCG online budget:
  D2/D3 calibrated maximum relative residuals are 2.335/2.635.
- Removed a superseded aborted output, unused truncated download, obsolete
  endpoint-only patch and regenerable caches. Main net removal is about
  136 MiB before caches. Exact numerical failure/source evidence remains in a
  513566-byte archive. See [cleanup manifest](cleanup_20261030.md).
- Valid negative results, active dependencies, useful tests and immutable old
  result ledgers remain. New valid experiment outputs naturally consume space;
  cleanup is not a claim that total research disk usage decreased this round.
- Production DBA-DRANPP is unchanged. Research worktree is intentionally not
  git-clean: pre-existing uncommitted research code is not discarded.

Artifacts under `results/`:
`dranppr_extended_calibration_20261030`,
`dranppr_extended_online_baselines_20261030`, and
`dranppr_checkpoint_tracking_20261030`.
The online archive includes `five_dataset_comparison.json`, `verification.json`,
test XML, final source snapshots, exact validation commands and SHA256SUMS.
The calibration archive retains the D9 timeout; the October 29 archive is
unchanged. Neither calibration failure nor causal regression is hidden by cleanup.

## Next Minimal Research Step

First, test a weight-consistent finite-work solve/preconditioner on frozen D1
failing checkpoints at the SAME communication and work budget. Only after it
improves actual step residual/state tracking should it enter a new causal replay.
Second, replace the dense calibration bottleneck with a numerically equivalent
sparse/eliminated formulation and verify against the existing small QR cases
before attempting D9 again. Neither step needs extra gates or noise parameters.

The working hypothesis is now more precise: useful noise evidence must both
reweight measurements appropriately AND remain solvable under online budgets.
This is not yet a complete adaptive-robust algorithm. Continuous decentralized
learning, uninspected/full missions, retained high-noise failures, outlier and
association robustness, latency/memory bounds and ObjectPose3 transfer remain open.
