# Known Added Variance: A Mixed State-Quality Result

## Purpose

Test whether knowing the artificially injected Gaussian bearing variance is
enough to repair the previous noise-stress failure before implementing an online
noise learner. The answer is mixed: causal trajectory error improves slightly,
but final trajectory and object errors worsen substantially. Do not promote
scalar variance adaptation on the basis of this experiment.

This is an attribution experiment, not a new robust/adaptive algorithm. The
previous D2 stream, corruption realization, prefix, topology and optimizer
settings are retained. No variance sweep or ATE-based parameter selection occurs.

## Implementation and Fixed Protocol

The only estimator-side change is a diagnostic replay option:

```text
--suffix-added-bearing-variance 0.0225
```

It changes only newly inserted suffix bearing factors from variance 0.05^2 to
0.05^2 + 0.15^2 = 0.025 rad^2, giving SD 0.158113883 rad. The 0.15 value is the
predeclared injection SD, not a fitted parameter. Warm-prefix factors remain
nominal. Range, odometry, graph, identities, initialization and PCG budget stay
unchanged. No factors are removed or retroactively reweighted. Zero is the
default; the summary explicitly states that the parameter is externally
supplied and not learned. No production DRAN++ or library algorithm changes.

On the Gaussian-corrupted stream this is oracle knowledge of the *added noise
component only*. It is NOT knowledge of the actual original MR.CLAM covariance,
odometry bias, correlations, or truth state. On the original stream, the same
value is deliberately incorrect inflation, not an oracle. The variance addition
uses independence of the artificial perturbation; the wrapped bearing residual
retains the existing local Gaussian approximation.

Three predeclared cases each run nominal/added/added/nominal in serial ABBA order:

1. Original D2, lazy refresh: negative control for universal downweighting.
2. Gaussian-corrupted D2, lazy refresh: primary known-added-variance diagnostic.
3. Gaussian-corrupted D2, ordinary refresh: test dependence on refresh policy.

All 12 processes finish successfully within the 600 s per-process watchdog.
All use `joint_cached_one`, fixed chain 1-2-3-4-5, a common past cutoff of
21.877000093460083 s, and global elapsed end time 120 s. Each has 1222 suffix
frames, 597 prefix factors and 3538 final factors. All initial losses are exactly
131.69952464379733. Bootstrap preparation is excluded equally. GT evaluation
begins after all 12 estimator processes terminate. Two timing repetitions are
not independent data/noise samples.

## State-Quality Results

Nominal -> added variance, units m. These use the established one-global-SE2
alignment per final/causal trajectory, not per robot.

| Case | Causal trajectory RMSE | Final trajectory RMSE | Final object RMSE |
| --- | ---: | ---: | ---: |
| Original, lazy | 0.296666 -> 0.363259 | 0.311673 -> 0.413695 | 0.244900 -> 0.299087 |
| Gaussian, lazy | 0.402228 -> 0.386158 | 0.345087 -> 0.424311 | 0.273408 -> 0.311222 |
| Gaussian, ordinary | 0.401273 -> 0.388407 | 0.346063 -> 0.424248 | 0.272312 -> 0.311504 |

On noisy data, causal improvement is 4.00% with lazy refresh and 3.21% with
ordinary refresh. But final trajectory error increases 22.96%/22.59% and object
error increases 13.83%/14.39%. The ordinary-refresh control reproduces the
direction of the effect, so it is not explained simply by the lazy policy.
Both controls still use one PCG iteration per frame; this does not rule out
finite-response error shared by both.

On original data, indiscriminate inflation worsens causal, final trajectory,
and object errors by 22.45%, 32.73%, and 22.13%. Thus universal bearing
downweighting is not supported. The positive noisy causal result is also far
from restoring original-data quality (0.386 m versus 0.297 m).

### Alignment and Temporal Checks

A posthoc sensitivity check fits one transform using only the identical common
past, then freezes it for every suffix and every method. No suffix estimate
participates in this alignment fit; the transform is evaluation-only.

| Case | Frozen-prefix-alignment causal RMSE, nominal -> added, m |
| --- | ---: |
| Original, lazy | 0.343323 -> 0.401979 |
| Gaussian, lazy | 0.458600 -> 0.439138 |
| Gaussian, ordinary | 0.456628 -> 0.438521 |

The small causal benefit and original-data harm persist, so they are not merely
an artifact of independently aligning each suffix. Equal-frame-count quarters,
without re-alignment, show that the noisy lazy run worsens in Q1
(0.363680 -> 0.413994 m) but improves in Q4 (0.523371 -> 0.445922 m). The benefit
is not uniform over time. Quarter splitting is posthoc diagnosis, not an event
gate or a rule for changing the variance.

The final-error degradation is not restricted to old prefix poses. Under each
run's single final whole-trajectory alignment, Gaussian/lazy prefix RMSE rises
0.312250 -> 0.353367 m (247 poses), and suffix RMSE rises
0.351352 -> 0.437254 m (1222 poses). Their pooled squared errors reproduce the
reported final RMSE. Causal publication quality and later historical-state
quality are distinct requirements; neither should be substituted for the other.

## Computation and Audits

| Case | Nominal service s | Added service s | Nominal/added application MB |
| --- | ---: | ---: | ---: |
| Original, lazy | 7.073-7.131 | 7.437-7.531 | 24.152680 / 24.246312 |
| Gaussian, lazy | 7.032-7.041 | 7.250-7.312 | 24.138344 / 24.171048 |
| Gaussian, ordinary | 24.379-24.596 | 23.999-24.020 | 24.067560 / 24.067560 |

The lazy implementation remains about 3.3x faster than ordinary refresh with
the added variance on the noisy stream, but this is not a statistical robustness
result. Comparisons hold information and interface iteration count fixed, not
exactly bytes or computation. Model parameters are static experimental inputs;
an adaptive algorithm's estimation/communication cost is absent and cannot be
claimed free.

Lazy refresh requests change 32 -> 40 on original data and 29 -> 41 on noisy
data. Non-updating frames change 5 -> 12 and 4 -> 6; all are isolated and all
requests are followed by a successful full step next frame. There is no terminal
pending refresh. These are optimizer diagnostics, not observation-validity
scores. Every accepted step decreases its own current weighted physical loss.

Changing weights changes that objective. For example, the noisy lazy terminal
loss changes 6512.57 -> 1544.90, but those numbers refer to different objectives
and cannot be used to claim a 76% optimization improvement. This report ranks no
method by cross-covariance weighted loss. No calibrated predictive NLL, MSE
optimality, nonlinear stationarity, or asymptotic consistency is established.

## Interpretation and Next Decision

The experiment supports only a limited causal benefit from using the known
added variance. It does not pass the broader end-to-end state-quality gate.
The main lesson is not that variance addition is mathematically wrong. It is
that specifying one noise component correctly is not equivalent to having the
correct complete state-estimation model.

Downweighting suffix bearings changes their balance against range, odometry,
and the retained prefix. Remaining baseline covariance mismatch, bias and
correlation, finite nonlinear response, weakly constrained directions, and
single-realization variability remain competing explanations. Even a fully
correct stochastic model would not promise lower error on every realization.
These results do not identify any one explanation uniquely. The previous
covariance-learning transfer failure is therefore not resolved by this oracle.

**Decision:** keep this option diagnostic and keep the default unchanged. Do not
implement a residual-driven scalar variance learner or add another refresh gate
as the next fix. First freeze the identical terminal graphs under nominal and
added covariance and compare sufficiently optimized states while preserving
each factor's actual covariance. That separates an inferior finite-budget
response from an inferior optimum of the assumed model. A completion solve
must not silently reset all factors to nominal noise or become uncounted online
work. Only after this attribution should joint odometry/observation calibration
and its identifiability be reconsidered.

Sparse contamination remains a separate unresolved problem: knowing its second
moment does not give the correct contamination likelihood. No sparse-error
oracle, robust baseline rerun, additional sequence, or independent noise seed
was added in this stage.

## Verification and Reproduction

62 C++ and 119 Python tests pass. New integration checks verify analytic
first-frame bearing loss scaling for both normal and conditional replay,
unchanged prefix loss/factors, zero/default byte parity, future-mutation
causality, invalid/duplicate values, and required common-past scope. New audit
tests reject changed information while allowing different weighted losses.
The posthoc final prefix/suffix decomposition reproduces the full final RMSE.

All three nominal controls are byte-identical to the previous archived runs
for final states, causal states and control traffic. All repeats have the same
byte identity. Thus the new CLI does not silently alter default results.

Native unit-test fixed-gauge step error is 1.13e-16, full residual 3.25e-16,
and routed PCG relative residual 1.98e-16 after six iterations. Experiment runs
use exactly 1222 PCG iterations (one per frame); maximum reported relative
residual reaches 2.63, so they are deliberately inexact, not converged linear
solves. MPI: 0 ranks. CCI/D-CCI and production DBA-DRANPP remain unchanged.

```sh
cmake --build /tmp/dranppr_backend_build --target native-joint-replay -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_oracle_variance --output NEW_OUTPUT --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.analyze_oracle_variance --results NEW_OUTPUT --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_native_suffix_noise.py tests/test_audit_oracle_variance.py tests/test_analyze_oracle_variance.py
```

Archive: `results/dranppr_oracle_variance_20261003/`, with exact inputs, binary,
source snapshots, all commands and 12 outputs, `analysis.json`,
`common_alignment.json`, test logs/XML, provenance, and checksums. The original
experiment test snapshot is preserved; the final expanded test and posthoc
sources are archived separately. Full regression commands are in
`test_manifest.json`. No commit, push, or default-method replacement.

New replay SHA256:
`8c53e07cd5c23e2446595530f4bf13643e76fee85ebe2a2781dc5fb2100f032d`.
