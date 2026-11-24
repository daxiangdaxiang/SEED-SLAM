# riMESA Numerical Audit and Interpretation

## Scope

This audit separates implementation failures from limitations of a robust
statistical model. It is not a DRANPPR improvement or an end-to-end SLAM result.
The problem has two robots, known coordinate frames, and one shared Point2.
All noise/outlier scenarios here are controlled synthetic observations.

The unchanged pinned upstream sources remain clean. Repairs are applied only to
two isolated riSAM copies under `/tmp/dranpp_noise_audit/`. Neither packaged
DBA-DRANPP nor the DRAN++ solver implementation was changed.

Pinned sources:

- riMESA: `ffddc21dad530ba0b0e57c42495146b410581fc4`.
- riSAM-v2: `c6fa05eaece103efd7e44e640bda8df6eff30e8d`.
- Paper GTSAM fork: `03c263fe08b02c90a8861d36f77422765d39506d`.

Source references: [DoglegLineSearch](https://github.com/rpl-cmu/risam-v2/blob/c6fa05eaece103efd7e44e640bda8df6eff30e8d/risam/include/risam/DoglegLineSearch.h),
[ExtendedISAM2](https://github.com/rpl-cmu/risam-v2/blob/c6fa05eaece103efd7e44e640bda8df6eff30e8d/risam/src/ExtendedISAM2.cpp),
[GraduatedFactor](https://github.com/rpl-cmu/risam-v2/blob/c6fa05eaece103efd7e44e640bda8df6eff30e8d/risam/include/risam/GraduatedFactor.h).

## Failure 1: Dogleg Endpoint Cancellation

The unchanged paper-dependency smoke completed only 7 of 16 runs. GDB located
the first non-finite step inside `ComputeBlend`, reached at the Newton endpoint.
The captured Cauchy and Newton directions differ only at floating-point precision:

```text
u = [-0.0002548417886291125,  -0.00011734925498463766]
n = [-0.00025484178862911254, -0.00011734925498463767]
delta = norm(n) = 0.00028056226559737996
tau1 = NaN
```

The norm-to-square round trip can make `delta * delta < n.squaredNorm()` even
when `delta` was set to `n.norm()`. The code then enters the blend calculation
instead of returning the Newton endpoint. With almost equal directions, the
expanded quadratic coefficients suffer cancellation and produce NaN.

The isolated repair returns `n` when `delta >= n.norm()`, before squaring.
It introduces no noise parameter, damping parameter or numerical epsilon.

`experiments/rimesa_dogleg_endpoint_probe.cpp` fails on the captured point with
the unchanged header and passes with the repair. Four other checks cover the
gradient segment, blend segment, Newton segment and zero stationary direction.
All five pass after repair; the blend error is `2.22e-16`.

The exact original 16-run input/configuration smoke then completes 16/16.
An expanded audit also completes 672/672: six scenarios, eight seeds, two
covariance-reporting conditions and seven solver/consensus variants. This is
eight independent seeds per condition, not 672 independent datasets. Nominal
and oracle inputs coincide in scenarios where the declared noise is correct.
This tests the captured failure and these runs, not all possible Dogleg inputs.

## Failure 2: Inconsistent Line-Search Objective

Let `r` be a whitened observation residual and `rho(r)` a graduated kernel cost.
The local linear system has the gradient of

```text
sum_plain 0.5 * ||r||^2 + sum_graduated rho(r).
```

The upstream line-search function instead sums `sqrt(2 * factor.error(x))`.
For a plain Gaussian factor this gives `||r||`; for a graduated factor its
special error wrapper gives `rho(r)`. The resulting mixed merit is therefore
not the objective differentiated by the linear model.

The second isolated repair explicitly uses the kernel cost for a graduated
factor and ordinary `factor.error(x)` otherwise. It changes neither SIG weights,
graduation schedules, reported covariances nor distributed dual updates.

`experiments/rimesa_merit_gradient_probe.cpp` compares finite-difference merit
gradients with the actual factor-linearization gradient. The unchanged source
fails Gaussian and mixed tests with absolute discrepancies `0.9181` and
`0.9392`; the all-graduated test already passes. After repair, all three pass,
with maximum discrepancy `1.93e-11`.

The analytic Gaussian two-robot diagnostic, with observations at 0 and 0.2 and
equal precision, has common optimum 0.1. After 100 communications:

| Local implementation | Robot A | Robot B |
| --- | ---: | ---: |
| Unchanged upstream | 0.02 | 0.18 |
| Endpoint repair only | 0.02 | 0.18 |
| Endpoint plus merit repair | 0.10 | 0.10 |

This establishes a concrete cause of the Gaussian diagnostic's stagnation.
It does NOT explain every low-noise error of the full robust riMESA method.
In fact, across the expanded 96 paired `rimesa` runs, adding the merit repair
changes no output coordinate at any recorded step: maximum absolute change is
exactly zero. The same is true for `no_decay`; both use graduated observation
and consensus factors in this harness. The mixed `exact_consensus` ablation
does change, as expected. GN controls bypass this merit and remain unchanged.

Both repaired builds complete all 672 expanded runs without failures. They
remain explicitly labelled diagnostic variants, not unchanged official results.

## Statistical Interpretation After Numerical Repair

Mean final two-copy position error after 60 arriving observation pairs and 100
settling communication rounds, averaged over eight seeds:

| Scenario | Covariance supplied | Repaired riMESA (m) | Gaussian distributed control (m) |
| --- | --- | ---: | ---: |
| Nominal Gaussian | Correct | 0.02732 | 0.01531 |
| High Gaussian noise | Understated | 0.09454 | 0.06126 |
| Same high noise | Oracle correct | 0.07136 | 0.05492 |
| Isolated gross outliers | Nominal | 0.04014 | 0.16524 |

The Gaussian control is an explicit no-robustness, no-dual-decay, GN ablation,
not DRAN++. These rows are not equal-wall-time comparisons. The two methods
have the same observation and communication schedule, but their actual message
payloads and local compute are recorded separately. Batch WLS is only a final
objective reference and has access to the full stream.

The results preserve both sides of the tradeoff: robust estimation helps greatly
on gross contamination, while a fixed robust loss does not infer an unknown
sensor precision. With understated high-noise covariance, the post-hoc
chi-square surprise rate among genuine inliers is about 80.5%, versus 5.1%
with the oracle covariance. This is our explicit squared-Mahalanobis diagnostic,
NOT the official rejection count and NOT the fraction of discarded factors.
An oracle covariance is not a deployable adaptive estimate.

At 100 rounds, even the Gaussian control has nonzero copy disagreement.
Thus these errors combine statistical effects with finite distributed solving;
they cannot all be assigned to a permanent robustness bias. A longer fixed-data
settling audit is recorded separately to distinguish those effects.

### Longer Settling Does Not Fix Every Disagreement

The additional 160 runs use the identical observation streams, two scenarios,
eight seeds, two covariance conditions and five variants, but increase settling
from 100 to 2,000 communication rounds. All complete without numerical failures.

| Scenario / model | Variant | Error at 100 rounds (m) | Error at 2,000 rounds (m) | Copy disagreement at 2,000 (m) |
| --- | --- | ---: | ---: | ---: |
| Nominal / correct | riMESA, repaired | 0.02732 | 0.02732 | 0.03800 |
| Nominal / correct | Gaussian GN control | 0.01531 | 0.01403 | 0.01202 |
| High noise / understated | riMESA, repaired | 0.09454 | 0.09454 | 0.11050 |
| High noise / correct | riMESA, repaired | 0.07136 | 0.07136 | 0.07731 |
| High noise / correct | Exact consensus, robust observations | 0.05751 | 0.05656 | 0.000063 |
| High noise / correct | Gaussian GN control | 0.05492 | 0.04896 | 0.000276 |

For repaired riMESA, recorded coordinates do not change over the final 100
rounds in any of these runs. Removing dual decay alone also leaves substantial
disagreement (mean `0.08145 m` for correctly declared high noise); it does not
repair the whole mechanism. With ordinary consensus and robust observations,
disagreement in that condition nearly vanishes. This localizes a useful research
question to the treatment of consensus, not just the observation kernel.

There is also a finite-convergence effect: the Gaussian control nearly reaches
the batch WLS error `0.04896453 m` for correctly declared high noise, but still
has slow convergence for nominal low noise. The official Point2 penalty is 1
while accumulated observation precision grows as `count / sigma^2`; checking
that imbalance is more informative than equating 100 rounds with convergence.
This is a conditioning explanation suggested by the source and these results,
not a proven universal convergence-rate formula.

Not every exact-consensus variant is stable under understated high noise:
the GN ablation's mean final error rises from `0.12648` to `0.16816 m`, and its
largest coordinate change across the last 100 rounds is `0.42186 m`. Removing
robust consensus or switching solvers alone is therefore not a general solution.
All local fixed-point and global optimality claims remain bounded by these tests.

## Implications for DRANPPR

1. Fix and disclose baseline numerical defects before assessing a new method.
   Neither endpoint repair nor objective consistency is research novelty.
2. Keep the physical observation likelihood, local solver merit and noise-learning
   score consistent. Do not treat optimization failure as sensor degradation.
3. Distinguish known-covariance robustness from unknown precision learning.
   Do not remove robust protection merely because it loses on clean Gaussian data.
4. The native Dataset 1 transfer failure of our own point-estimated noise model
   remains unresolved. Repairing riMESA does not validate our adaptation.
5. A persistent incremental DRAN++ backend remains unimplemented. Existing
   `CachedTranslationSystem::matches` checks the entire fixed-size Q, and
   `QuadraticProblem` fixes n/d/r. Reusing identical-Q factorizations is not
   selective graph-growth factorization. `ManualObjectDran` still loads the
   full object graph; its event schedule controls communication, not factor arrival.

No superiority or completed-research claim follows from this audit.

## Artifacts and Reproduction

Results: `results/dranppr_rimesa_numeric_audit_20260922/`.

- `first_blend.log`: unchanged-source debugger trace.
- `endpoint_original.csv`, `endpoint_patched.csv`: red/green endpoint probe.
- `merit_original.csv`, `merit_patched.csv`: red/green objective-gradient probe.
- `consensus_endpoint*.txt`: the two repaired-build Gaussian diagnostics.
- `endpoint_smoke/`: exact 16-run repeat.
- `endpoint_expanded/`, `endpoint_merit_expanded/`: complete paired 672-run audits.
- `endpoint_merit_settle2000/`: longer fixed-data settling diagnostic.
- `source_snapshot/`: exact harness, probes and patches with verified SHA256SUMS.

The historical endpoint-only patch is retained under
`results/dranppr_rimesa_numeric_audit_20260922/source_snapshot/rimesa_endpoint_diagnostic.patch`.
Its superseded active-tree duplicate was removed on 2026-10-30. The combined patch is
`experiments/rimesa_endpoint_merit_diagnostic.patch`. Both pass `git apply --check`
against the pinned clean riSAM-v2 source. They are alternatives, not patches to
apply one after the other. Original failure artifacts were not overwritten.

Configure each isolated source using the same paper GTSAM dependency. Example
for the endpoint-plus-merit copy, after applying its patch:

```sh
cmake -S /tmp/dranpp_noise_audit/rimesa -B /tmp/dranpp_noise_audit/paper-endpoint-merit-build -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_RISAM=/tmp/dranpp_noise_audit/risam-endpoint-merit-diagnostic -DGTSAM_DIR=/tmp/dranpp_noise_audit/gtsam-paper-build -DGTSAM_INCLUDE_DIR=/tmp/dranpp_noise_audit/gtsam-paper
cmake --build /tmp/dranpp_noise_audit/paper-endpoint-merit-build -j2
cmake -S ../tools/robust_noise_audit -B /tmp/dranpp_noise_audit/paper-endpoint-merit-adapter -DCMAKE_BUILD_TYPE=Release -DRIMESA_SOURCE_DIR=/tmp/dranpp_noise_audit/rimesa -DRISAM_SOURCE_DIR=/tmp/dranpp_noise_audit/risam-endpoint-merit-diagnostic -DUPSTREAM_BUILD_DIR=/tmp/dranpp_noise_audit/paper-endpoint-merit-build -DGTSAM_DIR=/tmp/dranpp_noise_audit/gtsam-paper-build -DGTSAM_INCLUDE_DIR=/tmp/dranpp_noise_audit/gtsam-paper
cmake --build /tmp/dranpp_noise_audit/paper-endpoint-merit-adapter -j2
```

The audit harness commands and parameters are in each run's `metadata.json`.
For the second copy, `--upstream-root` is a directory of symlinks to the actual
linked sources, so its dirty-source record includes both repairs. The first
copy's metadata explicitly corrects the harness's default clean-source probe
and names its actual linked source and patch. All binary hashes are recorded.

The endpoint build passes both existing CTests. The merit build passes the
kernel/GN CTest and the new consistency probe. Its old `upstream-merit-probe`
deliberately expects the original bug and is not a passing repair test; do not
claim that the entire unchanged CTest suite passes for the merit build.
