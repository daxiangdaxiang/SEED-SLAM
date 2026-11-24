# Fixed-Chart Mean Evidence and Noise Calibration

Date: 2026-10-27. Research branch only; no estimator or production change.

## Decision

The new experiment supports inspecting the *direction* of residual evidence,
not only its average magnitude. It does NOT support immediately adding one
bias state per robot or treating a structural score as a fault classifier.
Three findings change the design constraints:

1. A variance fit can normalize average residual energy while leaving a strong
   concentration in coherent mean-error directions.
2. Even an identifiable five-robot offset family can be weakly observed. Most
   of an injected range offset can be absorbed by state/map changes.
3. A correct extra mean family helps strong, observable injected means, but
   harms prediction when the true mean is zero or too weak. More model freedom
   is not automatically a safer adaptive estimator.

No new ATE/RMSE improvement, robust-baseline win, or online adaptation result
is claimed here. The prior D2 high-noise negative result remains unresolved.

## Purpose and Frozen Setting

Use the same archived D1/D4 prefix Jacobians as the previous noise-calibration
study, after verifying all 360 source hashes. D1 has a 1800-by-966 Jacobian and
834 residual degrees of freedom; D4 has an 847-by-475 Jacobian and 372 residual
degrees of freedom. The five-robot chain and declared object identities stay
fixed. No ground truth or future measurements enter this experiment.

Two separate assumed generative covariance laws are tested: nominal weights,
and the original fitted five-vector treated as KNOWN. The latter is a simulation
law, not proof that fitted real-world covariance is correct. The chart is frozen;
no nonlinear observation is generated and no nonlinear state is solved.

There are 4096 paired Gaussian training/check draws per graph/law, seed
20261027. Cases are zero mean with variance multiplier 1 or 4, a nonzero
state-equivalent discrepancy, and +0.10/+0.30 m range offsets for EVERY robot
and EVERY observed object. All 192 case/law rows are retained. These rows share
random draws and are not 192 independent real-world experiments. The verified
producer completed in 13.92 seconds under concurrent regression-test load;
this timing is not a runtime benchmark.

## What Is Being Measured

For a whitened, fixed linear model, write

    y = A delta + B beta + epsilon,    epsilon ~ N(0, q I).

`B` contains physically labelled constant range-offset directions. An object
offset here means a subject-specific additive measurement discrepancy, NOT a
geometric landmark displacement. Let `C` have orthonormal columns with
`C' A = 0`; then `z = C' y` removes linear state adjustments.

The scalar-only noise control estimates

    q_hat = ||z||^2 / d,
    E[q_hat] = q + ||C' B beta||^2 / d.

This is ONE global scale, not the existing five-component Q/R fit. It isolates
the inability of magnitude alone to distinguish residual mean from variance.
If a discrepancy lies in the column space of A, it disappears from z entirely.
That is an observation-equivalence example, not evidence that all sensor biases
are unobservable. Explicit nonzero A*delta vectors were checked numerically.

Let U span the five robot-offset columns AFTER state elimination. The diagnostic

    F = (||U' z||^2 / r) / (||(I-U U') z||^2 / (d-r))

has the F(r,d-r) null law under the stated independent Gaussian model. Uniform
unknown variance cancels. The 0.05 reference level is used ONLY for controlled
simulation size/power, never inserted as a gate in the backend. Object-offset
alternatives need not lie in U and cannot be labelled robot faults by this test.

For a second control, fit a mean in U and estimate scale on its orthogonal
complement, using d-r degrees of freedom. Evaluate both this model and the
scalar-only model on an independent Gaussian draw with the same fixed mean.
Scores are plug-in NLL per residual coordinate, not trajectory error or a
parameter-uncertainty-integrated forecast. Real temporal/geometry transfer is
not tested by this independent same-design control.

## Result 1: Global Normalization Hides Structure

Descriptive projections of the ORIGINAL empirical prefix residuals:

| Prefix, fitted covariance | Average contrast energy | Energy density in robot-offset subspace / complement |
| --- | ---: | ---: |
| D1 | 0.99999995 | 9.7799 |
| D4 | 1.00000001 | 10.1407 |

Thus an aggregate close to one does not imply isotropic, zero-mean residual
evidence. These are not valid real-data F significance tests: the weights were
estimated on these residuals, the chart is nonlinear/data-dependent, and the
actual correlation law is unknown. No hardware bias or offending robot is
identified by this table. The apparent global calibration is itself partly a
consequence of the fitted model's scale score equation.

## Result 2: Rank Is Not Sufficient Evidence Strength

The robot-only family has rank 5 for both graphs and both covariance laws.
Nevertheless, the fraction of squared offset-direction norm remaining after
state elimination is only:

| Assumed covariance | D1, range across robots | D4, range across robots |
| --- | ---: | ---: |
| Nominal | 0.048%-4.134% | 0.059%-2.085% |
| Fitted treated as known | 0.181%-7.301% | 0.101%-6.123% |

This is an energy fraction in whitened observation space, not a probability
or fraction of actual physical bias removed. It means a state adjustment can
explain most of these particular perturbations in the fixed chart.

Simulated detection frequency at the predeclared 5% reference level, using the
fitted covariance as the correct generative law:

| Graph / injected robot | +0.10 m | +0.30 m |
| --- | ---: | ---: |
| D1 / R1 | 5.62% | 10.74% |
| D1 / R2 | 22.66% | 98.95% |
| D1 / R3 | 48.24% | 100.00% |
| D1 / R4 | 37.94% | 100.00% |
| D1 / R5 | 16.94% | 94.82% |
| D4 / R1 | 7.40% | 27.66% |
| D4 / R2 | 9.50% | 58.94% |
| D4 / R3 | 5.22% | 5.52% |
| D4 / R4 | 12.65% | 80.54% |
| D4 / R5 | 32.54% | 99.90% |

Even with the correct Gaussian covariance supplied, the same 30 cm offset is
almost undetectable on D4/R3 but reliably detected on D4/R5. At 10 cm, most
cases are weak. For zero-mean noise, observed false positive rates are
4.96%-5.76%; quadrupling variance leaves the paired decisions EXACTLY unchanged.
Monte Carlo standard error near 5% is about 0.34 percentage points.

## Result 3: Detecting a Discrepancy Does Not Identify Its Source

On every range-observation row,

    sum(robot offset columns) = sum(object offset columns).

Consequently, adding a common value to every robot offset and subtracting it
from every object offset leaves the predicted observations unchanged. This
particular enlarged model has an exact parameter ambiguity even before any
state is eliminated. Fixing a parameter convention removes the algebraic
ambiguity, but cannot tell which family physically caused a common discrepancy.

After state elimination:

| Graph | Robot + object columns | Rank | Nullity |
| --- | ---: | ---: | ---: |
| D1 | 23 | 22 | 1 |
| D4 | 22 | 18 | 4 |

D4's three additional invisible directions correspond to object clusters
2, 4 and 15, each observed once in the prefix. Their range discrepancy is
absorbed by their free point coordinates; projected squared norms are below
7e-29. These cluster IDs are internal graph IDs, not benchmark subject labels.
Several other object-offset directions overlap substantially with the robot
subspace, so even a positive structural test is not a causal attribution.

## Result 4: Blindly Adding Mean States Is Not a Fix

The mean-plus-scale control improves independent-check NLL in only 19 of the
96 fitted-covariance case rows; it worsens the rest, including all zero-mean
controls. This count is descriptive, not a statistical win rate across datasets.
For zero-mean data, the added mean model costs approximately 0.0030 NLL per
coordinate on D1 and 0.0068 on D4. It estimates noise as a mean unnecessarily.

For fitted-covariance D4/R5 at +0.30 m, the scalar-only estimate rises to 1.1080
despite true random variance 1, while the mean-complement estimate is 0.9997.
The mean model improves independent-check NLL by 0.0440 per coordinate. At
D4/R3 +0.30 m, however, it worsens NLL by 0.0067. Removing the wrong or weakly
supported mean family cannot be justified by the large-offset success alone.

## Verification and Environment Limitation

- New tests first failed for the missing implementation. A further edge-case
  test caught numerical projection dust being treated as an observable mean;
  rank tolerances now use the ORIGINAL direction scale, not only the projected
  near-zero singular value. Real-graph ranks and reported conclusions did not
  change. First-pass outputs are retained separately.
- Focused Python regressions: **76 passed**, JUnit in the verified result
  folder. These include the nine new tests, prior linear evidence, variance,
  adaptation, range-attribution and frozen-weight-factorial checks.
- All 192 simulated mean scale estimates lie within 0.957 Monte Carlo standard
  errors of their analytic expectations. All 52 robot/null detection checks
  agree with the central/noncentral F reference within five Monte Carlo standard
  errors; maximum absolute frequency difference is 1.341 percentage points.
- Maximum C' A relative norm is below 7e-16. Across 2906 repeated numeric
  fields, the maximum absolute difference is 6.40e-14 and all simulated
  decision frequencies agree exactly. Details are in `repeat_verification.json`;
  floating SVD/BLAS output is not claimed to be bitwise identical.
- An initial wider test invocation returned 47 passed / 1 failed because
  `/tmp/dranppr_backend_build/bin/native-joint-audit` no longer exists. The native
  build and old pinned temporary dependency tree have been cleared. The archived
  binary currently resolves GTSAM from `/usr/local/lib` with TBB, unlike the old
  pinned no-TBB build. It was NOT used as a replacement. The native 62-test suite
  was not rerun, and C++/ABI parity is not newly certified.
- This audit executes Python matrix operations on checksummed exports only.
  No C++ solver, PCG iterations, MPI ranks, baseline, or new SLAM trajectory
  experiment is involved. Production DBA-DRANPP remains unchanged.

Commands and source snapshots are in
`results/dranppr_mean_evidence_verified_20261027`. Main reproduction:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300 \
  python3 -m experiments.audit_mean_evidence \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output results/dranppr_mean_evidence_reproduction
```

## Next Decision

Do not promote the F test or five bias states to the backend. First test whether
past-only coherent evidence persists across new geometry/time blocks, using a
fixed independent-check protocol and accounting for estimated noise parameters.
Only observable, predictive combinations justify expanding the mean model;
weak evidence must not be reinterpreted as proof of zero bias or precision.
Restore a reproducible pinned native build before the next nonlinear/online
experiment. D2 high noise, wrong associations, continuous adaptation, and
ObjectPose3 remain open. The present result narrows the modeling problem; it
does not close the robust-online claim.
