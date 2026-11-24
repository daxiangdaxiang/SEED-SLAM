# Noise-Parameter Boundary and Prior Audit

## Question and Scope

The preceding chronological audit failed on Dataset 1 after a six-component
REML plug-in fit. Does integrating noise parameters automatically avoid that
overconfidence? This diagnostic rejects one tempting implementation: integrating
uniformly over log variance with a very small numerical lower bound.

Use the same first 30 seconds, fixed local linearization, and fitted nuisance
components as the preceding experiment. Select Dataset 1 Robot 5 (failed
transfer) and Dataset 4 Robot 3 (positive case). Neither ground truth nor future
observations enters this calculation. These are post-selected diagnostics, not
additional independent validation data. No noise estimate is fed into a solver.

## Why a Numerical Bound Can Become a Hidden Prior

For one nonnegative variance multiplier q, write the residual-contrast model as

```text
z ~ N(0, S(q)),    S(q) = S0 + q K.
```

The other five components are fixed at their fitted values in S0. If S0 is
positive definite, the Gaussian likelihood L(q) approaches a finite positive
L(0) as q approaches zero. A log-uniform prior is pi(q) proportional to 1/q.
Consequently the integral of L(q)/q diverges at zero. A finite integration range
will produce numbers, but those numbers depend on its lower cutoff; reducing
the cutoff is not evidence of a more precise sensor.

This is a direct derivation for the present fixed-linear model, not a new
statistical result. The analogous issue and lower-support sensitivity for
hierarchical variance parameters are discussed in
[Gelman (2006), Section 4.2](https://sites.stat.columbia.edu/gelman/bayescomputation/Gelman2006.pdf).

A one-dimensional conditional Jeffreys reference density instead uses

```text
I(q) = 0.5 tr[S(q)^(-1) K S(q)^(-1) K],
pi_J(q) proportional to sqrt(I(q)).
```

It is finite near zero when S0 is positive definite and K is nonzero. It is
not generally 1/q for an additive variance component. In log-q quadrature,
the log density is -NLL(q) + log pi_J(q) + log q; the last term is the
change-of-variable Jacobian. Omitting it would change the model.

The comparison below is conditional on five fitted nuisance values. It is
neither a profile likelihood nor a marginalized six-dimensional posterior.
Separately computed one-dimensional Jeffreys densities must not be multiplied
and silently called the joint Jeffreys prior.

## Measured Lower-Cutoff Sensitivity

Use 4,097 equally spaced log-q points on [-64, 0]. Truncate that same grid at
-16, -32 or -64. The upper bound q=1 is fixed for this diagnostic only, not a
recommended prior support or deployed covariance bound. q multiplies the
nominal bearing variance 0.05^2 rad^2; it is not itself a variance in rad^2.

The minimum eigenvalue of S0 is 0.0123329 on D1/R5 and 0.000130178 on D4/R3.
Both satisfy the stated nonintegrability condition numerically. Their NLLs
approach finite limits 4.79344 and 85.04522 respectively at log q=-64.

| Case | Density | Median, lower=-16 | Median, lower=-32 | Median, lower=-64 |
| --- | --- | ---: | ---: | ---: |
| D1/R5 | Log-uniform | 4.95078e-5 | 1.65745e-8 | 1.86522e-15 |
| D1/R5 | Conditional Jeffreys | 0.0187489 | 0.0187488 | 0.0187488 |
| D4/R3 | Log-uniform | 7.36437e-5 | 5.84565e-8 | 6.58348e-15 |
| D4/R3 | Conditional Jeffreys | 0.000581901 | 0.000581777 | 0.000581777 |

The data are unchanged. In the log-uniform calculation, moving a numerical
cutoff makes the apparent median precision change by many orders of magnitude.
This happens even in the previously successful D4 case, whose fitted bearing
component is interior rather than on the optimizer boundary.

At the widest tested interval, conditional Jeffreys means are 0.0364696 and
0.00163231. These are diagnostic conditional summaries, not proposed replacement
noise values. Stability to this lower cutoff proves neither joint posterior
propriety, useful predictive calibration, upper-tail insensitivity, nor improved
state accuracy.

## Do Not Overstate Nonidentifiability

Both six-component contrast models have structural rank six. Normalize the
expected Fisher matrix in variance coordinates by its diagonal to examine
similarity of covariance responses without the vanishing log-q derivative.

| Case | Residual degrees of freedom | Smallest normalized Fisher eigenvalue | Largest |
| --- | ---: | ---: | ---: |
| D1/R5 | 32 | 0.160773 | 2.352129 |
| D4/R3 | 120 | 0.324673 | 1.980789 |

These matrices are not algebraically rank deficient. D1's bearing/yaw-odometry
response correlation is 0.838601, whereas D4's is 0.0208803. The local
nuisance-adjusted bearing-information fraction, the reciprocal of the bearing
diagonal of the inverse normalized Fisher matrix, is 0.288765 and 0.714449.
These are local geometry diagnostics, not coverage-calibrated uncertainty
intervals at a boundary.

Therefore do not claim that the D1 failure proves an absolutely unidentifiable
noise model. The stronger existing evidence is finite-data likelihood support:
after nuisance refitting, increasing D1 bearing variance about 442,000-fold costs
only 1.00882 training nats. Sparse calibration, nonlinear/model mismatch and
temporal dependence remain possible contributors. The present audit identifies
an invalid proposed prior, not the unique cause of the earlier online failure.

## Relation to Existing Research

Joint state/noise estimation is already established. Khosoussi and Shames
explicitly discuss unbounded joint maximum likelihood when empirical residual
covariance becomes singular, and use covariance constraints or a Wishart prior.
That singular-likelihood issue is related to covariance degeneracy but is not
identical to the additive-component improper-prior calculation above.
[Joint State and Noise Covariance Estimation](https://arxiv.org/html/2502.04584v2).

Wong et al. also model uncertainty over covariance parameters within variational
trajectory estimation. Parameter marginalization or inverse-Wishart learning
alone cannot be claimed as our novelty.
[Variational Inference with Parameter Learning Applied to Vehicle Trajectory Estimation](https://arxiv.org/html/2003.09736v2).

The research target remains deciding which evidence supports changing the state,
changing a shared precision model, or changing a relation, while retaining a
consistent, computationally feasible distributed online state estimator. This
audit rules out one apparently parameter-free shortcut; it does not establish
that target as solved.

## Reproduction and Runtime Status

Producer entrypoint: `experiments/audit_variance_boundary.py`. For example:

```sh
env PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 experiments/audit_variance_boundary.py \
  --run results/dranppr_noise_factorial_20260922/dataset1/run \
  --stream /tmp/dranppr_causal_streams_v1/dataset1 \
  --robot 5 --components 3 --output /tmp/dranppr_variance_boundary_new_run
```

Each output preserves the contrast matrices, calibration, grid, summaries,
source snapshots and producer/event hashes. The saved fit must reproduce the
original calibration before any new calculation is accepted.

| Producer attempt | Process exit | Status |
| --- | ---: | --- |
| D1/R5 first | 139 | Outputs written; abnormal shutdown |
| D1/R5 repeat with faulthandler | 0 | Normal completion |
| D4/R3 first | 139 | Outputs written; abnormal shutdown |
| D4/R3 GDB run | 1 | Debugger lost process/thread at shutdown; not a clean run |
| D4/R3 repeat with faulthandler | 139 | Outputs written; abnormal shutdown |

At the time of these five original attempts, the Python/system-GTSAM shutdown
fault was unresolved. No root cause was inferred from the inconclusive debugger
output, and no failed run was relabeled as successful merely because it wrote
results. A subsequent runtime audit now supplies two normal D4 reconstructions
with identical arrays and CSVs, using a process-scoped TBB allocator preload.
See `docs/dranppr_gtsam_runtime_audit_20260922.md` for the isolated failure,
controlled intervention and its limits. The original statuses above are
unchanged; the new runs resolve the numerical-reconstruction caveat for this
diagnostic without establishing a joint posterior or online improvement.

Two independent checks separate arithmetic from that runtime issue:

- In a fresh process importing only NumPy/SciPy and the audit's mathematical
  functions, the first/repeat NPZ arrays are bitwise equal and summary CSVs are
  byte-identical on each case. Recompute the distributions on 4,097 and 8,193
  grid points. The maximum relative change of the mean is 4.99e-11 on D1 and
  2.45e-15 on D4. The process confirms GTSAM was not imported and exits zero.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python3 -m pytest -q tests/test_variance_boundary_audit.py
  tests/test_dranppr_variance_components.py` gives 8 passed, exit zero. Tests
  include scalar likelihood/Fisher identities, an analytic integral mean, the
  log-coordinate Jacobian, cutoff sensitivity and invalid inputs.

This validates the conditional matrix calculation, not the native Python
integration or a Bayesian SLAM estimator. Both abnormal and normal artifacts
are retained under `results/dranppr_variance_boundary_20260922/`, with explicit
process statuses in `run_status.json`.

## Decision

Do not insert a variance floor, select the best cutoff by ATE, or promote this
conditional Jeffreys calculation to an online adaptation method. A future joint
uncertainty model must declare its prior or uncertainty-set assumptions, verify
boundary/tail behavior, and pass the same chronological D1/D4 gates without
per-dataset retuning. Native incremental DRAN development with fixed physical
noise can proceed independently after architecture review; it must not depend
on pretending that this failed plug-in adaptation is validated.
