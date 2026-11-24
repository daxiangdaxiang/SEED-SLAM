# Mean Evidence Audit Plan

> Continue the approved same-graph controlled investigation inline. Use
> executing-plans and experiment-craft; no agents, commits, production changes,
> bias-state implementation, new gate, or parameter search.

**Goal:** Determine what fixed-chart residual evidence can distinguish between
random variance, coherent range offsets, and state-equivalent discrepancies.

**Architecture:** Reuse archived D1/D4 prefix Jacobians and the existing
`LinearResidualEvidence`. Project physically labelled offset directions into
the residual space. Analyze rank and ambiguity, then run paired Gaussian
simulations in that exact residual space. This is an experiment, not an update
to the estimator or a claim about the true cause of real residuals.

**Tech Stack:** NumPy, SciPy SVD and F distribution, existing native export
adapter, pytest, JSON/CSV with SHA256 provenance.

## Frozen Protocol

- Source `results/dranppr_network_noise_transfer_qr_20261003`; verify SHA256SUMS.
- D1/D4 training prefix only, both nominal and original fitted five-vector
  covariance as separate assumed generative laws. Never refit or select laws.
- Use the exact same A, graph, labels and linearization. No GT or future data.
- Build one constant physical range-offset column per robot and per object.
  Object columns are phenomenological subject offsets, NOT moved landmarks.
- Verify raw robot-column sum equals object-column sum. Report ranks after
  removing state directions; never invert an unidentifiable joint bias model.
- For C' A=0, z=C' y, use qhat=||z||^2/d as the one-global-scale control.
  Compare qhat after projecting out the entire robot-offset subspace, with its
  lost degrees of freedom explicitly counted. This is NOT the five-component
  production fit and is not an automatic model selector.
- 4096 paired Gaussian draws, seed 20261027, per dataset/law. Cases: zero mean
  with variance multipliers 1 and 4; a nonzero A*delta mean; each robot and each
  object with a fixed +0.10 m or +0.30 m range offset. No result-based changes.
- Use an F statistic for concentration in the predeclared robot-offset
  subspace versus its orthogonal residual space. A 0.05 significance level is
  a simulation reference, NOT an estimator threshold. Valid calibration only
  for this fixed linear independent spherical Gaussian generative model.
- Report ALL simulated cases; do not label object offsets as robot faults.
  Use independent training/check Gaussian draws to evaluate plug-in predictive
  likelihood of scalar-only versus robot-mean+scalar models. Validation is in
  residual coordinates of the SAME fixed graph, not unseen real-world data.
- Original empirical residual projections are descriptive only: learned weights,
  nonlinear chart choice and unknown dependence invalidate simple F p-values.
- Budget: one process, one BLAS thread, 300-second outer watchdog. No nonlinear
  solver, PCG or MPI runs. Stop on failed input or numerical invariants.

## Tasks

- [x] Write tests for physical row mapping, gauge-like robot/object ambiguity,
  state absorption, projection/OLS parity, scale-invariant F, predictive NLL,
  invalid inputs and deterministic paired sampling; observe expected failures.
- [x] Implement `experiments/audit_mean_evidence.py` with no estimator changes.
- [x] Run focused tests, then the frozen audit; retain every case and all hashes.
- [x] Inspect results, write the report, and distinguish detection from cause
  attribution, identifiability from model correctness, and simulation from SLAM.
- [x] Rerun related Python regressions (76 passed), read back result hashes,
  and check production unchanged. Record commands and no new baseline/ATE claim.
- [ ] Native tests deferred: the temporary pinned build/dependencies are gone.
  Initial wider regression had one FileNotFoundError, not an algorithm failure.
  Do not execute archived binaries against a different system GTSAM as a fix.

## Reproduction

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  timeout 300 python3 -m experiments.audit_mean_evidence \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output results/dranppr_mean_evidence_20261027
```
