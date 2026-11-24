# Network Noise Support Diagnostic Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, inline. Keep research branch; no commits, push, subagents or production edits.

**Goal:** Test whether network shared-object constraints improve the evidence for past-only noise calibration and subsequent state estimation.

**Architecture:** Export the existing native physical Jacobian after completing the common prefix. Reuse classical fixed-linear REML with either shared or split object columns, then apply the fitted covariance to future factors in the existing frozen-completion harness. This is offline statistical attribution, not distributed online calibration.

**Tech Stack:** Native GTSAM audit, existing NumPy/SciPy ContrastVarianceModel, JSON packages and pytest.

## Predeclared Scope

D2 common prefix and the four streams from the frozen-noise-completion archive: original and Gaussian seeds 20261003--20261005. Fit once on the original common prefix only. Five robot-shared multipliers: odometry longitudinal/lateral/yaw, bearing, range. No distance-dependent term or new prior/floor. Keep the existing REML numerical log bounds [-16,10], explicitly report boundary fits rather than interpreting them as supported precision.

Four arms: nominal; joint observation-only REML with Q fixed; joint five-component REML; independent-map five-component REML with all object copies split. The latter keeps the same states, factors, and row weights, but relaxes network equality columns; all non-root poses remain free. Its contrasts differ from joint contrasts, so do NOT compare their raw NLL values as model preference scores.

Apply each fit to suffix factors only, prefix unchanged. In injected streams add the known 0.0225 rad^2 after multiplying the nominal bearing variance. This is an oracle-added-component diagnostic, not detection of a noise change. Use each stream's existing nominal endpoint for every completion. Sixteen LM solves plus eight GN checks on original/seed20261003, all capped at 100 iterations/600s. Fit timeout 600s per arm. No selection by ATE. Ground truth loaded only after estimators finish.

## Tasks

- [x] Write failing `tests/test_network_noise_support.py` for native optional export, exact residual/gradient parity, default-output equality, shared/split column mapping and suffix covariance updates.
- [x] Add optional `--linearization` to `experiments/native_local/joint.cpp`; export whitened per-factor blocks, b, key identities, robot and factor type, excluding only the root's fixed columns. Call after state optimization; do not feed exports back to native solver. Reject other extra flags.
- [x] Add `experiments/network_noise_support.py` to assemble matrices, reuse existing REML, and update v2 suffix covariances. Verify `0.5*b^T*b == final_cost` and `||A^T*b|| == posthoc_gradient_norm`. Fixed Q uses base variance 1 on odometry rows; other group entries are indicator columns. Reject nonfinite/nonpositive multipliers.
- [x] Implement/run bounded `experiments/audit_network_noise_support.py`, snapshot all sources/binaries/packages, record failures and boundary estimates. Verify nominal completions reproduce previous same-objective results before evaluating fitted arms.
- [x] Run regression tests, review code and report model assumptions, calibration support, physical state results and unresolved transfer/online boundaries. D1/D4 remain required subsequent transfer gates; this D2 diagnostic does not supersede their prior negative/positive records.

## Post-Selected Attribution Extension

After the primary 24 solves showed a joint Q/R gain over R-only and split calibration, add Q-only interventions without refitting: keep the first three fitted multipliers from joint_qr or split_qr, reset bearing/range multipliers to one. Preserve known injection variance and all other contracts. Run four data cases x two interventions in LM, plus GN for original/seed20261003: 12 solves. This is an explicit post-selected mechanism audit, not additional independent validation or an ATE-selected covariance candidate. Keep it in a separate archive with a link to primary fits.

Completed: all 40 processes exited normally (one prefix solve, three fits, 36 terminal solves). 62 C++ and 139 Python tests passed. Report: `docs/dranppr_network_noise_support_report_20261003.md`. No promotion to online adaptation or production default.
