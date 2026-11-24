# Historical Relinearization Cause Audit, 2026-10-03

## Decision

The previous online failure is not explained by one simple cause. Common rigid
motion contributes substantially on D9, but removing it does not eliminate
historical deformation. More importantly, the displacement trigger repeatedly
relinearizes factors whose cached residual prediction remains accurate at the
requested new linearization point.

This supports investigating model-accuracy-controlled inexact incremental
optimization. It does not justify freezing history, adding a larger displacement
threshold, promoting the conditional implementation, or claiming a new online
speedup. No solver policy was changed in this audit.

## Setup and Passivity

The same D4/D9 common-past inputs, suffixes, likelihoods, known identities,
topology, chart threshold `1e-4`, and PCG policies as the 2026-10-02 report were
used. D4 has 84 suffix frames through elapsed time 20 s; D9 has 75 through 136 s.
The four final audited runs are under
`results/dranppr_chart_cause_20261003/origin_check/`.

Each actual refresh transaction is observed before it changes the model.
`optimizer_refresh` and `identity_chart` coordination are recorded separately.
The audit sees only historical factors and the proposed relinearization values.
Its target is the transaction's new chart base: the old base with requested keys
replaced, not necessarily the complete current estimate on unrefreshed keys.

For all four cases, final-state CSVs, causal-state CSVs and control-message
JSONL files are byte-identical to the archived uninstrumented candidate runs.
Physical costs, accepted/rejected decisions, chart-refresh counts, factor
linearizations, PCG counts, and application bytes match exactly. Per-frame factor
incidence counts match the optimizer's existing counters; byte ledgers and
fixed-graph physical descent also pass.

Diagnostic calculation time is included in the instrumented service time, which
must not be used as a performance result. The audit does not transmit statistics
as part of the algorithm or use ground truth.

## Two Independent Coordinate Checks

For old bases theta and the proposed target x, the first diagnostic fits a common
SE(2) transform using all historical pose translations. It then compares x with
G theta. If pose translations cannot determine rotation, the pose yaw changes
provide the fallback. Position-fit energy refers only to squared translation
displacement; it does not combine meters and radians.

The second diagnostic uses the exact local first-pose coordinate change:

\[
G = X_0^{\mathrm{target}}(X_0^{\mathrm{base}})^{-1}.
\]

This is a coordinate diagnostic, not an added anchor measurement, an extra prior,
or an estimated identity. On the network-root robot, the fixed first pose gives
G = I. No actual root, state, or constraint is transformed by the solver.

The second check was added after self-review: fitting translations alone can
reduce trajectory-position discrepancy while worsening yaw/object linearization.
It is not a likelihood-optimal coordinate choice. Therefore the fraction of
translation energy explained by a fit must not be interpreted as the fraction
of model updates that can safely be skipped.

### Measured Coordinate Contribution

Counts below are repeated pose-refresh events, not distinct poses.

| Case | Pose translation energy removed by position fit | Raw pose triggers | After position fit | After first-pose coordinate change | First-pose trigger-count reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| D4, one PCG | 75.93% | 15,558 | 15,515 | 14,202 | 8.72% |
| D4, tight PCG | 85.83% | 12,891 | 12,409 | 9,844 | 23.64% |
| D9, one PCG | 84.43% | 152,696 | 152,669 | 81,284 | 46.77% |
| D9, tight PCG | 94.20% | 135,396 | 119,840 | 51,913 | 61.66% |

All counts use the unchanged existing componentwise `1e-4` test, excluding fixed
keys. This existing test mixes translation and angle components; we did not
recalibrate it. Alignments can move previously subthreshold variables above the
threshold, so the reported reduction is a net count difference, not an exact
set of safe-to-skip updates or a time saving.

The D9 one-step origin-frame counts by robot are informative:

| Robot | Raw triggers | Origin-frame triggers |
| --- | ---: | ---: |
| 1, network root | 33,897 | 33,897 |
| 2 | 28,714 | 5,594 |
| 3 | 29,447 | 28,935 |
| 4 | 33,170 | 8,853 |
| 5 | 27,468 | 4,005 |

Robots 2/4/5 have a large coordinate-motion contribution, but robot 1 and robot 3
retain substantial deformation. A single rigid correction is not a complete
replacement for local state estimation.

## Are the Old Models Actually Inaccurate?

For each touched historical factor, the cached whitened Jacobian predicts

\[
\widehat r_f(x)=A_f\delta_f-b_f,\qquad
e_f=\|r_f(x)-\widehat r_f(x)\|_2.
\]

The diagnostic compares this with the actual whitened nonlinear residual at the
requested new chart base. These errors are in units of the assumed sensor
noise, not meters, ATE, or ground-truth error. Histogram boundaries are reporting
bins only; none are used as acceptance or relinearization parameters.

| Case | Optimizer factor-refresh events | Events with e <= 0.01 | Events with e > 1 | Direct identity-chart factor refreshes |
| --- | ---: | ---: | ---: | ---: |
| D4, one | 37,172 | 99.3651% | 1 | 114 |
| D4, tight | 31,587 | 98.4519% | 1 | 114 |
| D9, one | 319,091 | 99.9787% | 6 | 9 |
| D9, tight | 289,197 | 99.6106% | 6 | 9 |

These are repeated evaluations of the same histories, not independent samples
or evidence of a statistical success probability. Direct identity-chart updates
account for 0.306%/0.360% of D4 and 0.00282%/0.00311% of D9 refresh events.
Thus their direct relinearization workload is small. This does not exclude
indirect effects of network coupling on later optimizer state changes.

The reference already refreshes aggressively. Small single-transition errors
may partly be a consequence of that policy; they do not establish that errors
stay small after many consecutive skipped refreshes. A persistent-cache trial
is still needed before claiming that these updates are unnecessary.

Large-error cases are retained rather than averaged away. D4's e > 1 event is
an odometry factor at epoch 15, robot 1. D9 has large optimizer errors at epoch 1
(robots 3/4) and epoch 72 (robot 2); the largest raw error is 9.22 in the tight
variant. Its initial identity-chart update also has three observation errors
above 1, with a maximum of 2.564. Even on these short suffixes, indefinitely
freezing all historical factors is not supported.

### Coordinate Transport Is Not a Universal Remedy

For native Pose2 local increments, a common left SE(2) action preserves the pose
tangent coordinates. Point2 tangent coordinates rotate. The audit transports
old observation columns by `A_point R^T` and leaves the old residual unchanged.
Deterministic pure-rigid-motion tests verify negligible transported prediction
error; they also check nonlinear physical gauge invariance and snapshot immutability.

On real refreshes, however, transport does not uniformly improve prediction:

| Case | Raw observation error RMS per residual dimension | After position-fit transport | After first-pose transport |
| --- | ---: | ---: | ---: |
| D4, one | 0.005465 | 0.179405 | 0.007003 |
| D4, tight | 0.005410 | 0.042498 | 0.006724 |
| D9, one | 0.004209 | 0.007759 | 0.004130 |
| D9, tight | 0.005695 | 0.011426 | 0.005882 |

The position-fit failure is consistent with optimizing a pose-translation metric
that ignores yaw and remote objects. The origin-frame check removes much of that
artifact but still does not dominate the raw predictor. Therefore neither
transport alone nor position normalization alone is selected as the new solver.

## Research Implication

The useful question is not only "did the state move?", but "is the cached model
still accurate enough for the optimization progress sought in this update?"
This is related to existing work, not by itself a novel contribution:

- iSAM2 already combines selective relinearization, incremental reordering and
  Bayes-tree updates. Our single-key conditional prototype is not an implementation
  of all those mechanisms. Its negative result cannot be generalized to iSAM2.
  [Kaess et al., iSAM2](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf).
- Inexact Newton methods relate linear-solve accuracy to nonlinear progress,
  rather than oversolving every intermediate model. Model agreement can inform
  adaptive forcing terms. This is relevant theoretical guidance, not a theorem
  for the current distributed stale-model solver.
  [Eisenstat and Walker, 1996](https://users.wpi.edu/~walker/Papers/forcing_terms,SISC_17,1996,16-32.pdf).

For the fixed physical graph in one update, define a cached model m and predicted
decrease D = m(0)-m(s). The elementary identity

\[
F(x)-F(x\oplus s)=D-E(s),\quad
E(s)=[F(x\oplus s)-m(s)]-[F(x)-m(0)]
\]

shows why model error should be compared with predicted progress, not only with
a fixed displacement. If D > |E(s)|, the true objective decreases. This is only
a one-step identity. The current physical-cost backtracking already tests actual
descent, so this observation alone is not an algorithmic innovation.

The next substantive problem is obtaining a cheap, local model-error control
that preserves useful conditional work, with finite communication and repeated
online updates. A candidate must address accumulated drift, fresh/late strong
observations and recovery after a poor model, not only small average errors.
The pointwise residual audit does not bound gradient/Hessian error or establish
stationarity. In particular, noisy least squares generally has nonzero residual
at its optimum; Newton root-finding theorems for F(x)=0 cannot simply be applied
by substituting the measurement residual for F.

Proposed next experiment, not yet implemented: compare a model-accuracy-driven
refresh policy against the unchanged threshold policy using persistent bases,
the same physical likelihood, and strict matched communication/time accounting.
Refresh work should be tied to model failure rather than a manually enlarged
displacement threshold. Start with the observed large-error epochs and delayed
measurement recovery; only then extend to full streams. Do not add separate
heuristic gates for each dataset. Coordinate-aware bookkeeping may help, but
the present audit does not establish that it is necessary or sufficient.

## Verification and Archive

Implementation is confined to experimental observer code and tests:
`experiments/native_local/chart_audit.h`, `joint_step.h`, `joint_replay.cpp`,
`experiments/audit_native_chart_refresh.py`, and focused regression tests.
No production solver, likelihood, CCI/D-CCI routine, or default policy changed.

59 C++ tests and 74 Python tests passed. New tests first failed for missing
diagnostic implementation/flag and then for the absent independent origin-frame
check. Final runs verify passivity against the pre-audit executable's archived
outputs, not only against another instrumented run.

```sh
cmake --build /tmp/dranppr_backend_build --target testIncrementalLocalModel native-joint-replay -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_native_chart_replay.py \
  tests/test_audit_native_chart_refresh.py tests/test_native_matched_online.py \
  tests/test_native_conditional_replay.py tests/test_audit_conditional_replay.py \
  tests/test_native_conditional_audit.py tests/test_native_joint_audit.py \
  tests/test_native_joint_replay.py tests/test_native_interface_support.py \
  tests/test_audit_native_online.py
python3 -m experiments.audit_native_chart_refresh \
  --binary /tmp/dranppr_backend_build/bin/native-joint-replay \
  --baseline-root results/dranppr_nonlinear_conditional_20261002 \
  --output /tmp/new_chart_cause_audit
```

The final four-case analysis and complete per-transaction logs are in
`results/dranppr_chart_cause_20261003/origin_check/`. The parent directory retains
the initial position-fit-only diagnostic and its original executable/source;
the origin check is the final analysis. Inputs, reference state/message files,
source snapshots, commands, executed binaries, tests/XML and SHA256 manifests
are retained. No result directory was overwritten.

Final executed binary SHA256:
`e4aebf40517a95ef383ea0bcd12b277cb409ff149764386bc6649140d39036a1`.
Pinned GTSAM library SHA256:
`280ba9f45a49464f3ad1c4385526fc355300644147cc0a798d2f0a2185bdd28e`.

Tight PCG maximum verified relative residuals remain below 9.91e-11; totals are
D4 2988 and D9 2226 iterations. One-step totals are D4 84 and D9 75, with maximum
relative residuals 1.21859 and 2.08483, respectively: these are not converged
linear solves. MPI ranks: 0. Production `DBA-DRANPP` remains clean; no commit or
push was made. No new ATE, robustness, asynchronous online, or SOTA claim follows
from this diagnostic.
