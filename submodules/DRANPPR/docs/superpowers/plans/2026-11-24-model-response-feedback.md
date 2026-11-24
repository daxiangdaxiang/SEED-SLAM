# Model-Response Feedback Experiment

> Execute inline using executing-plans and TDD. User approved execution of the
> conditional release plan. This experiment is its first research acceptance
> step, not authorization to promote a failed candidate.

**Goal:** Test whether correcting a frozen-boundary local quadratic response
model resolves the long-delay instability of time-series extrapolation and
improves objective progress per actual communicated byte.

**Architecture:** A sender provides its actual post-update state, held-cache
local gradient, and last update metric/gain. A receiver predicts repeated
updates with this frozen local model. New packets reset the true state anchor
and correct the model forcing. The existing causal error budget controls the
receiver's actual step; unsupported predictions retain the directional hold
fallback. No physical weight, CCI, native backend, or production default change.

**Tech Stack:** Existing Python/NumPy/SciPy fixed-quadratic diagnostic.

## Frozen Candidate

For A=B_jj and packet state x_s, gradient g_s, inverse metric P, and gain a,
the frozen recurrence is x_{k+1}=x_k-a P(A x_k+b_s), b_s=g_s-A x_s.
If A=L L^T and L^T P L=U diag(lambda) U^T, its n-step response is

```
W = L^-T U
xhat(s+n) = x_s - W diag(1-(1-a*lambda)^n) W^T g_s
```

The eigenvalues are in (0,1] for the existing majorizers. This response
saturates for a fixed model rather than growing linearly with horizon.
It is NOT a stability proof for the coupled delayed system, where forcing
and adaptive gains change. The signed forcing innovation is
g_new-g_old-A(x_new-x_old). Corrected mode replaces b with the new value;
frozen mode learns b once but still refreshes actual state and update policy.
This isolates correction from merely updating a pose cache.

Use source update opportunities, not ticks, for n. Do not provide initial
gradients for free. Sender gradients use only its actual received cache, not
its predictions. Keep received and predicted state separate. L=8 remains the
unchanged causal error memory. No hyperparameter tuning in this experiment.

## Payload And Comparison

Legacy packet: 40-byte header plus 8*d state bytes. Model packet additionally
contains 8*d gradient bytes and 16 policy/gain bytes, charged at every hop.
A static descriptor (A, delayed P, fresh P, period) costs 48+24*d*d bytes per
dependency per routed hop. Count it at initialization. Static descriptor setup
is pre-distributed and has no simulated latency/retry; disclose that assumption.
Common offline row/chart setup remains excluded for all arms, as previously.

Arms: directional hold, adaptive hold, adaptive linear prediction, adaptive
corrected model response, adaptive frozen-forcing response. Preserve bitwise
legacy state parity. Compare common-byte endpoints with last available states
(no state interpolation), equal opportunities separately, and all prefix curves.
Frozen/corrected model arms must have identical prefix byte budgets.

## Attempt Budget And Gates

One structural candidate plus ablations. First D1/D4/SE3_31/coupled-SPD,
2000 ticks, seed34, delay12--20 and outage200--600; these reproduce the existing
counterexamples. Also ideal and ordinary delay/heterogeneous/loss/outage.
Expansion uses all nine archived models and independent seeds only if there is
no catastrophic regression and communication evidence is promising. No short
horizon may substitute for the 2000-tick stress. Analyze failures before any
new idea. A negative result blocks production release, not documentation.

## Tasks

- [x] Add failing tests for exact recurrence, stable saturation, signed forcing
  correction, frozen ablation, payload copying/accounting, stale packets,
  causal prefixes, ideal parity, and common-byte alignment.
- [x] Implement `experiments/quadratic_response.py` and optional model packets
  in `experiments/async_interface.py`; preserve all legacy numerical paths.
- [x] Add `experiments/audit_model_response.py`, a sealed driver with independent
  update/objective checks, source snapshots, and communication-aligned results.
- [x] Execute predeclared stress and ordinary controls. Expand only if justified.
- [x] Write a report and update the parent release checklist honestly. Do not
  publish or remove useful negative evidence to satisfy a desired outcome.

## Outcome

Completed 40 stress and 100 ordinary-control runs with 119 passing tests.
`review_model_response.py` independently verifies all prefix traffic, updates,
objectives, and 24 bitwise legacy pairs. Corrected response improves all 24
nonideal frozen-model ablations and does not reproduce the three old divergences.
Against adaptive hold at equal bytes it wins 0, ties 2, loses 26. Removing setup
bytes does not change this outcome. Do not expand or port a failed candidate
merely to finish a release checklist. Parent release gate remains unpassed.
See `docs/dranppr_model_response_feedback_report_20261124.md`.
