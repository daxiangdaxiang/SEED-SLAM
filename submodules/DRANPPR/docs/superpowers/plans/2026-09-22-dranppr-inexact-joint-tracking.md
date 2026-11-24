# Causal Joint Tracking: Reuse and Finite Interface Work

## Purpose

The previous D9/300 causal joint replay has accurate terminal estimates but a
53.22 s serial delivery backlog. Test the cause without changing likelihoods,
identity evidence, topology, initialization policy, or the physical acceptance
objective. This is an efficiency/tracking investigation, not noise adaptation.

## Fixed Ablations

- `joint`: fresh charts, interface PCG relative tolerance 1e-10, cap 500.
- `joint_cached`: reuse charts below the existing local/iSAM2 comparison
  threshold 1e-4; same interface accuracy.
- `joint_one`: fresh charts, at most one PCG iteration per arriving frame.
- `joint_cached_one`: both changes.

The one-iteration budget is a deliberate extreme diagnostic, not an adaptive
rule or a tuned default. Keep all failures. Use D4/120 and D9/300, then decide
whether an intermediate or evidence-driven inner accuracy rule is warranted.

## Algebra and Invariants

At declared chart theta, keep a current estimate theta.retract(d). A cached
quadratic has current gradient g+H d. Eliminating private variables gives the
boundary gradient b+S d_B. Solve S s_B=-(b+S d_B) to finite or tight accuracy;
recover q_P from the original response at q_B=d_B+s_B, then take s=q-d.
Backtracking evaluates theta.retract(d+alpha*s) against the true physical loss.

Shared Point2 charts AND current points must agree. Identity coordination
transmits both for cached modes, with counted payload. New measurements use
the current estimate for initialization. Gauge merges transform current values
and fully rebase affected local models. Cached responses are reusable only if
the local revision, private-key set, and fixed gauge all match.

## Verification and Gates

- [x] Dense algebra tests: nonzero chart displacement, full and one-step PCG,
  private recovery, inconsistent shared displacements rejected.
- [x] Stream tests: all four modes, delayed connectivity, causality, same-time
  ordering, alias separation, nonlinear physical descent, response reuse.
- [x] Old fresh-mode deterministic state parity.
- [x] Paired real replay with stage timings, recomputation counters, bytes,
  causal matched timestamps, frozen-completion gap, virtual delivery error.
- [x] Record outcome and remaining limits; no SOTA claim without equal-budget
  native-model riMESA/iMESA comparison.

Transport is still a synchronous routed simulator with scalar collectives.
Caching is not incremental sparse elimination, and one CG step is not a bound
on total per-frame work. Statistical cause separation remains a separate gate.

## Outcome

All 8 replays and 8 frozen completions exit 0. Tight cached solves preserve final
and causal accuracy, reduce D9 serial work 226.2 -> 167.5 s and backlog 54.7 ->
12.4 s. One-step solves reduce bytes by more than 91% with near-identical terminal
states, but not enough CPU work. Combined caching/one-step solves invalidate
responses on every joint update and are slower than tight caching. Do not promote
this combination or hide the failure with threshold tuning. A total-work-aware
inner solve and genuinely incremental local response remain next investigations.
See `docs/dranppr_inexact_joint_report_20260922.md`.
