# Frozen Noise Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Inline execution; no subagents, commits, or production edits.

**Goal:** Distinguish finite online optimization error from noise-weighting effects without changing measurements, covariance, topology, or gauge.

**Architecture:** Add an explicit weighted frozen-input schema to the diagnostic parser. Export terminal online states without registration or averaging. Complete identical frozen graphs under two fixed covariance models and cross-starts, then load ground truth only for evaluation.

**Tech Stack:** Existing native GTSAM Pose2/Point2 factors, Python/pytest, JSON, CSV.

## Predeclared Experiment

- D2 original and Gaussian bearing injection, sigma 0.15 rad, seeds 20261003, 20261004, 20261005. Existing seed and original online endpoints are reused with hashes; four new lazy online runs for the other two seeds.
- Two objectives: nominal covariance, or add 0.0225 rad^2 to suffix bearing variance only. Nominal prefix, range, and odometry remain unchanged. Original-data inflation is NOT an oracle.
- Four LM completions per dataset: each objective from each online endpoint. Two GN completions per dataset: own matched start. Two extra routed tight-PCG completions on seed 20261003. Total 26 frozen solves, cap 100 outer iterations and 600 seconds per process. No performance-based parameter selection.
- Compare costs only within the same objective. Report final gradient, solver/start agreement and state errors; no global optimality or across-dataset claim. Frozen completion is offline, not a new online algorithm.

## Tasks

- [x] Add failing tests in `tests/test_frozen_noise_completion.py`: analytic per-factor loss scaling, covariance validation, replay rejection of weighted bootstrap, and terminal export cost parity.
  Run `python3 -m pytest -q tests/test_frozen_noise_completion.py`; expect unsupported schema/export failure.
- [x] Extend `experiments/native_local/joint_package.h`: v2 requires positive finite per-factor `sigmas`; v1 rejects explicit `sigmas`. Preserve v1 numeric path. Reject v2 warm starts in `joint_replay.cpp` so covariance is never silently reset.
  Rebuild `native-joint-audit native-joint-replay native-conditional-audit` in `/tmp/dranppr_backend_build` with `-j2`.
- [x] Implement `experiments/prepare_frozen_completion.py::prepare(events, states, warm, duration, variance)` preserving exact terminal poses and shared points, validating frame/state support and prefix. Explicit sigma arrays: odom `[.1,.1,.05]*sqrt(dt)`, observation `[sqrt(.05**2 + suffix_variance),.1]`.
  Require exact shared charts; reject duplicates, missing states, invalid variance and truth-marked warm input. Test exporter on deterministic replay fixture; frozen initial loss must match online physical loss.
- [x] Implement fixed-grid runner `experiments/audit_frozen_noise_completion.py`. Snapshot sources, binaries, hashes, input packages and full commands; sequential subprocesses only. Assert matched-start initial loss equals online loss at relative tolerance 1e-10 and absolute tolerance 1e-8. Use original hash-verified streams and deterministic seed perturbations.
- [x] Run predeclared grid, retaining failures rather than changing settings. Only after all estimators terminate, compute final and causal metrics and common-prefix aligned causal metrics using existing evaluation helpers.
- [x] Run focused and full existing native regression suites plus C++ suite. Record actual commands, counts, residuals and any failures. Produce report, provenance checksums and a clear explanation distinguishing covariance calibration, outlier robustness, bias, identifiability, and solver lag.

## Stop / Interpretation Rules

Schema/export parity failures stop the grid. Numerical failures remain diagnostic outcomes, not reasons to sweep parameters. Small gradients and cross-solver agreement support a local stationary explanation, not global optimality. A weighted objective decrease is not an accuracy improvement across covariance models. Additional seeds on one sequence are not independent datasets.

## Execution Record

All 4 online and 26 frozen estimator processes succeeded without retries. Added a posthoc-only reference-residual analysis after inspecting the grid, explicitly exploratory and never fed into an estimator. New test first failed for the missing analysis helper, then passed after implementation. Final verification: 62 C++ tests, 132 Python tests. Original plain pytest invocation encountered unrelated Dash/Jinja plugin incompatibility; disable plugin autoload as in previous audits. Report: `docs/dranppr_frozen_noise_completion_report_20261003.md`. Preserve the research branch without merge or push.
