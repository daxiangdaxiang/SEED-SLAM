# Validated Model-Error Feedback And Release Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans inline, as requested by the user. Track completed steps below. Do not delegate or publish before the research acceptance gate passes.

**Goal:** After demonstrating that ongoing model-error correction improves asynchronous second-order stability and benefit per communicated byte, consolidate the validated algorithm and publish an accurately scoped SEED-SLAM release.

**Architecture:** Keep research in `projects/seed_slam/DRANPPR` until a frozen candidate passes controlled comparisons and implementation checks. Integrate only the validated changes into the existing DBA/DRAN++ source snapshot in `projects/seed_slam/SEED-SLAM`; preserve DBA behavior and existing data. Release claims must match the tested implementation, not merely the fixed-quadratic diagnostic.

**Tech Stack:** Existing C++17/CMake backend, Python/NumPy/SciPy experiment tools, Git, and the existing vendored-source layout.

## Authorization And Current Status

The user authorized this sequence on 2026-11-24: achieve the research objective,
organize the latest algorithm code, update GitHub, publish a version, and explain
the algorithm changes. This is conditional publication authorization, not a
request to publish the current prototype immediately.

- Research directory: `/home/jackal/zyx/dpgo_ICRA/projects/seed_slam/DRANPPR`.
- Research branch inspected: `research/dranppr-evidence-online`; no Git remote.
- Release directory: `/home/jackal/zyx/dpgo_ICRA/projects/seed_slam/SEED-SLAM`.
- Release remote inspected: `git@github.com:daxiangdaxiang/SEED-SLAM.git`.
- Current local release HEAD inspected: `d319cc1`; this is not a fresh remote check.
- Existing package: `submodules/DBA-DRANPP`, a vendored snapshot, not a Git submodule.
- Snapshot provenance currently records source commit
  `104a200a7478a6aa6df19b93719edfe5b453480a` from 2026-09-08.
- No local release tags were present when inspected. Select the version only
  after fetching remote tags and reviewing compatibility at release time.

The research acceptance gate is currently **NOT PASSED**. The sealed report
`docs/dranppr_horizon_error_budget_report_20261124.md` records useful ordinary-
network gains, but three catastrophic long-delay failures of the predictive
variant. The hold/adaptive-update ablation is the stronger tested reference.
Completing a finite simulation horizon does not establish convergence.

## Stage 1: Establish The Research Result

- [x] Freeze one next model-correction candidate and its evaluation protocol
  before confirmatory runs. Do not replace failed stress cases with easier cases
  or retune memory length separately for each dataset.
- [x] Distinguish actual received state, predicted state, known model innovation,
  and unknown current delay error in both code and explanation. A cache refresh
  alone must not be presented as a new error-feedback algorithm.
- [x] Test continuous correction against a stopped/frozen-correction ablation
  with identical model payload and update opportunities. Compare additionally
  against the original directional baseline and the adaptive hold variant.
- [ ] Count all additional gradient/model payload, routed hops, and incremental
  setup information. Report equal-byte objective curves, time to matched
  accuracy, and fixed-opportunity results separately. Include setup assumptions.
- [ ] Re-run the existing long-delay and long-outage counterexamples through
  at least their original horizon, plus independent network seeds, loss,
  heterogeneous activation, and initial-state controls. Report all regressions.
- [ ] Demonstrate gains on both 2D and 3D problems without per-case sweeps.
  Separate synthetic/frozen SE3 quadratic evidence from native nonlinear SE3
  evidence; test the native path before claiming a deployed backend improvement.
- [ ] Independently reconstruct updates, objectives, packet budgets, and causal
  error labels; verify legacy and ideal-network parity where applicable.
- [x] Write an acceptance report listing the frozen protocol, retained failures,
  correction ablation, strongest-reference comparison, overhead, and scope.

Acceptance requires a reproducible correction-specific benefit, a demonstrated
stability improvement on the predeclared stresses, and a useful communication/
accuracy tradeoff against the strongest current reference, not just the oldest
baseline. A lower prediction error alone, aggregate wins that conceal divergence,
or a fixed-quadratic result alone is insufficient for a production claim. Do not
claim universal robustness or a new convergence guarantee from empirical tests.
If the candidate fails, preserve the negative result and keep release pending.

## Stage 2: Consolidate Only Validated Code

- [ ] Inventory modified, untracked, and generated files; distinguish user work
  from research artifacts. Use an explicit file-level inclusion/removal list.
- [ ] Select one canonical implementation and entry point. Keep useful ablations
  as explicit research options rather than a chain of production heuristics.
- [ ] Remove obsolete duplicate implementations and disposable logs only after
  checking references. Preserve minimal reproducible counterexamples, regression
  tests, configurations, and sealed evidence. A negative result is not junk.
- [ ] Preserve existing DBA workflows, physical objectives, CCI weight defaults,
  and published datasets unless a documented validated change requires otherwise.
- [ ] Add deterministic numerical tests and native 2D/3D integration tests for
  the released path. Check stale/duplicate/reordered packets, history reset, and
  unsupported-model fallback when applicable.
- [ ] Document the actual default, remaining parameters, communication contract,
  dependencies, build/run commands, and differences from the GitHub baseline.

## Stage 3: Prepare The SEED-SLAM Snapshot

Files to review/update after Stage 1 passes:

- `SEED-SLAM/submodules/DBA-DRANPP/`: validated source and required tests.
- `SEED-SLAM/submodules/DBA-DRANPP.version`: actual clean source provenance.
- `SEED-SLAM/scripts/sync_dba_dranpp.sh`: safe, reviewed packaging behavior.
- `SEED-SLAM/scripts/build_dba_dranpp.sh`: supported build path if changed.
- `SEED-SLAM/README.md` and `SEED-SLAM/CHANGELOG.md`: version and user-facing changes.
- Release-scoped experiment manifest and concise report within the package.

The current sync script uses `rsync --delete`, excludes only a few generated
directories, and records HEAD even if the source has uncommitted modifications.
Do **not** point it at the dirty research directory. First prepare a clean,
reviewed release source containing only approved files; inspect a dry-run diff
and correct provenance handling before synchronizing. Do not overwrite the
canonical DBA package wholesale with an experimental fork.

- [ ] Inspect local changes and fetch remote history/tags without resetting or
  force-pushing. Reconcile any remote changes before selecting the release base.
- [ ] Prepare and review the source snapshot; exclude build trees, caches,
  temporary outputs, credentials, and large unrequested datasets.
- [ ] Preserve `data/Indoor_10_robots` and unrelated existing repository content.
- [ ] Verify the snapshot has no machine-local path or missing-file dependency.
- [ ] Build and test from a clean checkout/staging copy using documented commands.
  Reproduce a small deterministic example and the key release comparisons.

## Stage 4: Publish And Verify

- [ ] Select a compatible version after inspecting the actual remote tags.
  A research-only artifact must be labeled experimental, not a production
  replacement or an implicit assertion that Stage 1 passed.
- [ ] Commit the reviewed source, tests, documentation, and provenance; inspect
  the final diff and ensure all required tests pass on that exact commit.
- [ ] Push the intended branch without force, create the version tag on the
  tested commit, and publish a GitHub Release with the same version.
- [ ] Verify remote commit/tag identity and retrieve the published release.
  Check a fresh clone/build. Report authentication or publishing blockers
  accurately; a local commit or tag is not a completed GitHub publication.
- [ ] Provide the release URL, version, commit, test commands, evidence location,
  and concise algorithm update notes to the user.

## Algorithm Update Notes Outline

1. Problem addressed: stale neighbor information corrupts the local second-order
   model; explain the concrete instability or wasted-communication mechanism.
2. Core algorithm change: what model information is corrected, by which arriving
   observations/messages, and how that changes the actual optimization step.
3. What remains unchanged: physical objective, measurement interpretation, and
   baseline/DBA behavior outside the validated scope.
4. Evidence: same-byte and same-accuracy comparisons, correction ablation,
   stress results, 2D/3D scope, and measured computation/payload overhead.
5. Limits: remaining assumptions, parameters, adverse cases, and non-claims.
6. Usage and migration: entry point, defaults, compatibility, and reproduction.

## Progress

- [x] Record the conditional release request and verify the local release remote/layout.
- [x] Identify the existing synchronization/provenance hazards without executing it.
- [ ] Pass the research acceptance gate.
- [ ] Consolidate and verify the release candidate.
- [ ] Publish and verify the GitHub version.

## First Execution Result

On 2026-11-24, the approved next candidate was implemented and evaluated in
140 fixed-quadratic runs; 119 focused tests pass. Continuous forcing correction
beats frozen forcing in all 24 nonideal conditions, and the three previous
long-delay divergences are not reproduced. But equal-byte comparison against
adaptive hold gives 0 wins, 2 ties, and 26 losses. Removing incremental setup
bytes does not resolve the failure. Thus the research acceptance gate is NOT
passed and no release is authorized by these results.

The first-cycle plan is `2026-11-24-model-response-feedback.md`; evidence is
reported in `../../dranppr_model_response_feedback_report_20261124.md`. All
traffic prefixes, declared updates and objectives were audited; 24 legacy
stress runs retain exact archived state/counter parity. Native nonlinear SE3,
new riMESA comparisons, broader seed/initial-state confirmation and controlled
matched-accuracy timing remain unperformed because the candidate already fails
the prerequisite communication-efficiency test.

Research artifacts and counterexamples are retained. No production code cleanup,
snapshot synchronization, commit, tag, push, or GitHub Release was performed.
The overall plan remains incomplete; do not mark publication or the research
milestone complete based solely on this diagnostic implementation.
