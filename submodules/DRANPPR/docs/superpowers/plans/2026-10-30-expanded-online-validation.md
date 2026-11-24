# Expanded Online Validation and Scoped Cleanup

> Execute inline in the existing research worktree. User approved continuation,
> dataset expansion and cleanup. No agents, commits, pushes or production edits.

**Goal:** Extend the frozen protocol to all five locally available MRCLAM
sequences and separate finite-work tracking from past-data/model limitations.

**Architecture:** Reuse the existing matched online runner, accepting a declared
dataset/duration manifest instead of hard-coded D1/D4. Prepare D2/D3/D9 from
archived, checksummed common-past inputs, with the SAME five-component first-
prefix REML procedure. Preserve baseline failures and do not tune or silently
substitute parameters. A separate D1 checkpoint audit compares the native online
state with completed LM/GN on exactly the same arrived graph.

**Tech Stack:** Existing C++17 native backend, pinned repaired riMESA/iMESA,
Python orchestration, JSON provenance, pytest and GoogleTest.

## Fixed Design

- D2/D3 end at 120 s; D9 at 300 s because its first connected prefix ends at
  131.723 s. These are additional previously inspected sequences, not a blind
  test or full-mission validation. No noise injection or new model choice.
- First-prefix nominal LM and joint_qr REML as in D1/D4; freeze all five scales.
  Record fit conditioning/boundaries/time. Resource watchdog 600 s per process.
- Native nominal/calibrated plus paper-iMESA/repaired-riMESA, each baseline with
  one-sweep and causal common-byte-ceiling schedules. Same common primal states,
  cold baseline internals, topology, events and gauge. No cross-weight cost rank.
- Evaluate after all estimators terminate. Report final AND first-published
  trajectory/object errors, bytes, compute, and failures. Calibration overhead
  remains separately measured, not hidden in suffix timing.
- D1 diagnosis at 30,60,90 s: replay both weights with unchanged one-step budget,
  freeze arrived graph, complete LM and GN from that endpoint. Compare full
  state plus latest poses using shared prefix and terminal alignments. Completion
  is diagnostic and not a fair real-time competitor. No future observations.
- Cleanup only within DRANPPR: aborted pre-QR output, truncated riMESA download,
  regenerable caches, and superseded endpoint-only patch. Preserve minimal
  failure provenance, complete numerical-regression evidence, active dependencies,
  referenced source, useful tests, and valid negative scientific results.

## Tasks

- [x] Test dataset/duration contract and invalid inputs before generalizing
  `experiments/audit_online_calibration_baselines.py`. Legacy D1/D4 defaults
  remain unchanged; evaluation follows declared records, not hard-coded IDs.
- [x] Add `experiments/prepare_extended_calibration.py`: validate archived
  inputs, run prefix training and existing fit entrypoint, freeze a source
  manifest plus SHA256SUMS. No new statistical implementation.
- [x] Execute new datasets sequentially under the old six-arm protocol and
  evaluate separately. Keep original D1/D4 archives immutable.
- [x] Add and execute `experiments/audit_checkpoint_tracking.py`, with input
  equality, truncation-causality and physical-objective parity checks.
- [x] Inventory cleanup targets, archive small abort/source evidence and
  checksums, remove only explicit verified paths, and update stale documentation.
- [x] Run regression tests, verify result hashes, production clean status and
  active build. Publish one combined five-dataset report and cleanup manifest.

Tests: extend `tests/test_online_calibration_baselines.py` for default and
expanded schedules, duplicate/missing/nonfinite duration rejection. Test the
checkpoint latest-state selector on unordered rows, non-pose exclusion and
missing/duplicate robot support. Run these RED then GREEN, then the existing
187-test selection and native 62-test suite. Do not remove failing tests to
obtain a green suite.

## Outcome

198 Python and 62 native tests pass. All 17 launched expanded online arms
finish; D9 calibration times out at the unchanged 600 s limit and is explicitly
unavailable. D2/D3 improve all three primary error metrics; D1 causal regression
is retained. Eighteen D1 checkpoint calls establish a finite-work tracking
component without future observations. Ten state-file parities and five
physical-objective checks pass. See the expanded report and cleanup manifest.
Completing this validation plan does not mean the adaptive-online algorithm
has met deployment or high-noise robustness requirements.
