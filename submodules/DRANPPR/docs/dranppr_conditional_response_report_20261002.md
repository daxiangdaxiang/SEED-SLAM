# Persistent Conditional Response: First Component Gate

## What Was Implemented

`IncrementalConditionalModel` maintains a declared Gaussian local model by
stable factor ID. GTSAM `EliminateQR` supplies the numerical elimination;
the new code maintains dependency ownership, cached private conditionals and
their remaining factors. An edit rebuilds its old/new owning elimination nodes
and the affected paths toward the boundary. Unaffected numeric objects survive
by immutable sharing. Private recovery accepts any boundary increment without
re-eliminating the local model.

The response implements the same `NativeLocalResponse` interface as the existing
fresh `NativeSchurModel`. Topology-local PCG is unchanged. The nonlinear joint
replay still uses its existing response implementation; no new default, chart
refresh rule, robust kernel, noise adaptation, or window length was installed.

Factor append, replacement and removal are transactions. Invalid inputs or
singular private elimination leave the previous published response usable.
Reordering or changing existing private/boundary membership triggers an explicit
full rebuild. Existing-key tangent dimensions cannot silently change. Fixed
gauges are exact column removal in the audit, not soft priors. Boundary rank
deficiency is allowed; the response is not an observability certificate.

This is a scalar-node elimination tree using library QR, not the optimized
multifrontal iSAM2 optimizer. It retains a dense LOCAL boundary Hessian and
copies O(graph) metadata for rollback. It does not yet provide bounded work or
incremental nonlinear state publication. Incremental elimination is established
prior work, not a new algorithmic novelty claim.

## Verification Before Data Experiments

The first native build fails for the missing conditional-model header after
adding contract tests. The first driver tests fail because the new executable
does not exist. Those red states precede implementation. A later adversarial
test exposes a real numerical bug: testing a trailing pivot only relative to
its own nearly zero scale can accept an unanchored gauge after roundoff. The
rank check now uses the original Jacobian coefficient scale and private tangent
dimension, together with finite conditional checks. It adds no prior or noise
floor. The failing and passing test XML are retained.

Final checks: 50 native tests (39 existing, 11 conditional-response tests) and
60 focused Python tests pass. Coverage includes mixed dimensions, changed factor
support, repeated replacements, removal, appending independent branches, boundary
membership changes, no-op updates, immutable snapshots, invalid/overflow rollback,
exact gauge handling and shifted-chart routed solves.

## Native-Data Protocol

Use the archived D4/15.534 s and D9/131.723 s common-past packages from the
previous matched-backend experiment. They contain native Pose2 odometry and
polar object factors with unchanged declared noise. All packages pass the same
strict loader used by `native-joint-audit`.

**This is a frozen-chart factor-arrival replay, not a causal online estimator.**
Coordinates are the declared package coordinates, which can contain information
from later in that package's prefix. We insert factors in timestamp/pose groups
but hold those charts fixed, expressly isolating Gaussian response maintenance.
No truth file is read and no ATE is measured. Known identities and the chain
topology are unchanged.

The reference is `IncrementalLocalModel::update` plus sparse export and a fresh
`NativeSchurModel`. The candidate prepares native Gaussian factors, removes the
same fixed columns and updates `IncrementalConditionalModel`. Every valid
prefix compares the complete boundary operator, reduced gradient and private
recovery. Neither arm uses a different likelihood or a damped fake gauge.

There are 140 valid / 8 private-singular D4 prefixes and 1140 valid / 887
private-singular D9 prefixes. Singularity is determined by the existing fresh
private-system check. Early singular prefixes are recorded with their reason,
and their factors are accumulated until a valid response exists; no priors are
added. These counts are NOT benchmark solver failures or a from-scratch bootstrap
result. The candidate is not independently timed solving each singular prefix.

For fair local timing, exclude singular prefixes AND the first valid prefix
of each robot. The latter has different accumulated bootstrap work. This leaves
135 paired steady updates on D4 and 1135 on D9. Candidate work includes native
Gaussian preparation, response update and ONE full private recovery. Reference
work includes normal-model update, sparse export, fresh response and ONE full
private recovery. Validation products, extra checks, input loading and output
serialization are excluded symmetrically. This is not total SLAM frame service.

Execute three deterministic repeats on one host, sequentially without concurrent
builds or estimator jobs. Reference precedes candidate inside each paired update;
timing order is not randomized. These are timing repeats, not independent data
seeds, confidence intervals, or an isolated-hardware benchmark. D4's total work
is only milliseconds, so its ratio deserves particular caution.

## Results

| Case | Paired updates | Reused / total private conditionals | Reference local work s, three repeats | Candidate local work s, three repeats | Speed ratio |
| --- | ---: | ---: | --- | --- | --- |
| D4 | 135 | 2369 / 2638 (89.80%) | .019138 / .017192 / .017292 | .010866 / .009700 / .009713 | 1.76--1.78x |
| D9 | 1135 | 276833 / 279102 (99.19%) | 1.264306 / 1.262985 / 1.261930 | .582437 / .580686 / .580293 | 2.17--2.18x |

These ratios already include the candidate's slower private recovery. They are
not the speedup of the earlier 204.52 s nonlinear replay and do not establish
reduced delivery age. Original normal-block accumulation is absent from the
candidate component, but its transaction metadata copy and full private recovery
remain linear in history size.

Maximum numerical discrepancies over all prefixes and the controlled edits:

| Case | Boundary product relative error | Reduced-gradient relative error | Private recovery relative error | Private equation relative residual |
| --- | ---: | ---: | ---: | ---: |
| D4 | 5.30e-14 | 9.06e-14 | 3.56e-14 | 2.87e-13 |
| D9 | 6.91e-13 | 1.97e-12 | 2.68e-13 | 2.82e-13 |

The relative errors divide by `max(1, reference norm)`; they are not relative
to an arbitrarily small zero-gradient norm. These values are identical across
the timing repeats, supporting deterministic arithmetic for this run, not a
general error bound.

### Where Reuse Fails

After each terminal local model, replace one late and one early physical factor
with information multiplied by 1.1, in that order. This is a declared controlled
edit, NOT noise learning, a robustness rule, or a tuned performance setting.

- Each late-factor edit rebuilds only one private conditional on these chains.
- The five early edits together rebuild 143/147 private conditionals on D4 and
  2025/2026 on D9. The latter essentially revisits the complete local histories.
- Early-edit update work is worse: D4 candidate .00148--.00156 s versus reference
  .00081--.00086 s; D9 candidate .01267--.01272 s versus .00882--.00891 s.
- Full private recovery is also slower in this straightforward library-map
  implementation. On steady D9 prefixes it costs .1333--.1339 s versus
  .01213--.01215 s for the reference, roughly eleven times more.

Thus, a small number of changed factors need not imply a small elimination
update. Updating an early factor can affect all later conditionals. This negative
result directly constrains later relinearization and robustness claims.

### Distributed Compatibility

Both terminal responses, including the identical controlled information edits,
are fed into the unchanged routed interface solver:

| Case | Reference / conditional PCG iterations | Reference / conditional application bytes | Conditional verified residual | Relative step difference |
| --- | ---: | ---: | ---: | ---: |
| D4 | 33 / 33 | 106208 / 106208 | 2.11e-11 | 5.34e-14 |
| D9 | 33 / 32 | 102312 / 99352 | 3.88e-11 | 1.93e-10 |

The D9 one-iteration difference is an arithmetic/termination-path difference
between equivalent numerical representations, not a new communication mechanism
or a guaranteed communication saving. No network protocol was changed. The
deterministic small routed test has exactly matching iteration and byte counts.
MPI: 0 ranks. CCI/D-CCI and the packaged DBA-DRANPP repository are untouched.

## Decision And Next Step

The FIRST algebra/locality gate passes: the implementation genuinely reuses
unaffected elimination work, preserves the declared response and reduces paired
append-local component work on these inputs. It is more than Jacobian caching
followed by a complete new private factorization.

The online research gate remains open. These charts are frozen, whereas the
previous cached-plus-one-PCG run refreshed nearly all historical factors. Simply
plugging the new factorization behind that schedule could reproduce the slow
early-edit case. Neither loosening the existing 1e-4 chart threshold nor hiding
history with a fixed window is justified by this result.

Next integrate the Gaussian cache at the nonlinear model boundary, so updating
the published state does not automatically rebuild a duplicate full normal
model. First preserve the existing declared-chart refresh decisions in a matched
control, including all invalidation and whole-history recovery costs. Measure
the actual affected paths under those decisions. Only a total-service and
state-quality comparison on causal D4/D9 slices can establish a useful online
improvement. Adaptive statistical modeling, false-identity robustness and
held-out transfer remain separate unresolved objectives.

## Artifacts And Reproduction

Archive: `results/dranppr_conditional_response_20261002/`. It includes six JSON
traces, copied inputs, executed binary, source snapshots, test XML and checksums.
Source entry: `include/DPGO/IncrementalConditionalModel.h` and
`src/IncrementalConditionalModel.cpp`; driver:
`experiments/native_local/conditional_audit.cpp`.

```sh
cmake -S . -B /tmp/dranppr_backend_build
cmake --build /tmp/dranppr_backend_build --target \
  testIncrementalLocalModel native-conditional-audit native-joint-audit \
  native-joint-replay -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_native_conditional_audit.py \
  tests/test_native_joint_audit.py tests/test_native_joint_replay.py \
  tests/test_native_interface_support.py tests/test_native_matched_online.py \
  tests/test_audit_native_online.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 120 \
  results/dranppr_conditional_response_20261002/native-conditional-audit \
  results/dranppr_conditional_response_20261002/inputs/d9.json /tmp/d9_response.json
```

GTSAM revision: `03c263fe08b02c90a8861d36f77422765d39506d`.
Executed binary SHA256:
`b8560981b8c0ce6a51b5e6ac23e21bb21ace5fb254426556a73a87500bd9df63`.
The existing root CMake dependency/deprecation warnings remain; they are not
new native compiler errors. One initial new-target build required explicit CMake
regeneration; one test invocation required quoting its zsh wildcard. Both were
corrected and final commands completed normally. No commit or push was made.
