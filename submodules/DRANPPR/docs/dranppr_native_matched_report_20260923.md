# Common-Past Native Backend Controls

## Scope and Implementation

The new isolated `experiments/native_matched` adapter accepts the existing
native frozen-joint package. It preloads the same primal values, performs the
baseline's standard neighbor declarations and handshakes, then inserts the
physical factors and updates persistent local solvers. Only robot 1's root
has a constrained gauge prior. No other robot root is fixed. Preloading does
not run an optimizer or import another backend's duals/robust weights.

These are **common-primal-start, cold-internal-state, frozen-prefix controls**.
They are not a replay of the baseline's full historical internal state, a
from-scratch test, or an online-suffix comparison. The common start is from the
past-only DRAN replay, which is a declared conditioning choice. No new DRAN
solver or statistical adaptation mechanism was introduced in this experiment.

The current original RSS24 iMESA source (`693c1ad2e20d88f97b480060c4830f0545599f7b`)
is different from the iMESA control in the riMESA paper's experiments. The latter
uses the newer riMESA implementation with Gaussian factors, Gauss-Newton local
updates and dual decay 0.9. Its wrapper can also use oracle outlier labels.
The primary `imesa_paper` arm matches that solver configuration but supplies
**no oracle labels and removes no measurements**. Original iMESA and its
single-change Dogleg diagnostic are archived separately, not presented as the
paper's current control. Official configuration source:
[IMESAAgent](https://github.com/rpl-cmu/rimesa-experiments/blob/1744174e5e2aa5b16a8a3bd84b19b2a3e44072b0/agents/include/agents/imesa_agent.h).

riMESA is pinned at `ffddc21dad530ba0b0e57c42495146b410581fc4` and riSAM at
`c6fa05eaece103efd7e44e640bda8df6eff30e8d`, using the previously disclosed
isolated endpoint/merit numerical repairs. The official source remains clean.
All methods use the same full-chain native odometry Jacobian and polar factors;
this common numerical correction is not a DRAN innovation. Gaussian riMESA is
an explicitly labelled ablation disabling both robust factor families, not
another implementation of the original iMESA baseline.

## Protocol

- D4 stops at first full connection, 15.534 s: 148 poses, 209 observations and
  143 odometry factors; evaluation has 148 supported poses and 28 object copies.
- D9 stops at first full connection, 131.723 s: 2027 poses, 2200 observations and
  2022 odometry factors; evaluation has 1590 supported poses and 26 object copies.
  Its 437 unsupported truth timestamps are excluded and reported identically
  for every method. These are development prefixes, not held-out sequences.
- Physical inputs, primal values and the single gauge are identical. No new
  measurements, noise fitting, relation changes or truth enter any solve.
- Set two application-byte ceilings to the measured traffic of 1 and 10 routed
  DRAN updates. Baselines stop before a complete handshake/update round would
  exceed each ceiling. Report actual bytes, not fictitious exact equality.
- Count declarations, solution values and 40-byte logical headers per directed
  message. Startup handshakes and local updates are charged. Common bootstrap
  generation and physical transport/ACK/retransmission costs remain excluded.
- Evaluate all states only after the estimator processes exit, with one global
  SE(2) alignment per run, not independent robot alignments. Each run fits its
  own evaluation alignment; there is no oracle information in the estimator.

## Primary Results: Lower Communication Ceiling

RMSE and copy disagreement are in meters. Bytes are application bytes.

| Prefix | Method | Actual bytes | Physical cost | Copy disagreement | Trajectory RMSE | Object-copy RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D4 | DRAN routed | 107168 | 70.230104 | 0 | .197430 | .245777 |
| D4 | iMESA paper configuration, no oracle | 105968 | 62.362608 | .249001 | .247128 | .304800 |
| D4 | riMESA, numerically repaired | 105968 | 95.606106 | .407777 | .171912 | .276715 |
| D9 | DRAN routed | 109192 | 109.438504 | 0 | .363952 | 1.448008 |
| D9 | iMESA paper configuration, no oracle | 108880 | 91.698046 | .433224 | .522441 | 1.529746 |
| D9 | riMESA, numerically repaired | 108880 | 234.195039 | 1.176524 | .531083 | 1.653379 |

Baseline slack is 1200 bytes on D4 and 312 bytes on D9. They receive 62/64
complete rounds respectively, not one round. DRAN's one update requires 33/35
PCG iterations with verified residuals 2.47e-12/1.11e-11. MPI: 0 ranks.

The common initial trajectory/object RMSEs are .197643/.245921 on D4 and
.392773/1.452153 on D9. Thus D9's large physical cost reduction corresponds to
a modest trajectory improvement and a very small object improvement; it is
not a 79% truth-error improvement.

## Higher Communication Ceiling

| Prefix | Method | Actual bytes | Copy disagreement | Trajectory RMSE | Object-copy RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| D4 | DRAN routed | 995264 | 0 | .197432 | .245826 |
| D4 | iMESA paper configuration, no oracle | 994496 | .248859 | .246872 | .305006 |
| D4 | riMESA, numerically repaired | 994496 | .407979 | .171930 | .276693 |
| D9 | DRAN routed | 1014496 | 0 | .364450 | 1.451139 |
| D9 | iMESA paper configuration, no oracle | 1013704 | .431743 | .523232 | 1.531674 |
| D9 | riMESA, numerically repaired | 1013704 | 1.481255 | .504703 | 1.574070 |

More communication does not uniformly improve these metrics. On D9 it improves
riMESA trajectory/object RMSE while copy disagreement increases. D4 barely
changes. This is not proof of an asymptotic error floor or a global optimum.

The Gaussian riMESA ablation also keeps nonzero copy disagreement: .24882/.45122
at the lower D4/D9 ceilings, and .24869/.42769 at the higher ceilings. Therefore
disagreement cannot be attributed solely to robust observation kernels.

## What This Supports, and What It Does Not

1. Native distributed joint information can provide a useful budgeted state
   correction on the D9 joining transient. DRAN has better trajectory and
   object RMSE than both primary controls at these two ceilings.
2. This is not uniform dominance: riMESA has better D4 trajectory RMSE, while
   DRAN has better object RMSE and exact shared-copy consistency.
3. An iMESA sum of physical costs below DRAN's constrained cost is not evidence
   of a better solution to the same constrained problem. Its copies disagree,
   so that state is not feasible for the exact-consensus reference. Robust
   optimization also changes which measurement errors are prioritized.
4. The results are consistent with a benefit from coupled interface responses
   and exact consistency, but do not isolate their respective contributions.
   They do not establish improved noise adaptation or residual-cause recovery.
5. Common primal initialization is not matched historical internal state. Both
   baseline duals and GNC states start fresh; a causal persistent suffix and a
   full from-scratch comparison are still necessary.

Raw setup/update times are recorded, but there is no equal-wall-time accuracy
claim. Timings are single-host measurements, not isolated-hardware statistics;
baseline message-trace writes are included in its service time, whereas DRAN
uses its existing in-memory message ledger. Instrumentation and state-delivery
work need a common timing audit before asserting a speedup. This is a matched
communication-ceiling experiment, not a completed real-time benchmark.

## Verification and Artifacts

24 focused Python/integration checks pass: the new adapter/runner contracts and
the unchanged native joint audit tests. They cover physical initial cost and
factor parity, one gauge/free other roots, rigid-gauge equivariance, exact byte
totals, byte ceilings, malformed inputs and a small noiseless consensus case.
The latter reaches 1.24e-7 m copy disagreement with the original iMESA after
2000 rounds, with approximate local cost .000653 under its default relinearization.
No claim of exact batch convergence is made.

The primary matrix has 16 normal estimator exits (8 per prefix). The separate
RSS24/step-type diagnostic has 20 normal exits. Those deterministic runs are
not independent statistical seeds. Earlier smoke results are not extra evidence
for significance. Canonical archive in the research worktree:

`results/dranppr_native_matched_20260923/`

It contains `d4`, `d9`, both `*_rss24_diagnostic` folders, test XML and the
official paper's iMESA wrapper sources. Each primary folder includes full
messages/traces/states, evaluation, manifest, immutable executable copies,
linked-library hashes, static-library hashes and pinned baseline source snapshots.

```sh
cmake --build /tmp/dranppr_matched_build -j2
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_native_matched.py tests/test_audit_native_matched.py tests/test_native_joint_audit.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_native_matched --package /tmp/dranppr_matched_bootstrap_d9_20260923/frozen.json --output NEW_OUTPUT --steps 1 10 --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
```

Use the archived `baseline_CMakeCache.txt` for explicit pinned configure paths.
The packaged DBA-DRANPP repository is unchanged and clean. No commit/push was
made. CCI/D-CCI is unchanged. Online suffix continuation, timing fairness,
held-out transfer and native statistical adaptation remain open research gates.
