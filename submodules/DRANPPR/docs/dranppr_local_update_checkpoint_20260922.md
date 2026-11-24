# Local Update Work: Checkpoint, Not a Performance Result

Historical checkpoint. The completed paired audit is now recorded in
`dranppr_local_update_report_20260922.md`. The numerical-only path was withdrawn
from active code after failing to demonstrate an end-to-end speedup; its source,
binaries and traces remain in the experiment artifacts.

The instrumented cached-tight control completes D4/120 and D9/300, plus both
frozen completions, with exit zero. Final and causal state CSVs are byte-identical
to the previous inexact-joint cached-tight outputs. Sources, baseline binaries,
linked-library hashes, commands and exits are under
`/tmp/dranppr_local_update_control_20260922`.

| Control | Total service s | Snapshot s | Factor linearization s | Contribution maintenance s |
| --- | ---: | ---: | ---: | ---: |
| D4/120 | 33.09324 | 1.48566 | 2.94190 | 4.87745 |
| D9/300 | 229.31421 | 14.56944 | 15.68703 | 24.68646 |

The last three columns are nested joint-model update timings, not additional
costs. Insertion, gauge changes and independent singleton updates are excluded
from these nested counters. Contribution maintenance accounts for 14.7%/10.8%
of total service, supporting the planned single numerical-update intervention.

Timing is not stable across runs: the preceding cached-tight D9 run reported
167.477 s, while this numerically identical control takes 229.314 s. Do not
attribute that difference to the algorithm or quote a reliable speedup without
interleaved unchanged-binary repetitions. No CPU-affinity/isolation experiment
has established the source of the timing variability.

The research implementation now preserves incidence, Hessian block support
counts and factor-map entries for relinearized/replaced factors whose ordered
keys and dimensions do not change. It subtracts all old numerical contributions
before adding new contributions. Structural edits retain the generic path.
Copy-on-publish rollback remains. No solver/noise/identity tolerance changes.

Three focused C++ tests first failed on the missing numerical-update behavior;
their XML is retained as `red_tests.xml`. A streaming test also failed on the
missing counter. After implementation, all 39 native C++ and 24 streaming Python
checks pass, exit zero. Coverage includes repeated covariance/chart changes,
mixed remove/rekey/refresh, extreme source IDs, remove-after-refresh, aggregate
overflow rollback and unchanged source-factor ownership. Native XML is retained
as `green_tests.xml`. This is correctness evidence, not a speedup measurement.

Still pending: real treatment replays, interleaved timing controls, full Python
and legacy regression coverage, fresh completions and delivery comparison.
Arithmetic cancellation differs where the old structural path discarded a
temporarily unsupported block; real numerical parity must be measured, not
assumed from unit tests. This implementation does not provide incremental sparse
elimination, bounded update latency, statistical adaptation or a SOTA comparison.
The overall research goal and the paired-verification plan remain open.
