# Native riSAM Bridge: Strong Baseline and a Longer-Sequence Failure

## Purpose and Scope

Replace inference from a two-robot Point2 toy problem with native observation-time
Pose2/Point2 factor graphs. This is the local solver used by riMESA, not a complete
distributed riMESA experiment. Five independent robot graphs are replayed without
communication, frame registration or relationship edits. Each local graph fixes
only its own coordinate gauge and is evaluated with its own rigid alignment.
Do not present the aggregate errors as globally aligned multi-robot ATE.

The C++ bridge uses the pinned paper GTSAM dependency and the isolated riSAM copy
with the endpoint and mixed-merit repairs described in the preceding numerical
audit. It preserves native range/bearing factors, not an isotropic Cartesian
surrogate. Barcode identities are provided by the input; no evaluation-only label
permutation enters estimation. Truth is loaded by a separate post-run evaluator.

## Implementation and Checks

- `experiments/native_risam/replay.cpp`: persistent local graph per robot,
  timestamp-atomic insertion, raw polar factors, causal/final states and timings.
- `experiments/native_risam/CMakeLists.txt`: explicit pinned upstream dependency
  paths; no change to the production DRAN++ build or default solver.
- `experiments/audit_native_risam.py`: sequential bounded runs, complete failure
  retention, source/binary/library hashes, and post-run evaluation.
- `tests/test_native_risam_replay.py`: exact geometry, Gaussian parity, future
  invariance, simultaneous-robot frame atomicity, and malformed-stream rejection.

The first bridge failed the simultaneous-robot test because packets belonging
to one physical robot/timestamp could be separated by another robot's initial
pose packet. It now collects one timestamp bucket and inserts each robot's whole
frame atomically. The complete new test file passes 12/12; with existing local
incremental and evaluation tests, 36/36 pass. On Dataset 4, all 1,710 causal pose
outputs of the Gaussian C++ bridge exactly match the existing Python reference:
maximum absolute x/y/yaw difference is zero. This validates the factor and
chronology bridge, not distributed operation.

The producer is a streaming consumer with a one-timestamp buffer. It does not
inspect future observations to initialize landmarks. Physical source IDs are
preserved and duplicated observations are rejected. A failed process leaves its
partial output and is not resumed with an unknown solver state.

## Fixed Protocol

Observation sigmas are `(0.05 rad, 0.1 m)` in bearing/range order; odometry sigmas
are `(0.1, 0.1, 0.05) * sqrt(dt)`. These remain declared nominal models, not known
true sensor covariances. No tuning or corruption is introduced in this audit.

- `isam2`: Gaussian factors, QR, relinearization threshold `1e-4`, wildfire `1e-3`.
- `risam`: Gaussian odometry, official SIG graduated observation factors,
  upstream local optimizer defaults, and the disclosed numerical repairs.
- `risam_gn`: same graduated observation factors and GNC, but the Gaussian
  control's QR/relinearization settings with ordinary GN rather than extended
  Dogleg. This changes more than the step formula and is labelled an ablation.

SIG uses the already pinned harness convention `shapeParamFromInfThresh(0.1,2,0.95)`,
`STABLE` graduation and `EQUAL_5` initialization increments. In `risam`, upstream
relinearization threshold is `0.1` and factorization is Cholesky. Do not attribute
every difference from `isam2` to robustness alone.

Dataset 1/4 run from 0 to 120 seconds; Dataset 9 runs from 0 to 300 seconds.
The main accuracy rows below evaluate poses after second 30, using all previous
observations for estimation, matching the earlier adaptation study. Full-prefix
accuracy is also retained. Every binary runs serially with a 180-second timeout.
All nine runs, including the later Dataset 9 GN control, complete without
failures or timeouts.

Per-frame time includes factor construction, solver update and extracting the
current full estimate. Solve and extraction times are logged separately. JSON
parsing, CSV IO and post-run truth evaluation are excluded from that timer;
whole-process wall time is also recorded. Backlog is a simulated serial five-robot
work queue, not a measured distributed critical path. Initialization and all
pre-30-second updates are included in the work totals.

## Dataset 1 and 4: Robust Baseline Is Strong

Final and causal trajectory errors are independently aligned local RMSEs.

| Dataset | Method | Final trajectory (m) | Causal trajectory (m) | Object copies (m) | Total update work (s) | Per-frame p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Gaussian iSAM2 | 0.24138 | 0.31749 | 0.80231 | 1.482 | 4.283 |
| 1 | riSAM, repaired | 0.18315 | 0.28408 | 0.68918 | 8.106 | 15.772 |
| 1 | riSAM GN ablation | 0.17707 | 0.28769 | 0.68541 | 18.574 | 34.622 |
| 4 | Gaussian iSAM2 | 0.22275 | 0.22595 | 0.92346 | 2.465 | 4.770 |
| 4 | riSAM, repaired | 0.16726 | 0.17789 | 0.83038 | 10.338 | 12.793 |
| 4 | riSAM GN ablation | 0.16904 | 0.17594 | 0.84922 | 26.926 | 38.489 |

Default-local riSAM improves final trajectory RMSE on all five Dataset 4 robots
and four of five Dataset 1 robots. Dataset 1 Robot 2 is a small counterexample:
`0.19193 -> 0.19974 m`. The GN control is not uniformly better, despite higher
computation. Retain both benefits and regressions rather than selecting one row.

The aggregate work ratio of default-local riSAM to Gaussian is about 5.47x on
Dataset 1 and 4.19x on Dataset 4. Nevertheless, their maximum serial backlogs
are only 31.8 and 29.8 ms in these short streams. This is additional computation,
not a demonstrated inability to keep up with acquisition in these prefixes.
The runs are sequential but not repeated timing trials; ratios are descriptive.

### Comparison With Our Existing Noise Adaptation

The prior joint Q/R adaptation, evaluated on the same after-30-second poses,
reached final RMSE `0.16221 m` on Dataset 4 and `0.25031 m` on Dataset 1. The
new strong robust reference reaches `0.16726` and `0.18315 m`, respectively.
Thus the large Dataset 4 gain over Gaussian alone is not strong evidence of
an advantage over robust SOTA: the remaining gain is small, while transfer to
Dataset 1 is substantially worse. The native adaptation gate remains failed.
No equal-compute comparison is claimed; calibration/prediction work of the
adaptive arms is accounted for in their separate report.

## Dataset 9: A Reproducible Local Failure, Not Yet a riMESA Failure

The 300-second prefix contains 4,541 frames and 9,599 input packets across all
five robots. It uses the same code and parameters as Dataset 1/4.

| Method | Final trajectory (m) | Causal trajectory (m) | Object copies (m) | Total update work (s) | p99 (ms) | Max serial backlog (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian iSAM2 | 0.77756 | 1.11697 | 1.26194 | 9.435 | 13.223 | 0.0203 |
| riSAM, repaired | 1.34445 | 2.03831 | 2.31070 | 68.635 | 52.805 | 0.1073 |
| riSAM GN ablation | 1.27328 | 2.07616 | 2.37487 | 155.717 | 117.510 | 22.7001 |

The largest final-trajectory regression is Robot 5: `0.46968 -> 2.16851 m`.
Robot 3 improves final trajectory (`0.39989 -> 0.33277 m`) but worsens its
object error (`0.42374 -> 0.84571 m`). This is not a uniform failure on all robots.

The Robot 5 causal traces initially agree. Their independently aligned prefix
RMSEs at 120 seconds are `0.02038` and `0.02030 m`; at 180 seconds they are
`0.33765` and `0.54511 m`; at 300 seconds they are `1.64403` and `3.50331 m`.
Each prefix has a separately fitted evaluation alignment, so these values are
not instantaneous online error estimates. In the identical estimator gauge,
the two Robot 5 causal positions first differ by over 0.5 m at 161.917 seconds
(pose index 459), with maximum difference 8.659 m at 225.050 seconds.

This is a native, uncorrupted observation-stream counterexample to uniform
local robust improvement. It does not yet establish the cause. Candidate
explanations include incorrect noise scaling, nonlinear state/model error,
premature downweighting of informative observations, and local optimizer
settings. No relationship corruption was injected. Dataset 9's previously
audited reference-yaw jumps also prohibit treating every truth-conditioned
residual as a sensor fault label. The later GN control also remains poor and
costs more computation, so merely using its tighter relinearization/QR/GN
configuration does not recover this case. A same-optimizer Gaussian control
and same-target completion remain necessary before attributing it to the kernel.

Despite 7.27x additional work, this 300-second riSAM prefix still finishes its
updates well within acquisition duration. The older full-history backlog result
cannot be extrapolated into a demonstrated riSAM deadline failure here.
The explicit GN ablation does accumulate 22.7 seconds of serial backlog despite
its total work being less than the 300-second acquisition interval. Bursty
latency and total throughput are therefore different measures. This is an
ablation failure, not the default-local riSAM latency result.

### Frozen-Objective Completion: Original Python Diagnostic

`experiments/audit_risam_checkpoint.py` reconstructs the identical past-only
Robot 5 graph, starts from its saved riSAM state, and runs up to 100 LM iterations
without adding data or changing the noise model. At final graduation `mu=1`,
the official SIG cost equals Geman--McClure with the same shape. A unit test
checks both cost and linearized weight equality and preservation of the ordinary
odometry term. The gauge remains fixed rather than fitted against truth.

The repeatable numerical output is: robust objective `414.66070 -> 411.71077`,
free-variable gradient norm `745.22 -> 0.000598`, but independently aligned
after-30-second trajectory RMSE `2.16851 -> 2.19777 m` and object RMSE
`4.16558 -> 4.20620 m`. This suggests that completing the current local solve
alone does not recover the desired state. It does not distinguish a wrong local
basin from a misspecified robust objective and is not a global optimum certificate.

This diagnostic is NOT yet a clean validation gate: its first Python process
printed and wrote the result, then exited with signal 11 (status 139). An
unchanged repeat exited zero with identical numerical values; a GDB run also
exited normally. Two combined test invocations printed 37 passed assertions but
then exited 139; the same tests under GDB exited normally. The crash site was
not captured, so no root-cause or repair claim is made. This Python binding
loads system `/usr/local/lib/libgtsam.so.4` and TBB, whereas the native C++ runs
use the pinned paper library. The native C++ runs and their evaluation harnesses
all exited zero. Preserve all three completion attempts and verify completion
again in the pinned C++ environment before using it as primary evidence.

**Subsequent native revalidation:** the pinned C++ completion now exits normally
and reproduces these values. A two-start/two-target check additionally shows
that SIG completion from the much better Gaussian state lowers robust cost
670.496 -> 473.058 but worsens trajectory RMSE 0.46968 -> 0.84491 m. The target
gain trades odometry decrease for observation disagreement, increasing weights
below 0.1 from 4 to 91. A raw co-observation consistency audit supplies independent
geometric evidence for only 13 of those 91 observations, so neither universal
fault classification nor a new rejection gate is justified. See
`docs/dranppr_risam_fixed_objective_audit_20260922.md` for controls, results,
limitations and reproduction. The old Python shutdown failure remains unresolved.

## Next Causal Test and Remaining Gates

Freeze the Dataset 9 Robot 5 measurements around the observed divergence, then
separately change local solver completion and robustness while preserving the
factor likelihood. Inspect factor weights and the support for the disputed state
using already-arrived observations. Test whether additional state correction
restores consistency before interpreting large residuals as sensor degradation.
Do not tune a new rejection threshold or covariance floor against the truth.

Full distributed comparison still needs causal coordinate-frame initialization
and one coherent network gauge. Fixing five separately registered root poses as
five exact global priors would change the problem and hide registration error.
The ordinary riMESA `update` wrapper also does not expose factor removals,
although its underlying riSAM does; an initialization-prior release must therefore
be designed and verified rather than silently inserted into a baseline.

The actual persistent incremental DRAN++ implementation, calibrated parameter
uncertainty, identity-ambiguity response, and matched communication/time-budget
network experiments remain open. This bridge is necessary input/baseline
validation, not a replacement for any of those requirements.

## Reproduction and Artifacts

```sh
cmake -S experiments/native_risam -B /tmp/dranppr-native-risam-build -DCMAKE_BUILD_TYPE=Release -DRISAM_SOURCE_DIR=/tmp/dranpp_noise_audit/risam-endpoint-merit-diagnostic -DRISAM_LIBRARY=/tmp/dranpp_noise_audit/paper-endpoint-merit-build/_deps/risam-build/librisam.a -DGTSAM_DIR=/tmp/dranpp_noise_audit/gtsam-paper-build -DGTSAM_INCLUDE_DIR=/tmp/dranpp_noise_audit/gtsam-paper
cmake --build /tmp/dranppr-native-risam-build -j2
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DRANPPR_NATIVE_RISAM_BINARY=/tmp/dranppr-native-risam-build/native-risam-replay python3 -m pytest -q tests/test_native_risam_replay.py tests/test_dranppr_incremental.py tests/test_evaluate_incremental_reference.py
```

Archived results are under `results/dranppr_native_risam_bridge_20260922/`:
`datasets1_4_120s/`, `dataset9_300s/` and `dataset9_gn_300s/`. Each contains complete causal/final
states, per-frame work, summaries, per-robot evaluations, source snapshots and
a manifest with the exact run arguments, linked library hash, patched-source
diff, input digests and binary hash. Original outputs under `/tmp` are retained;
the first overlapping smoke runs are not used for timing claims.

`completion_first_exit139/`, `completion_repeat_exit0/`, and
`completion_gdb_normal/` preserve the provisional completion diagnostic rather
than discarding its failed process. The source script and unit test are copied
beside those attempts. Normal GDB exits do not prove the intermittent shutdown
failure has been repaired.

No production DRAN++/DBA code, CCI/D-CCI behavior or MPI path was changed.
