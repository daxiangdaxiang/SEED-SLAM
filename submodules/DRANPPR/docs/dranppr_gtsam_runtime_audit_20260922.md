# Python/GTSAM Shutdown Runtime Audit

## Scope and Decision

The preceding noise experiments sometimes wrote complete outputs and then
exited with SIGSEGV. This audit isolates a dependency-lifecycle failure and
validates a process-scoped mitigation. It changes neither the estimator nor
the packaged DRAN++ backend. Historical abnormal runs remain abnormal.

For this host's Python/GTSAM experiments, explicitly preload the system TBB
allocator in the experiment process only:

```sh
env LD_PRELOAD=/lib/x86_64-linux-gnu/libtbbmalloc.so.2 \
  PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 experiments/gtsam_shutdown_probe.py --mode linearize --dump-maps
```

This is a documented host-specific dependency workaround, not a portable
upstream fix. Do not export it globally, replace the user's installed libraries,
apply it to the separate no-TBB native baseline binary, or suppress an abnormal
exit with `os._exit`, a sleep, or an exception handler.

## Evidence

The active environment uses Conda Python 3.9, NumPy 1.21.5, SciPy 1.9.1, the
locally installed GTSAM 4.2a9 binding and `/usr/local/lib/libgtsam.so.4`.
System TBB is Debian package `2020.1-2`; Conda's allocator comes from TBB
`2021.6.0`. The Python binding's former build-directory RUNPATH no longer exists.

Actual process mappings, rather than shell `ldd` alone, reveal three libraries:

- `/usr/lib/x86_64-linux-gnu/libtbb.so.2`
- `/usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2`
- `/home/jackal/anaconda3/lib/libtbbmalloc.so.2.6`

The loader trace shows GTSAM's dependency resolving to the Conda allocator,
followed by system TBB dynamically loading the system allocator by its path.
Thus two allocator implementations coexist in one process.

On the real D4 audit, GDB catches SIGSEGV in a TBB worker while the main thread
is inside process finalization (`_dl_fini`, through an MKL finalizer). The
worker stack contains system `libtbb.so.2`; it does not contain our fitting
routine. A separate address-mapping diagnostic records worker PC
`0x7fffe55cd630`, which belongs to the pre-shutdown system allocator mapping
at base `0x7fffe55b8000`. Offset `0x15630` is `scalable_aligned_free` in that
binary. The debugger loses the process during the latter teardown, so that run
does not establish the exact unmapping/destructor order.

The supported mechanism is therefore a TBB/allocator teardown failure in the
mixed-library environment. The precise upstream lifetime defect has not been
proved. A similar TBB-dependent wrapper exit crash is reported upstream on a
different platform; that is corroborating context, not proof that it has the
same root cause.
[GTSAM issue 2489](https://github.com/borglab/gtsam/issues/2489).

## Minimal Reproducer and Controlled Intervention

`experiments/gtsam_shutdown_probe.py` uses a deterministic 100-pose odometry
chain, with no MRCLAM data or DRANPPR statistical code. Four modes separately
import GTSAM, evaluate graph error, linearize the graph, and run LM. The parent
checks each child process exit after its JSON output, not just the printed
calculation result.

The saved controlled run has ten serial subprocesses per cell:

| Operation | Default environment: SIGSEGV / runs | System allocator preloaded: SIGSEGV / runs |
| --- | ---: | ---: |
| Import | 0 / 10 | 0 / 10 |
| Graph error | 0 / 10 | 0 / 10 |
| Linearization | 3 / 10 | 0 / 10 |
| LM optimization | 4 / 10 | 0 / 10 |

All 80 subprocesses finish their calculation and print a result; seven then
fail at shutdown. Error is exactly `209.89819849614494`, linear graph size is
100, and optimized error is exactly `6.4459040332432e-11` in both environments.
All preloaded subprocesses map only the system runtime and system allocator.
There are no per-dataset or algorithm parameter changes.

Earlier exploratory runs also reproduce the failure: a five-repeat probe has
5/5 linearization and 3/5 optimization shutdown failures; another ten-repeat
linearization probe has 8/10. Preloading either the system or Conda allocator
gives 0/10 failures in that latter probe and removes the duplicate allocator
mapping. This means the evidence favors library loading/lifetime interaction,
not a claim that the Conda allocator itself is intrinsically broken. Failure
frequency depends on timing; these are runtime repetitions, not independent
SLAM accuracy seeds or a universal reliability guarantee.

Two debugger commands mistakenly supplied redirection through GDB's `run`
argument, resetting the inferior's Python arguments; they executed no probe.
Another wrapper attempt failed to add `experiments/` to `sys.path`. Their logs
are retained but excluded from the counts and from numerical validation. A
minimal probe paused at exit also terminated normally; the debugger returned
one because a later mapping query had no live process. A debugger exit code is
not used as a substitute for the inferior's recorded outcome.

## Native Reproduction and Tests

The original D4/R3 30-second warmup variance-boundary audit was rerun twice with
only the process-scoped system allocator preload. Both processes exit zero.
All four contrast-model NPZ arrays are bitwise equal to the saved reconstruction,
and `summary.csv` and `slices.csv` are byte-identical. The statistical findings,
including the negative log-uniform-prior result, are unchanged. These are new
normal runs; they do not retroactively turn old failed runs into passes.

The following targeted tests also terminate normally, with 42 passed:

```sh
env LD_PRELOAD=/lib/x86_64-linux-gnu/libtbbmalloc.so.2 \
  PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q \
  tests/test_dranppr_incremental.py \
  tests/test_dranppr_sparse_linearization.py \
  tests/test_dranppr_sparse_prediction.py \
  tests/test_dranppr_variance_components.py \
  tests/test_audit_online_lower.py \
  tests/test_audit_online_checkpoint.py \
  tests/test_variance_boundary_audit.py
```

This covers existing local factor lifecycle, sparse linearization/prediction,
variance components and audit reconstruction helpers. It is not a test of a
new native distributed incremental DRAN backend, which remains unimplemented.

## Artifacts and Limits

`results/dranppr_gtsam_runtime_audit_20260922/` preserves the exact probe, all
80 child exit codes/stdout/stderr/mappings, the valid crash stack, loader trace,
inconclusive/invalid debugger logs, both normal D4 outputs and binary hashes.
The initial malformed runpy invocation's traceback was recorded in the tool
output; its saved GDB log only contains the inferior exit and debugger output.

No globally installed dependency, CCI behavior, production DRAN C++ source, or
original packaged repository was changed. The no-TBB pinned C++ riSAM/DRAN
builds are a separate environment and do not receive this workaround.

Thread-count caveat: BLAS/OpenMP environment limits do not constrain the legacy
TBB worker pool. Do not reinterpret prior Python timing as a strictly
single-threaded GTSAM benchmark, or directly rank its timing against the
separately configured no-TBB C++ backend. Future matched runtime comparisons
must pin the whole dependency and threading configuration.
