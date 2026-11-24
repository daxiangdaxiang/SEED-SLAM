# Incremental Response Feasibility Audit

## Scope

This is a source audit, reanalysis of an existing measured trace, and rerun of
existing native contract tests. No estimator implementation, noise model,
communication schedule, or benchmark state is changed. No new speedup has been
measured. The proposed next intervention below requires design approval.

The input is the completed common-past D9 one-PCG suffix archived under
`results/dranppr_native_online_20260923/d9_one/dran/`. It has 2514 frames and
9594 terminal physical factors, of which 4222 precede the suffix. All frames
accept one joint correction. Setup is 0.038085920 s; warm cutoff is
131.72300004959106 s. The first/last suffix frames arrive at
131.74699997901917 / 299.98400020599365 s, relative to the stream origin.

## Where The Work Actually Goes

The recorded 204.515367205 s of frame service includes:

| Disjoint stage | Seconds | Fraction of service |
| --- | ---: | ---: |
| Accepted-state publication/model refresh | 127.260998 | 62.23% |
| Local model export and response construction | 31.094145 | 15.20% |
| Interface solve | 7.977961 | 3.90% |
| Physical-loss evaluation and scalar collectives | 18.902168 | 9.24% |

Other frame work includes insertion, identity coordination and uninstrumented
overhead. The publication category has these NESTED counters; do not add them
again to the table above:

- Full model snapshot: 19.819212 s.
- Factor preparation/linearization: 29.850538 s. This timer also covers factor
  cloning, validation and normal-contribution construction, not Jacobians alone.
- Subtracting old and inserting new factor contributions: 48.155902 s.
- Remaining publication work: approximately 29.435346 s, not isolated further
  by these counters. It includes work outside the three nested intervals; do
  not attribute it all to a particular allocator or validation operation.

There are 17,416,801 repeated factor linearizations, 12,570 response builds
(five robots per frame), and zero response reuses. These are repeated work
counts, not unique measurements or additional evidence.

The source explains the behavior:

- `experiments/native_local/joint_step.h:163`: the fresh arm sends every free
  key to `relinearizeValues` after each accepted step.
- `src/IncrementalLocalModel.cpp:147`: each nonempty transaction first copies
  the complete state, including incidence and accumulated normal blocks.
- `src/IncrementalLocalModel.cpp:180`: all factors incident to rebased keys are
  refreshed. This is necessary for the current single-chart contract.
- `src/IncrementalLocalModel.cpp:198`: refreshed contributions are subtracted
  and reinserted; `linearSystem()` subsequently exports the sparse matrix again.
- `src/NativeSchurModel.cpp`: each new response extracts the blocks and calls a
  fresh sparse private factorization.

Thus, calling this simply a slow linear solver or slow nonlinear Jacobian
evaluation would misidentify the problem. The whole publication/model lifecycle
is still batch-like. The prior incidence-only micro-optimization already failed
the end-to-end gate and is not reinstated by this audit.

## Optimistic Timing Subtraction

Recompute the existing virtual serial queue with
`finish_t = max(finish_previous, arrival_t) + service_t`, initialized at
`warm_cutoff + setup_seconds`. The unchanged recurrence reproduces every stored
`serial_backlog_s` value exactly in the executed awk calculation (maximum
absolute difference zero).

Then subtract selected disjoint measured stages from each frame's service,
leaving the arrival schedule and all other measured work unchanged:

| Counterfactual | Frame service s | Maximum state age s | Mean state age s |
| --- | ---: | ---: | ---: |
| Reproduced original trace | 204.515367 | 46.608870 | 10.418296 |
| Entire local model/response stage costs zero | 173.421222 | 24.958048 | 4.018413 |
| Model/response AND interface solve both cost zero | 165.443260 | 20.843306 | 3.294795 |

These are timing counterfactuals, NOT executions of faster estimators. They do
not predict how cache locality, allocations or state trajectories would change
under an actual implementation. Under the explicitly fixed-other-work
assumption, even eliminating the entire model stage gives only a 1.1793x
total-service speedup and does not remove the queue. Total work smaller than the
arrival interval also does not guarantee a small queue, because arrivals and
service costs are uneven. The serial model has no network latency, robot-level
parallelism or motion extrapolation.

## Library And Mathematical Constraints

The pinned GTSAM source was inspected, not just the public API names.
`GaussianISAM`/`ISAM::updateInternal` appends new factors to the removed top of
the existing tree; this is not an arbitrary factor-replacement API. Appending a
new linearization without removing the old contribution double-counts data.

`ISAM2::recalculate` removes affected cliques and their paths to the root,
preserves unaffected subtrees, and reconstructs the affected region from
original factors plus cached boundary contributions. Its source explicitly
warns against reconstructing replacement caches from inappropriate old
conditionals. In the pinned implementation, a sufficiently large affected set
can trigger a batch path. Merely using this data structure cannot make global
relinearization local.

The primary literature establishes selective relinearization and incremental
elimination as existing techniques, not prospective DRANPPR innovations:
[Kaess et al., iSAM2, IJRR 2012](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf).
The official [ISAM2 API](https://gtsam.org/doxygen/a04947.html) documents factor
removal and incremental update separately from estimate extraction. Local
inspection uses the pinned GTSAM revision already recorded in the online
experiment manifest, not a claim that current website code is that revision.

For a fixed declared quadratic, partitioning private and shared variables gives

```text
q(d_P, d_B) = q_B(d_B)
           + 0.5 (d_P - d_P_star(d_B))^T H_PP (d_P - d_P_star(d_B)),

d_P_star(d_B) = -H_PP^{-1}(g_P + H_PB d_B),
q_B(d_B) = c_B + b^T d_B + 0.5 d_B^T S d_B.
```

This completion-of-the-square identity assumes H_PP is positive definite;
it is not a nonlinear marginalization identity. Changing the published d_B
does not, by itself, change that quadratic's elimination. Changing a factor,
noise model, or declared linearization base does. This is the useful reuse
boundary: state recovery and model rebuilding must not be conflated.

## Alternatives And Proposed Next Gate

1. Recommended: a persistent, separator-conditioned local elimination model,
   using the pinned library's elimination primitives. Keep original factors and
   stable IDs; retain private conditionals and the reduced shared-variable model.
   Append/replacement/removal/relinearization updates rebuild their actual
   dependency closure, not a complete duplicate normal-block map. Publishing a
   solution uses conditional recovery without automatically rebasing all keys.
   No new statistical threshold or fixed history window is introduced.
2. Only accelerate the current sparse factorization or normal-block bookkeeping.
   This is easier but misses the dominant lifecycle work. The timing subtraction
   and withdrawn micro-optimization make it a weak primary research direction.
3. Fixed-lag marginalization. This limits active history but introduces a window
   choice and freezes historical nonlinear information. It changes the comparison
   and complicates later association/noise edits, so it is not the next control.

The recommended direction is NOT simply installing iSAM2 as a replacement
network algorithm. The DRAN shared-variable response and topology-local solve
remain. Established incremental inference is implementation infrastructure; any
novelty claim must come from the information-consistent distributed formulation
and subsequently demonstrated robustness/online properties.

First compare the incremental response with a fresh reconstruction at IDENTICAL
declared charts and factor revisions: H/g equivalence, Schur products, private
recovery and gauge handling. Include local edits, old-factor removal/noise
replacement, boundary membership changes, and a deliberately nonlocal update.
Require visible reused/reeliminated work counts. Keep nonlinear scheduling
separate: the current cached-plus-one-PCG result is negative, and reusing its
unchanged refresh rule does not guarantee useful locality. Do not conceal an
all-history affected set by relaxing the existing 1e-4 threshold.

Only after those checks should small D4/D9 causal slices test total service,
communication and state parity. A speedup in a nested timer is insufficient.
Full replays, held-out sequences and statistical adaptation follow only if this
gate passes. Whole-history response recovery and physical-loss checks remain
possible O(history) costs; no constant-time claim is made.

## Verification In This Turn

```sh
cmake --build /tmp/dranppr_backend_build --target testIncrementalLocalModel -j2
/tmp/dranppr_backend_build/bin/testIncrementalLocalModel \
  --gtest_output=xml:/tmp/dranppr_incremental_feasibility_20260923.xml
```

All 39 existing native tests pass. These cover current-model factor lifecycle,
chart consistency, fresh Schur parity and routed interface algebra; they do not
validate an unimplemented incremental tree. No new PCG run was performed.
The reanalyzed D9 trace retains its 2514 PCG iterations and maximum relative
residual 4.242834; it is a capped solve, not convergence. CCI/D-CCI unchanged;
MPI: 0 ranks. Packaged DBA-DRANPP is clean and untouched. No commit or push.
