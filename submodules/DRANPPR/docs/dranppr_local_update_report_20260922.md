# Local Numerical Replacement: Negative End-to-End Result

## Intervention and Protocol

Change only maintenance of same-layout relinearized/replaced factors: retain
incidence, block support counts and factor-map nodes while replacing numerical
contributions. Keep copy-on-publish rollback, physical likelihood, identity,
chart tolerance, interface precision and arrival schedule unchanged.

Run control A and treatment B on D4/120 and D9/300, then reverse-order B/A
repeats using archived unchanged binaries. This gives eight causal replays and
four separate frozen LM completions, all exit zero. It is two deterministic
runs per arm/case, not independent data seeds or isolated-hardware benchmarking.
Small read-only development/test activity occurred between repeated runs; no
heavy build or other estimator overlapped measured replays. Do not claim timing
confidence intervals. GT is accessed only by postprocessing.

## Measured Work

| Case | Arm | First service s | Repeat service s | First contribution s | Repeat contribution s |
| --- | --- | ---: | ---: | ---: | ---: |
| D4/120 | control | 33.093 | 31.346 | 4.877 | 4.788 |
| D4/120 | treatment | 34.228 | 32.658 | 2.242 | 2.159 |
| D9/300 | control | 229.314 | 172.508 | 24.686 | 19.810 |
| D9/300 | treatment | 229.271 | 183.466 | 13.719 | 10.874 |

The nested contribution stage improves, but total service does not establish a
win. The unchanged D9 control itself varies by 56.806 s. Other work stages also
change; neither machine effects nor allocation/sparsity effects have been
isolated. The evidence does not justify attributing all variation to either.

All repeated final/causal state files are byte-identical within each binary.
Across control/treatment, maximum final/causal positional differences are
9.47e-14/4.68e-12 m on D4 and 2.61e-12/3.18e-10 m on D9. Accepted-step decisions
are unchanged. D4/D9 final physical costs are 1230.3885441654/3280.2840141647;
gaps to separate frozen completions are 2.42e-6/9.82e-5. D9 final evaluation
excludes 437 unsupported GT timestamps, as before; causal connected tails have
2515 supported timestamps and no omissions.

PCG totals are 59543/59543 on D4 and 101606/101609 on D9 (control/treatment).
Roundoff changes individual iteration counts on 2/413 frames. Both arms retain
verified maximum relative residual below 1e-10. D9 application bytes change
1026727008 -> 1026617104, not exact communication parity and not a meaningful
compression result. D4 stays at 585202432 bytes. MPI: 0 ranks; CCI unchanged.

## Delivery and Decision

The same control D9 state outputs have virtual serial-delivery RMSE 2.319 m in
the first run and .824 m in the repeat, while measurement-time RMSE remains
.559461 m. Treatment gives 2.734/1.163 m. These differences arise from measured
service schedules in a zero-network-latency serial queue with no motion
extrapolation, not changed sensing or association. They are not deployed
multi-robot timing measurements. Smaller maximum backlog alone does not imply
smaller mean state age or better stale-delivery error.

Withdraw the numerical-only path rather than retain a new runtime switch or
promote a local microbenchmark. Active core source and joint replay/audit
binaries are restored exactly to the instrumented control. Preserve profiling
and the additional lifecycle/overflow tests. The broader chart-reuse work is
unchanged; only this latest intervention is withdrawn.

Both treatment and restored code pass 39 native tests and 69 Python replay/
evaluation checks; six unchanged legacy Schur/owner-PCG checks pass. Restored
code plus the separately repaired predictive-evidence tests passes 86 Python
checks. These are scoped regressions, not the entire repository test suite.

## Artifacts

- `/tmp/dranppr_local_update_control_20260922`: immutable control sources,
  binaries, two streams/completions, input/library hashes and initial red tests.
- `/tmp/dranppr_local_update_treatment_20260922`: corresponding treatment,
  retained numerical-path tests and passing native/Python/legacy test XML.
- `/tmp/dranppr_local_update_repeats_20260922`: exact repeat commands/exits,
  source-manifest hashes, `analyze.py`, `analysis.json`, restored test XML.

Restored replay SHA256:
`a50a4e57ba772d6f2ca1f0f66355c1f1a8a561c6b877c5384e92bde2917c2bd1`.
Treatment replay SHA256:
`61c5919b59312d47fcc96bee8339a5be1b2d25cd43193bbdaac228e0035bf7a3`.

This closes the local-update experiment negatively, not the research goal.
Do not continue adding cache mechanisms in place of native matched-budget
baselines, cause-aware prediction/adaptation, or genuine incremental elimination.
