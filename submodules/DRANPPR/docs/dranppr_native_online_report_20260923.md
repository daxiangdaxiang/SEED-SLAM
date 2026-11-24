# Common-Past Persistent Online Suffix

## Protocol

This extends the frozen-prefix control, not a new estimator or a completed
adaptive DRANPPR system. D4 resumes after 15.534 s through 120 s; D9 resumes
after 131.723 s through 300 s. The common primal states are the previously
archived past-only DRAN bootstrap. Every backend starts with cold internal
models, duals and robust weights. This deliberately conditions on the same
past state; it is not an unbiased from-scratch startup comparison.

The native solvers persist across frames. A frame contains one robot's new pose
and all observations at that timestamp. Equal-time robot frames use ascending
robot ID. The prefix is checked against raw pose times, odometry, observation
source IDs, ranges, bearings and labels before skipping it. No future state or
GT is supplied to an estimator. Truth is opened only after all runs terminate.

Only one network root pose is fixed. Stable local barcode keys survive arrival
of new objects and merging observer components. Same-label copies are shared
only across edges whose two endpoints have observed that label. Consensus
error compares copies inside each observer-connected component, not copies
separated by nonobserving robots. These are known-barcode association controls,
not tests of the DBA upper level or rejection of false identities.

The D4 high-accuracy reference uses existing `joint_cached`: one GN correction
per frame, cached charts with the already-fixed 1e-4 refresh threshold and tight
interface PCG capped at 500 iterations. Its 584 MB suffix motivated the SAME
existing `joint_one` control on BOTH D4 and D9: one PCG iteration with fresh
local models, not cached-plus-inexact (that combination already failed an earlier
audit). This is one declared computational cap, not a per-dataset parameter
search or a newly adaptive stopping rule. Neither mode bounds local elimination
or total per-frame work.
The baseline configurations and isolated riSAM numerical repairs are the same
as the frozen native matched report. The paper-configured iMESA uses no outlier
oracle. `rimesa_gaussian` is an explicit ablation, not another named baseline.

Two schedules are recorded:

- `one`: one complete all-edge handshake sweep per frame, followed by local
  updates. The active robot first inserts its new physical factors. This is a
  matched arrival/opportunity schedule, **not equal communication** to DRAN.
- `bytes`: after the same arrived frame, perform complete baseline sweeps until
  the next sweep would exceed DRAN's cumulative application bytes at that frame.
  Unspent credit carries forward. The baseline's startup sweep is also charged.
  This is a matched causal ceiling, not literal byte equality or a deployable
  scheduler. It never uses future observations or GT to choose the credit.

The byte-filling schedule is intentionally a budget-response diagnostic. Forcing
a method to spend almost all credit can waste work after it has stabilized.
Its larger runtime must not be advertised as the runtime of the baseline's
preferred online policy. Read the one-sweep arms alongside the byte-matched arms.
The driver has a declared 900 s process watchdog; failures/timeouts are retained,
not ranked using partial graphs.

All methods count startup/update work, in-memory messages and logical 40-byte
headers. JSON input, trace serialization/file I/O and posthoc evaluation are
excluded from service time. Setup is separate and included in the virtual
serial queue's initial delay. Common bootstrap generation and transport framing,
ACKs, retransmission and latency are excluded for all methods. These are single
serial service realizations on one host, not replicated or distributed latency
benchmarks. Final accuracy and causal accuracy use separate single-global-SE(2)
posthoc alignments. Delivery error additionally compares the unextrapolated
causal state with truth at its virtual completion time.

## D4 Results

The suffix has 1562 new frames and 3679 packets. Every terminal graph contains
4031 physical factors, including the same 352-factor prefix. Final evaluation
has 1710 poses and 68 object copies. Application MB below means bytes / 1e6.

| Matched low-byte arm | MB | Trajectory RMSE m | Object-copy RMSE m | Causal RMSE m | Copy disagreement m |
| --- | ---: | ---: | ---: | ---: | ---: |
| DRAN, one PCG | 49.555912 | .285604 | .184824 | .272676 | 0 |
| Paper iMESA, no oracle | 49.555152 | .281234 | .255591 | .290106 | .569857 |
| Repaired riMESA | 49.555152 | .368693 | .536705 | .411337 | 3.377565 |
| Gaussian riMESA ablation | 49.555152 | .281146 | .255356 | .291349 | .567605 |

DRAN improves object and causal quality against the Gaussian controls but has
slightly worse final trajectory RMSE. It improves all three against the robust
riMESA configuration on this case. This is not uniform metric dominance.
Baseline budget slack is 760 bytes. DRAN's cost is 1230.453317; paper iMESA's
955.971922 is evaluated on inconsistent copies and is not a feasible lower
exact-consensus objective value.

The high-accuracy DRAN reference uses 583.975640 MB and reaches trajectory/object
RMSE .285678/.184898, causal RMSE .277174 and cost 1230.388544. One-PCG thus
reduces communication by 91.51% with small terminal state/cost changes, but
service work is 32.96 s versus 31.21 s, not faster. Full chart publication and
relinearization take 20.88 s in the fresh one-PCG path. This reconfirms the
previous finding: bounding interface iterations does not bound local work.

| Schedule | Paper iMESA | riMESA | Gaussian riMESA |
| --- | ---: | ---: | ---: |
| One sweep/frame, actual MB | 5.795528 | 5.795528 | 5.795528 |
| One sweep/frame, trajectory/object RMSE m | .281216 / .254721 | .371779 / .539241 | .281712 / .255417 |
| Large byte ceiling, actual MB | 583.973472 | 583.973472 | 583.973472 |
| Large byte ceiling, trajectory/object RMSE m | .281412 / .255363 | .371788 / .539724 | .281186 / .256000 |

Spending roughly 100 times the communication barely changes terminal state
quality and does not remove copy disagreement. The large-credit arms consume
153873 complete sweeps, versus 1563 including startup in the one-sweep arms.
Their long virtual queues are properties of this forced credit-filling policy,
not proof that the original algorithms necessarily require that much online
work. The repeated D4 one-sweep arms produce identical final and causal states;
they are protocol/timing repeats, not additional independent accuracy evidence.

## D9 Results And Delivery Failure

There are 2514 new frames and 5372 packets. Every terminal graph has 9594
physical factors, including the same 4222-factor prefix. Final GT evaluation
has 4104 supported poses, 437 omitted prefix timestamps and 73 object copies.
All 2514 suffix poses support both measurement-time and delivery-time evaluation.

| Matched low-byte arm | MB | Trajectory RMSE m | Object-copy RMSE m | Causal RMSE m | Copy disagreement m |
| --- | ---: | ---: | ---: | ---: | ---: |
| DRAN, one PCG | 88.730152 | .401417 | .229958 | .508964 | 0 |
| Paper iMESA, no oracle | 88.725480 | .495113 | .417262 | 1.207279 | 1.124526 |
| Repaired riMESA | 88.725480 | 1.746472 | 2.806287 | 3.191046 | 8.480520 |
| Gaussian riMESA ablation | 88.725480 | .494515 | .418057 | .919652 | 1.128246 |

At these per-frame byte ceilings, DRAN improves all three state metrics. Slack
is 4672 bytes; each baseline executes 21995 complete sweeps including startup.
DRAN's physical cost is 3280.434946 versus paper iMESA 2799.489030 on infeasible
copies. Robust riMESA reaches physical cost 199039.546533: its robust target
is different, and all its outputs are finite, so this is not a numerical-process
failure. Disabling BOTH robust observation and consensus factors while retaining
its local optimization configuration greatly improves this case. This does not
identify which robust family or which nonlinear basin caused the regression.
The common warm start and cold GNC states also remain important conditioning
choices; do not generalize this to an original from-scratch riMESA run.

| D9 execution arm | MB | Service s | Max serial state age s | Causal RMSE m | Unextrapolated delivery RMSE m |
| --- | ---: | ---: | ---: | ---: | ---: |
| DRAN, one PCG | 88.730 | 204.52 | 46.609 | .509 | 1.790 |
| Paper iMESA, one sweep/frame | 9.780 | 30.33 | .076 | 1.176 | 1.176 |
| Paper iMESA, byte ceiling | 88.725 | 198.59 | 51.082 | 1.207 | 3.298 |
| riMESA, one sweep/frame | 9.780 | 154.77 | 15.250 | 3.299 | 3.321 |
| riMESA, byte ceiling | 88.725 | 508.22 | 339.970 | 3.191 | 3.660 |
| Gaussian riMESA, one sweep/frame | 9.780 | 57.45 | .108 | .956 | .956 |
| Gaussian riMESA, byte ceiling | 88.725 | 415.81 | 249.709 | .920 | 3.930 |

**The current DRAN prototype fails the serial real-time gate.** Its much better
measurement-time estimate does not ensure the best timely state. In this
unextrapolated delivery diagnostic, the low-communication Gaussian controls beat
DRAN. Those arms use different compute and bytes, so this is a practical tradeoff,
not an equal-budget accuracy ranking. It is also not a proof of deployed
multi-processor infeasibility: no robot-parallel execution, network latency or
odometric forward prediction is modeled.

DRAN's one-PCG interface work is only 7.98 s, versus 127.26 s for accepted-state
publication/full relinearization, 31.09 s for local model/response work and
18.90 s for cost evaluation/collectives. Publication accounts for 62.2% of
service. Further reducing PCG iterations cannot fix the dominant workload.
Repeated historical relinearization/model maintenance needs genuinely incremental
reuse; the previously withdrawn incidence-only micro-optimization is not revived.

## Verified Gaussian Mechanism

The pinned Gaussian paper baseline updates each edge dual as
`u_next = beta*u + rho*(x-z)`; beta is 0.9 by default and the Point2 biased
prior uses unit noise, independently of the physical measurement precision.
Sources: pinned `rimesa/src/rimesa.cpp` lines 127--149 and
`rimesa/include/rimesa/biased_prior-inl.h` Point2 factory/evaluateError.

Consider two scalar coordinates of Point2 copies with physical costs
`0.5*w*(x1-0)^2` and `0.5*w*(x2-1)^2`, Gaussian factors, fixed rho and no
outliers. At a stationary synchronized state, `z=(x1+x2)/2`. For beta < 1,
the dual recurrence implies `u=rho*(x-z)/(1-beta)`. Substituting in the exact
local first-order equations gives an effective finite coupling
`kappa = rho*(1 + 1/(1-beta))/2` and disagreement
`x2-x1 = w/(w+2*kappa)`. For beta=1 the stationary dual equation instead
requires `x=z`. This is a fixed-point calculation, not a convergence theorem.

Thus, under these explicit assumptions, more precise or more numerous local
measurements increase disagreement relative to a fixed decayed-dual coupling.
More rounds alone need not recover the exact consensus solution. This is a
candidate explanation for an information-scale mismatch, independent of robust
measurement rejection. It is not yet an attribution of all nonlinear SLAM error
to decay, nor an argument to simply disable decay in a dynamic robust system.
The separate pinned-library `native-consensus-bias` diagnostic tests the
calculation without changing any real-data baseline parameters. It uses 5000
official API handshakes per case:

| Dual decay | Declared sigma | Predicted copy gap | Native copy gap |
| --- | ---: | ---: | ---: |
| 0.9 | 1 | .0833333333 | .0833333333 |
| 0.9 | .1 | .9009009009 | .9009009009 |
| 1, diagnostic | 1 | 0 | 0 |
| 1, diagnostic | .1 | 0 | 6.99e-15 |

The two local measurement means remain 0 and 1, so their exact constrained
solution is always .5. The calculation matches the native implementation within
7e-15 absolute gap error. This is a deterministic information-scaling example,
not random low-noise data or a statistical frequency of failure. Changing sigma
changes the assumed physical precision while keeping the numerical coupling
fixed. The prior experiment already showed that removing dual decay alone does
NOT cure robust riMESA. This Gaussian isolation explains one mechanism; robust
consensus weights, local approximation and nonlinear basins remain separate.

## Remaining Research Gates

Even successful suffix runs do not validate native noise adaptation, false
association handling, causal fault attribution, unknown-frame online bootstrap
for all baselines, or held-out transfer. Exact shared-copy consistency does not
imply smaller truth error. Conversely, a lower sum of physical losses with
disagreeing copies is not a lower feasible value of the exact-consensus problem.

The next algorithmic question is how to preserve statistically meaningful
joint information at bounded incremental work, while not interpreting temporary
state/consensus lag as sensor noise or an incorrect relation. This experiment
does not add another gate, kernel, threshold sweep or learned noise mechanism.

## Verification And Artifacts

All 21 formal estimator processes exit normally: seven D4 tight-reference runs
and seven one-PCG-budget runs on each of D4 and D9. No timeout or failed run is
silently dropped. The three repeated D4 one-sweep arms have byte-identical final
and causal states, so these are not 21 independent trials or held-out seeds.
There are 68 passing focused checks. Red tests preceded the new interfaces,
runner, byte ceilings and pinned-library mechanism diagnostic.

A final malformed-input check found that the baseline loader could silently
reinterpret an illegally split neighbor identity when switching to stable
labels. It now applies the same neighbor-identity validation as the native
loader. Both formal packages already passed that native check. The formal
binary/source snapshots are retained unchanged; two post-fix D4 replays
(paper-iMESA and robust riMESA) have byte-identical final and causal states.
This guard changes input rejection, not any valid solver update.

Tight D4 uses 57977 total PCG iterations with maximum relative residual
9.985e-11. One-PCG D4/D9 use 1562/2514 iterations, with maximum relative
residuals 3.155/4.243. These inexact residuals are NOT numerical convergence
claims; fixed-graph physical descent is separately checked. The Gaussian dual
diagnostic matches its fixed-point calculation within 7e-15. MPI: 0 ranks.
CCI/D-CCI, the packaged DBA-DRANPP repository and upstream riMESA are unchanged.
No commit, push, Overleaf edit or new default algorithm was made.

Canonical archive: `results/dranppr_native_online_20260923/`, containing
`d4_tight`, `d4_one`, `d9_one`, full traces/states/ledgers, source and binary
snapshots, input/library hashes, test XML, the Gaussian diagnostic and post-run
input-validation records. Pinned upstream source snapshots are included once
under `upstream/`. Each run's own manifest identifies its exact executed code;
use the archived binaries for bit-level reproduction.

```sh
cmake -S experiments/native_matched -B /tmp/dranppr_matched_build
cmake --build /tmp/dranppr_matched_build --target native-matched-baseline native-consensus-bias -j2
cmake --build /tmp/dranppr_backend_build --target native-joint-replay native-joint-audit -j2
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_native_consensus_bias.py tests/test_native_matched_online.py tests/test_audit_native_online.py tests/test_native_matched.py tests/test_audit_native_matched.py tests/test_native_joint_replay.py tests/test_native_joint_audit.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m experiments.audit_native_online --package /tmp/dranppr_matched_bootstrap_d9_20260923/frozen.json --events /tmp/dranppr_causal_streams_v1/dataset9/events.jsonl --duration 300 --dran-mode joint_one --output NEW_OUTPUT --raw /home/jackal/zyx/dpgo_ICRA/resources/datasets/MRCLAM/raw
```

The first configure command reuses the pinned dependency paths in the existing
CMake cache; fresh environments must supply the explicit paths recorded in the
archived `baseline_CMakeCache.txt`. This remains a native 2D development-data
control, not validation of the complete SEED-SLAM system or ObjectPose3 transfer.
