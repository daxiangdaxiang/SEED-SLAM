# Complete Benchmark-Suffix Prediction Audit

Date: 2026-10-28. Research-only DRANPPR experiment. Production DBA-DRANPP unchanged.

## Decision

Frozen Q/R calibration retains better average predictive NLL on both discovery
datasets when all time intervals of the declared 120-second benchmark are
included. However, its uncertainty is substantially less reliable than the
previous three-window audit suggested. Do not promote this plug-in Gaussian
predictor as a reliable robust-online estimator.

This round improves the evidence and identifies missed failures; it introduces
no new estimator, gate, mean parameter, noise fit or backend setting. Average
NLL improvement is not a new trajectory-accuracy result.

## Protocol and Coverage

- Original MRCLAM D1/D4 streams, five robots, fixed chain topology, known barcode
  identities, same first-connected-past fits and nominal prefix factors.
- From warm cutoff to 120 s: D1 (27.153,120], D4 (15.534,120]. Disjoint five-second
  windows on the existing time grid, including each initial partial window.
  This covers the ENTIRE declared benchmark suffix, NOT the full raw missions
  (approximately 1490.6 and 1390.0 seconds).
- 19 D1 and 21 D4 windows, two arms each: nominal and frozen joint-QR scales.
  All 40 workers and 118 native subprocesses complete, including 80 batch LM
  solves and 38 causal initializer replays. Cumulative worker wall time 546.04 s;
  longest worker 48.37 s. This dense-QR audit is not an online timing benchmark.
- Every checkpoint uses only arrived data. Both arms start from the SAME nominal
  causal replay state. The first partial window uses the original nominal
  training graph in both state solves, with arm-specific future noise laws.
- Predict using subsequently arriving odometry, but assimilate NO object
  observations within a forecast window. Those observations can enter the graph
  at the next checkpoint. No GT is loaded by this experiment or analysis.
- Unchanged single-thread setting, LM cap 100, no new damping/floor/threshold,
  no failures dropped, no favorable windows selected, no parameter retries.

| Dataset | Suffix observations | Scored | No local object copy | Unseen globally | Scored fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 | 1491 | 1365 | 125 | 1 | 91.55% |
| D4 | 2117 | 1917 | 193 | 7 | 90.55% |
| Total | 3608 | 3282 | 318 | 8 | 90.96% |

Every suffix event is accounted for exactly once. There are 2482 scored
robot/timestamp frames, not 3282 independent experimental trials. The old
three-window protocol scored 533/3608 events (14.77%); this audit scores 6.16
times as many while exposing the remaining 326 unsupported observations.

"No local object copy" means another robot has previously observed the label,
but the forecasting robot has no corresponding checkpoint copy. It is a
limitation of the current predictor's support contract, NOT proof that the
quantity is globally unobservable. Expanding that support would require an
explicit information/communication contract; it is not silently done here.

## Main Results

Joint polar predictive NLL is lower-is-better. Range RMSE is error in predicting
future measured range, NOT trajectory RMSE, ATE, or pure range-sensor noise.
Coverage is the empirical fraction inside the declared 95% 2D ellipse.

| Dataset / arm | Mean NLL | Range RMSE (m) | Bearing RMSE (rad) | 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| D1 nominal | -1.519630 | 0.147281 | 0.073061 | 97.88% |
| D1 joint-QR | -1.805303 | 0.150941 | 0.068262 | 91.65% |
| D4 nominal | -1.317767 | 0.171448 | 0.094543 | 97.24% |
| D4 joint-QR | -1.369218 | 0.169233 | 0.090006 | 87.27% |

The adapted predictor is better in average NLL, but under-covers on both
datasets. Its range prediction worsens on D1 and improves only slightly on D4.
The old selected-window adapted coverages were 94.02%/91.49%; those should no
longer represent full benchmark-suffix reliability.

The NLL gain keeps its sign under alternative descriptive weighting:

| Dataset | Observation-equal gain | Frame-equal gain | Window-equal gain |
| --- | ---: | ---: | ---: |
| D1 | -0.285673 | -0.286179 | -0.075524 |
| D4 | -0.051451 | -0.123384 | -0.023064 |

These are adapted minus nominal, not significance tests. Equal-window scoring
gives the initial shorter window the same weight as other windows; it is a
sensitivity analysis, not an unbiased independent-trial estimator.

## Failures That the Old Windows Missed

### Temporal Concentration

D1 improves in 18/19 windows, yet its (50,55] window worsens severely:
mean NLL 1.303858 -> 7.751236 across 28 scored observations. A high fraction of
improved windows therefore does not establish safety against episodic errors.
Across the full 30--60 s band, D1's observation-weighted NLL worsens by 0.207076.

Post-selected inspection of the failed D1 window shows:

| Robot / subject | Count | Range RMSE, nominal -> adapted (m) | Mean predicted range SD (m) |
| --- | ---: | ---: | ---: |
| R5 / 6 | 14 | 0.633887 -> 0.683725 | 0.194219 -> 0.113361 |
| R5 / 7 | 4 | 0.580800 -> 0.604507 | 0.211501 -> 0.133352 |

For R5/6 the adapted mean signed range error is +0.675274 m, while the mean
claimed marginal SD is only 0.113361 m. Both arms already fail on these
observations; adaptation makes the mean slightly worse and the distribution
narrower. This is evidence of a coherent predictive discrepancy, NOT a
diagnosis of a constant hardware bias, wrong association, or a unique physical
cause. Map error, motion error, noise-law mismatch and temporal dependence
remain possible. Both subjects use the same known-identity contract.

D4 improves in 15/21 windows. The six negative windows begin at 20, 40, 50, 55,
75 and 85 s. Its 30--60 s band also worsens (NLL difference +0.120391).
Window counts share fitted parameters and overlapping histories; no binomial
independence or significance claim is appropriate.

### Robot-Level Conclusions Change with Coverage

| Group | Mean NLL, nominal -> adapted | Adapted 95% coverage | Interpretation |
| --- | ---: | ---: | --- |
| D1/R5 | -1.423043 -> -1.300280 | 86.99% | Previous failure persists |
| D4/R2 | -1.595713 -> -1.702357 | 94.52% | Aggregate score now improves; range RMSE still worsens |
| D4/R3 | -1.170508 -> -0.985379 | 81.19% | Newly exposed major conditional failure |
| D4/R4 | -1.035653 -> -0.990592 | 83.25% | Observation-equal score worsens; frame-equal score improves |

Seven of ten robot-level mean NLLs improve, three worsen. R4's sign sensitivity
reinforces the need to disclose observation multiplicity. D4/R2 was a negative
case in the three old windows, not a uniformly bad robot across time. Choosing
a permanent per-robot correction from those short windows would have been
poorly supported. The prior constant-mean-response negative remains relevant.

## What Improves, and What Does Not

A post-selected numerical swap holds predicted means fixed while changing
covariance; this is NOT a new fitted estimator or a causal decomposition:

| Mean / covariance | D1 NLL | D4 NLL |
| --- | ---: | ---: |
| Nominal / nominal | -1.519630 | -1.317767 |
| Nominal / adapted | -1.802775 | -1.371072 |
| Adapted / nominal | -1.521240 | -1.319468 |
| Adapted / adapted | -1.805303 | -1.369218 |

Most average score benefit is reproduced by covariance reassignment. For D4,
the adapted mean slightly worsens the score when combined with adapted
covariance. Better average NLL can therefore coexist with harmful overconfidence
in particular robot/time groups. A score rewards both concentration and matching
the observation; aggregate concentration gains do not guarantee calibrated tails.

This does not contradict the earlier GT trajectory gains: those evaluate the
joint fitted state after measurement assimilation, whereas this audit evaluates
future measurements under odometry-only propagation. Neither metric substitutes
for the other. No new state-accuracy or baseline-superiority claim is made here.

## Verification

- All 360 source archive hashes verified before and after execution. Native
  binary and linked-library hashes unchanged; pinned GTSAM, no TBB fallback.
- Twelve old-window/arm forecasts reproduce means, covariance and NLL EXACTLY.
- Independent checks validate identical non-weight inputs and initialization,
  unchanged physical measurements, past-only support, nominal prefix weights,
  and declared suffix multipliers. Maximum sigma discrepancy 5.55e-17.
- 6564 arm-specific predictions reconstruct source innovations exactly.
  Independent inverse/log-determinant scoring differs by at most 1.49e-13 in
  energy and 7.46e-14 in NLL. Maximum QR solve residual 2.35e-15.
- LM uses at most 16 iterations; maximum posthoc gradient 2.50e-4. This is not
  zero-gradient/global-optimality certification. Old LM/GN controls do not
  automatically certify every newly included endpoint.
- 11 new tests were observed failing before implementation/fix; final selected
  Python regression suite: **118 passed, zero skips**. Native suite: **62 passed**.
  The independent checker initially assumed explicit v2 weights; a regression
  test corrected its handling of the legacy v1 nominal prefix. Estimator inputs,
  forecasts and parameters did not change.
- Native fixture Schur residual 3.25e-16; matrix-free product error 3.53e-17.
  Basic routed PCG: 6 iterations, residual 1.98e-16; 2/3/5/10-robot fixtures:
  12/16/23/39 iterations. MPI ranks: 0. These are unit fixtures, not a new
  distributed communication experiment. CCI/D-CCI were not modified.

Artifacts: `results/dranppr_rolling_prediction_20261028`. Main files:
`manifest.json`, `summary.json`, per-window inputs/outputs and support ledgers,
`verification_analysis.json`, JUnit XML, source snapshots and `SHA256SUMS`.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 2400 \
  python3 -m experiments.audit_rolling_prediction \
  --source results/dranppr_network_noise_transfer_qr_20261003 \
  --output results/dranppr_rolling_prediction_reproduction

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m experiments.analyze_rolling_prediction \
  --output results/dranppr_rolling_prediction_reproduction

build/native/bin/testIncrementalLocalModel
```

The exact Python regression command is archived in `validation_commands.json`.
The worktree is kept on `research/dranppr-evidence-online`; no commit/push.

## Cumulative Progress and Next Gate

1. **State-quality signal exists.** The earlier first-prefix five-component
   calibration reduced D1/D4 120-second trajectory RMSE by 41.33%/34.61%, and
   object-copy RMSE by 33.48%/12.95%, versus nominal weights. Those are historical
   batch-checkpoint results, not newly measured distributed online gains.
2. **An online-compute component exists.** Earlier descent-triggered historical
   refresh reduced service time by 3.24--3.52 times on the declared D4/D9 suffixes,
   with terminal state-metric changes below 0.2 mm. It is a persistent native
   Pose2/Point2 prototype, not yet the integrated adaptive production backend.
3. **The reliability gap is now clearer.** Full benchmark-suffix evaluation
   preserves average score gains but exposes stronger conditional/tail failures.
   Numerical parity and input audits do not make the likelihood family correct.
4. **Blind mean expansion is not justified.** The previous constant per-robot
   first-order mean response failed to improve future prediction; robot-plus-
   object offsets also admit exact ambiguities. This round further shows that
   a short-window robot label is not a stable diagnosis.

The minimal next diagnostic is to test whether uncertainty in estimated Q/R
parameters explains a meaningful part of the overconfidence, using past-only
parameter uncertainty rather than arbitrary covariance inflation. If not, the
conditional mean/dependence/noise family needs reconsideration. Do not add
robot/object biases or gates before distinguishing these explanations. A
negative result must be retained, not repaired by sweeping a global safety factor.

Still open: an untouched-sequence confirmation; the retained D2 high-Gaussian-
noise negative; unknown outliers and wrong associations; continuous causal
parameter updates; distributed covariance/communication implementation; bounded
latency and memory; ObjectPose3 transfer; and a new matched end-to-end baseline
comparison. Average success does not close these risks.
