# Directional Delay Control: Fixed-Quadratic Derivation

Research note, 2026-11-23. This is not a convergence claim for the nonlinear
online backend. Novelty relative to matrix-metric asynchronous optimization
has NOT been established. No sensor weights or physical factors are changed.

## Setting

After exact private elimination and fixed block-diagonal coordinate scaling,
the immutable model is

\[
q(y)=\tfrac12 y^T B y+c^T y,\qquad B=B^T\succ0.
\]

One robot owns each block. At tick t, an active owner i uses its current block
and received versions of other blocks:

\[
\widehat g_i(t)=B_{ii}y_i(t)+c_i+
  \sum_{j\ne i} B_{ij}y_j(\tau_{ij}(t)),\qquad
0\le t-\tau_{ij}(t)\le D_{ij}.
\]

The constants D are supplied valid bounds, not estimates from observed age.
Inactive blocks have zero increment. Every block must activate within a finite
number of ticks. All actors share one fixed tangent chart and objective.

## Matrix Instead Of Scalar Delay Damping

Let B_ij=U Sigma V^T and define C_ij=U Sigma U^T and C_ji=V Sigma V^T.
These are the positive semidefinite square roots of B_ij B_ij^T and its
transpose-side counterpart. For any u,v,

\[
2u^T B_{ij}v\le u^T C_{ij}u+v^T C_{ji}v,
\]

because the difference equals the squared norm of
Sigma^(1/2)(U^T u-V^T v). This holds for rectangular and rank-deficient blocks.

Choose a fixed local metric and update:

\[
M_i=B_{ii}+\sum_{j\ne i}(1+D_{ij}+D_{ji})C_{ij},\qquad
s_i(t)=-M_i^{-1}\widehat g_i(t).
\]

B_ii is positive definite, so M_i is invertible without an added ridge. The
scalar reference replaces C_ij by ||B_ij||_2 I and B_ii by its spectral bound.
Our normalized B_ii is I up to roundoff. Therefore the matrix metric is no
larger in Loewner order than that scalar metric. This preserves uncoupled
directions, but does NOT prove better finite-horizon objective values.

## History-Energy Calculation

Write ||v||_C^2=v^T C v. Extend s(k)=0 for k<0. Define a diagnostic Lyapunov
function, which is never communicated or evaluated by the optimizer:

\[
V_t=q(y(t))-q(y^*)+
\tfrac12\sum_{i\ne j}\sum_{k=t-D_{ij}}^{t-1}
(k-t+D_{ij}+1)\|s_j(k)\|_{C_{ji}}^2.
\]

Expand the quadratic change. Its linear term at an active block is
-||s_i(t)||_(M_i)^2 plus the stale-gradient cross terms. Apply the preceding
matrix inequality to each delayed increment and to the simultaneous-step
cross terms. The history difference cancels the past-increment sums and adds
the reverse-direction D_ji terms. Consequently,

\[
V_{t+1}-V_t\le-\tfrac12\sum_i\|s_i(t)\|_{M_i}^2.
\]

This is a bound on objective PLUS recent update history, not monotone physical
objective at every asynchronous event. Under the stated fixed-model/bounded-
delay/fair-activation assumptions, this gives square-summable increments.
Bounded delays then make stale-gradient errors vanish; finite activation gaps
make every true gradient block tend to zero. Positive definiteness identifies
the unique fixed quadratic solution. This argument does not apply unchanged
to indefinite Hessians, changing graphs/charts/metrics, or permanent partitions.

## Internal Audit

- Direction of indices checked: D_ij bounds j's age at receiver i; its history
  term acts on j with C_ji. Reverse delays appear only after reindexing.
- Unequal block sizes and zero singular values are supported by the SVD bound.
- Concurrent active blocks are included in the quadratic cross-term bound.
- Inactive blocks contribute zero; they do not need fictitious local solves.
- A physical gauge must be removed before assuming B positive definite.
- The cached own block is NOT stale. The actor always uses its current value.
- No claim that a smaller metric always yields a better realized iterate.
- Numeric history-energy checks are regression evidence, not a proof by trials.

The scalar-delay starting point is [Ubl and Hale, 2022](https://arxiv.org/html/2203.11307).
Matrix-metric coordinate updates are established tools, reviewed in
[Nutini et al., JMLR 2022](https://www.jmlr.org/papers/volume23/18-045/18-045.pdf).
The adaptation above needs independent theoretical and novelty review before
being presented as a new result. Its main purpose here is a concrete,
testable alternative to uniformly damping every local direction.

final-status: VERIFIED_NUMERICALLY for the implementation and stated energy
inequality; independently reviewed nonlinear/online convergence remains open.
