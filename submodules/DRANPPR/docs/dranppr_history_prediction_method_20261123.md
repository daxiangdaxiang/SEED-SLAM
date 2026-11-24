# History Prediction Without Replacing The Optimizer

Date: 2026-11-23. Fixed-model research implementation, not the production backend.

## Motivation And Information Boundary

Keep DRAN's physical objective and the previous directional interface update.
Replace the assumption that an uncommunicative neighbor has stopped with a
prediction based exclusively on received history. The variable's velocity is
optimization motion of its estimate, not the robot's physical velocity.

CORD already uses pose/optimization-velocity prediction on Lie groups. We do
not claim to invent that idea or replace DRAN with its inertial dynamics. Its
paper explicitly limits its theorem to synchronous constant-mass dynamics:
https://arxiv.org/html/2605.11210v1 . Our question is whether received history
can estimate changing velocity without extra velocity packets or new mass and
damping coefficients.

Predictions are temporary inputs to the local gradient. They NEVER overwrite
the last received state. No global state, future message, truth, global cost
test or global restart is available to the predictor. In the simulator packets
label post-update states: before updating at tick t, predict the neighbor at
t-1. This preserves exact zero-delay behavior. A real implementation must map
source time to the receiver's clock or a defined optimization-time coordinate;
the synthetic shared tick does not solve clock synchronization.

## Why Higher Order Alone Is Not Robust

Jointly fitting history gives a local expansion

\[
\widehat y(t+h)=y(t)+h\widehat v+\tfrac12h^2\widehat a
                    +\tfrac16h^3\widehat j.
\]

Estimating derivatives from multiple samples avoids taking differences of just
two noisy values. Nevertheless, an error in acceleration is multiplied by h^2,
and an error in the third derivative by h^3. Corresponding variance terms can
grow as h^4 and h^6. Better interpolation is not necessarily better prediction.
Locally polynomial smoothing/derivative estimation is established, e.g.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html .
Our irregular source times require an explicit design matrix, not uniform-time
filter coefficients.

## A Single Forecast-Risk Criterion

Use up to eight received samples, centered at the latest source time and scaled
by the observed time span. Fit nested degrees p=0,1,2,3 by QR least squares.
At least two residual degrees of freedom are retained; with insufficient
history the predictor holds the received state. Eight samples and maximum
degree three are explicit structural choices, not a claim of no parameters.

Let Z_p contain powers of normalized source time and phi_p(h) the powers at the
requested prediction horizon. Denote Y the matrix of received states and e_n
the selector for the last sample. Forecast only the fitted increment:

\[
 a_p(h)=e_n+(Z_p^+)^T[\phi_p(h)-\phi_p(0)],\qquad
 \widehat y_p(t+h)=a_p(h)^T Y.
\]

Thus p=0 is exactly hold, h=0 returns the exact received state, and historical
samples jointly estimate derivatives without forcing a noisy last sample to
be the fitted polynomial intercept. Implementation uses QR solves rather than
normal-equation inversion. A numerical roundoff floor proportional to eps^2
and history energy only resolves exact-fit floating-point degeneracy.

Receiver i already has cross block B_ij. For model residual R_p=Y-Z_p Z_p^+Y,
estimate unresolved variation in the receiver's coupling-gradient space:

\[
 \widehat\sigma_{ij,p}^2
 =\frac{\|R_p B_{ij}^T\|_F^2}{n-p-1},\qquad
 p^*=\arg\min_p\widehat\sigma_{ij,p}^2\|a_p(h)\|^2.
\]

The norm of ABSOLUTE weights a, not just increment weights, includes the
correlation between the fitted increment and the retained last sample. If a
local polynomial is correct and sample errors are zero-mean, temporally
independent with common covariance Sigma, the coupling forecast noise variance
is exactly tr(B_ij Sigma B_ij^T) times ||a||^2. Actual optimizer histories are
correlated and can have model bias. The implemented criterion is therefore a
variance/fit-mismatch proxy, NOT an unbiased full-risk estimator, confidence
interval, posterior covariance, or bound on future prediction error.

This criterion can choose lower order as noise or horizon increases, while
retaining second/third-order behavior when a smooth history supports it. No
manual mixing gain, per-dataset order sweep, momentum or variance reweighting
of sensor factors is added.

## Relation To The Optimization Step

On the fixed quadratic q(y)=0.5 y^T B y+c^T y, prediction changes only the
estimated block gradient:

\[
 \widehat g_i-g_i=\sum_{j\ne i}B_{ij}(\widehat y_j-y_j).
\]

For the active blocks let M be the previous block-diagonal directional metric.
It majorizes the corresponding B principal submatrix by the matrix Young
inequality used in the earlier note. With e=ghat-g and s=-M^{-1}(g+e):

\[
 q(y+s)-q(y)\le g^T s+\tfrac12s^TMs
 =-\tfrac12\|g\|_{M^{-1}}^2+\tfrac12\|e\|_{M^{-1}}^2.
\]

The equality follows by expanding the two quadratic forms; cross terms
cancel. Inactive increments are zero, so the same calculation applies to an
active principal block. A sufficient condition for descent is error energy
smaller than true-gradient energy. This explains why the important prediction
error is curvature-scaled GRADIENT error, not unweighted state error.

This is a conditional fixed-quadratic statement, not a theorem that our
predictor satisfies the condition, and not a claim of monotone asynchronous
objective decrease. The true error is unavailable during execution. The review
reconstructs ghat=-M s from saved increments ONLY AFTER execution and checks
the inequality; no diagnostic is used as an acceptance gate. The previous
stale-cache Lyapunov guarantee cannot simply be copied after extrapolation.

## Remaining Boundaries

- Forecast quality alone does not order complete coupled trajectories or
  guarantee faster convergence in weak global directions.
- Unseen changes after the latest packet cannot be inferred from history.
  Sudden stop/reversal controls intentionally demonstrate this limitation.
- No prediction can replace information lost during an arbitrary partition.
  Existing bounded-delay, reliable-relay and offline-row-setup assumptions stay.
- This version predicts fixed-chart coordinates. Nonlinear SE3 deployment needs
  consistent retraction/transport, gauge and model-version handling; the three
  SE3 Jacobians in these experiments do not establish that deployment.
- Runtime includes a Python prototype and offline diagnostics. Equal bytes and
  update counts do not imply equal CPU cost or lower end-to-end wall time.
- Risk-based polynomial selection and inexact-gradient analysis are established
  ideas. Novelty of this application or a future complete method is not yet
  established by these numerical results.

The verified algebra supports an error-control research direction. Empirical
claims and retained failures belong in the companion experiment report.
