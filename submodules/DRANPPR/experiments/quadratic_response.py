"""Packet-corrected response of a frozen local quadratic, not a future bound."""
import numpy as np
from scipy.linalg import solve_triangular


class QuadraticResponse:
    def __init__(self, curvature, delayed_step, fresh_step, period=1, correct=True):
        self.a = np.asarray(curvature, float).copy()
        if (self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1] or
                not np.isfinite(self.a).all() or
                not np.allclose(self.a, self.a.T, rtol=1e-12, atol=1e-12) or
                period < 1 or int(period) != period):
            raise ValueError('invalid local response descriptor')
        lower = np.linalg.cholesky(self.a)
        self.spectra = {}
        for fresh, step in ((False, delayed_step), (True, fresh_step)):
            step = np.asarray(step, float)
            if (step.shape != self.a.shape or not np.isfinite(step).all() or
                    not np.allclose(step, step.T, rtol=1e-12, atol=1e-12)):
                raise ValueError('invalid local inverse metric')
            k = lower.T@step@lower
            eigenvalues, basis = np.linalg.eigh(.5*(k+k.T))
            if np.min(eigenvalues) <= 0. or np.max(eigenvalues) > 1.+1e-12:
                raise ValueError('local response must be contractive')
            self.spectra[fresh] = (np.minimum(eigenvalues, 1.),
                                  solve_triangular(lower.T, basis, lower=False))
        self.period, self.correct = int(period), bool(correct)
        self.reset(np.zeros(len(self.a)))

    @staticmethod
    def descriptor_bytes(dimension):
        return 48+24*dimension*dimension

    def reset(self, value):
        self.value = np.asarray(value, float).copy()
        if self.value.shape != (len(self.a),) or not np.isfinite(self.value).all():
            raise ValueError('invalid response state')
        self.stamp, self.gain, self.fresh = -1, 1., False
        self.forcing = self.observed_forcing = self.gradient = None
        self.last_innovation = np.zeros_like(self.value)
        self.corrections, self.innovation_sq = 0, 0.

    def receive(self, stamp, value, gradient, gain=1., fresh=False):
        value, gradient = np.asarray(value, float), np.asarray(gradient, float)
        if (stamp <= self.stamp or int(stamp) != stamp or stamp < 0 or
                value.shape != self.value.shape or gradient.shape != self.value.shape or
                not np.isfinite(value).all() or not np.isfinite(gradient).all() or
                not np.isfinite(gain) or not 0. <= gain <= 1. or fresh not in (False, True)):
            raise ValueError('invalid or nonmonotone model packet')
        forcing = gradient-self.a@value
        if self.observed_forcing is not None:
            self.last_innovation = forcing-self.observed_forcing
            self.innovation_sq += float(self.last_innovation@self.last_innovation)
            self.corrections += int(self.correct)
        if self.forcing is None or self.correct:
            self.forcing = forcing.copy()
        self.observed_forcing = forcing.copy()
        self.value, self.stamp = value.copy(), int(stamp)
        self.gain, self.fresh = float(gain), bool(fresh)
        # Keep the transmitted gradient verbatim in corrected mode to avoid
        # subtracting and re-adding a large local curvature contribution.
        self.gradient = gradient.copy() if self.correct else self.a@value+self.forcing

    def predict(self, target):
        if target < self.stamp:
            raise ValueError('prediction target precedes actual state')
        count = int(target)//self.period-self.stamp//self.period
        if self.gradient is None or count == 0 or self.gain == 0.:
            return self.value.copy()
        eigenvalues, basis = self.spectra[self.fresh]
        scaled = self.gain*eigenvalues
        weights = np.ones_like(scaled)
        interior = scaled < 1.
        weights[interior] = -np.expm1(count*np.log1p(-scaled[interior]))
        return self.value-basis@(weights*(basis.T@self.gradient))
