"""Causal receiver-side prediction in a fixed coordinate chart.

History is received state, never recursively predicted state. Source times can
be irregular. This is not a replacement for manifold transport across charts.
"""
from collections import deque

import numpy as np


class HistoryPrediction:
    def __init__(self, mode, size=8, influence=None):
        if mode not in ('hold', 'secant', 'poly1', 'poly2', 'poly3', 'risk3') or size < 2:
            raise ValueError('invalid prediction mode or history size')
        self.mode, self.size = mode, size
        self.times, self.values = deque(maxlen=size), deque(maxlen=size)
        self.influence = None if influence is None else np.asarray(influence, float).copy()
        self.selected_degree = 0
        self.degree_counts = np.zeros(4, dtype=np.int64)

    def reset(self, time, value):
        self.times.clear()
        self.values.clear()
        self.add(time, value)

    def add(self, time, value):
        value = np.asarray(value, float)
        if (not np.isfinite(time) or (self.times and time <= self.times[-1])):
            raise ValueError('history time must increase')
        if (value.ndim != 1 or not np.isfinite(value).all() or
                (self.values and value.shape != self.values[-1].shape)):
            raise ValueError('invalid history state')
        self.times.append(float(time))
        self.values.append(value.copy())

    def predict(self, time):
        if not self.times or not np.isfinite(time):
            raise ValueError('missing history or invalid prediction time')
        latest = self.values[-1]
        if time <= self.times[-1] or len(self.times) < 2 or self.mode == 'hold':
            return latest.copy()
        if self.mode == 'secant':
            velocity = (latest-self.values[-2])/(self.times[-1]-self.times[-2])
            return latest+(time-self.times[-1])*velocity
        if self.mode == 'risk3':
            return self._risk_polynomial(time)
        times = np.asarray(self.times)
        scale = times[-1]-times[0]
        t = (times[:-1]-times[-1])/scale
        degree = min(int(self.mode[-1]), len(t))
        design = np.column_stack([t**power for power in range(1, degree+1)])
        coefficients = np.linalg.lstsq(design, np.asarray(self.values)[:-1]-latest, rcond=None)[0]
        horizon = (time-times[-1])/scale
        return latest+np.array([horizon**power for power in range(1, degree+1)])@coefficients

    def _risk_polynomial(self, time):
        latest = self.values[-1]
        # Retain two residual degrees of freedom to estimate unresolved history
        # variation. With insufficient history, hold rather than invent noise.
        maximum_degree = min(3, len(self.times)-3)
        if maximum_degree < 1:
            self.degree_counts[0] += 1
            return latest.copy()
        times = np.asarray(self.times)
        scale = times[-1]-times[0]
        t = (times-times[-1])/scale
        values = np.asarray(self.values)-latest
        coupled_values = values if self.influence is None else values@self.influence.T
        roundoff = np.finfo(float).eps**2*np.sum(coupled_values**2)
        horizon = (time-times[-1])/scale
        best_risk, best_increment = np.inf, np.zeros_like(latest)
        for degree in range(maximum_degree+1):
            design = np.vander(t, N=degree+1, increasing=True)
            q, r = np.linalg.qr(design, mode='reduced')
            residual = coupled_values-q@(q.T@coupled_values)
            noise = max(roundoff, np.sum(residual**2)/(len(times)-degree-1))
            basis_delta = np.array([0.]+[horizon**k for k in range(1,degree+1)])
            weights = q@np.linalg.solve(r.T, basis_delta)
            # The forecast retains the actual last sample: its noise is
            # correlated with the fitted increment and must be counted once.
            absolute_weights = weights.copy()
            absolute_weights[-1] += 1.
            risk = noise*(absolute_weights@absolute_weights)
            if risk < best_risk:
                best_risk, best_increment = risk, weights@values
                self.selected_degree = degree
        self.degree_counts[self.selected_degree] += 1
        return latest+best_increment


class ResponsePrediction(HistoryPrediction):
    """Packet-fitted first-order contribution forecast with causal gain learning.

    The metric is the receiver's inverse majorizer. Discounted regression uses
    observed average velocities, not ground truth or fitted in-sample residuals.
    It does not guarantee calibration at unseen, longer forecast horizons.
    """
    def __init__(self, mode, size=8, metric=None):
        if mode not in ('response_linear', 'response_calibrated'):
            raise ValueError('invalid response prediction mode')
        super().__init__('hold', size)
        self.mode = mode
        self.metric = None if metric is None else np.asarray(metric, float).copy()
        if self.metric is not None:
            if not np.isfinite(self.metric).all():
                raise ValueError('invalid response metric')
            if self.metric.ndim == 0:
                if self.metric <= 0.:
                    raise ValueError('response metric must be positive')
            elif (self.metric.ndim != 2 or self.metric.shape[0] != self.metric.shape[1] or
                  not np.allclose(self.metric, self.metric.T) or np.linalg.eigvalsh(self.metric).min() <= 0.):
                raise ValueError('response metric must be positive definite')
        self.discount = (size-1.)/(size+1.)
        self._clear_statistics()

    def _clear_statistics(self):
        self.velocity = None
        self.sum_pp = self.sum_pd = 0.
        self.gain = 1. if self.mode == 'response_linear' else 0.
        self.fits = self.validations = 0
        self.last_innovation = self.last_raw_innovation = None
        self.validated_error = self.validated_hold_error = 0.

    def reset(self, time, value):
        self._clear_statistics()
        super().reset(time, value)

    def _weighted(self, value):
        if self.metric is None:
            return value
        return self.metric*value if self.metric.ndim == 0 else self.metric@value

    def add(self, time, value):
        old_time = self.times[-1] if self.times else None
        old_value = self.values[-1] if self.values else None
        if self.metric is not None and self.metric.ndim == 2 and np.shape(value) != (len(self.metric),):
            raise ValueError('response metric dimension mismatch')
        # Validation/copying can fail without mutating learned statistics.
        super().add(time, value)
        if self.velocity is not None:
            dt = time-old_time
            delta = self.values[-1]-old_value
            observed_velocity = delta/dt
            weighted_velocity = self._weighted(self.velocity)
            self.last_raw_innovation = delta-dt*self.velocity
            self.last_innovation = delta-dt*self.gain*self.velocity
            self.validated_error += float(self.last_innovation@self._weighted(self.last_innovation))
            self.validated_hold_error += float(delta@self._weighted(delta))
            self.sum_pp = self.discount*self.sum_pp+float(self.velocity@weighted_velocity)
            self.sum_pd = self.discount*self.sum_pd+float(observed_velocity@weighted_velocity)
            if self.mode == 'response_calibrated':
                self.gain = float(np.clip(self.sum_pd/self.sum_pp, 0., 1.)) if self.sum_pp > 0. else 0.
            self.validations += 1
        if len(self.times) >= 2:
            t = np.asarray(self.times)-time
            centered = t-t.mean()
            values = np.asarray(self.values)-self.values[-1]
            self.velocity = (centered@values)/(centered@centered)
            self.fits += 1

    def predict(self, time):
        if not self.times or not np.isfinite(time):
            raise ValueError('missing history or invalid prediction time')
        latest = self.values[-1]
        if time <= self.times[-1] or self.velocity is None:
            return latest.copy()
        return latest+self.gain*(time-self.times[-1])*self.velocity
