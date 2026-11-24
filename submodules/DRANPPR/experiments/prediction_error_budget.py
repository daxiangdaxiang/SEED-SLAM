"""Causal horizon-matched error envelopes, not guaranteed future error bounds."""
from collections import deque

import numpy as np


def robust_step(gradient, inverse_metric, epsilon):
    if not np.isfinite(epsilon) or epsilon < 0.:
        raise ValueError('invalid error budget')
    direction = inverse_metric@gradient
    norm = np.sqrt(max(0., float(gradient@direction)))
    alpha = float(max(0., 1.-epsilon/norm)) if norm > 0. else 0.
    return -alpha*direction, alpha


class HorizonErrorMemory:
    def __init__(self, inverse_metric, size, max_age):
        self.metric = np.asarray(inverse_metric, float).copy()
        if (size < 1 or int(size) != size or max_age < 0 or int(max_age) != max_age or
                self.metric.ndim != 2 or self.metric.shape[0] != self.metric.shape[1] or
                not np.isfinite(self.metric).all() or not np.allclose(self.metric, self.metric.T) or
                np.linalg.eigvalsh(self.metric).min() <= 0.):
            raise ValueError('invalid error memory')
        self.size, self.max_age = int(size), int(max_age)
        self.seed(-1, 0, np.zeros(len(self.metric)))

    def _value(self, value):
        value = np.asarray(value, float)
        if value.shape != (len(self.metric),) or not np.isfinite(value).all():
            raise ValueError('invalid response value')
        return value

    def norm(self, value):
        return np.sqrt(max(0., float(value@self.metric@value)))

    def seed(self, stamp, revision, value):
        self.last_stamp, self.last_revision = stamp, revision
        self.last_value = self._value(value).copy()
        self.pending, self.errors = deque(), {}
        self.last_target = -np.inf
        self.validated = self.unlabelled = self.expired = 0
        self.scored_bounds = self.bound_violations = self.max_pending = 0

    def bound(self, age):
        if age == 0:
            return 0.
        samples = self.errors.get(age, ())
        return max(samples) if len(samples) == self.size else None

    def record(self, target, base_stamp, forecast):
        value = self._value(forecast)
        age = target-base_stamp
        if age < 0 or age != int(age) or target <= self.last_target:
            raise ValueError('invalid forecast target or source age')
        self.last_target = target
        while self.pending and target-self.pending[0][0] > self.max_age:
            self.pending.popleft()
            self.expired += 1
        if 0 < age <= self.max_age:
            self.pending.append((target, age, value.copy(), self.bound(age)))
            self.max_pending = max(self.max_pending, len(self.pending))

    def observe(self, revision, stamp, value):
        value = self._value(value)
        if revision <= self.last_revision or stamp <= self.last_stamp:
            return
        consecutive = revision == self.last_revision+1
        while self.pending and self.pending[0][0] <= stamp:
            target, age, forecast, prior_bound = self.pending.popleft()
            if target == stamp:
                actual = value
            elif consecutive and target >= self.last_stamp:
                # A source optimizer's state is constant between consecutive
                # revisions; missing revisions do NOT license interpolation.
                actual = self.last_value
            else:
                self.unlabelled += 1
                continue
            error = self.norm(actual-forecast)
            if prior_bound is not None:
                self.scored_bounds += 1
                roundoff = 64.*np.finfo(float).eps*max(self.norm(actual), self.norm(forecast))
                self.bound_violations += int(error > prior_bound+roundoff)
            self.errors.setdefault(age, deque(maxlen=self.size)).append(error)
            self.validated += 1
        self.last_revision, self.last_stamp = revision, stamp
        self.last_value = value.copy()
