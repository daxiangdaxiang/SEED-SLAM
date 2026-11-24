"""Actor-local fixed-quadratic updates with routed delayed absolute messages.

Time is a synthetic compute-opportunity tick, NOT measured milliseconds.
There is no link contention model. All global diagnostics are posthoc.
"""
from collections import deque
from dataclasses import dataclass
import heapq

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.neighbor_prediction import HistoryPrediction, ResponsePrediction
from experiments.prediction_error_budget import HorizonErrorMemory, robust_step
from experiments.quadratic_response import QuadraticResponse


@dataclass
class Packet:
    source: int
    destination: int
    revision: int
    stamp: int
    value: np.ndarray
    model_gradient: object = None
    model_gain: float = 1.
    model_fresh: bool = False

    @property
    def wire_bytes(self):
        return 40+8*len(self.value)+(0 if self.model_gradient is None else 16+8*len(self.model_gradient))


class Actor:
    def __init__(self, robot, rows, c, step, prediction='hold', history_size=8,
                 fresh_step=None, error_age_limits=None):
        self.robot = robot
        self.rows = {j: block.copy() for j, block in rows.items()}
        self.c = c.copy()
        self.step = float(step) if np.ndim(step) == 0 else np.asarray(step, float).copy()
        self.value = np.zeros_like(c)
        self.revision = 0
        self.cache = {j: np.zeros(block.shape[1]) for j, block in rows.items() if j != robot}
        self.response_cache = {j: np.zeros_like(c) for j in self.cache}
        self.versions = {j: (0, 0) for j in self.cache}
        response = prediction in ('response_linear', 'response_calibrated')
        model_response = prediction in ('model_response', 'model_frozen')
        self.histories = {j: HistoryPrediction(prediction, history_size, influence=self.rows[j])
                          for j in self.cache} if prediction != 'hold' and not response and not model_response else {}
        self.responses = {j: ResponsePrediction(prediction, history_size, self.step)
                          for j in self.cache} if response else {}
        self.fresh_step = None if fresh_step is None else np.asarray(fresh_step, float).copy()
        self.error_memories = {j: HorizonErrorMemory(fresh_step, history_size, error_age_limits[j])
                               for j in self.cache} if fresh_step is not None else {}
        self.error_ready, self.error_epsilon, self.error_alpha = False, -1., 0.
        self.planned_step = None
        self.supported_steps = self.fallback_steps = self.zero_steps = 0
        self.models = {}
        self.last_model_fresh, self.last_model_gain = False, 1.

    def receive(self, packet):
        if packet.destination != self.robot or packet.source not in self.cache:
            raise ValueError('unexpected message recipient or dependency')
        if packet.value.shape != self.cache[packet.source].shape or not np.isfinite(packet.value).all():
            raise ValueError('invalid message payload')
        version = (packet.revision, packet.stamp)
        if version <= self.versions[packet.source]:
            return False
        if self.models:
            self.models[packet.source].receive(packet.stamp, packet.value, packet.model_gradient,
                                               packet.model_gain, packet.model_fresh)
        if self.histories:
            self.histories[packet.source].add(packet.stamp, packet.value)
        contribution = self.rows[packet.source]@packet.value
        if self.error_memories:
            self.error_memories[packet.source].observe(packet.revision, packet.stamp, contribution)
        if self.responses:
            self.responses[packet.source].add(packet.stamp, contribution)
        self.cache[packet.source] = packet.value.copy()
        self.response_cache[packet.source] = contribution
        self.versions[packet.source] = version
        return True

    def gradient(self, tick=None):
        gradient = self.c+self.rows[self.robot]@self.value
        envelopes = []
        for j, cached in self.cache.items():
            # A stamp labels a post-update state. At the start of tick t the
            # contemporaneous neighbor state is post-update t-1, not t.
            if self.models and tick is not None:
                contribution = self.rows[j]@self.models[j].predict(tick-1)
            elif self.responses and tick is not None:
                contribution = self.responses[j].predict(tick-1)
            elif self.histories and tick is not None:
                contribution = self.rows[j]@self.histories[j].predict(tick-1)
            else:
                contribution = self.response_cache[j]
            gradient += contribution
            if self.error_memories and tick is not None:
                memory = self.error_memories[j]
                stamp = self.versions[j][1] if self.versions[j][0] else -1
                age = tick-1-stamp
                envelopes.append(memory.bound(age))
                memory.record(tick-1, stamp, contribution)
        if self.fresh_step is not None and tick is not None:
            self.error_ready = all(epsilon is not None for epsilon in envelopes)
            self.error_epsilon = float(sum(envelopes)) if self.error_ready else -1.
            self.error_alpha, self.planned_step = 0., None
            if self.error_ready:
                self.planned_step, self.error_alpha = robust_step(gradient, self.fresh_step, self.error_epsilon)
            else:
                return self.gradient()
        return gradient

    def update(self, gradient=None):
        if gradient is None:
            self.planned_step = None
            gradient = self.gradient()
        if self.fresh_step is not None and self.planned_step is not None:
            self.last_model_fresh, self.last_model_gain = True, self.error_alpha
            self.value += self.planned_step
            self.supported_steps += 1
            self.zero_steps += self.error_alpha == 0.
        else:
            self.last_model_fresh, self.last_model_gain = False, 1.
            self.value -= self.step*gradient if np.ndim(self.step) == 0 else self.step@gradient
            if self.fresh_step is not None:
                self.fallback_steps += 1
        self.planned_step = None
        self.revision += 1


@dataclass(frozen=True)
class Scenario:
    name: str
    delay_min: int = 1
    delay_max: int = 5
    seed: int = 31
    outage: tuple = (200, 300)

    def __post_init__(self):
        if (self.name not in ('ideal', 'delay', 'heterogeneous', 'loss', 'outage') or
                self.delay_min < 0 or self.delay_max < self.delay_min or
                self.outage[0] < 0 or self.outage[1] < self.outage[0]):
            raise ValueError('invalid communication scenario')

    def period(self, rank):
        return (1, 2, 3, 1, 4)[rank % 5] if self.name == 'heterogeneous' else 1

    def latency(self, source, destination, tick):
        if self.name == 'ideal':
            return 0
        value = (self.seed*73856093 + source*19349663 + destination*83492791 + tick*2654435761)
        value ^= value >> 13
        return self.delay_min + value % (self.delay_max-self.delay_min+1)

    def lost(self, source, destination, tick, outage_edge):
        if self.name == 'loss':
            return (tick+self.seed+3*source+destination) % 12 < 4
        return (self.name == 'outage' and tuple(sorted((source, destination))) == outage_edge and
                self.outage[0] <= tick < self.outage[1])


class Network:
    def __init__(self, links, scenario, deliver):
        self.links = list(links)
        self.adjacency = {}
        for a, b in links:
            if a == b:
                raise ValueError('self link')
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)
        if not self.adjacency:
            raise ValueError('empty topology')
        self.paths = {}
        for origin in self.adjacency:
            paths, queue = {origin: [origin]}, deque([origin])
            while queue:
                robot = queue.popleft()
                for other in sorted(self.adjacency[robot]):
                    if other not in paths:
                        paths[other] = paths[robot]+[other]
                        queue.append(other)
            if len(paths) != len(self.adjacency):
                raise ValueError('topology is not connected')
            for destination, path in paths.items():
                self.paths[origin, destination] = path
        self.scenario, self.deliver = scenario, deliver
        self.outage_edge = tuple(sorted(links[len(links)//2]))
        self.events, self.sequence = [], 0
        self.application_bytes = self.hop_messages = self.dropped_packets = self.delivered_packets = 0

    def _enqueue(self, time, packet, hop):
        self.sequence += 1
        heapq.heappush(self.events, (time, self.sequence, packet, hop))

    def send(self, packet, time):
        if packet.stamp != time or packet.source == packet.destination:
            raise ValueError('invalid packet time or endpoint')
        copy = Packet(packet.source, packet.destination, packet.revision, packet.stamp, packet.value.copy(),
                      None if packet.model_gradient is None else packet.model_gradient.copy(),
                      packet.model_gain, packet.model_fresh)
        self._enqueue(time, copy, 0)

    def advance(self, time):
        while self.events and self.events[0][0] <= time:
            now, _, packet, hop = heapq.heappop(self.events)
            path = self.paths[packet.source, packet.destination]
            if hop == len(path)-1:
                self.deliver(packet)
                self.delivered_packets += 1
                continue
            a, b = path[hop:hop+2]
            self.application_bytes += packet.wire_bytes
            self.hop_messages += 1
            # The bounded-loss control drops at the source link only. Arbitrary
            # independent multi-hop losses need a different delivery contract.
            can_drop = self.scenario.name != 'loss' or hop == 0
            if can_drop and self.scenario.lost(a, b, now, self.outage_edge):
                self.dropped_packets += 1
                continue
            self._enqueue(now+self.scenario.latency(a, b, now), packet, hop+1)


def delay_bounds(network, ids, periods):
    """Conservative end-to-end age contract, not an online age estimator.

    Loss is a four-tick source-link burst in each twelve-tick cycle; senders
    heartbeat every tick in that scenario. Relay links do not drop packets.
    """
    bounds = {}
    for i in ids:
        for j in ids:
            if i == j or network.scenario.name == 'ideal':
                bounds[i, j] = 0
                continue
            hops = len(network.paths[j, i])-1
            extra = 0
            if network.scenario.name == 'loss':
                extra = 4
            if network.scenario.name == 'outage':
                path = network.paths[j, i]
                if any(tuple(sorted(edge)) == network.outage_edge for edge in zip(path[:-1], path[1:])):
                    extra = network.scenario.outage[1]-network.scenario.outage[0]
            bounds[i, j] = hops*network.scenario.delay_max+periods[j]+extra+1
    return bounds


def directional_majorizer(rows, robot, bounds):
    """Matrix-valued cross-term bound, not a change to sensor information."""
    metric = rows[robot].copy()
    for j, block in rows.items():
        if j == robot:
            continue
        u, singular, _ = np.linalg.svd(block, full_matrices=False)
        curvature = (u*singular)@u.T
        metric += (1+bounds[robot,j]+bounds[j,robot])*curvature
    return .5*(metric+metric.T)


def simulate(model, links, scenario, mode, ticks, initial=None, prediction='hold', history_size=8):
    if mode not in ('synchronous', 'stale', 'bounded', 'directional', 'error_budget', 'fresh_unchecked') or ticks <= 0:
        raise ValueError('invalid simulation mode or horizon')
    model_response = prediction in ('model_response', 'model_frozen')
    if model_response and mode != 'error_budget':
        raise ValueError('model response requires the error-budget policy')
    ids = sorted(model.indices)
    initial = np.zeros_like(model.c) if initial is None else np.asarray(initial, float).copy()
    if initial.shape != model.c.shape or not np.isfinite(initial).all():
        raise ValueError('invalid common initial state')
    periods = {i: scenario.period(k) for k, i in enumerate(ids)}
    rows = model.rows()
    actors = {}
    ignored = 0

    def deliver(packet):
        nonlocal ignored
        if not actors[packet.destination].receive(packet):
            ignored += 1

    network = Network(links, scenario, deliver)
    if not set(ids) <= set(network.adjacency):
        raise ValueError('owner missing from topology')
    bounds = delay_bounds(network, ids, periods)
    steps = {}
    for i in ids:
        fresh_step = None
        if mode in ('directional', 'error_budget', 'fresh_unchecked'):
            zero_bounds = {key: 0 for key in bounds}
            metric = directional_majorizer(rows[i], i, zero_bounds if mode == 'fresh_unchecked' else bounds)
            step = cho_solve(cho_factor(metric, lower=True), np.eye(len(metric)))
            eigenvalues = np.linalg.eigvalsh(step)
            steps[i] = [float(eigenvalues[0]),float(eigenvalues[-1])]
            if mode == 'error_budget':
                fresh = directional_majorizer(rows[i], i, zero_bounds)
                fresh_step = cho_solve(cho_factor(fresh, lower=True), np.eye(len(fresh)))
        else:
            denominator = sum(np.linalg.norm(block, 2)*(1+(bounds[i, j]+bounds[j, i] if mode == 'bounded' else 0))
                              for j, block in rows[i].items())
            steps[i] = step = 1./denominator
        actors[i] = Actor(i, rows[i], model.c[model.indices[i]], step, prediction, history_size,
                          fresh_step, {j: bounds[i, j] for j in ids})
        actors[i].value[:] = initial[model.indices[i]]
        for j in actors[i].cache:
            actors[i].cache[j][:] = initial[model.indices[j]]
            actors[i].response_cache[j] = rows[i][j]@initial[model.indices[j]]
            if actors[i].histories:
                actors[i].histories[j].reset(-1., initial[model.indices[j]])
            if actors[i].responses:
                actors[i].responses[j].reset(-1., actors[i].response_cache[j])
            if actors[i].error_memories:
                actors[i].error_memories[j].seed(-1, 0, actors[i].response_cache[j])
    model_setup_bytes = 0
    if model_response:
        for i, receiver in actors.items():
            for j in receiver.cache:
                sender = actors[j]
                response = QuadraticResponse(sender.rows[j], sender.step, sender.fresh_step,
                                             periods[j], correct=prediction == 'model_response')
                response.reset(initial[model.indices[j]])
                receiver.models[j] = response
                model_setup_bytes += (len(network.paths[j, i])-1)*response.descriptor_bytes(len(sender.c))
        network.application_bytes += model_setup_bytes
    states, counters = [], []
    gradient_records, held_records, active_records = [], [], []
    budget_records = []
    last_sync = -max(periods.values())
    violations = max_age = barriers = 0
    for tick in range(ticks):
        network.advance(tick)
        if mode == 'synchronous':
            ready = tick-last_sync >= max(periods.values()) and all(
                actor.versions[j][0] == actors[j].revision
                for actor in actors.values() for j in actor.cache)
            active = ids if ready else []
            if ready:
                barriers += 1
                last_sync = tick
        else:
            active = [i for i in ids if tick % periods[i] == 0]
        # Simultaneous eligible reads prevent loop-order advantages at zero delay.
        gradients = {i: actors[i].gradient(tick=tick) for i in active}
        if mode == 'error_budget':
            budget_records.append([[float(actors[i].error_ready), actors[i].error_epsilon, actors[i].error_alpha]
                if i in active else [0., -1., 0.] for i in ids])
        estimated, held, mask = np.zeros_like(model.c), np.zeros_like(model.c), np.zeros(len(model.c), dtype=bool)
        for i in active:
            indices = model.indices[i]
            estimated[indices], held[indices], mask[indices] = gradients[i], actors[i].gradient(), True
        gradient_records.append(estimated)
        held_records.append(held)
        active_records.append(mask)
        for i in active:
            for j, (revision, stamp) in actors[i].versions.items():
                age = 0 if revision == actors[j].revision else tick-stamp
                max_age = max(max_age, age)
                violations += age > bounds[i, j]
        for i in active:
            actors[i].update(gradients[i])
        for i in ids:
            if tick % periods[i] == 0:
                model_gradient = actors[i].gradient() if model_response else None
                for j in actors[i].cache:
                    network.send(Packet(i, j, actors[i].revision, tick, actors[i].value,
                        model_gradient, actors[i].last_model_gain, actors[i].last_model_fresh), tick)
        network.advance(tick)
        state = np.zeros_like(model.c)
        for i, actor in actors.items():
            state[model.indices[i]] = actor.value
        states.append(state)
        counters.append((network.application_bytes, sum(a.revision for a in actors.values())))
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e100:
            break
    # No references or global diagnostics are visible to Actor.update.
    states = np.asarray(states)
    optimum = np.linalg.solve(model.b, -model.c)
    errors = states-optimum
    initial_error = initial-optimum
    initial_gap = .5*initial_error@model.b@initial_error
    denominator = max(initial_gap, np.finfo(float).tiny)
    gaps = .5*np.einsum('ij,ij->i', errors@model.b, errors)/denominator
    residuals = np.linalg.norm(states@model.b+model.c, axis=1)/max(np.linalg.norm(model.b@initial+model.c), np.finfo(float).tiny)
    preupdate = np.vstack([initial, states[:-1]])
    actual_gradients = preupdate@model.b+model.c
    gradient_error_sq = np.column_stack([np.sum(np.square((np.asarray(record)-actual_gradients)*active_records), axis=1)
                                         for record in (gradient_records, held_records)])
    hit = np.flatnonzero(gaps < 1e-6)
    summary = dict(mode=mode, scenario=scenario.name, seed=scenario.seed, ticks_completed=len(states),
        relative_gap=float(gaps[-1]), relative_residual=float(residuals[-1]),
        first_gap_1e6_tick=int(hit[0]) if len(hit) else None,
        application_bytes=network.application_bytes, hop_messages=network.hop_messages,
        dropped_packets=network.dropped_packets, delivered_packets=network.delivered_packets,
        obsolete_or_duplicate_packets=ignored, updates=counters[-1][1],
        global_reductions=0, synchronous_barriers=barriers, max_cache_age=max_age,
        bound_violations=int(violations), steps=steps, period_ticks=periods,
        prediction=prediction, history_size=history_size,
        model_setup_bytes=model_setup_bytes,
        model_statistics=dict(corrections=sum(m.corrections for a in actors.values() for m in a.models.values()),
            innovation_sq=sum(m.innovation_sq for a in actors.values() for m in a.models.values())),
        model_setup_contract='Additional descriptors charged, setup delivery assumed; common row/chart setup excluded',
        forecast_degree_counts=sum((history.degree_counts for actor in actors.values()
            for history in actor.histories.values()), np.zeros(4,dtype=np.int64)).tolist(),
        response_statistics=dict(
            fits=sum(p.fits for a in actors.values() for p in a.responses.values()),
            validations=sum(p.validations for a in actors.values() for p in a.responses.values()),
            final_gains=[p.gain for a in actors.values() for p in a.responses.values()],
            validated_error=sum(p.validated_error for a in actors.values() for p in a.responses.values()),
            validated_hold_error=sum(p.validated_hold_error for a in actors.values() for p in a.responses.values())),
        error_budget_statistics=dict(
            supported_steps=sum(a.supported_steps for a in actors.values()),
            fallback_steps=sum(a.fallback_steps for a in actors.values()),
            zero_steps=sum(a.zero_steps for a in actors.values()),
            validated=sum(m.validated for a in actors.values() for m in a.error_memories.values()),
            unlabelled=sum(m.unlabelled for a in actors.values() for m in a.error_memories.values()),
            expired=sum(m.expired for a in actors.values() for m in a.error_memories.values()),
            scored_bounds=sum(m.scored_bounds for a in actors.values() for m in a.error_memories.values()),
            retrospective_violations=sum(m.bound_violations for a in actors.values() for m in a.error_memories.values()),
            max_pending=max([m.max_pending for a in actors.values() for m in a.error_memories.values()]+[0])),
        gradient_error_sq_sum=float(gradient_error_sq[:,0].sum()),
        held_gradient_error_sq_sum=float(gradient_error_sq[:,1].sum()),
        delay_bounds={f'{i}:{j}': bound for (i,j), bound in bounds.items()},
        finite=bool(np.isfinite(states).all()), horizon_complete=len(states) == ticks,
        initial_state_norm=float(np.linalg.norm(initial)),
        setup_bytes_included=False, simulated_time_unit='compute-opportunity tick, not milliseconds',
        synchronous_control='Optimistic free barrier, not original DRAN PCG',
        loss_guarantee='Four-tick source-link burst; reliable delayed relay hops' if scenario.name == 'loss' else 'Configured bounded-delay contract')
    return dict(summary=summary, states=states, relative_gaps=gaps, relative_residuals=residuals,
                counters=np.asarray(counters, dtype=np.int64), gradient_error_sq=gradient_error_sq,
                estimated_gradients=np.asarray(gradient_records),
                error_budget_trace=np.asarray(budget_records).reshape(-1, len(ids), 3))
