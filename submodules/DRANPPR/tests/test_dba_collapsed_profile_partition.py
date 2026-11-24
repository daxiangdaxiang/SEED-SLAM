import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dba_collapsed_profile_partition.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "dba_collapsed_profile_partition", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy(module, robot, object_id, semantic_class=10):
    return module.ObjectCopy(robot, object_id, semantic_class)


def _observation(module, copy, mean, variance):
    mean = np.asarray(mean, dtype=float)
    return module.GaussianObjectObservation(
        copy=copy,
        mean=mean,
        covariance=np.eye(mean.size) * variance,
    )


def test_component_evidence_matches_direct_gaussian_marginal():
    module = load_module()
    first = _observation(module, _copy(module, 0, 0), [0.2], 0.4)
    second = _observation(module, _copy(module, 1, 0), [-0.1], 0.7)
    prior = module.GaussianObjectPrior(
        mean=np.array([0.3]), covariance=np.array([[1.2]])
    )

    actual = module.component_negative_log_evidence(
        (first, second), prior
    )
    covariance = np.array([[1.6, 1.2], [1.2, 1.9]])
    residual = np.array([0.2, -0.1]) - 0.3
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0.0
    expected = 0.5 * (
        residual @ np.linalg.solve(covariance, residual)
        + logdet
        + 2.0 * np.log(2.0 * np.pi)
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_distributed_component_packets_equal_centralized_evidence():
    module = load_module()
    observations = (
        _observation(module, _copy(module, 0, 0), [0.2, -0.1], 0.3),
        _observation(module, _copy(module, 1, 2), [0.4, 0.0], 0.2),
        _observation(module, _copy(module, 2, 7), [0.1, -0.2], 0.5),
    )
    prior = module.GaussianObjectPrior(
        mean=np.array([0.0, 0.0]), covariance=np.eye(2) * 4.0
    )

    packets = tuple(
        module.make_component_packet(observation) for observation in observations
    )
    aggregate = module.aggregate_component_packets(packets)

    assert module.packet_component_negative_log_evidence(
        aggregate, prior
    ) == pytest.approx(
        module.component_negative_log_evidence(observations, prior),
        abs=1e-12,
    )


def test_merge_delta_is_one_shared_object_vs_two_singletons():
    module = load_module()
    first_copy = _copy(module, 0, 0)
    second_copy = _copy(module, 1, 0)
    observations = {
        first_copy: _observation(module, first_copy, [0.0], 0.05),
        second_copy: _observation(module, second_copy, [0.1], 0.05),
    }
    prior = module.GaussianObjectPrior(
        mean=np.array([0.0]), covariance=np.array([[25.0]])
    )
    singleton = module.NetworkPartition.build(((first_copy,), (second_copy,)))
    merged = module.NetworkPartition.build(((first_copy, second_copy),))

    delta = module.partition_objective(merged, observations, prior) - (
        module.partition_objective(singleton, observations, prior)
    )

    assert module.merge_delta(
        singleton, first_copy, second_copy, observations, prior
    ) == pytest.approx(delta, abs=1e-12)
    assert delta < 0.0


def test_partition_score_difference_is_unit_and_gauge_invariant():
    module = load_module()
    first_copy = _copy(module, 0, 0)
    second_copy = _copy(module, 1, 0)
    observations = {
        first_copy: _observation(module, first_copy, [1.0, -2.0], 0.2),
        second_copy: _observation(module, second_copy, [1.2, -1.9], 0.3),
    }
    prior = module.GaussianObjectPrior(
        mean=np.array([0.5, -1.0]),
        covariance=np.array([[9.0, 0.7], [0.7, 4.0]]),
    )
    singleton = module.NetworkPartition.build(((first_copy,), (second_copy,)))
    merged = module.NetworkPartition.build(((first_copy, second_copy),))
    reference = module.partition_objective(merged, observations, prior) - (
        module.partition_objective(singleton, observations, prior)
    )

    angle = 0.73
    scale = 100.0
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    linear = scale * rotation
    offset = np.array([400.0, -170.0])
    transformed_observations, transformed_prior = (
        module.affine_transform_model(observations, prior, linear, offset)
    )
    transformed = module.partition_objective(
        merged, transformed_observations, transformed_prior
    ) - module.partition_objective(
        singleton, transformed_observations, transformed_prior
    )

    assert transformed == pytest.approx(reference, abs=1e-10)


def test_coordinate_descent_recovers_variable_cardinality_partition():
    module = load_module()
    a = _copy(module, 0, 0)
    b = _copy(module, 1, 0)
    c = _copy(module, 2, 0)
    unmatched = _copy(module, 1, 1)
    observations = {
        a: _observation(module, a, [0.0, 0.0], 0.04),
        b: _observation(module, b, [0.1, -0.1], 0.04),
        c: _observation(module, c, [-0.1, 0.1], 0.04),
        unmatched: _observation(module, unmatched, [8.0, 7.5], 0.04),
    }
    prior = module.GaussianObjectPrior(
        mean=np.zeros(2), covariance=np.eye(2) * 100.0
    )
    initial = module.NetworkPartition.build(tuple((copy,) for copy in observations))
    candidate_edges = ((a, b), (a, c), (b, c), (a, unmatched), (c, unmatched))

    result = module.optimize_partition(
        initial, observations, prior, candidate_edges
    )

    assert result.partition == module.NetworkPartition.build(
        ((a, b, c), (unmatched,))
    )
    assert result.objective_trace[-1] < result.objective_trace[0]
    assert all(
        after < before
        for before, after in zip(
            result.objective_trace, result.objective_trace[1:]
        )
    )


def test_distributed_min_consensus_matches_central_edit_sequence():
    module = load_module()
    copies = tuple(_copy(module, robot, 0) for robot in range(3))
    observations = {
        copy: _observation(module, copy, [0.05 * copy.robot, 0.0], 0.04)
        for copy in copies
    }
    prior = module.GaussianObjectPrior(
        mean=np.zeros(2), covariance=100.0 * np.eye(2)
    )
    initial = module.NetworkPartition.build(tuple((copy,) for copy in copies))
    edges = ((copies[0], copies[1]), (copies[1], copies[2]))

    central = module.optimize_partition(initial, observations, prior, edges)
    distributed = module.optimize_partition(
        initial,
        observations,
        prior,
        edges,
        selection_topology=((0, 1), (1, 2)),
    )

    assert distributed.partition == central.partition
    assert distributed.objective_trace == pytest.approx(central.objective_trace)
    assert distributed.accepted_edits == central.accepted_edits
    assert distributed.consensus_rounds > 0
    assert distributed.consensus_payload_bytes > 0


def test_merge_rejects_two_copies_from_the_same_robot():
    module = load_module()
    first = _copy(module, 0, 0)
    second = _copy(module, 0, 1)
    partition = module.NetworkPartition.build(((first,), (second,)))

    with pytest.raises(ValueError, match="same robot"):
        module.merge_partition_components(partition, first, second)


def test_coordinate_descent_moves_copy_out_of_a_wrong_incumbent_component():
    module = load_module()
    source = _copy(module, 0, 0)
    wrong = _copy(module, 1, 0)
    correct = _copy(module, 1, 1)
    observations = {
        source: _observation(module, source, [0.0], 0.03),
        wrong: _observation(module, wrong, [4.0], 0.03),
        correct: _observation(module, correct, [0.05], 0.03),
    }
    prior = module.GaussianObjectPrior(
        mean=np.array([0.0]), covariance=np.array([[16.0]])
    )
    initial = module.NetworkPartition.build(((source, wrong), (correct,)))

    result = module.optimize_partition(
        initial,
        observations,
        prior,
        candidate_edges=((source, wrong), (source, correct)),
    )

    assert result.partition == module.NetworkPartition.build(
        ((source, correct), (wrong,))
    )
    assert any(move.kind == "move" for move in result.accepted_edits)


def test_profiled_objective_recovers_relative_frame_and_correct_partition():
    module = load_module()
    first = tuple(_copy(module, 0, index) for index in range(3))
    second = tuple(_copy(module, 1, index) for index in range(3))
    world = np.array([[0.0, 0.0], [2.0, 0.5], [-0.3, 2.2]])
    angle = 0.45
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    translation = np.array([3.0, -1.2])
    second_local = (world - translation) @ rotation
    observations = {
        **{
            copy: _observation(module, copy, point, 0.01)
            for copy, point in zip(first, world)
        },
        **{
            copy: _observation(module, copy, point, 0.01)
            for copy, point in zip(second, second_local)
        },
    }
    prior = module.GaussianObjectPrior(
        mean=np.array([0.5, 0.5]), covariance=np.eye(2) * 25.0
    )
    correct = module.NetworkPartition.build(
        tuple((first[index], second[index]) for index in range(3))
    )
    wrong = module.NetworkPartition.build(
        (
            (first[0], second[1]),
            (first[1], second[0]),
            (first[2],),
            (second[2],),
        )
    )

    correct_profile = module.profile_partition_transforms(
        correct, observations, prior, root_robot=0
    )
    wrong_profile = module.profile_partition_transforms(
        wrong, observations, prior, root_robot=0
    )

    assert correct_profile.objective < wrong_profile.objective
    recovered_angle, recovered_translation = correct_profile.transforms[1]
    assert recovered_angle == pytest.approx(angle, abs=2e-3)
    assert recovered_translation == pytest.approx(translation, abs=3e-3)


def test_alternating_partition_and_lower_profile_monotonically_improves_one_j():
    module = load_module()
    first = tuple(_copy(module, 0, index) for index in range(2))
    second = tuple(_copy(module, 1, index) for index in range(3))
    observations = {
        first[0]: _observation(module, first[0], [0.0, 0.0], 0.02),
        first[1]: _observation(module, first[1], [2.0, 1.0], 0.02),
        second[0]: _observation(module, second[0], [-1.0, 0.5], 0.02),
        second[1]: _observation(module, second[1], [1.0, 1.5], 0.02),
        second[2]: _observation(module, second[2], [7.0, 6.0], 0.02),
    }
    prior = module.GaussianObjectPrior(
        mean=np.zeros(2), covariance=np.eye(2) * 100.0
    )
    initial = module.NetworkPartition.build(
        tuple((copy,) for copy in observations)
    )
    candidates = (
        (first[0], second[0]),
        (first[0], second[2]),
        (first[1], second[1]),
        (first[1], second[2]),
    )

    result = module.run_collapsed_profile_partitioning(
        initial,
        observations,
        prior,
        candidates,
        root_robot=0,
        initial_transforms={
            0: (0.0, np.zeros(2)),
            1: (0.0, np.array([1.0, -0.5])),
        },
        transform_priors={
            1: module.GaussianTransformPrior(
                mean=np.array([0.0, 1.0, -0.5]),
                covariance=np.diag([0.05, 0.05, 0.05]),
            )
        },
    )

    assert result.partition == module.NetworkPartition.build(
        ((first[0], second[0]), (first[1], second[1]), (second[2],))
    )
    assert all(
        after <= before + 1e-12
        for before, after in zip(result.objective_trace, result.objective_trace[1:])
    )


def test_profiled_state_rebuilds_reciprocal_support_after_frame_alignment():
    module = load_module()
    anchor_a = _copy(module, 0, 0, semantic_class=9)
    first_a = _copy(module, 0, 1)
    second_a = _copy(module, 0, 2)
    anchor_b = _copy(module, 1, 0, semantic_class=9)
    first_b = _copy(module, 1, 1)
    second_b = _copy(module, 1, 2)
    observations = {
        anchor_a: _observation(module, anchor_a, [0.0, 0.0], 0.01),
        first_a: _observation(module, first_a, [1.0, 0.0], 0.01),
        second_a: _observation(module, second_a, [10.0, 0.0], 0.01),
        anchor_b: _observation(module, anchor_b, [-5.0, 0.0], 0.01),
        first_b: _observation(module, first_b, [-4.0, 0.0], 0.01),
        second_b: _observation(module, second_b, [5.0, 0.0], 0.01),
    }
    initial = module.NetworkPartition.build(
        (
            (anchor_a, anchor_b),
            (first_a,),
            (second_a,),
            (first_b,),
            (second_b,),
        )
    )
    prior = module.GaussianObjectPrior(np.zeros(2), 100.0 * np.eye(2))

    result = module.run_collapsed_profile_partitioning(
        initial,
        observations,
        prior,
        candidate_edges=(),
        root_robot=0,
        initial_transforms={0: (0.0, np.zeros(2)), 1: (0.0, np.zeros(2))},
        candidate_topology=((0, 1),),
        selection_topology=((0, 1),),
    )

    assert result.partition == module.NetworkPartition.build(
        ((anchor_a, anchor_b), (first_a, first_b), (second_a, second_b))
    )


def test_exact_partition_oracle_confirms_coordinate_solution():
    module = load_module()
    a = _copy(module, 0, 0)
    b = _copy(module, 1, 0)
    c = _copy(module, 2, 0)
    unmatched = _copy(module, 1, 1)
    observations = {
        a: _observation(module, a, [0.0], 0.04),
        b: _observation(module, b, [0.1], 0.04),
        c: _observation(module, c, [-0.05], 0.04),
        unmatched: _observation(module, unmatched, [5.0], 0.04),
    }
    prior = module.GaussianObjectPrior(
        mean=np.array([0.0]), covariance=np.array([[25.0]])
    )
    initial = module.NetworkPartition.build(tuple((copy,) for copy in observations))
    candidate_edges = ((a, b), (a, c), (b, c), (a, unmatched), (c, unmatched))

    exact = module.solve_exact_partition(
        observations, prior, candidate_edges=candidate_edges
    )
    coordinate = module.optimize_partition(
        initial, observations, prior, candidate_edges
    )

    assert exact.partition == module.NetworkPartition.build(
        ((a, b, c), (unmatched,))
    )
    assert coordinate.partition == exact.partition
    assert coordinate.objective_trace[-1] == pytest.approx(
        exact.objective, abs=1e-12
    )
    assert exact.feasible_partition_count > 1


def test_empirical_prior_is_affine_covariant_without_a_scale_knob():
    module = load_module()
    copies = tuple(_copy(module, robot, 0) for robot in range(3))
    observations = {
        copies[0]: _observation(module, copies[0], [0.0, 1.0], 0.2),
        copies[1]: _observation(module, copies[1], [2.0, -1.0], 0.3),
        copies[2]: _observation(module, copies[2], [1.0, 3.0], 0.4),
    }
    prior = module.empirical_object_prior(observations)
    linear = np.array([[0.0, -10.0], [10.0, 0.0]])
    offset = np.array([4.0, -7.0])
    transformed_observations, expected = module.affine_transform_model(
        observations, prior, linear, offset
    )
    actual = module.empirical_object_prior(transformed_observations)

    assert actual.mean == pytest.approx(expected.mean, abs=1e-12)
    assert actual.covariance == pytest.approx(expected.covariance, abs=1e-10)


def test_category_complete_support_uses_only_communication_neighbors():
    module = load_module()
    copies = (
        _copy(module, 0, 0, 10),
        _copy(module, 0, 1, 71),
        _copy(module, 1, 0, 10),
        _copy(module, 1, 1, 71),
        _copy(module, 2, 0, 10),
    )

    edges = module.category_complete_candidate_edges(
        copies, topology=((0, 1), (1, 2))
    )

    assert edges == (
        (copies[0], copies[2]),
        (copies[1], copies[3]),
        (copies[2], copies[4]),
    )


def test_reciprocal_physical_support_has_no_distance_threshold():
    module = load_module()
    copies = tuple(
        _copy(module, robot, object_id)
        for robot, object_id in ((0, 0), (0, 1), (1, 2), (1, 3))
    )
    observations = {
        copies[0]: _observation(module, copies[0], [0.0, 0.0], 1.0),
        copies[1]: _observation(module, copies[1], [10000.0, 0.0], 1.0),
        copies[2]: _observation(module, copies[2], [1.0, 0.0], 1.0),
        copies[3]: _observation(module, copies[3], [11000.0, 0.0], 1.0),
    }

    edges = module.reciprocal_physical_candidate_edges(
        observations, topology=((0, 1),)
    )

    assert edges == ((copies[0], copies[2]), (copies[1], copies[3]))


def test_partition_association_rows_are_directed_and_topology_limited():
    module = load_module()
    a = _copy(module, 0, 0)
    b = _copy(module, 1, 2)
    c = _copy(module, 2, 4)
    singleton = _copy(module, 1, 9)
    partition = module.NetworkPartition.build(((a, b, c), (singleton,)))

    rows = module.partition_association_rows(
        partition, topology=((0, 1), (1, 2))
    )

    assert rows == (
        {
            "receiver_robot": 0,
            "sender_robot": 1,
            "receiver_object": 0,
            "sender_object": 2,
            "weight": 1.0,
        },
        {
            "receiver_robot": 1,
            "sender_robot": 0,
            "receiver_object": 2,
            "sender_object": 0,
            "weight": 1.0,
        },
        {
            "receiver_robot": 1,
            "sender_robot": 2,
            "receiver_object": 2,
            "sender_object": 4,
            "weight": 1.0,
        },
        {
            "receiver_robot": 2,
            "sender_robot": 1,
            "receiver_object": 4,
            "sender_object": 2,
            "weight": 1.0,
        },
    )


def test_component_mean_gradients_match_finite_differences():
    module = load_module()
    copies = tuple(_copy(module, robot, 0) for robot in range(3))
    observations = tuple(
        _observation(module, copy, mean, variance)
        for copy, mean, variance in (
            (copies[0], [0.2, -0.3], 0.2),
            (copies[1], [0.5, 0.1], 0.4),
            (copies[2], [-0.1, 0.2], 0.3),
        )
    )
    prior = module.GaussianObjectPrior(
        mean=np.array([0.0, 0.0]),
        covariance=np.array([[4.0, 0.3], [0.3, 2.0]]),
    )

    _, gradients = module.component_evidence_with_mean_gradients(
        observations, prior
    )
    epsilon = 1e-6
    for observation_index, observation in enumerate(observations):
        for axis in range(2):
            perturbed = []
            for index, value in enumerate(observations):
                mean = np.array(value.mean, copy=True)
                if index == observation_index:
                    mean[axis] += epsilon
                perturbed.append(
                    module.GaussianObjectObservation(
                        value.copy, mean, value.covariance
                    )
                )
            plus = module.component_negative_log_evidence(perturbed, prior)
            perturbed[observation_index] = module.GaussianObjectObservation(
                observation.copy,
                observation.mean - np.eye(2)[axis] * epsilon,
                observation.covariance,
            )
            minus = module.component_negative_log_evidence(perturbed, prior)
            finite_difference = (plus - minus) / (2.0 * epsilon)
            assert gradients[observation.copy][axis] == pytest.approx(
                finite_difference, abs=2e-8
            )


def test_transform_preserves_symmetry_for_large_scale_covariance():
    module = load_module()
    copy = _copy(module, 0, 1)
    observation = module.GaussianObjectObservation(
        copy, np.array([2.0, -1.0]), 1228594.2318667606 * np.eye(2)
    )

    transformed = module.transform_robot_observations(
        {copy: observation}, {0: (-999.986, np.array([3.0, 4.0]))}
    )

    assert np.array_equal(
        transformed[copy].covariance, transformed[copy].covariance.T
    )


def test_transform_changes_physical_coordinates_but_not_attributes():
    module = load_module()
    copy = _copy(module, 0, 1)
    observation = module.GaussianObjectObservation(
        copy,
        np.array([1.0, 0.0, 4.0, -2.0]),
        np.diag([0.2, 0.2, 0.3, 0.4]),
    )

    transformed = module.transform_robot_observations(
        {copy: observation},
        {0: (np.pi / 2.0, np.array([2.0, 3.0]))},
    )[copy]

    assert np.allclose(transformed.mean, [2.0, 4.0, 4.0, -2.0])
    assert np.allclose(transformed.covariance, observation.covariance)


def test_empirical_component_intensities_are_semantic_counts():
    module = load_module()
    copies = (
        _copy(module, 0, 0, semantic_class=4),
        _copy(module, 1, 0, semantic_class=4),
        _copy(module, 0, 1, semantic_class=7),
    )
    partition = module.NetworkPartition.build(
        ((copies[0],), (copies[1],), (copies[2],))
    )

    assert module.empirical_component_intensities(partition) == {4: 2.0, 7: 1.0}


def test_point_process_intensity_blocks_weak_merge_but_not_strong_merge():
    module = load_module()
    first = _copy(module, 0, 0)
    second = _copy(module, 1, 0)
    initial = module.NetworkPartition.build(((first,), (second,)))
    prior = module.GaussianObjectPrior(np.zeros(2), 100.0 * np.eye(2))

    weak = {
        first: _observation(module, first, [0.0, 0.0], 10.0),
        second: _observation(module, second, [0.0, 0.0], 10.0),
    }
    blocked = module.optimize_partition(
        initial,
        weak,
        prior,
        ((first, second),),
        component_intensities={10: 100.0},
    )
    assert blocked.partition == initial

    strong = {
        first: _observation(module, first, [0.0, 0.0], 0.01),
        second: _observation(module, second, [0.0, 0.0], 0.01),
    }
    accepted = module.optimize_partition(
        initial,
        strong,
        prior,
        ((first, second),),
        component_intensities={10: 100.0},
    )
    assert accepted.partition.component_count == 1


def test_feasible_edit_enumeration_is_deterministic_and_score_free():
    module = load_module()
    first = _copy(module, 0, 0)
    incumbent = _copy(module, 1, 0)
    alternative = _copy(module, 1, 1)
    third = _copy(module, 2, 0)
    partition = module.NetworkPartition.build(
        ((first, incumbent), (alternative,), (third,))
    )
    edges = ((third, alternative), (first, alternative), (alternative, first))

    forward = module.enumerate_feasible_partition_edits(partition, edges)
    reverse = module.enumerate_feasible_partition_edits(
        partition, tuple(reversed(edges))
    )

    assert forward == reverse
    assert any(edit.kind == "detach" for edit in forward)
    assert any(edit.kind == "move" for edit in forward)
    assert any(edit.kind == "merge" for edit in forward)
    assert all(
        edit.target is None or edit.copy.robot != edit.target.robot
        for edit in forward
    )
