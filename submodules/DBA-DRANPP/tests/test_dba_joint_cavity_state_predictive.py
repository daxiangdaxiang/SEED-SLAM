import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scripts.dba_collapsed_profile_partition import (
    GaussianObjectObservation,
    GaussianObjectPrior,
    partition_objective,
)
from scripts.dba_network_quotient_reference import (
    NetworkPartition,
    NetworkQuotientProblem,
    ObjectCopy,
    RobotCavityState,
    relabel_robots,
)


def _two_robot_problem(off_diagonal: float = 0.0):
    first = ObjectCopy(0, 0, 10)
    first_context = ObjectCopy(0, 1, 71)
    second = ObjectCopy(1, 0, 10)
    second_context = ObjectCopy(1, 1, 71)
    states = (
        RobotCavityState(
            0,
            (first, first_context),
            np.array([[0.2, -0.1], [1.2, 0.5]]),
            np.array([[0.4, off_diagonal], [off_diagonal, 0.6]]),
        ),
        RobotCavityState(
            1,
            (second, second_context),
            np.array([[-0.3, 0.4], [1.0, 0.8]]),
            np.array([[0.7, -off_diagonal], [-off_diagonal, 0.5]]),
        ),
    )
    partition = NetworkPartition.build(
        ((first, second), (first_context,), (second_context,))
    )
    problem = NetworkQuotientProblem(states, partition)
    transforms = {
        0: (0.0, np.zeros(2)),
        1: (0.2, np.array([0.4, -0.2])),
    }
    prior = GaussianObjectPrior(
        np.array([0.1, 0.2]), np.array([[2.0, 0.3], [0.3, 1.5]])
    )
    return problem, partition, transforms, prior


def _explicit_marginal(problem, partition, transforms, prior):
    copies = tuple(copy for state in problem.robots for copy in state.copies)
    component_by_copy = {
        copy: index
        for index, component in enumerate(partition.components)
        for copy in component
    }
    assignment = np.zeros((len(copies), len(partition.components)))
    for row, copy in enumerate(copies):
        assignment[row, component_by_copy[copy]] = 1.0
    expanded = np.kron(assignment, np.eye(2))
    latent_mean = np.tile(prior.mean, len(partition.components))
    latent_covariance = np.kron(
        np.eye(len(partition.components)), prior.covariance
    )
    noise_blocks = []
    transformed_means = []
    for state in problem.robots:
        yaw, translation = transforms[state.robot]
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        )
        transformed_means.append(state.means @ rotation.T + translation)
        noise_blocks.append(np.kron(state.covariance, np.eye(2)))
    noise = np.zeros((2 * len(copies), 2 * len(copies)))
    offset = 0
    for block in noise_blocks:
        width = block.shape[0]
        noise[offset : offset + width, offset : offset + width] = block
        offset += width
    return (
        expanded @ latent_mean,
        noise + expanded @ latent_covariance @ expanded.T,
        np.concatenate(transformed_means).reshape(-1),
    )


def test_joint_objective_matches_direct_multivariate_gaussian():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )

    actual = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        {10: 1.0, 71: 1.0},
    )
    mean, covariance, observation = _explicit_marginal(
        problem, partition, transforms, prior
    )
    expected = -multivariate_normal.logpdf(
        observation, mean=mean, cov=covariance
    )

    assert actual.state_nll == pytest.approx(expected, abs=1e-10)
    assert actual.total == pytest.approx(expected, abs=1e-10)


def test_joint_objective_can_disable_component_intensity():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    intensities = {10: 2.0, 71: 3.0}
    with_intensity = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        include_component_intensity=True,
    )
    without_intensity = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        include_component_intensity=False,
    )

    assert without_intensity.cardinality == 0.0
    assert with_intensity.total - without_intensity.total == pytest.approx(
        with_intensity.cardinality
    )


def test_diagonal_robot_covariance_matches_manually_diagonalized_problem():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    diagonal_problem = NetworkQuotientProblem(
        tuple(
            RobotCavityState(
                state.robot,
                state.copies,
                state.means,
                np.diag(np.diag(state.covariance)),
            )
            for state in problem.robots
        ),
        partition,
    )
    intensities = {10: 2.0, 71: 3.0}

    actual = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        robot_covariance_mode="diagonal",
    )
    expected = joint_cavity_partition_objective(
        partition,
        diagonal_problem,
        transforms,
        prior,
        intensities,
    )

    assert actual == expected


def test_independent_pair_relations_report_network_conflict_on_a_cycle():
    from scripts.dba_joint_cavity_state_predictive import (
        relation_components,
        relation_partition_diagnostics,
    )

    zero = ObjectCopy(0, 0, 10)
    one_first = ObjectCopy(1, 0, 10)
    one_second = ObjectCopy(1, 1, 10)
    two = ObjectCopy(2, 0, 10)
    copies = (zero, one_first, one_second, two)
    independently_selected = (
        tuple(sorted((zero, one_first))),
        tuple(sorted((zero, two))),
        tuple(sorted((one_second, two))),
    )

    feasible, duplicate_count = relation_partition_diagnostics(
        copies, independently_selected
    )

    assert feasible is False
    assert duplicate_count == 1
    assert relation_components(copies, independently_selected) == (
        tuple(sorted(copies)),
    )


def test_canonical_workspace_matches_dense_value_and_observation_gradient():
    from scripts.dba_joint_cavity_state_predictive import (
        _JointCavityCanonicalWorkspace,
        _joint_objective_and_observation_gradient,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    intensities = {10: 2.0, 71: 3.0}
    dense, dense_gradient = _joint_objective_and_observation_gradient(
        partition, problem, transforms, prior, intensities
    )
    canonical_value, canonical_blocks = _JointCavityCanonicalWorkspace(
        partition, problem, prior, intensities
    ).evaluate(transforms)
    canonical_gradient = np.concatenate(
        [canonical_blocks[state.robot].reshape(-1) for state in problem.robots]
    )

    assert canonical_value == pytest.approx(dense.total, abs=1e-10)
    assert canonical_gradient == pytest.approx(dense_gradient, abs=1e-10)


def test_structured_and_dense_workspace_backends_are_equivalent():
    from scripts.dba_joint_cavity_state_predictive import (
        _JointCavityCanonicalWorkspace,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    intensities = {10: 2.0, 71: 3.0}
    structured = _JointCavityCanonicalWorkspace(
        partition,
        problem,
        prior,
        intensities,
        precision_backend="structured",
    ).evaluate(transforms)
    dense = _JointCavityCanonicalWorkspace(
        partition,
        problem,
        prior,
        intensities,
        precision_backend="dense",
    ).evaluate(transforms)

    assert structured[0] == pytest.approx(dense[0], rel=1e-11, abs=1e-11)
    for robot in structured[1]:
        assert structured[1][robot] == pytest.approx(
            dense[1][robot], rel=1e-11, abs=1e-11
        )


@pytest.mark.parametrize("robot_count", (2, 3, 6))
def test_distributed_canonical_reduction_matches_dense_for_any_tree_root(
    robot_count,
):
    from scripts.dba_joint_cavity_state_predictive import (
        distributed_joint_cavity_partition_objective,
        joint_cavity_partition_objective,
    )

    states = []
    components = [[], []]
    transforms = {}
    for robot in range(robot_count):
        copies = (
            ObjectCopy(robot, 0, 10),
            ObjectCopy(robot, 1, 71),
        )
        components[0].append(copies[0])
        components[1].append(copies[1])
        states.append(
            RobotCavityState(
                robot,
                copies,
                np.array(
                    [[0.2 * robot, -0.1], [1.0 + 0.1 * robot, 0.8]]
                ),
                np.array(
                    [
                        [0.3 + 0.01 * robot, 0.04],
                        [0.04, 0.5 + 0.02 * robot],
                    ]
                ),
            )
        )
        transforms[robot] = (
            0.03 * robot,
            np.array([0.1 * robot, -0.05 * robot]),
        )
    partition = NetworkPartition.build(components)
    problem = NetworkQuotientProblem(tuple(states), partition)
    prior = GaussianObjectPrior(
        np.array([0.1, -0.2]), np.array([[2.0, 0.2], [0.2, 1.5]])
    )
    intensities = {10: 2.0, 71: 3.0}
    topology = tuple((robot, robot + 1) for robot in range(robot_count - 1))
    expected = joint_cavity_partition_objective(
        partition, problem, transforms, prior, intensities
    )

    first = distributed_joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        topology=topology,
        root_robot=0,
    )
    second = distributed_joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        intensities,
        topology=tuple(reversed(topology)),
        root_robot=robot_count - 1,
    )

    for actual in (first, second):
        assert actual.objective.total == pytest.approx(expected.total, abs=1e-10)
        assert actual.objective.state_nll == pytest.approx(
            expected.state_nll, abs=1e-10
        )
        assert actual.objective.mahalanobis == pytest.approx(
            expected.mahalanobis, abs=1e-10
        )
        assert actual.objective.log_determinant == pytest.approx(
            expected.log_determinant, abs=1e-10
        )
        assert actual.reduction_messages == robot_count - 1
        assert actual.broadcast_messages == robot_count - 1
        if robot_count > 1:
            assert actual.reduction_payload_bytes > 0
            assert actual.broadcast_payload_bytes == 8 * (robot_count - 1)


def test_leave_pair_composite_profiles_one_shared_network_transform():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
        profile_joint_cavity_leave_pair_composite_transforms,
    )

    copies = tuple(ObjectCopy(robot, 0, 10) for robot in range(3))
    partition = NetworkPartition.build(((copies[0], copies[1]), (copies[2],)))
    problem = NetworkQuotientProblem(
        tuple(
            RobotCavityState(
                robot,
                (copies[robot],),
                np.array([[1.5 * robot, -0.2 * robot]]),
                np.array([[0.2 + 0.05 * robot]]),
            )
            for robot in range(3)
        ),
        partition,
    )
    leave_pairs = {(0, 1): problem, (1, 2): problem}
    prior = GaussianObjectPrior(np.zeros(2), 10.0 * np.eye(2))
    initial_transforms = {
        robot: (0.0, np.zeros(2)) for robot in range(3)
    }
    initial_objective = sum(
        joint_cavity_partition_objective(
            partition,
            active,
            initial_transforms,
            prior,
            {10: 1.0},
        ).total
        for pair, active in leave_pairs.items()
    )

    result = profile_joint_cavity_leave_pair_composite_transforms(
        partition,
        leave_pairs,
        prior,
        {10: 1.0},
        root_robot=0,
        initial_transforms=initial_transforms,
    )

    assert set(result.transforms) == {0, 1, 2}
    assert result.objective.total <= initial_objective + 1e-10


def test_crossfit_fold_changes_only_its_heldout_association_block():
    from scripts.dba_joint_cavity_state_predictive import (
        _crossfit_fold_partition,
    )

    copies = tuple(ObjectCopy(robot, 0, 10) for robot in range(3))
    reference = NetworkPartition.build(tuple((copy,) for copy in copies))
    candidate = NetworkPartition.build((copies,))
    topology = ((0, 1), (1, 2))

    first = _crossfit_fold_partition(
        reference, candidate, (0, 1), topology
    )
    second = _crossfit_fold_partition(
        reference, candidate, (1, 2), topology
    )

    assert first == NetworkPartition.build(
        ((copies[0], copies[1]), (copies[2],))
    )
    assert second == NetworkPartition.build(
        ((copies[0],), (copies[1], copies[2]))
    )


def test_crossfit_partition_from_cached_edges_matches_direct_construction():
    from scripts.dba_joint_cavity_state_predictive import (
        _crossfit_fold_partition_from_edges,
        _topology_partition_edges,
    )
    from scripts.dba_network_quotient_reference import partition_from_edges

    first = tuple(ObjectCopy(robot, 0, 10) for robot in range(3))
    second = tuple(ObjectCopy(robot, 1, 10) for robot in range(3))
    reference = NetworkPartition.build(
        (
            (first[0], first[1], first[2]),
            (second[0], second[1]),
            (second[2],),
        )
    )
    candidate = NetworkPartition.build(
        (
            (first[0], first[1]),
            (first[2],),
            (second[0], second[1], second[2]),
        )
    )
    topology = ((0, 1), (1, 2))
    reference_edges = _topology_partition_edges(reference, topology)
    candidate_edges = _topology_partition_edges(candidate, topology)

    for fold in topology:
        selected = []
        for pair in sorted(reference_edges):
            selected.extend(
                candidate_edges[pair]
                if pair == tuple(sorted(fold))
                else reference_edges[pair]
            )
        expected = partition_from_edges(reference.copies, selected)
        actual = _crossfit_fold_partition_from_edges(
            reference.copies,
            reference_edges,
            candidate_edges,
            fold,
        )
        assert actual == expected


def test_diagonal_joint_objective_reduces_to_component_evidence():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem()
    observations = {}
    for state in problem.robots:
        yaw, translation = transforms[state.robot]
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        )
        means = state.means @ rotation.T + translation
        for index, copy in enumerate(state.copies):
            observations[copy] = GaussianObjectObservation(
                copy,
                means[index],
                np.eye(2) * state.covariance[index, index],
            )

    actual = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        {10: 2.0, 71: 3.0},
    )
    expected = partition_objective(
        partition,
        observations,
        prior,
        component_intensities={10: 2.0, 71: 3.0},
    )

    assert actual.total == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("support", "support"),
        ("transforms", "cover every robot"),
        ("prior", "two-dimensional"),
        ("intensities", "cover every semantic class"),
    ),
)
def test_joint_objective_rejects_inconsistent_inputs(mutation, message):
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem()
    intensities = {10: 1.0, 71: 1.0}
    if mutation == "support":
        partition = NetworkPartition.build((partition.components[0],))
    elif mutation == "transforms":
        transforms = {0: transforms[0]}
    elif mutation == "prior":
        prior = GaussianObjectPrior(np.zeros(1), np.eye(1))
    else:
        intensities = {10: 1.0}

    with pytest.raises(ValueError, match=message):
        joint_cavity_partition_objective(
            partition, problem, transforms, prior, intensities
        )


def test_joint_transform_gradient_matches_centered_finite_difference():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_transform_objective_with_gradient,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    value, gradient = joint_cavity_transform_objective_with_gradient(
        partition,
        problem,
        transforms,
        prior,
        {10: 1.0, 71: 1.0},
        root_robot=0,
    )
    assert np.isfinite(value)
    vector = np.array(
        [transforms[1][0], transforms[1][1][0], transforms[1][1][1]]
    )
    numerical = np.zeros(3)
    epsilon = 1e-6
    for index in range(3):
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_transforms = {
            0: transforms[0],
            1: (plus[0], plus[1:]),
        }
        minus_transforms = {
            0: transforms[0],
            1: (minus[0], minus[1:]),
        }
        plus_value = joint_cavity_transform_objective_with_gradient(
            partition,
            problem,
            plus_transforms,
            prior,
            {10: 1.0, 71: 1.0},
            root_robot=0,
        )[0]
        minus_value = joint_cavity_transform_objective_with_gradient(
            partition,
            problem,
            minus_transforms,
            prior,
            {10: 1.0, 71: 1.0},
            root_robot=0,
        )[0]
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)

    assert gradient == pytest.approx(numerical, abs=2e-7)


def test_joint_profile_does_not_increase_the_same_objective():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
        profile_joint_cavity_transforms,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    initial = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        {10: 1.0, 71: 1.0},
    )

    result = profile_joint_cavity_transforms(
        partition,
        problem,
        prior,
        {10: 1.0, 71: 1.0},
        root_robot=0,
        initial_transforms=transforms,
    )

    assert result.objective.total <= initial.total + 1e-12
    assert result.evaluations > 0
    assert np.isfinite(result.gradient_norm)
    assert isinstance(result.status, int)
    assert result.message


def test_joint_objective_is_invariant_to_common_global_gauge():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    original = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        {10: 1.0, 71: 1.0},
    )
    yaw = -0.63
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    translation = np.array([4.0, -2.5])
    gauged_transforms = {
        robot: (
            yaw + transform[0],
            rotation @ transform[1] + translation,
        )
        for robot, transform in transforms.items()
    }
    gauged_prior = GaussianObjectPrior(
        rotation @ prior.mean + translation,
        rotation @ prior.covariance @ rotation.T,
    )
    gauged = joint_cavity_partition_objective(
        partition,
        problem,
        gauged_transforms,
        gauged_prior,
        {10: 1.0, 71: 1.0},
    )

    assert gauged.total == pytest.approx(original.total, abs=1e-10)


def test_joint_objective_is_invariant_to_robot_permutation():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    original = joint_cavity_partition_objective(
        partition,
        problem,
        transforms,
        prior,
        {10: 1.0, 71: 1.0},
    )
    mapping = {0: 1, 1: 0}
    permuted = relabel_robots(problem, mapping)
    permuted_transforms = {
        mapping[robot]: transform for robot, transform in transforms.items()
    }
    actual = joint_cavity_partition_objective(
        permuted.partition,
        permuted,
        permuted_transforms,
        prior,
        {10: 1.0, 71: 1.0},
    )

    assert actual.total == pytest.approx(original.total, abs=1e-10)


def test_joint_partition_delta_is_invariant_to_unit_change():
    from scripts.dba_joint_cavity_state_predictive import (
        joint_cavity_partition_objective,
    )

    problem, partition, transforms, prior = _two_robot_problem(
        off_diagonal=0.12
    )
    singleton = NetworkPartition.build(
        tuple((copy,) for copy in partition.copies)
    )

    def delta(active_problem, active_transforms, active_prior):
        merged_value = joint_cavity_partition_objective(
            partition,
            active_problem,
            active_transforms,
            active_prior,
            {10: 1.0, 71: 1.0},
        ).total
        singleton_value = joint_cavity_partition_objective(
            singleton,
            active_problem,
            active_transforms,
            active_prior,
            {10: 1.0, 71: 1.0},
        ).total
        return merged_value - singleton_value

    expected = delta(problem, transforms, prior)
    scale = 100.0
    scaled_problem = NetworkQuotientProblem(
        tuple(
            RobotCavityState(
                state.robot,
                state.copies,
                scale * state.means,
                scale * scale * state.covariance,
            )
            for state in problem.robots
        ),
        problem.partition,
    )
    scaled_transforms = {
        robot: (transform[0], scale * transform[1])
        for robot, transform in transforms.items()
    }
    scaled_prior = GaussianObjectPrior(
        scale * prior.mean, scale * scale * prior.covariance
    )

    assert delta(
        scaled_problem, scaled_transforms, scaled_prior
    ) == pytest.approx(expected, abs=1e-8)


def _observable_leave_pair_case():
    first = (ObjectCopy(0, 0, 10), ObjectCopy(0, 1, 10))
    second = (ObjectCopy(1, 0, 10), ObjectCopy(1, 1, 10))
    anchors = (
        ObjectCopy(0, 2, 9),
        ObjectCopy(1, 2, 9),
        ObjectCopy(0, 3, 9),
        ObjectCopy(1, 3, 9),
    )
    initial = NetworkPartition.build(
        tuple((copy,) for copy in first + second)
        + ((anchors[0], anchors[1]), (anchors[2], anchors[3]))
    )
    problem = NetworkQuotientProblem(
        (
            RobotCavityState(
                0,
                first + (anchors[0], anchors[2]),
                np.array(
                    [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]
                ),
                np.array(
                    [
                        [0.02, 0.005, 0.0, 0.0],
                        [0.005, 0.02, 0.0, 0.0],
                        [0.0, 0.0, 0.01, 0.002],
                        [0.0, 0.0, 0.002, 0.01],
                    ]
                ),
            ),
            RobotCavityState(
                1,
                second + (anchors[1], anchors[3]),
                np.array(
                    [[0.05, 0.0], [5.05, 0.0], [0.05, 5.0], [5.05, 5.0]]
                ),
                np.array(
                    [
                        [0.02, -0.004, 0.0, 0.0],
                        [-0.004, 0.02, 0.0, 0.0],
                        [0.0, 0.0, 0.01, -0.002],
                        [0.0, 0.0, -0.002, 0.01],
                    ]
                ),
            ),
        ),
        initial,
    )
    prior = GaussianObjectPrior(np.zeros(2), 100.0 * np.eye(2))
    candidates = tuple((a, b) for a in first for b in second)
    expected = NetworkPartition.build(
        (
            (first[0], second[0]),
            (first[1], second[1]),
            (anchors[0], anchors[1]),
            (anchors[2], anchors[3]),
        )
    )
    arguments = {
        "root_robot": 0,
        "initial_transforms": {
            0: (0.0, np.zeros(2)),
            1: (0.0, np.zeros(2)),
        },
        "component_intensities": {9: 1.0, 10: 1.0},
        "selection_topology": ((0, 1),),
        "leave_pair_problems": {(0, 1): problem},
    }
    return initial, problem, prior, candidates, expected, arguments


def test_leave_pair_optimizer_monotonically_recovers_observable_matches():
    from scripts.dba_joint_cavity_state_predictive import (
        run_joint_cavity_partitioning,
    )

    initial, problem, prior, candidates, expected, arguments = (
        _observable_leave_pair_case()
    )

    result = run_joint_cavity_partitioning(
        initial,
        problem,
        prior,
        candidates,
        **arguments,
    )

    assert result.partition == expected
    assert len(result.accepted_edits) == 2
    assert len(result.candidate_edge_counts) == 1
    assert len(result.objective_trace) == 2
    assert all(
        after < before
        for before, after in zip(
            result.objective_trace, result.objective_trace[1:]
        )
    )
    assert result.consensus_rounds == 3
    assert result.objective_evaluations > 0
    assert result.canonical_state_sync_messages > 0
    assert result.canonical_state_sync_payload_bytes > 0


def test_leave_pair_optimizer_structured_and_dense_backends_are_equivalent(
    monkeypatch,
):
    import scripts.dba_joint_cavity_state_predictive as module

    initial, problem, prior, candidates, _, arguments = (
        _observable_leave_pair_case()
    )
    structured = module.run_joint_cavity_partitioning(
        initial, problem, prior, candidates, **arguments
    )
    structured_workspace = module._JointCavityCanonicalWorkspace

    class DenseWorkspace(structured_workspace):
        def __init__(self, *args, **kwargs):
            kwargs["precision_backend"] = "dense"
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        module, "_JointCavityCanonicalWorkspace", DenseWorkspace
    )
    dense = module.run_joint_cavity_partitioning(
        initial, problem, prior, candidates, **arguments
    )

    assert structured.partition == dense.partition
    assert structured.accepted_edits == dense.accepted_edits
    assert structured.candidate_edge_counts == dense.candidate_edge_counts
    assert structured.objective_trace == pytest.approx(
        dense.objective_trace, rel=1e-10, abs=1e-10
    )
    assert structured.objective.total == pytest.approx(
        dense.objective.total, rel=1e-10, abs=1e-10
    )
