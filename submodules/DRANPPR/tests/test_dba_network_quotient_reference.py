import importlib.util
import csv
import inspect
import json
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dba_network_quotient_reference.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "dba_network_quotient_reference", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _two_robot_problem(module):
    first_copies = tuple(module.ObjectCopy(0, index, 10) for index in range(3))
    second_copies = tuple(module.ObjectCopy(1, index, 10) for index in range(3))
    first = module.RobotCavityState(
        robot=0,
        copies=first_copies,
        means=np.array([[0.0, 0.0], [1.2, 0.2], [0.3, 1.4]]),
        covariance=np.array(
            [
                [0.08, 0.01, -0.01],
                [0.01, 0.07, 0.005],
                [-0.01, 0.005, 0.09],
            ]
        ),
    )
    second = module.RobotCavityState(
        robot=1,
        copies=second_copies,
        means=np.array([[2.0, -1.0], [2.9, -0.2], [1.1, -0.1]]),
        covariance=np.array(
            [
                [0.06, -0.004, 0.006],
                [-0.004, 0.08, 0.01],
                [0.006, 0.01, 0.07],
            ]
        ),
    )
    partition = module.NetworkPartition.build(
        tuple((first_copies[index], second_copies[index]) for index in range(3))
    )
    return module.NetworkQuotientProblem((first, second), partition)


@pytest.mark.parametrize(
    "identifiers",
    (
        (True, 4, 10),
        (0.0, 4, 10),
        (0, False, 10),
        (0, 4.0, 10),
        (0, 4, True),
        (0, 4, 10.0),
    ),
)
def test_object_copy_rejects_bool_and_float_identifiers(identifiers):
    module = load_module()

    with pytest.raises(ValueError, match="integral"):
        module.ObjectCopy(*identifiers)


def test_object_copy_normalizes_numpy_integer_identifiers():
    module = load_module()

    copy = module.ObjectCopy(np.int64(2), np.int64(4), np.int64(10))

    assert copy == module.ObjectCopy(2, 4, 10)
    assert type(copy.robot) is int
    assert type(copy.object_id) is int
    assert type(copy.semantic_class) is int


def test_partition_build_canonicalizes_members_and_components():
    module = load_module()
    a = module.ObjectCopy(1, 4, 10)
    b = module.ObjectCopy(0, 7, 10)
    c = module.ObjectCopy(2, 3, 71)

    first = module.NetworkPartition.build(((a, b), (c,)))
    second = module.NetworkPartition.build(((c,), (b, a)))

    assert first == second
    assert first.components == ((b, a), (c,))
    assert first.copy_count == 3
    assert first.component_count == 2


def test_partition_copies_are_globally_sorted():
    module = load_module()
    first = module.ObjectCopy(0, 4, 10)
    middle = module.ObjectCopy(1, 7, 71)
    last = module.ObjectCopy(2, 3, 10)

    partition = module.NetworkPartition.build(((first, last), (middle,)))

    assert partition.copies == (first, middle, last)


def test_partition_copy_support_is_cached_after_canonicalization():
    module = load_module()
    first = module.ObjectCopy(0, 1, 10)
    second = module.ObjectCopy(1, 2, 10)
    partition = module.NetworkPartition.build(((first, second),))

    assert partition.copies == (first, second)
    assert partition.copies is partition.copies


@pytest.mark.parametrize(
    ("components", "message"),
    [
        (
            (((0, 4, 10), (0, 7, 10)),),
            "more than one copy",
        ),
        (
            (((0, 4, 10), (1, 7, 71)),),
            "semantic",
        ),
        (
            (((0, 4, 10),), ((0, 4, 10),)),
            "appears in multiple components",
        ),
    ],
)
def test_partition_build_rejects_invalid_structure(components, message):
    module = load_module()
    copies = tuple(
        tuple(module.ObjectCopy(*copy) for copy in component)
        for component in components
    )

    with pytest.raises(ValueError, match=message):
        module.NetworkPartition.build(copies)


def test_fixed_cardinality_validation_rejects_component_merge():
    module = load_module()
    a = module.ObjectCopy(0, 4, 10)
    b = module.ObjectCopy(1, 7, 10)
    current = module.NetworkPartition.build(((a,), (b,)))
    candidate = module.NetworkPartition.build(((a, b),))

    result = module.validate_fixed_cardinality_candidate(current, candidate)

    assert result == module.ValidationResult(
        valid=False, status="cardinality_change_unsupported"
    )


def test_fixed_cardinality_validation_rejects_copy_support_mismatch():
    module = load_module()
    current_copy = module.ObjectCopy(0, 4, 10)
    candidate_copy = module.ObjectCopy(1, 7, 10)
    current = module.NetworkPartition.build(((current_copy,),))
    candidate = module.NetworkPartition.build(((candidate_copy,),))

    result = module.validate_fixed_cardinality_candidate(current, candidate)

    assert result == module.ValidationResult(
        valid=False, status="copy_support_mismatch"
    )


def test_fixed_cardinality_validation_accepts_same_support_and_cardinality():
    module = load_module()
    a = module.ObjectCopy(0, 4, 10)
    b = module.ObjectCopy(1, 7, 10)
    c = module.ObjectCopy(2, 3, 10)
    d = module.ObjectCopy(3, 8, 10)
    current = module.NetworkPartition.build(((a, b), (c, d)))
    candidate = module.NetworkPartition.build(((a, c), (b, d)))

    result = module.validate_fixed_cardinality_candidate(current, candidate)

    assert result == module.ValidationResult(valid=True, status="ok")


def test_robot_cavity_rejects_non_spd_covariance():
    module = load_module()
    copies = (
        module.ObjectCopy(0, 0, 10),
        module.ObjectCopy(0, 1, 10),
    )

    with pytest.raises(ValueError, match="positive definite"):
        module.RobotCavityState(
            robot=0,
            copies=copies,
            means=np.zeros((2, 2)),
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_exact_local_hessian_matches_finite_difference_gradient():
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.array([0.12, -1.7, 0.8, 0.1, 0.2, 1.0, 0.4, 0.2, 1.3])

    evaluation = module.evaluate_network_canonical(problem, layout, vector)
    finite_difference = module.finite_difference_hessian(
        lambda value: module.evaluate_network_canonical(
            problem, layout, value
        ).gradient,
        vector,
    )

    assert evaluation.hessian == pytest.approx(
        finite_difference, rel=2e-6, abs=2e-6
    )
    assert evaluation.hessian == pytest.approx(
        evaluation.hessian.T, abs=1e-12
    )


def test_robot_cavity_copies_and_freezes_input_arrays():
    module = load_module()
    copies = (module.ObjectCopy(0, 0, 10),)
    means = np.array([[1.0, 2.0]])
    covariance = np.array([[0.25]])
    state = module.RobotCavityState(0, copies, means, covariance)

    means[0, 0] = 9.0
    covariance[0, 0] = 4.0

    np.testing.assert_allclose(state.means, [[1.0, 2.0]])
    np.testing.assert_allclose(state.covariance, [[0.25]])
    with pytest.raises(ValueError):
        state.means[0, 0] = 3.0
    with pytest.raises(ValueError):
        state.covariance[0, 0] = 3.0


def test_robot_cavity_rejects_negative_robot():
    module = load_module()

    with pytest.raises(ValueError, match="nonnegative"):
        module.RobotCavityState(
            robot=-1,
            copies=(),
            means=np.zeros((0, 2)),
            covariance=np.zeros((0, 0)),
        )


def test_robot_cavity_rejects_empty_copy_set():
    module = load_module()

    with pytest.raises(ValueError, match="must not be empty"):
        module.RobotCavityState(
            robot=0,
            copies=(),
            means=np.zeros((0, 2)),
            covariance=np.zeros((0, 0)),
        )


def test_local_evaluation_does_not_form_covariance_inverse(monkeypatch):
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.array([0.12, -1.7, 0.8, 0.1, 0.2, 1.0, 0.4, 0.2, 1.3])

    def reject_inverse(*_args, **_kwargs):
        raise AssertionError("explicit covariance inverse is forbidden")

    monkeypatch.setattr(module.np.linalg, "inv", reject_inverse)

    module.evaluate_local_canonical(problem.robots[1], layout, vector)


def test_network_value_gradient_matches_central_finite_difference():
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.array([0.12, -1.7, 0.8, 0.1, 0.2, 1.0, 0.4, 0.2, 1.3])
    evaluation = module.evaluate_network_canonical(problem, layout, vector)
    finite_difference = np.empty(layout.dimension)
    base_step = np.finfo(float).eps ** (1.0 / 3.0)

    for index in range(layout.dimension):
        step = base_step * (1.0 + abs(vector[index]))
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            module.evaluate_network_canonical(problem, layout, plus).value
            - module.evaluate_network_canonical(problem, layout, minus).value
        ) / (2.0 * step)

    assert evaluation.gradient == pytest.approx(
        finite_difference, rel=2e-7, abs=2e-7
    )


def test_local_value_matches_manual_full_covariance_solve():
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.array([0.12, -1.7, 0.8, 0.1, 0.2, 1.0, 0.4, 0.2, 1.3])
    state = problem.robots[1]
    components = vector[layout.component_offset :].reshape((-1, 2))
    residual = (
        state.means @ module.rotation2(vector[0]).T
        + vector[1:3]
        - components
    )
    information_residual = np.linalg.solve(state.covariance, residual)
    expected_chi2 = float(np.sum(residual * information_residual))
    _, covariance_logdet = np.linalg.slogdet(state.covariance)
    expected_value = (
        0.5 * expected_chi2
        + covariance_logdet
        + len(state.copies) * np.log(2.0 * np.pi)
    )

    evaluation = module.evaluate_local_canonical(state, layout, vector)

    assert evaluation.residual_chi2 == pytest.approx(expected_chi2, abs=1e-12)
    assert evaluation.value == pytest.approx(expected_value, abs=1e-12)


def test_network_evaluation_equals_sum_of_local_evaluations():
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.array([0.12, -1.7, 0.8, 0.1, 0.2, 1.0, 0.4, 0.2, 1.3])
    local = tuple(
        module.evaluate_local_canonical(state, layout, vector)
        for state in problem.robots
    )

    evaluation = module.evaluate_network_canonical(problem, layout, vector)

    assert evaluation.value == pytest.approx(sum(item.value for item in local))
    assert evaluation.gradient == pytest.approx(
        sum((item.gradient for item in local), np.zeros(layout.dimension))
    )
    assert evaluation.hessian == pytest.approx(
        sum(
            (item.hessian for item in local),
            np.zeros((layout.dimension, layout.dimension)),
        )
    )
    assert evaluation.residual_chi2 == pytest.approx(
        sum(item.residual_chi2 for item in local)
    )


def test_unpack_rejects_non_finite_vector():
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.zeros(layout.dimension)
    vector[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        module._unpack(layout, vector)


def test_variable_layout_mappings_are_read_only():
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)

    with pytest.raises(TypeError):
        layout.transform_offset[1] = 99
    with pytest.raises(TypeError):
        layout.component_by_copy[problem.partition.copies[0]] = 99


@pytest.mark.parametrize(
    "slogdet_result",
    [(-1.0, 0.0), (np.nan, 0.0), (1.0, np.inf)],
)
def test_local_evaluation_requires_positive_finite_covariance_determinant(
    monkeypatch, slogdet_result
):
    module = load_module()
    problem = _two_robot_problem(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.zeros(layout.dimension)

    monkeypatch.setattr(
        module.np.linalg, "slogdet", lambda _value: slogdet_result
    )

    with pytest.raises(ValueError, match="positive determinant"):
        module.evaluate_local_canonical(problem.robots[0], layout, vector)


def test_near_symmetric_covariance_is_normalized_before_evaluation():
    module = load_module()
    copies = (
        module.ObjectCopy(0, 0, 10),
        module.ObjectCopy(0, 1, 10),
    )
    state = module.RobotCavityState(
        robot=0,
        copies=copies,
        means=np.array([[0.2, -0.1], [-0.3, 0.4]]),
        covariance=np.array([[1e-12, 9e-13], [0.0, 1e-12]]),
    )
    partition = module.NetworkPartition.build(((copies[0],), (copies[1],)))
    problem = module.NetworkQuotientProblem((state,), partition)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.array([0.5, -0.2, -0.4, 0.7])

    assert np.array_equal(state.covariance, state.covariance.T)

    evaluation = module.evaluate_network_canonical(problem, layout, vector)
    finite_difference = np.empty(layout.dimension)
    base_step = np.finfo(float).eps ** (1.0 / 3.0)
    for index in range(layout.dimension):
        step = base_step * (1.0 + abs(vector[index]))
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            module.evaluate_network_canonical(problem, layout, plus).value
            - module.evaluate_network_canonical(problem, layout, minus).value
        ) / (2.0 * step)

    assert evaluation.gradient == pytest.approx(
        finite_difference, rel=2e-8, abs=1e-3
    )


def _three_robot_fixture(module, false_partition=False):
    rng = np.random.default_rng(20260718)
    world_points = np.array(
        [
            [-4.0, -1.0],
            [-1.5, 3.2],
            [0.7, -2.8],
            [3.8, 2.1],
            [6.2, -0.4],
            [1.9, 6.0],
            [-3.1, 6.8],
        ]
    )
    yaws = (0.0, 0.63, -0.91)
    translations = (
        np.array([0.0, 0.0]),
        np.array([7.0, -4.0]),
        np.array([-5.0, 8.0]),
    )
    states = []
    copies_by_robot = []
    for robot, (yaw, translation) in enumerate(zip(yaws, translations)):
        copies = tuple(module.ObjectCopy(robot, index, 10) for index in range(7))
        design = rng.normal(size=(7, 7))
        covariance = 0.002 * (design @ design.T / 7.0) + 0.004 * np.eye(7)
        noise = np.column_stack(
            (
                rng.multivariate_normal(np.zeros(7), covariance),
                rng.multivariate_normal(np.zeros(7), covariance),
            )
        )
        local_means = (
            module.rotation2(yaw).T @ (world_points - translation).T
        ).T + noise
        states.append(
            module.RobotCavityState(robot, copies, local_means, covariance)
        )
        copies_by_robot.append(copies)

    robot_two_indices = (
        np.roll(np.arange(7), 1) if false_partition else np.arange(7)
    )
    partition = module.NetworkPartition.build(
        (
            (
                copies_by_robot[0][index],
                copies_by_robot[1][index],
                copies_by_robot[2][robot_two_indices[index]],
            )
            for index in range(7)
        )
    )
    return module.NetworkQuotientProblem(tuple(states), partition)


def _apply_local_gauges(module, problem, seed):
    rng = np.random.default_rng(seed)
    states = []
    for state in sorted(problem.robots, key=lambda item: item.robot):
        yaw = rng.uniform(-np.pi, np.pi)
        translation = rng.normal(scale=20.0, size=2)
        means = (
            module.rotation2(yaw) @ state.means.T + translation[:, None]
        ).T
        states.append(
            module.RobotCavityState(
                state.robot, state.copies, means, state.covariance
            )
        )
    return module.NetworkQuotientProblem(tuple(states), problem.partition)


def _explicit_root_spread(score):
    values = [value for _, value in score.root_scores]
    return max(values) - min(values)


@pytest.mark.parametrize("seed", range(20))
def test_score_is_invariant_to_independent_local_robot_gauges(seed):
    module = load_module()
    correct_problem = _three_robot_fixture(module)
    false_problem = _three_robot_fixture(module, false_partition=True)
    original_correct = module.score_network_partition(correct_problem)
    original_false = module.score_network_partition(false_problem)
    gauged_correct = module.score_network_partition(
        _apply_local_gauges(module, correct_problem, seed)
    )
    gauged_false = module.score_network_partition(
        _apply_local_gauges(module, false_problem, seed)
    )

    scores = (original_correct, original_false, gauged_correct, gauged_false)
    assert all(score.valid for score in scores)
    assert abs(gauged_correct.score - original_correct.score) < 1e-8
    assert abs(gauged_false.score - original_false.score) < 1e-8
    original_margin = original_false.score - original_correct.score
    gauged_margin = gauged_false.score - gauged_correct.score
    assert abs(gauged_margin - original_margin) < 1e-8
    for score in scores:
        assert score.root_score_spread == pytest.approx(
            _explicit_root_spread(score), abs=0.0, rel=0.0
        )


@pytest.mark.parametrize("seed", (0, 7, 19))
def test_independent_local_gauges_preserve_candidate_ranking(seed):
    module = load_module()
    correct_problem = _three_robot_fixture(module)
    false_problem = _three_robot_fixture(module, false_partition=True)
    original_correct = module.score_network_partition(correct_problem)
    original_false = module.score_network_partition(false_problem)
    gauged_correct = module.score_network_partition(
        _apply_local_gauges(module, correct_problem, seed)
    )
    gauged_false = module.score_network_partition(
        _apply_local_gauges(module, false_problem, seed)
    )

    assert all(
        result.valid
        for result in (
            original_correct,
            original_false,
            gauged_correct,
            gauged_false,
        )
    )
    original_margin = original_false.score - original_correct.score
    gauged_margin = gauged_false.score - gauged_correct.score
    assert original_margin > 0.0
    assert gauged_margin > 0.0
    assert np.sign(gauged_margin) == np.sign(original_margin)


def test_robot_relabel_preserves_score_and_rebuilds_sorted_roots():
    module = load_module()
    problem = _three_robot_fixture(module)
    mapping = {0: 2, 1: 0, 2: 1}

    original = module.score_network_partition(problem)
    relabeled_problem = module.relabel_robots(problem, mapping)
    relabeled = module.score_network_partition(relabeled_problem)

    assert original.valid
    assert relabeled.valid
    assert abs(relabeled.score - original.score) < 1e-8
    assert tuple(state.robot for state in relabeled_problem.robots) == (
        0,
        1,
        2,
    )
    assert tuple(root for root, _ in relabeled.root_scores) == (0, 1, 2)
    for old_state in problem.robots:
        new_state = next(
            state
            for state in relabeled_problem.robots
            if state.robot == mapping[old_state.robot]
        )
        np.testing.assert_array_equal(new_state.means, old_state.means)
        np.testing.assert_array_equal(
            new_state.covariance, old_state.covariance
        )


@pytest.mark.parametrize(
    "mapping",
    (
        {0: 2, 1: 0},
        {0: 0, 1: 0, 2: 1},
        {0: -1, 1: 0, 2: 1},
    ),
)
def test_robot_relabel_rejects_invalid_mapping(mapping):
    module = load_module()

    with pytest.raises(ValueError):
        module.relabel_robots(_three_robot_fixture(module), mapping)


@pytest.mark.parametrize(
    "mapping",
    (
        [2, 0, 1],
        {False: 2, 1: 0, 2: 1},
        {0.0: 2, 1: 0, 2: 1},
    ),
)
def test_robot_relabel_rejects_nonmapping_or_nonintegral_source_ids(mapping):
    module = load_module()

    with pytest.raises(ValueError):
        module.relabel_robots(_three_robot_fixture(module), mapping)


def test_common_local_gauge_preserves_score_and_problem_structure():
    module = load_module()
    problem = _three_robot_fixture(module)

    original = module.score_network_partition(problem)
    gauged_problem = module.apply_common_local_gauge(
        problem, 1.17, np.array([31.0, -19.0])
    )
    gauged = module.score_network_partition(gauged_problem)

    assert original.valid
    assert gauged.valid
    assert abs(gauged.score - original.score) < 1e-8
    assert gauged_problem.partition is problem.partition
    yaw = 1.17
    translation = np.array([31.0, -19.0])
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
    )
    for original_state, gauged_state in zip(
        problem.robots, gauged_problem.robots
    ):
        assert gauged_state.copies == original_state.copies
        expected_means = original_state.means @ rotation.T + translation
        inverse_action = original_state.means @ rotation + translation
        np.testing.assert_array_equal(gauged_state.means, expected_means)
        assert not np.allclose(gauged_state.means, inverse_action)
        np.testing.assert_array_equal(
            gauged_state.covariance, original_state.covariance
        )


@pytest.mark.parametrize(
    ("yaw", "translation"),
    (
        (np.array([1.17]), np.array([31.0, -19.0])),
        (np.nan, np.array([31.0, -19.0])),
        (1.17, np.array([31.0])),
        (1.17, np.array([31.0, np.inf])),
    ),
)
def test_common_local_gauge_rejects_invalid_transform(yaw, translation):
    module = load_module()

    with pytest.raises(ValueError):
        module.apply_common_local_gauge(
            _three_robot_fixture(module), yaw, translation
        )


def _object_relabel_mapping(module, problem):
    return {
        copy: module.ObjectCopy(
            copy.robot, 100 - copy.object_id, copy.semantic_class
        )
        for copy in problem.partition.copies
    }


def test_object_relabel_preserves_score_and_cavity_row_order():
    module = load_module()
    problem = _three_robot_fixture(module)
    mapping = _object_relabel_mapping(module, problem)

    original = module.score_network_partition(problem)
    relabeled_problem = module.relabel_objects(problem, mapping)
    relabeled = module.score_network_partition(relabeled_problem)

    assert original.valid
    assert relabeled.valid
    assert abs(relabeled.score - original.score) < 1e-8
    for original_state, relabeled_state in zip(
        problem.robots, relabeled_problem.robots
    ):
        assert relabeled_state.copies == tuple(
            mapping[copy] for copy in original_state.copies
        )
        np.testing.assert_array_equal(
            relabeled_state.means, original_state.means
        )
        np.testing.assert_array_equal(
            relabeled_state.covariance, original_state.covariance
        )


def test_object_relabel_rejects_incomplete_mapping():
    module = load_module()
    problem = _three_robot_fixture(module)
    mapping = _object_relabel_mapping(module, problem)
    mapping.pop(problem.partition.copies[0])

    with pytest.raises(ValueError):
        module.relabel_objects(problem, mapping)


@pytest.mark.parametrize("changed_field", ("robot", "semantic_class"))
def test_object_relabel_rejects_robot_or_class_change(changed_field):
    module = load_module()
    problem = _three_robot_fixture(module)
    mapping = _object_relabel_mapping(module, problem)
    source = problem.partition.copies[0]
    destination = mapping[source]
    replacement = {
        "robot": destination.robot,
        "object_id": destination.object_id,
        "semantic_class": destination.semantic_class,
    }
    replacement[changed_field] += 1
    mapping[source] = module.ObjectCopy(**replacement)

    with pytest.raises(ValueError):
        module.relabel_objects(problem, mapping)


def test_object_relabel_rejects_destination_collision():
    module = load_module()
    problem = _three_robot_fixture(module)
    mapping = _object_relabel_mapping(module, problem)
    same_robot = problem.robots[0].copies[:2]
    mapping[same_robot[1]] = mapping[same_robot[0]]

    with pytest.raises(ValueError):
        module.relabel_objects(problem, mapping)


def test_object_relabel_requires_mapping():
    module = load_module()
    problem = _three_robot_fixture(module)
    destinations = list(_object_relabel_mapping(module, problem).values())

    with pytest.raises(ValueError):
        module.relabel_objects(problem, destinations)


def test_object_relabel_snapshots_stateful_mapping_once():
    module = load_module()
    problem = _three_robot_fixture(module)
    initial = _object_relabel_mapping(module, problem)

    class StatefulMapping(module.Mapping):
        def __init__(self, values):
            self.values = values
            self.reads = {copy: 0 for copy in values}

        def __getitem__(self, copy):
            self.reads[copy] += 1
            if self.reads[copy] == 1:
                return self.values[copy]
            return module.ObjectCopy(
                copy.robot, 200 - copy.object_id, copy.semantic_class
            )

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

    mapping = StatefulMapping(initial)

    relabeled = module.relabel_objects(problem, mapping)

    assert relabeled.partition.copies == tuple(sorted(initial.values()))
    assert all(reads == 1 for reads in mapping.reads.values())


def test_production_source_has_no_generalized_inverse_escape_hatch():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "pinv(" not in source
    assert "pseudo" not in source.lower()


def test_root_determinant_has_no_regularization_or_relaxed_gate():
    module = load_module()
    determinant_source = inspect.getsource(module._root_consistent_logdet).lower()
    score_source = inspect.getsource(module.score_network_partition).lower()

    for forbidden in ("damping", "jitter", "pinv", "pseudo"):
        assert forbidden not in determinant_source
    assert "root_score_spread >= 1e-8" in score_source


def test_singular_yaw_has_no_finite_score_fallback():
    module = load_module()
    first_copies = tuple(module.ObjectCopy(0, index, 10) for index in range(2))
    second_copies = tuple(
        module.ObjectCopy(1, index, 10) for index in range(2)
    )
    states = (
        module.RobotCavityState(
            0, first_copies, np.zeros((2, 2)), np.eye(2)
        ),
        module.RobotCavityState(
            1, second_copies, np.zeros((2, 2)), np.eye(2)
        ),
    )
    partition = module.NetworkPartition.build(
        tuple(
            (first_copies[index], second_copies[index])
            for index in range(2)
        )
    )

    result = module.score_network_partition(
        module.NetworkQuotientProblem(states, partition)
    )

    assert not result.valid
    assert result.status == "indefinite_exact_hessian"
    assert np.isnan(result.score)
    assert result.root_scores == ()


def test_network_quotient_score_prefers_gt_partition():
    module = load_module()
    correct = module.score_network_partition(_three_robot_fixture(module))
    false = module.score_network_partition(
        _three_robot_fixture(module, false_partition=True)
    )

    assert correct.valid
    assert false.valid
    assert false.root_score_spread < 1e-8
    assert correct.score < false.score
    assert false.score - correct.score > 1000.0


def test_every_root_has_the_same_quotient_score():
    module = load_module()
    result = module.score_network_partition(_three_robot_fixture(module))

    assert result.valid
    assert len(result.root_scores) == 3
    assert tuple(root for root, _ in result.root_scores) == (0, 1, 2)
    explicit_scores = [score for _, score in result.root_scores]
    assert result.root_score_spread == pytest.approx(
        max(explicit_scores) - min(explicit_scores), abs=0.0, rel=0.0
    )
    assert result.root_score_spread < 1e-8
    assert result.stationarity_norm <= np.sqrt(np.finfo(float).eps)
    assert result.profiles[0].objective == pytest.approx(
        0.5 * result.profile_chi2
    )


def test_current_transforms_are_regauged_for_every_quotient_root():
    module = load_module()
    current_transforms = {
        0: (0.0, np.zeros(2)),
        1: (0.63, np.array([7.0, -4.0])),
        2: (-0.91, np.array([-5.0, 8.0])),
    }

    result = module.score_network_partition(
        _three_robot_fixture(module), current_transforms=current_transforms
    )

    assert result.valid
    assert tuple(root for root, _ in result.root_scores) == (0, 1, 2)
    assert result.root_score_spread < 1e-8


def test_nonstationary_score_preserves_complete_root_mean(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_stationarity_norm", lambda *_args: 1.0)

    result = module.score_network_partition(_three_robot_fixture(module))

    assert not result.valid
    assert result.status == "profile_not_stationary"
    assert np.isfinite(result.score)
    assert result.score == pytest.approx(
        np.mean([score for _, score in result.root_scores])
    )


@pytest.mark.parametrize(
    "malformed",
    [
        0.0,
        (np.array([0.0, 0.1]), np.zeros(2)),
    ],
)
def test_malformed_current_transforms_return_invalid_status(malformed):
    module = load_module()
    transforms = {
        0: malformed,
        1: (0.63, np.array([7.0, -4.0])),
        2: (-0.91, np.array([-5.0, 8.0])),
    }

    result = module.score_network_partition(
        _three_robot_fixture(module), current_transforms=transforms
    )

    assert not result.valid
    assert result.status == "invalid_current_transforms"
    assert np.isnan(result.score)


def test_nonstationary_status_precedes_indefinite_hessian(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)

    monkeypatch.setattr(module, "_stationarity_norm", lambda *_args: 1.0)
    monkeypatch.setattr(
        module,
        "_root_consistent_logdet",
        lambda *_args: (_ for _ in ()).throw(
            np.linalg.LinAlgError("indefinite exact Hessian")
        ),
    )

    result = module.score_network_partition(problem)

    assert not result.valid
    assert result.status == "profile_not_stationary"
    assert np.isnan(result.score)
    assert len(result.profiles) == 3
    assert result.root_scores == ()


def test_regauged_transforms_preserve_points_in_root_frame():
    module = load_module()
    problem = _three_robot_fixture(module)
    transforms = {
        0: (0.23, np.array([1.1, -0.8])),
        1: (0.63, np.array([7.0, -4.0])),
        2: (-0.91, np.array([-5.0, 8.0])),
    }
    root_robot = 1

    regauged = module._regauge_current_transforms(
        problem, root_robot, transforms
    )

    root_yaw, root_translation = transforms[root_robot]
    root_inverse_rotation = module.rotation2(root_yaw).T
    for state in problem.robots:
        point = state.means[0]
        yaw, translation = transforms[state.robot]
        expected = root_inverse_rotation @ (
            module.rotation2(yaw) @ point + translation - root_translation
        )
        regauged_yaw, regauged_translation = regauged[state.robot]
        actual = (
            module.rotation2(regauged_yaw) @ point + regauged_translation
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    assert regauged[root_robot][0] == 0.0
    assert np.array_equal(regauged[root_robot][1], np.zeros(2))


def test_whitened_profile_derivatives_match_exact_canonical_model():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = module.identity_profile_start(problem, layout)
    vector[:6] = np.array([0.4, 2.0, -1.0, -0.7, -3.0, 4.0])

    residual, jacobian = module.profile_residual_and_jacobian(
        problem, layout, vector
    )
    evaluation = module.evaluate_network_canonical(problem, layout, vector)

    np.testing.assert_allclose(
        jacobian.T @ residual, evaluation.gradient, atol=2e-10, rtol=2e-12
    )
    gauss_newton_hessian = evaluation.hessian.copy()
    transforms, components = module._unpack(layout, vector)
    for state in problem.robots:
        if state.robot == layout.root_robot:
            continue
        yaw, translation = transforms[state.robot]
        rotated = state.means @ module.rotation2(yaw).T
        component_rows = np.array(
            [layout.component_by_copy[copy] for copy in state.copies]
        )
        local_residual = (
            rotated + translation - components[component_rows]
        )
        information_residual = np.linalg.solve(
            state.covariance, local_residual
        )
        correction = float(np.sum(-rotated * information_residual))
        offset = layout.transform_offset[state.robot]
        gauss_newton_hessian[offset, offset] -= correction
    np.testing.assert_allclose(
        jacobian.T @ jacobian,
        gauss_newton_hessian,
        atol=2e-9,
        rtol=2e-12,
    )


def test_exact_conditional_components_match_dense_normal_equations():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    workspace = module._build_reduced_profile_workspace(problem, layout)
    transform_vector = np.array([0.13, 0.4, -0.2, -0.09, -0.3, 0.5])

    full_vector = module._profile_components_for_transforms(
        problem, workspace, transform_vector
    )
    transformed = module._transformed_means_for_reduced_profile(
        problem, workspace, transform_vector
    )
    dense_rhs = np.zeros((problem.partition.component_count, 2))
    for information, component_rows, means in zip(
        workspace.information_by_robot,
        workspace.component_rows_by_robot,
        transformed,
    ):
        np.add.at(dense_rhs, component_rows, information @ means)
    dense_components = np.linalg.solve(
        workspace.component_information, dense_rhs
    )

    np.testing.assert_allclose(
        full_vector[layout.component_offset :].reshape((-1, 2)),
        dense_components,
        rtol=2e-13,
        atol=2e-13,
    )
    evaluation = module.evaluate_network_canonical(
        problem, layout, full_vector
    )
    np.testing.assert_allclose(
        evaluation.gradient[layout.component_offset :],
        0.0,
        rtol=0.0,
        atol=2e-12,
    )


def test_reduced_profile_residual_matches_full_profile_model():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=1)
    workspace = module._build_reduced_profile_workspace(problem, layout)
    transform_vector = np.array([-0.11, 0.3, 0.2, 0.08, -0.4, 0.1])

    residual, jacobian, full_vector = (
        module._reduced_profile_residual_and_jacobian(
            problem, workspace, transform_vector
        )
    )
    full_residual, full_jacobian = module.profile_residual_and_jacobian(
        problem, layout, full_vector
    )
    np.testing.assert_allclose(
        residual, full_residual, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        jacobian.T @ residual,
        (full_jacobian.T @ full_residual)[: layout.component_offset],
        rtol=2e-11,
        atol=2e-11,
    )


def test_reduced_profile_jacobian_matches_central_finite_difference():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=2)
    workspace = module._build_reduced_profile_workspace(problem, layout)
    vector = np.array([0.17, -0.2, 0.5, -0.07, 0.6, -0.3])
    residual, analytic, _ = module._reduced_profile_residual_and_jacobian(
        problem, workspace, vector
    )
    finite = np.empty_like(analytic)
    step = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(vector))
    for column in range(vector.size):
        delta = np.zeros_like(vector)
        delta[column] = step
        plus = module._reduced_profile_residual_and_jacobian(
            problem, workspace, vector + delta
        )[0]
        minus = module._reduced_profile_residual_and_jacobian(
            problem, workspace, vector - delta
        )[0]
        finite[:, column] = (plus - minus) / (2.0 * step)
    relative_error = np.linalg.norm(analytic - finite) / (
        1.0 + np.linalg.norm(finite)
    )
    assert relative_error < 2e-8
    assert residual.shape[0] == analytic.shape[0]


def _independent_weighted_yaw_optimum(module, source, target, information):
    ones = np.ones(source.shape[0])
    centroid_weights = information @ ones
    denominator = float(ones @ centroid_weights)
    source_centroid = source.T @ centroid_weights / denominator
    target_centroid = target.T @ centroid_weights / denominator

    def translation(yaw):
        return target_centroid - module.rotation2(yaw) @ source_centroid

    def objective(yaw):
        residual = (
            source @ module.rotation2(yaw).T + translation(yaw) - target
        )
        return float(np.sum(residual * (information @ residual)))

    result = minimize_scalar(
        objective,
        bounds=(-np.pi, np.pi),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert result.success
    return float(result.x), translation(result.x), objective


def test_weighted_procrustes_matches_independent_dense_weighted_optimum():
    module = load_module()
    source = np.array(
        [
            [-2.0, 0.5],
            [0.3, -1.7],
            [2.8, 1.1],
            [1.2, 4.0],
            [-3.0, -2.0],
        ]
    )
    yaw = 0.71
    translation = np.array([3.4, -2.2])
    noise = np.array(
        [
            [0.4, -0.2],
            [-0.1, 0.3],
            [0.2, 0.5],
            [-0.5, -0.1],
            [1.2, -0.8],
        ]
    )
    target = source @ module.rotation2(yaw).T + translation + noise
    design = np.array(
        [
            [1.0, 0.2, -0.1, 0.3, 0.4],
            [0.4, 1.3, 0.2, -0.2, 0.1],
            [-0.3, 0.1, 1.1, 0.5, -0.2],
            [0.2, -0.4, 0.3, 1.2, 0.1],
            [0.5, 0.2, -0.3, 0.1, 0.9],
        ]
    )
    information = design @ design.T + 0.5 * np.eye(5)

    recovered_yaw, recovered_translation = module._weighted_procrustes(
        source, target, information
    )
    optimum_yaw, optimum_translation, weighted_objective = (
        _independent_weighted_yaw_optimum(
            module, source, target, information
        )
    )
    unweighted_yaw, _ = module._weighted_procrustes(
        source, target, np.eye(source.shape[0])
    )

    assert recovered_yaw == pytest.approx(optimum_yaw, abs=5e-9)
    np.testing.assert_allclose(
        recovered_translation, optimum_translation, atol=5e-9, rtol=0.0
    )
    assert weighted_objective(recovered_yaw) == pytest.approx(
        weighted_objective(optimum_yaw), abs=1e-12, rel=1e-12
    )
    assert abs(unweighted_yaw - recovered_yaw) > 0.02
    assert (
        weighted_objective(unweighted_yaw)
        - weighted_objective(recovered_yaw)
        > 0.05
    )


def test_weighted_procrustes_enforces_so2_on_reflection_prone_data():
    module = load_module()
    source = np.array(
        [
            [-3.0, 0.1],
            [-1.0, 2.0],
            [0.2, -2.0],
            [2.7, 0.8],
            [1.1, 4.2],
        ]
    )
    target = (
        source @ np.diag([-1.0, 1.0])
        + np.array([2.0, -3.0])
        + np.array(
            [
                [0.1, 0.2],
                [-0.2, 0.05],
                [0.05, -0.1],
                [0.2, 0.15],
                [-0.15, -0.05],
            ]
        )
    )
    design = np.array(
        [
            [1.0, 0.2, -0.1, 0.3, 0.4],
            [0.4, 1.3, 0.2, -0.2, 0.1],
            [-0.3, 0.1, 1.1, 0.5, -0.2],
            [0.2, -0.4, 0.3, 1.2, 0.1],
            [0.5, 0.2, -0.3, 0.1, 0.9],
        ]
    )
    information = design @ design.T + 0.5 * np.eye(5)

    recovered_yaw, recovered_translation = module._weighted_procrustes(
        source, target, information
    )
    optimum_yaw, optimum_translation, weighted_objective = (
        _independent_weighted_yaw_optimum(
            module, source, target, information
        )
    )
    ones = np.ones(source.shape[0])
    weights = information @ ones
    source_centroid = source.T @ weights / (ones @ weights)
    target_centroid = target.T @ weights / (ones @ weights)
    cross = (
        (target - target_centroid).T
        @ information
        @ (source - source_centroid)
    )
    left, _, right_transpose = np.linalg.svd(cross)
    uncorrected = left @ right_transpose

    assert np.linalg.det(uncorrected) < 0.0
    assert recovered_yaw == pytest.approx(optimum_yaw, abs=5e-8)
    np.testing.assert_allclose(
        recovered_translation, optimum_translation, atol=5e-8, rtol=0.0
    )
    assert weighted_objective(recovered_yaw) == pytest.approx(
        weighted_objective(optimum_yaw), abs=1e-11, rel=1e-12
    )


def test_incidence_start_is_bitwise_deterministic():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=1)

    first = module.incidence_profile_start(problem, layout)
    second = module.incidence_profile_start(problem, layout)

    assert np.array_equal(first, second)


def test_incidence_start_rejects_disconnected_transform_graph():
    module = load_module()
    first_copies = tuple(module.ObjectCopy(0, index, 10) for index in range(2))
    second_copies = tuple(module.ObjectCopy(1, index, 10) for index in range(2))
    states = (
        module.RobotCavityState(
            0,
            first_copies,
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.2], [0.2, 1.0]]),
        ),
        module.RobotCavityState(
            1,
            second_copies,
            np.array([[0.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0, -0.1], [-0.1, 1.0]]),
        ),
    )
    partition = module.NetworkPartition.build(
        (
            (first_copies[0], second_copies[0]),
            (first_copies[1],),
            (second_copies[1],),
        )
    )
    problem = module.NetworkQuotientProblem(states, partition)
    layout = module.VariableLayout.build(problem, root_robot=0)

    with pytest.raises(
        ValueError, match="^partition transform graph is unobservable$"
    ):
        module.incidence_profile_start(problem, layout)


def test_profile_vectors_are_read_only():
    module = load_module()
    result = module.score_network_partition(_three_robot_fixture(module))

    with pytest.raises(ValueError):
        result.profiles[0].vector[0] = 1.0


def _raw_profile_result(module, problem, root_robot):
    layout = module.VariableLayout.build(problem, root_robot)
    start = module.incidence_profile_start(problem, layout)
    tolerance = np.finfo(float).eps ** (2.0 / 3.0)

    def residual(vector):
        return module.profile_residual_and_jacobian(
            problem, layout, vector
        )[0]

    def jacobian(vector):
        return module.profile_residual_and_jacobian(
            problem, layout, vector
        )[1]

    result = module.least_squares(
        residual,
        start,
        jac=jacobian,
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
    )
    return layout, result


def test_reduced_root_profile_matches_dense_reference():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout, reduced = module.profile_partition_for_root(
        problem, root_robot=0
    )
    dense_layout, dense = _raw_profile_result(module, problem, root_robot=0)
    dense = module._exact_newton_polish(problem, dense_layout, dense)
    reduced_evaluation = module.evaluate_network_canonical(
        problem, layout, reduced.x
    )
    dense_evaluation = module.evaluate_network_canonical(
        problem, dense_layout, dense.x
    )

    np.testing.assert_allclose(
        reduced.cost, dense.cost, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        reduced_evaluation.residual_chi2,
        dense_evaluation.residual_chi2,
        rtol=0.0,
        atol=1e-10,
    )
    assert reduced.x.shape == (layout.dimension,)
    assert np.linalg.norm(
        reduced_evaluation.gradient[layout.component_offset :]
    ) < 2e-10


def test_reduced_optimizer_result_is_restored_to_full_profile_coordinates():
    module = load_module()
    problem = _three_robot_fixture(module)
    layout, result = module.profile_partition_for_root(
        problem, root_robot=1
    )
    residual, jacobian = module.profile_residual_and_jacobian(
        problem, layout, result.x
    )

    assert result.x.shape == (layout.dimension,)
    assert result.jac.shape == jacobian.shape
    np.testing.assert_allclose(result.fun, residual, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(result.jac, jacobian, rtol=0.0, atol=2e-12)


def test_reduced_score_matches_dense_score_and_logdet(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)
    reduced = module.score_network_partition(problem)

    def dense_profile(centered_problem, root_robot, current_transforms=None):
        del current_transforms
        layout, result = _raw_profile_result(
            module, centered_problem, root_robot
        )
        return layout, module._exact_newton_polish(
            centered_problem, layout, result
        )

    monkeypatch.setattr(
        module, "profile_partition_for_root", dense_profile
    )
    dense = module.score_network_partition(problem)

    assert reduced.valid and dense.valid
    np.testing.assert_allclose(
        reduced.profile_chi2, dense.profile_chi2, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        reduced.logdet, dense.logdet, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        reduced.score, dense.score, rtol=0.0, atol=1e-10
    )
    assert reduced.status == dense.status


def test_exact_newton_polish_records_one_accepted_step():
    module = load_module()
    problem = _three_robot_fixture(module, false_partition=True)
    layout, result = _raw_profile_result(module, problem, root_robot=0)
    old_value = module.evaluate_network_canonical(
        problem, layout, result.x
    ).value
    old_stationarity = module._stationarity_norm(
        module.evaluate_network_canonical(problem, layout, result.x), result.x
    )
    old_vector = result.x.copy()
    old_nfev = result.nfev
    old_njev = result.njev

    polished = module._exact_newton_polish(problem, layout, result)
    new_evaluation = module.evaluate_network_canonical(
        problem, layout, polished.x
    )

    assert polished.exact_newton_polished
    assert not np.array_equal(polished.x, old_vector)
    assert polished.nfev == old_nfev + 1
    assert polished.njev == old_njev + 1
    assert new_evaluation.value < old_value
    assert module._stationarity_norm(
        new_evaluation, polished.x
    ) < old_stationarity
    expected_fun, expected_jacobian = module.profile_residual_and_jacobian(
        problem, layout, polished.x
    )
    expected_gradient = expected_jacobian.T @ expected_fun
    np.testing.assert_array_equal(polished.fun, expected_fun)
    np.testing.assert_array_equal(polished.jac, expected_jacobian)
    assert polished.cost == 0.5 * float(expected_fun @ expected_fun)
    np.testing.assert_array_equal(polished.grad, expected_gradient)
    assert polished.optimality == np.linalg.norm(expected_gradient, ord=np.inf)


def test_exact_newton_polish_rejection_preserves_result(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module, false_partition=True)
    layout, result = _raw_profile_result(module, problem, root_robot=0)
    original_evaluate = module.evaluate_network_canonical
    calls = 0

    def worsen_candidate(*args):
        nonlocal calls
        evaluation = original_evaluate(*args)
        calls += 1
        if calls == 2:
            return module.CanonicalEvaluation(
                evaluation.value + 1.0,
                evaluation.gradient,
                evaluation.hessian,
                evaluation.residual_chi2,
            )
        return evaluation

    old_vector = result.x.copy()
    old_cost = result.cost
    old_nfev = result.nfev
    old_njev = result.njev
    monkeypatch.setattr(module, "evaluate_network_canonical", worsen_candidate)

    rejected = module._exact_newton_polish(problem, layout, result)

    assert not rejected.exact_newton_polished
    assert np.array_equal(rejected.x, old_vector)
    assert rejected.cost == old_cost
    assert rejected.nfev == old_nfev
    assert rejected.njev == old_njev


def test_score_passes_a_regauged_current_start_to_every_root(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)
    current_transforms = {
        0: (0.23, np.array([1.1, -0.8])),
        1: (0.63, np.array([7.0, -4.0])),
        2: (-0.91, np.array([-5.0, 8.0])),
    }
    captured = []
    original_profile = module.profile_partition_for_root

    def capture_profile(problem, root_robot, current_transforms=None):
        captured.append(
            (
                root_robot,
                {
                    robot: (yaw, translation.copy())
                    for robot, (yaw, translation) in current_transforms.items()
                },
            )
        )
        return original_profile(problem, root_robot, current_transforms)

    monkeypatch.setattr(
        module, "profile_partition_for_root", capture_profile
    )

    result = module.score_network_partition(
        problem, current_transforms=current_transforms
    )

    assert result.valid
    assert tuple(root for root, _ in captured) == (0, 1, 2)
    point = np.array([0.7, -1.3])
    centers = {
        state.robot: np.mean(state.means, axis=0) for state in problem.robots
    }
    for root_robot, regauged in captured:
        assert regauged[root_robot][0] == 0.0
        assert np.array_equal(regauged[root_robot][1], np.zeros(2))
        root_yaw, root_translation = current_transforms[root_robot]
        inverse_root_rotation = module.rotation2(root_yaw).T
        for robot, (yaw, translation) in current_transforms.items():
            expected = inverse_root_rotation @ (
                module.rotation2(yaw) @ point
                + translation
                - root_translation
            ) - centers[root_robot]
            chart_yaw, chart_translation = regauged[robot]
            actual = (
                module.rotation2(chart_yaw) @ (point - centers[robot])
                + chart_translation
            )
            np.testing.assert_allclose(
                actual, expected, atol=1e-12, rtol=0.0
            )


def _map_vector_to_centered_chart(module, problem, layout, vector):
    centers = {
        state.robot: np.mean(state.means, axis=0)
        for state in problem.robots
    }
    root_center = centers[layout.root_robot]
    mapped = vector.copy()
    mapping_jacobian = np.eye(layout.dimension)
    skew = np.array([[0.0, -1.0], [1.0, 0.0]])
    for state in problem.robots:
        if state.robot == layout.root_robot:
            continue
        offset = layout.transform_offset[state.robot]
        yaw = vector[offset]
        rotated_center = module.rotation2(yaw) @ centers[state.robot]
        mapped[offset + 1 : offset + 3] += rotated_center - root_center
        mapping_jacobian[offset + 1 : offset + 3, offset] = (
            skew @ rotated_center
        )
    for component_index in range(problem.partition.component_count):
        offset = layout.component_offset + 2 * component_index
        mapped[offset : offset + 2] -= root_center
    return mapped, mapping_jacobian


def test_centered_chart_preserves_model_with_unit_jacobian():
    module = load_module()
    problem = _three_robot_fixture(module)
    root_robot = 1
    layout, profile = module.profile_partition_for_root(problem, root_robot)
    centered, _ = module._center_profile_problem(problem, None)
    mapped, mapping_jacobian = _map_vector_to_centered_chart(
        module, problem, layout, profile.x
    )

    original_residual, _ = module.profile_residual_and_jacobian(
        problem, layout, profile.x
    )
    centered_residual, _ = module.profile_residual_and_jacobian(
        centered, layout, mapped
    )
    np.testing.assert_allclose(
        centered_residual, original_residual, atol=1e-10, rtol=0.0
    )
    assert np.linalg.det(mapping_jacobian) == pytest.approx(1.0, abs=1e-14)


def test_public_profiles_are_restored_to_the_original_problem_chart():
    module = load_module()
    problem = _three_robot_fixture(module)
    centered, _ = module._center_profile_problem(problem, None)
    score = module.score_network_partition(problem)

    assert score.valid
    for profile in score.profiles:
        layout = module.VariableLayout.build(problem, profile.root_robot)
        original = module.evaluate_network_canonical(
            problem, layout, profile.vector
        )
        centered_vector, _ = _map_vector_to_centered_chart(
            module, problem, layout, profile.vector
        )
        centered_evaluation = module.evaluate_network_canonical(
            centered, layout, centered_vector
        )
        assert original.value == pytest.approx(
            centered_evaluation.value, abs=1e-10
        )
        assert original.residual_chi2 == pytest.approx(
            centered_evaluation.residual_chi2, abs=1e-10
        )
        assert profile.objective == pytest.approx(
            0.5 * original.residual_chi2, abs=1e-10
        )


def test_centered_structural_determinant_preserves_synthetic_score():
    module = load_module()
    problem = _three_robot_fixture(module)
    centered_score = module.score_network_partition(problem)
    uncentered_scores = []
    for state in problem.robots:
        layout, profile = module.profile_partition_for_root(
            problem, state.robot
        )
        evaluation = module.evaluate_network_canonical(
            problem, layout, profile.x
        )
        logdet = module._equilibrated_spd_logdet(evaluation.hessian)
        uncentered_scores.append(
            evaluation.value
            + 0.5 * logdet
            - 0.5 * layout.dimension * np.log(2.0 * np.pi)
        )

    assert centered_score.valid
    assert centered_score.score == pytest.approx(
        np.mean(uncentered_scores), abs=1e-10
    )


def test_profile_ignores_unsuccessful_optimizer_results(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    starts = (np.zeros(layout.dimension), np.ones(layout.dimension))
    unsuccessful = module.OptimizeResult(
        success=False,
        cost=-100.0,
        x=starts[0][: layout.component_offset].copy(),
        nfev=1,
        njev=1,
    )
    successful = module.OptimizeResult(
        success=True,
        cost=5.0,
        x=starts[1][: layout.component_offset].copy(),
        nfev=1,
        njev=1,
    )
    results = iter((unsuccessful, successful))

    monkeypatch.setattr(
        module, "deterministic_profile_starts", lambda *_args: starts
    )
    monkeypatch.setattr(
        module, "least_squares", lambda *_args, **_kwargs: next(results)
    )
    monkeypatch.setattr(
        module, "_restore_full_profile_result", lambda *_args: _args[-1]
    )
    monkeypatch.setattr(
        module, "_exact_newton_polish", lambda *_args: _args[-1]
    )

    returned_layout, returned = module.profile_partition_for_root(
        problem, root_robot=0
    )

    assert returned_layout == layout
    assert returned is successful


def test_profile_equal_cost_tie_chooses_first_start(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)
    layout = module.VariableLayout.build(problem, root_robot=0)
    starts = (np.zeros(layout.dimension), np.ones(layout.dimension))
    first = module.OptimizeResult(
        success=True,
        cost=5.0,
        x=starts[0][: layout.component_offset].copy(),
        nfev=1,
        njev=1,
    )
    second = module.OptimizeResult(
        success=True,
        cost=5.0,
        x=starts[1][: layout.component_offset].copy(),
        nfev=1,
        njev=1,
    )
    results = iter((first, second))

    monkeypatch.setattr(
        module, "deterministic_profile_starts", lambda *_args: starts
    )
    monkeypatch.setattr(
        module, "least_squares", lambda *_args, **_kwargs: next(results)
    )
    monkeypatch.setattr(
        module, "_restore_full_profile_result", lambda *_args: _args[-1]
    )
    monkeypatch.setattr(
        module, "_exact_newton_polish", lambda *_args: _args[-1]
    )

    _, returned = module.profile_partition_for_root(problem, root_robot=0)

    assert returned is first


def _frozen_profile_packets(module):
    problem = _three_robot_fixture(module)
    layout, profile = module.profile_partition_for_root(problem, 0)
    centralized = module.evaluate_network_canonical(
        problem, layout, profile.x
    )
    packets = module.build_canonical_packets(problem, layout, profile.x)
    return problem, layout, profile, centralized, packets


def test_packet_diagnostics_use_the_public_original_profile_chart(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)
    score = module.score_network_partition(problem)
    captured_means = []
    original_builder = module.build_canonical_packets

    def capture_builder(packet_problem, layout, vector):
        captured_means.extend(state.means for state in packet_problem.robots)
        return original_builder(packet_problem, layout, vector)

    monkeypatch.setattr(module, "build_canonical_packets", capture_builder)
    relative_error, _, packets, _ = module._packet_diagnostics(
        problem, ((0, 1), (1, 2)), score
    )

    assert packets
    assert relative_error < 1e-12
    assert len(captured_means) == len(problem.robots)
    for captured, original in zip(captured_means, problem.robots):
        np.testing.assert_array_equal(captured, original.means)


def _packet_wire_bytes(variable_ids):
    dimension = len(set(variable_ids))
    return 8 * (4 + 2 * dimension + dimension * (dimension + 1) // 2)


def _manual_packet(
    module,
    robot,
    variable_ids,
    packed_upper_hessian,
    gradient,
    quadratic_value,
    normalizer,
    *,
    ambient_dimension=3,
    layout_signature=(17, 3, 0),
    serialized_bytes=None,
):
    if serialized_bytes is None:
        serialized_bytes = _packet_wire_bytes(variable_ids)
    return module.CanonicalPacket(
        robot=robot,
        variable_ids=variable_ids,
        packed_upper_hessian=packed_upper_hessian,
        gradient=gradient,
        quadratic_value=quadratic_value,
        normalizer=normalizer,
        serialized_bytes=serialized_bytes,
        ambient_dimension=ambient_dimension,
        layout_signature=layout_signature,
    )


def test_routed_packets_match_centralized_exact_model():
    module = load_module()
    _, _, _, centralized, packets = _frozen_profile_packets(module)

    reduced = module.reduce_packets_on_tree(
        packets, topology=((0, 1), (1, 2)), root_robot=1
    )

    np.testing.assert_allclose(
        reduced.hessian,
        centralized.hessian,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        reduced.gradient,
        centralized.gradient,
        atol=1e-10,
        rtol=1e-10,
    )
    assert reduced.value == pytest.approx(centralized.value, abs=1e-10)
    relative_matrix_error = np.linalg.norm(
        reduced.hessian - centralized.hessian, ord="fro"
    ) / np.linalg.norm(centralized.hessian, ord="fro")
    assert relative_matrix_error < 1e-10
    assert reduced.link_payload_bytes > sum(
        packet.serialized_bytes for packet in packets
    )


def test_packet_byte_count_matches_wire_schema():
    module = load_module()
    _, _, _, _, packets = _frozen_profile_packets(module)
    packet = packets[1]
    dimension = len(packet.variable_ids)

    expected = 8 * (
        4 + 2 * dimension + dimension * (dimension + 1) // 2
    )

    assert packet.robot == 1
    assert packet.serialized_bytes == expected
    assert packet.ambient_dimension > dimension
    assert packet.layout_signature


def test_structural_packet_support_preserves_exact_zero_yaw_variable():
    module = load_module()
    first_copy = module.ObjectCopy(0, 0, 10)
    second_copy = module.ObjectCopy(1, 0, 10)
    problem = module.NetworkQuotientProblem(
        (
            module.RobotCavityState(
                0, (first_copy,), np.zeros((1, 2)), np.eye(1)
            ),
            module.RobotCavityState(
                1, (second_copy,), np.zeros((1, 2)), np.eye(1)
            ),
        ),
        module.NetworkPartition.build(((first_copy, second_copy),)),
    )
    layout = module.VariableLayout.build(problem, root_robot=0)
    vector = np.zeros(layout.dimension)
    centralized = module.evaluate_network_canonical(problem, layout, vector)

    packets = module.build_canonical_packets(problem, layout, vector)
    reduced = module.reduce_packets_on_tree(packets, ((0, 1),), 0)

    assert layout.dimension == 5
    assert 0 in packets[1].variable_ids
    assert np.count_nonzero(centralized.hessian[0]) == 0
    assert reduced.hessian.shape == (5, 5)
    np.testing.assert_array_equal(reduced.hessian, centralized.hessian)
    np.testing.assert_array_equal(reduced.gradient, centralized.gradient)
    assert reduced.value == centralized.value


def test_packet_build_evaluates_each_robot_exactly_once(monkeypatch):
    module = load_module()
    problem = _three_robot_fixture(module)
    layout, profile = module.profile_partition_for_root(problem, 0)
    original = module.evaluate_local_canonical
    calls = []

    def count_calls(state, *args):
        calls.append(state.robot)
        return original(state, *args)

    monkeypatch.setattr(module, "evaluate_local_canonical", count_calls)

    packets = module.build_canonical_packets(problem, layout, profile.x)

    assert tuple(packet.robot for packet in packets) == (0, 1, 2)
    assert calls == [0, 1, 2]


def test_packet_reduction_does_not_call_canonical_evaluators(monkeypatch):
    module = load_module()
    _, _, _, _, packets = _frozen_profile_packets(module)

    def reject_evaluation(*_args, **_kwargs):
        raise AssertionError("packet reduction must not evaluate the model")

    monkeypatch.setattr(
        module, "evaluate_network_canonical", reject_evaluation
    )
    monkeypatch.setattr(module, "evaluate_local_canonical", reject_evaluation)

    reduced = module.reduce_packets_on_tree(
        packets, topology=((0, 1), (1, 2)), root_robot=1
    )

    assert reduced.hessian.shape == reduced.gradient.shape * 2


def test_tree_routing_root_changes_only_manual_link_byte_accounting():
    module = load_module()
    _, _, _, _, packets = _frozen_profile_packets(module)
    packet_by_robot = {packet.robot: packet for packet in packets}
    all_ids = set().union(
        *(packet.variable_ids for packet in packets)
    )
    final_bytes = _packet_wire_bytes(all_ids)
    union_01_bytes = _packet_wire_bytes(
        set(packet_by_robot[0].variable_ids)
        | set(packet_by_robot[1].variable_ids)
    )
    union_12_bytes = _packet_wire_bytes(
        set(packet_by_robot[1].variable_ids)
        | set(packet_by_robot[2].variable_ids)
    )
    expected_bytes = {
        0: (
            packet_by_robot[2].serialized_bytes
            + union_12_bytes
            + 2 * final_bytes
        ),
        1: (
            packet_by_robot[0].serialized_bytes
            + packet_by_robot[2].serialized_bytes
            + 2 * final_bytes
        ),
        2: (
            packet_by_robot[0].serialized_bytes
            + union_01_bytes
            + 2 * final_bytes
        ),
    }

    reduced = {
        root: module.reduce_packets_on_tree(
            packets, topology=((0, 1), (1, 2)), root_robot=root
        )
        for root in (0, 1, 2)
    }

    for root in (0, 1, 2):
        np.testing.assert_allclose(
            reduced[root].hessian,
            reduced[0].hessian,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            reduced[root].gradient,
            reduced[0].gradient,
            atol=1e-12,
            rtol=1e-12,
        )
        assert reduced[root].value == pytest.approx(
            reduced[0].value, abs=1e-12, rel=1e-12
        )
        assert reduced[root].link_payload_bytes == expected_bytes[root]


def test_packet_routing_account_trace_reconciles_exact_link_bytes():
    module = load_module()
    _, _, _, _, packets = _frozen_profile_packets(module)

    reduced, bytes_by_transmitter = module.reduce_packets_on_tree_with_trace(
        packets, ((0, 1), (1, 2)), 0
    )

    assert reduced.link_payload_bytes == 7624
    assert sum(bytes_by_transmitter.values()) == 7624
    assert set(bytes_by_transmitter) == {0, 1, 2}
    assert bytes_by_transmitter[0] > 0


def test_packet_combine_scatter_preserves_sorted_union_and_shared_entries():
    module = load_module()
    first = _manual_packet(
        module,
        robot=0,
        variable_ids=(0, 2),
        packed_upper_hessian=np.array([1.0, 0.5, 2.0]),
        gradient=np.array([1.0, 2.0]),
        quadratic_value=3.0,
        normalizer=4.0,
        serialized_bytes=_packet_wire_bytes((0, 2)),
    )
    second = _manual_packet(
        module,
        robot=1,
        variable_ids=(1, 2),
        packed_upper_hessian=np.array([3.0, -0.25, 4.0]),
        gradient=np.array([5.0, 6.0]),
        quadratic_value=7.0,
        normalizer=8.0,
        serialized_bytes=_packet_wire_bytes((1, 2)),
    )

    combined = module._combine_canonical_packets(first, second)

    assert combined.variable_ids == (0, 1, 2)
    np.testing.assert_array_equal(combined.gradient, [1.0, 5.0, 8.0])
    np.testing.assert_array_equal(
        module._unpack_upper_hessian(combined),
        np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 3.0, -0.25],
                [0.5, -0.25, 6.0],
            ]
        ),
    )
    assert combined.quadratic_value == 10.0
    assert combined.normalizer == 12.0
    assert combined.serialized_bytes == _packet_wire_bytes((0, 1, 2))


def test_packet_combine_preserves_destination_owner_not_minimum_robot():
    module = load_module()
    destination = _manual_packet(
        module, 7, (0,), [1.0], [2.0], 3.0, 4.0,
        ambient_dimension=1,
        layout_signature=(99, 1),
    )
    subtree_sender = _manual_packet(
        module, 2, (0,), [5.0], [6.0], 7.0, 8.0,
        ambient_dimension=1,
        layout_signature=(99, 1),
    )

    combined = module._combine_canonical_packets(
        destination, subtree_sender
    )

    assert combined.robot == 7


@pytest.mark.parametrize("changed_metadata", ("dimension", "signature"))
def test_packet_combine_rejects_incompatible_layout_metadata(changed_metadata):
    module = load_module()
    first = _manual_packet(
        module, 0, (0,), [1.0], [0.0], 0.0, 0.0,
        ambient_dimension=2,
        layout_signature=(10, 2),
    )
    second_dimension = 3 if changed_metadata == "dimension" else 2
    second_signature = (
        (11, 2) if changed_metadata == "signature" else (10, 2)
    )
    second = _manual_packet(
        module, 1, (0,), [1.0], [0.0], 0.0, 0.0,
        ambient_dimension=second_dimension,
        layout_signature=second_signature,
    )

    with pytest.raises(ValueError, match="layout metadata"):
        module._combine_canonical_packets(first, second)

    with pytest.raises(ValueError, match="layout metadata"):
        module.reduce_packets_on_tree((first, second), ((0, 1),), 0)


def test_packets_from_different_root_fixed_layouts_are_incompatible():
    module = load_module()
    problem = _two_robot_problem(module)
    first_layout = module.VariableLayout.build(problem, root_robot=0)
    second_layout = module.VariableLayout.build(problem, root_robot=1)
    first_packets = module.build_canonical_packets(
        problem, first_layout, np.zeros(first_layout.dimension)
    )
    second_packets = module.build_canonical_packets(
        problem, second_layout, np.zeros(second_layout.dimension)
    )

    assert first_packets[0].ambient_dimension == second_packets[1].ambient_dimension
    assert first_packets[0].layout_signature != second_packets[1].layout_signature
    with pytest.raises(ValueError, match="layout metadata"):
        module._combine_canonical_packets(
            first_packets[0], second_packets[1]
        )


def test_preshared_packet_layout_context_is_excluded_from_wire_bytes():
    module = load_module()
    signature = [4, 2, 8]
    packet = _manual_packet(
        module, 0, (1,), [1.0], [0.0], 0.0, 0.0,
        ambient_dimension=2,
        layout_signature=signature,
    )
    signature[0] = 99

    assert packet.layout_signature == (4, 2, 8)
    transmitted_numeric_payload_bytes = _packet_wire_bytes((1,))
    assert packet.serialized_bytes == transmitted_numeric_payload_bytes
    with pytest.raises(ValueError, match="ambient dimension"):
        _manual_packet(
            module, 0, (1,), [1.0], [0.0], 0.0, 0.0,
            ambient_dimension=True,
        )
    with pytest.raises(ValueError, match="ambient dimension"):
        _manual_packet(
            module, 0, (2,), [1.0], [0.0], 0.0, 0.0,
            ambient_dimension=2,
        )
    with pytest.raises(ValueError, match="layout signature"):
        _manual_packet(
            module, 0, (0,), [1.0], [0.0], 0.0, 0.0,
            ambient_dimension=1,
            layout_signature=(1, False),
        )


@pytest.mark.parametrize(
    "variable_ids",
    ((1, 0), (0, 0), (-1, 0)),
)
def test_canonical_packet_rejects_invalid_variable_ids(variable_ids):
    module = load_module()
    dimension = len(variable_ids)

    with pytest.raises(ValueError, match="variable IDs"):
        _manual_packet(
            module,
            0,
            variable_ids,
            np.zeros(dimension * (dimension + 1) // 2),
            np.zeros(dimension),
            0.0,
            0.0,
            serialized_bytes=_packet_wire_bytes(variable_ids),
        )


def test_canonical_packet_rejects_malformed_upper_triangle():
    module = load_module()

    with pytest.raises(ValueError, match="packed upper"):
        _manual_packet(
            module, 0, (0, 1), np.zeros(2), np.zeros(2), 0.0, 0.0,
            serialized_bytes=72,
        )


def test_packet_models_copy_freeze_and_validate_dimensions():
    module = load_module()
    upper = np.array([2.0])
    gradient = np.array([3.0])
    packet = _manual_packet(
        module, 0, (0,), upper, gradient, 4.0, 5.0,
        ambient_dimension=1,
    )
    upper[0] = 20.0
    gradient[0] = 30.0

    np.testing.assert_array_equal(packet.packed_upper_hessian, [2.0])
    np.testing.assert_array_equal(packet.gradient, [3.0])
    with pytest.raises(ValueError):
        packet.gradient[0] = 0.0
    with pytest.raises(ValueError, match="serialized bytes"):
        _manual_packet(
            module, 0, (0,), [2.0], [3.0], 4.0, 5.0,
            ambient_dimension=1,
            serialized_bytes=8,
        )
    with pytest.raises(ValueError, match="dimension"):
        module.ReducedCanonicalModel(
            np.eye(2), np.zeros(1), 0.0, 0
        )


def test_canonical_packet_checks_finite_scalars_individually():
    module = load_module()

    packet = _manual_packet(
        module, 0, (), np.empty(0), np.empty(0), 1e308, 1e308,
        ambient_dimension=1,
    )

    assert np.isfinite(packet.quadratic_value)
    assert np.isfinite(packet.normalizer)
    with pytest.raises(ValueError, match="finite"):
        _manual_packet(
            module, 0, (), np.empty(0), np.empty(0), np.inf, 0.0,
            ambient_dimension=1,
        )


@pytest.mark.parametrize(
    ("packets", "topology", "root", "message"),
    (
        ("duplicate", ((0, 1),), 0, "one packet per robot"),
        ("normal", ((0, 1),), 0, "exactly packet robots"),
        ("normal", ((0, 1), (1, 2), (2, 0)), 0, "tree"),
        ("normal", ((0, 0), (0, 1), (1, 2)), 0, "self-loop"),
        ("normal", ((0, 1), (1, 2)), 7, "root"),
    ),
)
def test_tree_reduction_rejects_invalid_packets_or_topology(
    packets, topology, root, message
):
    module = load_module()
    _, _, _, _, canonical_packets = _frozen_profile_packets(module)
    if packets == "duplicate":
        canonical_packets = (
            canonical_packets[0],
            canonical_packets[0],
        )

    with pytest.raises(ValueError, match=message):
        module.reduce_packets_on_tree(canonical_packets, topology, root)


def test_tree_reduction_rejects_empty_inferred_dimension():
    module = load_module()
    packet = _manual_packet(
        module, 0, (), np.empty(0), np.empty(0), 0.0, 0.0,
        ambient_dimension=1,
    )

    with pytest.raises(ValueError, match="empty variable union"):
        module.reduce_packets_on_tree((packet,), (), 0)


def test_min_consensus_matches_central_argmin_with_canonical_ties():
    module = load_module()
    proposals = {
        0: ((4.0, "move_c"),),
        1: ((2.0, "move_b"),),
        2: ((2.0, "move_a"),),
    }

    result = module.run_min_consensus(
        proposals, topology=((0, 1), (1, 2))
    )

    assert result.selected == (2.0, "move_a")
    assert dict(result.selected_by_robot) == {
        0: (2.0, "move_a"),
        1: (2.0, "move_a"),
        2: (2.0, "move_a"),
    }
    assert result.rounds == 2
    assert result.payload_bytes == 112
    assert result.payload_bytes > 0
    with pytest.raises(TypeError):
        result.selected_by_robot[0] = (0.0, "changed")


def test_min_consensus_computes_diameter_on_connected_nontree_graph():
    module = load_module()
    proposals = {
        0: ((9.0, "long_move"),),
        1: ((5.0, "middle"),),
        2: ((3.0, "winner"),),
        3: ((7.0, "tail"),),
    }
    topology = ((0, 1), (1, 2), (2, 0), (2, 3))

    result = module.run_min_consensus(proposals, topology)

    assert result.rounds == 2
    assert result.selected == (3.0, "winner")
    assert set(result.selected_by_robot.values()) == {(3.0, "winner")}
    assert result.payload_bytes == 228


def test_min_consensus_result_normalizes_and_snapshots_public_state():
    module = load_module()
    selected_by_robot = {
        np.int64(2): (np.float64(1.5), "move_a"),
        np.int64(7): (np.float64(1.5), "move_a"),
    }

    result = module.MinConsensusResult(
        (np.float64(1.5), "move_a"),
        selected_by_robot,
        np.int64(3),
        np.int64(41),
    )
    selected_by_robot[np.int64(2)] = (0.0, "changed")

    assert result.selected == (1.5, "move_a")
    assert dict(result.selected_by_robot) == {
        2: (1.5, "move_a"),
        7: (1.5, "move_a"),
    }
    assert type(result.rounds) is int
    assert type(result.payload_bytes) is int


@pytest.mark.parametrize(
    ("selected", "selected_by_robot", "rounds", "payload_bytes", "message"),
    (
        ([1.0, "move"], {0: (1.0, "move")}, 0, 0, "selected"),
        ((True, "move"), {0: (1.0, "move")}, 0, 0, "score"),
        ((np.nan, "move"), {0: (np.nan, "move")}, 0, 0, "finite"),
        (
            (1.0, "\ud800"),
            {0: (1.0, "\ud800")},
            0,
            0,
            "UTF-8",
        ),
        ((1.0, "move"), {-1: (1.0, "move")}, 0, 0, "robot IDs"),
        ((1.0, "move"), {True: (1.0, "move")}, 0, 0, "robot IDs"),
        ((1.0, "move"), {0: (2.0, "other")}, 0, 0, "agree"),
        ((1.0, "move"), {}, 0, 0, "nonempty"),
        ((1.0, "move"), {0: (1.0, "move")}, -1, 0, "rounds"),
        ((1.0, "move"), {0: (1.0, "move")}, True, 0, "rounds"),
        ((1.0, "move"), {0: (1.0, "move")}, 0, -1, "payload"),
        ((1.0, "move"), {0: (1.0, "move")}, 0, False, "payload"),
    ),
)
def test_min_consensus_result_rejects_malformed_public_construction(
    selected, selected_by_robot, rounds, payload_bytes, message
):
    module = load_module()

    with pytest.raises(ValueError, match=message):
        module.MinConsensusResult(
            selected, selected_by_robot, rounds, payload_bytes
        )


@pytest.mark.parametrize(
    ("proposals", "topology", "message"),
    (
        ({0: ((1.0, "a"),), 1: ((2.0, "b"),)}, (), "connected"),
        ({0: (), 1: ((2.0, "b"),)}, ((0, 1),), "proposal"),
        ({0: ((np.nan, "a"),), 1: ((2.0, "b"),)}, ((0, 1),), "finite"),
        ({0: ((True, "a"),)}, (), "score"),
        ({0: ((np.array([1.0]), "a"),)}, (), "scalar"),
        ({0: ((1.0, "\ud800"),)}, (), "UTF-8"),
        ({0: ((1.0, "a"),)}, ((0, 1),), "proposal robots"),
    ),
)
def test_min_consensus_rejects_invalid_inputs(proposals, topology, message):
    module = load_module()

    with pytest.raises(ValueError, match=message):
        module.run_min_consensus(proposals, topology)


def test_cut_join_reassigns_identity_without_changing_cardinality():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(1, 1, 10)
    d = module.ObjectCopy(2, 0, 10)
    incumbent = module.partition_from_edge_components(
        ((a, b), (a, d), (c,))
    )

    move = module.make_cut_join_move(
        incumbent=incumbent,
        incumbent_edges=((a, b), (a, d)),
        remove_edge=(a, d),
        add_edge=(c, d),
    )

    assert move.valid is True
    assert move.status == "ok"
    assert move.removed_edge == (a, d)
    assert move.added_edge == (c, d)
    assert move.candidate is not None
    assert move.candidate.component_count == incumbent.component_count
    assert (c, d) in move.candidate.components


def test_canonical_edge_and_partition_recovery_are_representation_independent():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(2, 0, 10)
    singleton = module.ObjectCopy(3, 0, 10)

    assert module.canonical_edge(c, a) == (a, c)
    forward = module.partition_from_edges(
        (singleton, c, b, a), ((b, c), (a, b))
    )
    reverse = module.partition_from_edges(
        (a, b, c, singleton), ((c, b), (b, a))
    )

    assert forward == reverse
    assert forward.components == ((a, b, c), (singleton,))


@pytest.mark.parametrize(
    ("edge", "message"),
    (
        (("not-a-copy", "also-not-a-copy"), "ObjectCopy"),
        ("self", "distinct"),
        ("same_robot", "different robots"),
        ("semantic", "semantic"),
    ),
)
def test_canonical_edge_rejects_structural_conflicts(edge, message):
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    if edge == "self":
        endpoints = (a, a)
    elif edge == "same_robot":
        endpoints = (a, module.ObjectCopy(0, 1, 10))
    elif edge == "semantic":
        endpoints = (a, module.ObjectCopy(1, 0, 71))
    else:
        endpoints = edge

    with pytest.raises(ValueError, match=message):
        module.canonical_edge(*endpoints)


def test_partition_from_edges_validates_exact_support_and_edge_shape():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    outside = module.ObjectCopy(2, 0, 10)

    with pytest.raises(ValueError, match="unique"):
        module.partition_from_edges((a, a, b), ((a, b),))
    with pytest.raises(ValueError, match="support"):
        module.partition_from_edges((a, b), ((a, outside),))
    with pytest.raises(ValueError, match="two"):
        module.partition_from_edges((a, b), ((a,),))
    with pytest.raises(ValueError, match="ObjectCopy"):
        module.partition_from_edges((a, "not-a-copy"), ())


def test_partition_from_edge_components_preserves_singletons_and_validates_shape():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(2, 0, 10)

    partition = module.partition_from_edge_components(((b, a), (c,)))

    assert partition.components == ((a, b), (c,))
    with pytest.raises(ValueError, match="one or two"):
        module.partition_from_edge_components(((a, b, c),))


@pytest.mark.parametrize(
    ("remove_edge", "add_edge", "status"),
    (
        ("missing", "valid", "missing_removed_edge"),
        ("valid", "existing", "existing_added_edge"),
        ("valid", "removed", "added_edge_matches_removed_edge"),
        ("valid", "collision", "invalid_candidate"),
    ),
)
def test_cut_join_returns_deterministic_invalid_statuses(
    remove_edge, add_edge, status
):
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(1, 1, 10)
    d = module.ObjectCopy(2, 0, 10)
    incumbent_edges = ((a, b), (a, d))
    incumbent = module.partition_from_edge_components(
        incumbent_edges + ((c,),)
    )
    removes = {
        "missing": (b, d),
        "valid": (a, d),
    }
    additions = {
        "valid": (c, d),
        "existing": (a, b),
        "removed": (a, d),
        "collision": (a, c),
    }

    move = module.make_cut_join_move(
        incumbent,
        incumbent_edges,
        removes[remove_edge],
        additions[add_edge],
    )

    assert move.valid is False
    assert move.status == status
    assert move.candidate is None


def test_cut_join_public_result_is_immutable_and_validated():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    edge = (a, b)

    result = module.CutJoinMove(False, "missing", edge, edge, None)

    with pytest.raises(FrozenInstanceError):
        result.status = "changed"
    with pytest.raises(ValueError, match="candidate"):
        module.CutJoinMove(True, "ok", edge, edge, None)
    with pytest.raises(ValueError, match="status"):
        module.CutJoinMove(False, "ok", edge, edge, None)
    candidate = module.NetworkPartition.build(((a, b),))
    with pytest.raises(ValueError, match="different edges"):
        module.CutJoinMove(True, "ok", edge, edge, candidate)

    c = module.ObjectCopy(2, 0, 10)
    d = module.ObjectCopy(3, 0, 10)
    unchanged = module.NetworkPartition.build(((a, b, c), (d,)))
    with pytest.raises(ValueError, match="removed edge"):
        module.CutJoinMove(True, "ok", (a, b), (b, c), unchanged)
    disconnected_addition = module.NetworkPartition.build(((a, b), (c, d)))
    with pytest.raises(ValueError, match="added edge"):
        module.CutJoinMove(
            True,
            "ok",
            (a, c),
            (b, d),
            disconnected_addition,
        )


def test_cut_join_rejects_redundant_noncanonical_incumbent_edges():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(2, 0, 10)
    d = module.ObjectCopy(3, 0, 10)
    incumbent = module.NetworkPartition.build(((a, b, c), (d,)))

    move = module.make_cut_join_move(
        incumbent,
        ((a, b), (a, c), (b, c)),
        (a, c),
        (c, d),
    )

    assert move.valid is False
    assert move.status == "incumbent_edges_not_canonical_forest"
    assert move.candidate is None

    malformed = module.make_cut_join_move(
        incumbent,
        ((a, b), (a,)),
        (a, b),
        (b, d),
    )
    assert malformed.valid is False
    assert malformed.status == "incumbent_edges_not_canonical_forest"
    assert malformed.candidate is None


def test_cut_join_rejects_remove_add_that_leaves_partition_unchanged():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(2, 0, 10)
    d = module.ObjectCopy(3, 0, 10)
    incumbent = module.NetworkPartition.build(((a, b, c), (d,)))

    move = module.make_cut_join_move(
        incumbent,
        ((a, b), (a, c)),
        (a, b),
        (b, c),
    )

    assert move.valid is False
    assert move.status == "unchanged_candidate"
    assert move.candidate is None


def _partition_variant(module, shift_robot_1, shift_robot_2):
    base = _three_robot_fixture(module)
    by_robot = {state.robot: state.copies for state in base.robots}
    components = tuple(
        (
            by_robot[0][index],
            by_robot[1][(index + shift_robot_1) % 7],
            by_robot[2][(index + shift_robot_2) % 7],
        )
        for index in range(7)
    )
    partition = module.NetworkPartition.build(components)
    return module.NetworkQuotientProblem(base.robots, partition)


def _four_partition_variants(module):
    return {
        problem.partition: problem
        for problem in (
            _partition_variant(module, 0, 0),
            _partition_variant(module, 1, 0),
            _partition_variant(module, 0, 1),
            _partition_variant(module, 1, 1),
        )
    }


def test_exhaustive_oracle_finds_network_optimum():
    module = load_module()
    variants = _four_partition_variants(module)
    incumbent = _partition_variant(module, 1, 1).partition

    result = module.exhaustive_partition_oracle(
        variants, incumbent=incumbent
    )

    assert result.best_partition == _partition_variant(module, 0, 0).partition
    assert result.best_score < result.second_score
    assert result.local_descent_regret >= 0.0
    assert result.local_descent_score == pytest.approx(
        result.best_score + result.local_descent_regret,
        abs=1e-10,
    )
    assert tuple(score.score for _, score in result.ranked) == tuple(
        sorted(score.score for _, score in result.ranked)
    )
    with pytest.raises(FrozenInstanceError):
        result.best_score = 0.0


def test_exhaustive_oracle_requires_two_valid_candidates(monkeypatch):
    module = load_module()
    only = _partition_variant(module, 0, 0)

    with pytest.raises(ValueError, match="two valid partitions"):
        module.exhaustive_partition_oracle(
            {only.partition: only}, incumbent=only.partition
        )


def test_exhaustive_oracle_discards_invalid_scores(monkeypatch):
    module = load_module()
    variants = _four_partition_variants(module)
    invalid_partition = _partition_variant(module, 1, 1).partition
    original = module.score_network_partition

    def score(problem):
        if problem.partition == invalid_partition:
            return module._invalid_quotient_score("forced_invalid")
        return original(problem)

    monkeypatch.setattr(module, "score_network_partition", score)
    result = module.exhaustive_partition_oracle(
        variants, incumbent=_partition_variant(module, 1, 0).partition
    )

    assert invalid_partition not in {partition for partition, _ in result.ranked}
    assert all(item.valid for _, item in result.ranked)


def test_exhaustive_oracle_rejects_valid_nonfinite_score(monkeypatch):
    module = load_module()
    variants = _four_partition_variants(module)
    poisoned = _partition_variant(module, 1, 1).partition
    original = module.score_network_partition

    def score(problem):
        if problem.partition == poisoned:
            return _synthetic_valid_score(module, np.nan)
        return original(problem)

    monkeypatch.setattr(module, "score_network_partition", score)
    with pytest.raises(ValueError, match="valid quotient score must be finite"):
        module.exhaustive_partition_oracle(
            variants, incumbent=_partition_variant(module, 1, 0).partition
        )


@pytest.mark.parametrize("changed_field", ("means", "covariance"))
def test_exhaustive_oracle_requires_identical_robot_cavity_data(changed_field):
    module = load_module()
    variants = _four_partition_variants(module)
    changed_partition = _partition_variant(module, 1, 0).partition
    original = variants[changed_partition]
    states = list(original.robots)
    state = states[1]
    means = state.means.copy()
    covariance = state.covariance.copy()
    if changed_field == "means":
        means[0, 0] += 1e-12
    else:
        covariance[0, 0] += 1e-12
    states[1] = module.RobotCavityState(
        state.robot, state.copies, means, covariance
    )
    variants[changed_partition] = module.NetworkQuotientProblem(
        tuple(states), changed_partition
    )

    with pytest.raises(ValueError, match="common robot cavity"):
        module.exhaustive_partition_oracle(
            variants, incumbent=_partition_variant(module, 0, 0).partition
        )


class _SnapshotOnlyMapping(Mapping):
    def __init__(self, values):
        self._values = dict(values)
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        if self.items_calls != 1:
            raise AssertionError("mapping was read more than once")
        return tuple(self._values.items())

    def __getitem__(self, key):
        raise AssertionError("mapping was re-read after snapshot")

    def __iter__(self):
        raise AssertionError("mapping was re-read after snapshot")

    def __len__(self):
        return len(self._values)


def test_exhaustive_oracle_snapshots_mapping_once():
    module = load_module()
    variants = _SnapshotOnlyMapping(_four_partition_variants(module))

    result = module.exhaustive_partition_oracle(
        variants, incumbent=_partition_variant(module, 1, 1).partition
    )

    assert result.best_partition == _partition_variant(module, 0, 0).partition
    assert variants.items_calls == 1


def test_exhaustive_oracle_result_is_immutable_and_checks_regret():
    module = load_module()
    problem = _partition_variant(module, 0, 0)
    score = module.score_network_partition(problem)

    with pytest.raises(ValueError, match="regret"):
        module.ExhaustiveOracleResult(
            problem.partition,
            score.score,
            score.score + 1.0,
            ((problem.partition, score),),
            problem.partition,
            score.score - 1.0,
            -1.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("best_score", True),
        ("second_score", np.array([2.0])),
        ("local_descent_score", False),
        ("local_descent_regret", np.array([0.0])),
    ),
)
def test_exhaustive_oracle_result_rejects_bool_and_nonscalar_scores(
    field, value
):
    module = load_module()
    variants = _four_partition_variants(module)
    result = module.exhaustive_partition_oracle(
        variants, incumbent=_partition_variant(module, 1, 1).partition
    )
    arguments = {
        "best_partition": result.best_partition,
        "best_score": result.best_score,
        "second_score": result.second_score,
        "ranked": result.ranked,
        "local_descent_partition": result.local_descent_partition,
        "local_descent_score": result.local_descent_score,
        "local_descent_regret": result.local_descent_regret,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match="scalar non-bool"):
        module.ExhaustiveOracleResult(**arguments)


def test_exhaustive_oracle_result_rejects_cross_support_ranking():
    module = load_module()
    best = module.NetworkPartition.build(
        ((module.ObjectCopy(0, 0, 10), module.ObjectCopy(1, 0, 10)),)
    )
    cross = module.NetworkPartition.build(
        ((module.ObjectCopy(0, 0, 10), module.ObjectCopy(2, 0, 10)),)
    )
    best_score = _synthetic_valid_score(module, 1.0)
    cross_score = _synthetic_valid_score(module, 2.0)

    with pytest.raises(ValueError, match="fixed cardinality"):
        module.ExhaustiveOracleResult(
            best,
            1.0,
            2.0,
            ((best, best_score), (cross, cross_score)),
            best,
            1.0,
            0.0,
        )


def test_exhaustive_oracle_result_rejects_malformed_ranked_item():
    module = load_module()
    result = module.exhaustive_partition_oracle(
        _four_partition_variants(module),
        incumbent=_partition_variant(module, 1, 1).partition,
    )

    with pytest.raises(ValueError, match="partition-score pairs"):
        module.ExhaustiveOracleResult(
            result.best_partition,
            result.best_score,
            result.second_score,
            ((result.best_partition,),) + result.ranked[1:],
            result.local_descent_partition,
            result.local_descent_score,
            result.local_descent_regret,
        )


def _synthetic_valid_score(module, value):
    return module.QuotientScore(
        True,
        "ok",
        float(value),
        0.0,
        0.0,
        (),
        0.0,
        0.0,
        (),
    )


def _synthetic_problem_for_partition(module, partition):
    states = []
    for robot in sorted({copy.robot for copy in partition.copies}):
        copies = tuple(copy for copy in partition.copies if copy.robot == robot)
        means = np.array(
            [[float(copy.object_id), float(robot)] for copy in copies]
        )
        states.append(
            module.RobotCavityState(
                robot,
                copies,
                means,
                np.eye(len(copies)),
            )
        )
    return module.NetworkQuotientProblem(tuple(states), partition)


def test_canonical_forest_uses_lexicographically_smallest_star_center():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(2, 0, 10)
    singleton = module.ObjectCopy(3, 0, 10)
    partition = module.NetworkPartition.build(((c, a, b), (singleton,)))

    assert module.canonical_partition_forest(partition) == ((a, b), (a, c))


def test_local_cut_join_descent_uses_strict_score_and_canonical_ties():
    module = load_module()
    a = module.ObjectCopy(0, 0, 10)
    b = module.ObjectCopy(1, 0, 10)
    c = module.ObjectCopy(2, 0, 10)
    d = module.ObjectCopy(3, 0, 10)
    incumbent = module.NetworkPartition.build(((a, b, c), (d,)))
    first_tied_neighbor = module.NetworkPartition.build(((a, b), (c, d)))
    second_tied_neighbor = module.NetworkPartition.build(((a, c), (b, d)))
    scores = (
        (first_tied_neighbor, _synthetic_valid_score(module, 2.0)),
        (second_tied_neighbor, _synthetic_valid_score(module, 2.0)),
        (incumbent, _synthetic_valid_score(module, 3.0)),
    )

    selected, selected_score = module.local_cut_join_descent(
        {
            incumbent: _synthetic_problem_for_partition(module, incumbent),
            first_tied_neighbor: _synthetic_problem_for_partition(
                module, first_tied_neighbor
            ),
            second_tied_neighbor: _synthetic_problem_for_partition(
                module, second_tied_neighbor
            ),
        },
        incumbent,
        scores,
    )

    assert selected == first_tied_neighbor
    assert selected_score == 2.0


@pytest.mark.parametrize(
    ("ranked_builder", "message"),
    (
        (lambda module, partition, score: ((partition,),), "ranked"),
        (
            lambda module, partition, score: (
                (partition, module._invalid_quotient_score("forced")),
            ),
            "valid",
        ),
        (
            lambda module, partition, score: (
                (partition, _synthetic_valid_score(module, np.nan)),
            ),
            "finite",
        ),
        (
            lambda module, partition, score: (
                (partition, score),
                (partition, score),
            ),
            "unique",
        ),
    ),
)
def test_local_cut_join_descent_rejects_malformed_ranked_input(
    ranked_builder, message
):
    module = load_module()
    incumbent = _partition_variant(module, 0, 0).partition
    problem = _partition_variant(module, 0, 0)
    score = _synthetic_valid_score(module, 1.0)

    with pytest.raises(ValueError, match=message):
        module.local_cut_join_descent(
            {incumbent: problem},
            incumbent,
            ranked_builder(module, incumbent, score),
        )


def test_local_cut_join_descent_rejects_absent_and_cross_support_candidates():
    module = load_module()
    incumbent_problem = _partition_variant(module, 0, 0)
    absent_problem = _partition_variant(module, 1, 0)
    incumbent_score = _synthetic_valid_score(module, 1.0)
    absent_score = _synthetic_valid_score(module, 2.0)

    with pytest.raises(ValueError, match="supplied"):
        module.local_cut_join_descent(
            {incumbent_problem.partition: incumbent_problem},
            incumbent_problem.partition,
            (
                (incumbent_problem.partition, incumbent_score),
                (absent_problem.partition, absent_score),
            ),
        )

    extra = module.ObjectCopy(3, 99, 10)
    cross_partition = module.NetworkPartition.build(
        incumbent_problem.partition.components + ((extra,),)
    )
    cross_problem = _synthetic_problem_for_partition(module, cross_partition)
    with pytest.raises(ValueError, match="fixed cardinality"):
        module.local_cut_join_descent(
            {
                incumbent_problem.partition: incumbent_problem,
                cross_partition: cross_problem,
            },
            incumbent_problem.partition,
            (
                (incumbent_problem.partition, incumbent_score),
                (cross_partition, absent_score),
            ),
        )


def test_local_cut_join_descent_validates_problem_mapping_values_and_keys():
    module = load_module()
    problem = _partition_variant(module, 0, 0)
    other = _partition_variant(module, 1, 0)
    score = _synthetic_valid_score(module, 1.0)
    ranked = ((problem.partition, score),)

    with pytest.raises(ValueError, match="NetworkQuotientProblem"):
        module.local_cut_join_descent(
            {problem.partition: object()}, problem.partition, ranked
        )
    with pytest.raises(ValueError, match="mapping key"):
        module.local_cut_join_descent(
            {problem.partition: other}, problem.partition, ranked
        )


def test_local_cut_join_descent_requires_identical_robot_cavity_data():
    module = load_module()
    incumbent = _partition_variant(module, 0, 0)
    neighbor = _partition_variant(module, 1, 0)
    changed_states = list(neighbor.robots)
    changed = changed_states[1]
    changed_means = changed.means.copy()
    changed_means[0, 0] += 1e-12
    changed_states[1] = module.RobotCavityState(
        changed.robot,
        changed.copies,
        changed_means,
        changed.covariance,
    )
    changed_neighbor = module.NetworkQuotientProblem(
        tuple(changed_states), neighbor.partition
    )

    with pytest.raises(ValueError, match="common robot cavity"):
        module.local_cut_join_descent(
            {
                incumbent.partition: incumbent,
                neighbor.partition: changed_neighbor,
            },
            incumbent.partition,
            (
                (incumbent.partition, _synthetic_valid_score(module, 2.0)),
                (neighbor.partition, _synthetic_valid_score(module, 1.0)),
            ),
        )


def test_local_cut_join_descent_snapshots_problem_mapping_once():
    module = load_module()
    problem = _partition_variant(module, 0, 0)
    problems = _SnapshotOnlyMapping({problem.partition: problem})

    selected, score = module.local_cut_join_descent(
        problems,
        problem.partition,
        ((problem.partition, _synthetic_valid_score(module, 1.0)),),
    )

    assert selected == problem.partition
    assert score == 1.0
    assert problems.items_calls == 1


def test_network_score_has_nonadditive_three_robot_interaction():
    module = load_module()
    scores = {
        (shift_1, shift_2): module.score_network_partition(
            _partition_variant(module, shift_1, shift_2)
        ).score
        for shift_1 in (0, 1)
        for shift_2 in (0, 1)
    }
    mixed_difference = (
        scores[(1, 1)]
        - scores[(1, 0)]
        - scores[(0, 1)]
        + scores[(0, 0)]
    )

    # Independent 0-1 and 0-2 pair scores have zero mixed difference.
    assert abs(mixed_difference) > 1.0


def _write_network_quotient_cli_fixture(tmp_path):
    cavity = tmp_path / "cavity"
    cavity.mkdir()
    metadata = tmp_path / "local_object_metadata.csv"
    association = tmp_path / "final_association_weights.csv"
    topology = tmp_path / "topology.csv"
    manifest = tmp_path / "candidates.json"

    base = np.array([[0.0, 0.0], [2.0, 0.3], [0.4, 1.8]])
    gauges = ((0.2, (3.0, -2.0)), (-0.7, (-1.0, 4.0)), (1.1, (5.0, 2.0)))
    with (cavity / "final_pair_cavity_means.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("robot", "valid", "status", "object_id", "x", "y", "z"))
        for robot, (yaw, translation) in enumerate(gauges):
            local = (base - np.asarray(translation)) @ load_module().rotation2(yaw)
            for object_id, (x, y) in enumerate(local):
                writer.writerow((robot, 1, "ok", object_id, x, y, 0.0))
    covariance = np.array(
        [[0.12, 0.018, -0.009], [0.018, 0.09, 0.012], [-0.009, 0.012, 0.11]]
    )
    with (cavity / "final_pair_cavity_covariance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("robot", "valid", "status", "row_object", "column_object", "covariance"))
        for robot in range(3):
            for row in range(3):
                for column in range(3):
                    writer.writerow((robot, 1, "ok", row, column, covariance[row, column]))
    with metadata.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("robot_id", "local_slot_id", "true_object_id", "category", "ignored"))
        for robot in range(3):
            for object_id in range(3):
                writer.writerow((robot, object_id, 900 + object_id, 10, "not-read"))
    with association.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("receiver_robot", "sender_robot", "receiver_object", "sender_object", "weight"))
        for object_id in range(3):
            writer.writerow((0, 1, object_id, object_id, 0.9))
            writer.writerow((1, 0, object_id, object_id, 0.9))
            writer.writerow((1, 2, object_id, object_id, 0.9))
    with topology.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("first_robot", "second_robot"))
        writer.writerows(((0, 1), (1, 2)))
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {"name": "incumbent", "group": "fixed", "remove_edges": [], "add_edges": []},
                    {
                        "name": "two_edge_swap",
                        "group": "fixed",
                        "remove_edges": [[[1, 0], [2, 0]], [[1, 1], [2, 1]]],
                        "add_edges": [[[1, 0], [2, 1]], [[1, 1], [2, 0]]],
                    },
                ]
            },
            sort_keys=True,
        )
    )
    return cavity, metadata, association, topology, manifest


def test_scorer_cli_writes_frozen_schema_without_ground_truth_inputs(tmp_path):
    module = load_module()
    cavity, metadata, association, topology, manifest = (
        _write_network_quotient_cli_fixture(tmp_path)
    )
    output = tmp_path / "output"

    assert module.run_cli(
        [
            "--cavity-dir", str(cavity), "--metadata", str(metadata),
            "--association", str(association), "--candidate-manifest", str(manifest),
            "--topology", str(topology), "--output-dir", str(output),
            "--plane-axes", "xy",
        ]
    ) == 0

    candidate_path = output / "network_quotient_candidates.csv"
    packet_path = output / "network_quotient_packets.csv"
    summary_path = output / "network_quotient_summary.json"
    assert candidate_path.exists() and packet_path.exists() and summary_path.exists()
    with candidate_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == module.CANDIDATE_CSV_COLUMNS
    with packet_path.open(newline="") as handle:
        packet_rows = list(csv.DictReader(handle))
        assert tuple(packet_rows[0]) == module.PACKET_CSV_COLUMNS
        assert len(packet_rows) == 2 * 3
    summary = json.loads(summary_path.read_text())
    assert summary["ground_truth_inputs"] == 0
    assert summary["cardinality_change_candidates_scored"] == 0
    assert summary["configured_anchors"] == 0
    assert summary["configured_null_constants"] == 0
    assert summary["configured_damping_parameters"] == 0
    assert summary["score_ablation"] == "full"
    assert summary["plane_axes"] == "xy"
    assert summary["current_lower_start_used"] is False
    assert summary["candidate_count"] == 2
    assert summary["observed_copies"] == [
        [robot, object_id, 10]
        for robot in range(3)
        for object_id in range(3)
    ]
    assert summary["incumbent_edges"] == [
        [[0, object_id], [1, object_id]]
        for object_id in range(3)
    ] + [
        [[1, object_id], [2, object_id]]
        for object_id in range(3)
    ]
    assert summary["incumbent_components"] == [
        [[0, object_id], [1, object_id], [2, object_id]]
        for object_id in range(3)
    ]
    assert summary["incumbent_component_count"] == 3
    assert summary["topology_edges"] == [[0, 1], [1, 2]]
    assert set(summary["candidate_root_scores"]) == {
        "incumbent", "two_edge_swap"
    }
    assert summary["network_selection"]["candidate"] == "incumbent"
    assert summary["network_selection"]["canonical_move_id"] == (
        module.canonical_move_id([], [])
    )
    assert set(summary["network_selection"]["selected_by_robot"]) == {
        "0", "1", "2"
    }
    assert summary["network_selection"]["rounds"] == 2
    assert summary["group_selections"]["fixed"]["candidate"] == "incumbent"
    assert summary["group_selections"]["fixed"]["canonical_move_id"] == (
        module.canonical_move_id([], [])
    )
    for selection in (
        summary["network_selection"],
        summary["group_selections"]["fixed"],
    ):
        assert isinstance(selection["rounds"], int)
        assert selection["rounds"] >= 0
        assert isinstance(selection["payload_bytes"], int)
        assert selection["payload_bytes"] >= 0
        expected_payload = (
            2
            * 2
            * 2
            * (8 + len(selection["canonical_move_id"].encode("utf-8")))
        )
        assert selection["rounds"] == 2
        assert selection["payload_bytes"] == expected_payload
        assert all(
            value == [selection["score"], selection["canonical_move_id"]]
            for value in selection["selected_by_robot"].values()
        )
    assert all(
        float(row["packet_matrix_relative_error"]) < 1e-10
        for row in rows
        if row["valid"] == "1"
    )
    packet_bytes = {
        row["candidate"]: int(row["packet_link_payload_bytes"])
        for row in rows
    }
    for candidate in packet_bytes:
        assert sum(
            int(row["routed_link_bytes"])
            for row in packet_rows
            if row["candidate"] == candidate
        ) == packet_bytes[candidate]


def test_frozen_incumbent_snapshot_is_gauge_and_ablation_independent(tmp_path):
    module = load_module()
    cavity, metadata, association, topology, manifest_path = (
        _write_network_quotient_cli_fixture(tmp_path)
    )
    inputs = module.read_scorer_inputs(
        cavity, metadata, association, topology, "xy"
    )
    manifest = module.read_candidate_manifest(manifest_path)
    snapshots = []
    for name, seed, ablation in (
        ("full", None, "full"),
        ("gauge", 7, "full"),
        ("diagonal", None, "diagonal-covariance"),
    ):
        summary = module.score_candidate_manifest(
            inputs,
            manifest,
            tmp_path / name,
            plane_axes="xy",
            independent_gauge_seed=seed,
            score_ablation=ablation,
        )
        snapshots.append(
            {
                field: summary[field]
                for field in (
                    "observed_copies",
                    "incumbent_edges",
                    "incumbent_components",
                    "incumbent_component_count",
                    "topology_edges",
                )
            }
        )

    assert snapshots[1:] == snapshots[:1] * 2


def test_consensus_selected_move_uses_canonical_tie_breaking():
    module = load_module()
    candidates = (
        {
            "candidate": "move_z", "group": "g", "valid": 1,
            "score": 2.0, "remove_edges": [], "add_edges": [],
        },
        {
            "candidate": "move_a", "group": "g", "valid": 1,
            "score": 2.0, "remove_edges": [], "add_edges": [],
        },
        {
            "candidate": "worse", "group": "h", "valid": 1,
            "score": 3.0,
            "remove_edges": [[[0, 0], [1, 0]]], "add_edges": [],
        },
    )
    selection = module.consensus_select_move(
        candidates, ((0, 1), (1, 2)), (0, 1, 2)
    )
    assert selection["candidate"] == "move_a"
    assert selection["score"] == 2.0
    assert selection["rounds"] == 2
    assert selection["canonical_move_id"] == module.canonical_move_id([], [])
    assert set(selection["selected_by_robot"].values()) == {
        (2.0, selection["canonical_move_id"])
    }


def test_consensus_move_selection_is_rename_invariant():
    module = load_module()
    first_move = {
        "candidate": "z_name", "group": "g", "valid": 1, "score": 1.0,
        "remove_edges": [[[0, 0], [1, 0]]],
        "add_edges": [[[0, 0], [1, 1]]],
    }
    second_move = {
        "candidate": "a_name", "group": "g", "valid": 1, "score": 1.0,
        "remove_edges": [[[0, 1], [1, 1]]],
        "add_edges": [[[0, 1], [1, 0]]],
    }
    topology = ((0, 1), (1, 2))
    selected = module.consensus_select_move(
        (first_move, second_move), topology, (0, 1, 2)
    )
    renamed = module.consensus_select_move(
        (
            {**first_move, "candidate": "a_after_rename"},
            {**second_move, "candidate": "z_after_rename"},
        ),
        topology,
        (0, 1, 2),
    )
    assert renamed["canonical_move_id"] == selected["canonical_move_id"]


def test_consensus_deduplicates_identical_moves_and_checks_score():
    module = load_module()
    duplicate = {
        "group": "g", "valid": 1,
        "remove_edges": [[[0, 0], [1, 0]]], "add_edges": [],
    }
    selection = module.consensus_select_move(
        (
            {**duplicate, "candidate": "first", "score": 2.0},
            {**duplicate, "candidate": "second", "score": 2.0},
        ),
        ((0, 1),),
        (0, 1),
    )
    assert selection["candidate"] == "first"
    with pytest.raises(ValueError, match="consistent score"):
        module.consensus_select_move(
            (
                {**duplicate, "candidate": "first", "score": 2.0},
                {**duplicate, "candidate": "second", "score": 3.0},
            ),
            ((0, 1),),
            (0, 1),
        )


def test_scorer_metadata_ignores_true_object_id_and_manifest_bans_truth_names(tmp_path):
    module = load_module()
    _, metadata, _, _, manifest = _write_network_quotient_cli_fixture(tmp_path)
    first = module.read_observed_metadata(metadata)
    text = metadata.read_text().replace("900", "1900").replace("901", "1901").replace("902", "1902")
    metadata.write_text(text)
    second = module.read_observed_metadata(metadata)
    assert module.canonical_metadata_bytes(first) == module.canonical_metadata_bytes(second)

    manifest.write_text(json.dumps({"candidates": [], "nested": {"rmse_value": 1}}))
    with pytest.raises(ValueError, match="forbidden field"):
        module.read_candidate_manifest(manifest)


def test_true_object_mutation_preserves_all_frozen_scorer_bytes(
    tmp_path, monkeypatch
):
    module = load_module()
    cavity, metadata, association, topology, manifest = (
        _write_network_quotient_cli_fixture(tmp_path)
    )
    monkeypatch.setattr(module.time, "perf_counter", lambda: 123.0)

    def run(output):
        module.run_cli([
            "--cavity-dir", str(cavity), "--metadata", str(metadata),
            "--association", str(association),
            "--candidate-manifest", str(manifest),
            "--topology", str(topology), "--output-dir", str(output),
            "--plane-axes", "xy",
        ])
        return tuple(
            (output / name).read_bytes()
            for name in (
                "network_quotient_candidates.csv",
                "network_quotient_packets.csv",
                "network_quotient_summary.json",
            )
        )

    first = run(tmp_path / "first")
    with metadata.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    for index, row in enumerate(rows):
        row["true_object_id"] = str(70000 + index)
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    second = run(tmp_path / "second")

    assert second == first


def test_candidate_batches_are_atomic_and_strict(tmp_path):
    module = load_module()
    cavity, metadata, association, topology, _ = _write_network_quotient_cli_fixture(tmp_path)
    loaded = module.read_scorer_inputs(cavity, metadata, association, topology, "xy")
    incumbent_edges = module.canonical_partition_forest(loaded.problem.partition)
    invalid = {
        "name": "bad", "group": "fixed",
        "remove_edges": [[[1, 0], [2, 0]], [[1, 9], [2, 9]]],
        "add_edges": [[[1, 0], [2, 1]]],
    }
    result = module.apply_candidate_batch(
        loaded.problem.partition, incumbent_edges, invalid,
        loaded.metadata_by_key,
    )
    assert result.valid is False
    assert result.partition == loaded.problem.partition
    assert result.status == "nonexistent_removal"

    same_edge = {
        "name": "same", "group": "fixed",
        "remove_edges": [[[0, 0], [1, 0]]],
        "add_edges": [[[1, 0], [0, 0]]],
    }
    conflict = module.apply_candidate_batch(
        loaded.problem.partition, loaded.incumbent_edges, same_edge,
        loaded.metadata_by_key,
    )
    assert conflict.valid is False
    assert conflict.status == "conflicting_addition"


def test_candidate_manifest_builders_cover_swap_and_three_cycle(tmp_path):
    module = load_module()
    _, metadata, association, _, _ = _write_network_quotient_cli_fixture(tmp_path)
    alternative = tmp_path / "alternative.csv"
    with alternative.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("receiver_robot", "sender_robot", "receiver_object", "sender_object", "weight"))
        writer.writerows(((1, 2, 0, 1, 1.0), (1, 2, 1, 2, 1.0), (1, 2, 2, 0, 1.0)))

    single = module.build_candidate_manifest(
        "single-column", metadata, association, (1, 2), fixed_copy="2:0"
    )
    cyclic = module.build_candidate_manifest(
        "cyclic-by-class", metadata, association, (1, 2), semantic_classes=(10,)
    )
    symmetric = module.build_candidate_manifest(
        "symmetric-difference-permutations", metadata, association, (1, 2),
        frozen_alternative_association=alternative,
    )
    assert single["candidates"][0]["name"] == "incumbent"
    assert any(len(candidate["remove_edges"]) == 2 for candidate in single["candidates"])
    assert any(len(candidate["remove_edges"]) == 3 for candidate in cyclic["candidates"])
    assert any(len(candidate["remove_edges"]) == 3 for candidate in symmetric["candidates"])
    assert [
        (candidate["group"], candidate["remove_edges"], candidate["add_edges"])
        for candidate in symmetric["candidates"]
    ] == sorted(
        (candidate["group"], candidate["remove_edges"], candidate["add_edges"])
        for candidate in symmetric["candidates"]
    )


def _disjoint_pair_interaction_variant(module, shift_1, shift_2):
    base_a = np.array([[0.0, 0.0], [1.7, 0.2], [0.3, 1.6]])
    base_b = np.array([[4.0, 0.4], [5.6, 0.1], [4.3, 2.0]])
    copies0 = tuple(module.ObjectCopy(0, index, 10) for index in range(6))
    copies1 = tuple(module.ObjectCopy(1, index, 10) for index in range(3))
    copies2 = tuple(module.ObjectCopy(2, index, 10) for index in range(3))
    means0 = np.vstack((base_a, base_b))
    means1 = (base_a - np.array([2.0, -1.0])) @ module.rotation2(0.6)
    means2 = (base_b - np.array([-3.0, 2.0])) @ module.rotation2(-0.8)
    covariance0 = np.eye(6) * 0.12
    for row in range(3):
        for column in range(3, 6):
            covariance0[row, column] = 0.025 / (
                1 + abs(row - (column - 3))
            )
            covariance0[column, row] = covariance0[row, column]
    states = (
        module.RobotCavityState(0, copies0, means0, covariance0),
        module.RobotCavityState(1, copies1, means1, np.eye(3) * 0.09),
        module.RobotCavityState(2, copies2, means2, np.eye(3) * 0.1),
    )
    components = tuple(
        (copies0[index], copies1[(index + shift_1) % 3])
        for index in range(3)
    ) + tuple(
        (copies0[index + 3], copies2[(index + shift_2) % 3])
        for index in range(3)
    )
    return module.NetworkQuotientProblem(
        states, module.NetworkPartition.build(components)
    )


def test_score_ablation_determinism_gauge_and_interaction_contracts():
    module = load_module()
    problem = _partition_variant(module, 0, 0)
    topology = ((0, 1), (1, 2))
    for label in module.SCORE_ABLATIONS:
        first = module.score_partition_ablation(problem, label)
        second = module.score_partition_ablation(problem, label)
        assert first.valid and second.valid
        assert first.score == pytest.approx(second.score, abs=1e-10)

    gauged, _ = module.apply_independent_robot_gauge(problem, 17)
    assert module.score_partition_ablation(problem, "full").score == pytest.approx(
        module.score_partition_ablation(gauged, "full").score, abs=1e-8
    )
    assert abs(
        module.score_partition_ablation(problem, "common-frame").score
        - module.score_partition_ablation(gauged, "common-frame").score
    ) > 1e-3

    pairwise_scores = {
        shifts: module.score_partition_ablation(
            _disjoint_pair_interaction_variant(module, *shifts), "pairwise-sum"
        ).score
        for shifts in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    full_scores = {
        shifts: module.score_partition_ablation(
            _disjoint_pair_interaction_variant(module, *shifts), "full"
        ).score
        for shifts in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    pairwise_mixed = (
        pairwise_scores[(1, 1)]
        - pairwise_scores[(1, 0)]
        - pairwise_scores[(0, 1)]
        + pairwise_scores[(0, 0)]
    )
    full_mixed = (
        full_scores[(1, 1)]
        - full_scores[(1, 0)]
        - full_scores[(0, 1)]
        + full_scores[(0, 0)]
    )
    assert abs(pairwise_mixed) < 1e-8
    assert abs(full_mixed) > 1e-3


def test_pairwise_sum_includes_every_cooccurring_robot_pair():
    module = load_module()
    problem = _partition_variant(module, 0, 0)
    manual_scores = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        pair = module._restrict_problem_to_pair(problem, first, second)
        assert pair is not None
        manual_scores.append(module.score_network_partition(pair).score)
    pairwise = module.score_partition_ablation(problem, "pairwise-sum")
    assert pairwise.score == pytest.approx(sum(manual_scores), abs=1e-10)
    assert len(pairwise.profiles) == 6


def test_pairwise_sum_abstains_when_discovered_pair_is_unobservable():
    module = load_module()
    copies0 = tuple(module.ObjectCopy(0, index, 10) for index in range(5))
    copies1 = tuple(module.ObjectCopy(1, index, 10) for index in range(3))
    copies2 = tuple(module.ObjectCopy(2, index, 10) for index in range(3))
    states = (
        module.RobotCavityState(
            0, copies0,
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [3.0, 1.0]]),
            np.eye(5) * 0.1,
        ),
        module.RobotCavityState(
            1, copies1,
            np.array([[2.0, 1.0], [3.0, 1.0], [2.0, 2.0]]),
            np.eye(3) * 0.1,
        ),
        module.RobotCavityState(
            2, copies2,
            np.array([[-1.0, 2.0], [2.0, 2.0], [2.0, 3.0]]),
            np.eye(3) * 0.1,
        ),
    )
    partition = module.NetworkPartition.build(
        (
            (copies0[0], copies1[0], copies2[0]),
            (copies0[1], copies1[1]),
            (copies0[2], copies1[2]),
            (copies0[3], copies2[1]),
            (copies0[4], copies2[2]),
        )
    )
    score = module.score_partition_ablation(
        module.NetworkQuotientProblem(states, partition), "pairwise-sum"
    )
    assert score.valid is False
    assert score.status == (
        "pairwise_pair_invalid_partition_transform_graph_unobservable"
    )


def test_manifest_build_mode_exits_before_cavity_read(tmp_path):
    module = load_module()
    _, metadata, association, _, _ = _write_network_quotient_cli_fixture(tmp_path)
    output = tmp_path / "built.json"
    assert module.run_cli([
        "--metadata", str(metadata), "--association", str(association),
        "--candidate-manifest", str(output),
        "--build-candidate-manifest", "single-column",
        "--edit-robot-pair", "1", "2", "--fixed-copy", "2:0",
    ]) == 0
    assert json.loads(output.read_text())["candidates"]


def test_association_threshold_and_reciprocal_rows_are_frozen(tmp_path):
    module = load_module()
    _, metadata_path, association, _, _ = _write_network_quotient_cli_fixture(tmp_path)
    metadata = module.read_observed_metadata(metadata_path)
    edges = module.read_association_edges(association, metadata)
    assert len(edges) == 6
    with association.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((0, 2, 0, 0, 0.5))
        writer.writerow((2, 0, 0, 0, 0.5000000001))
        writer.writerow((0, 2, 0, 0, 0.9))
    changed = module.read_association_edges(association, metadata)
    assert len(changed) == 7


def test_cavity_reader_rejects_incomplete_covariance(tmp_path):
    module = load_module()
    cavity, metadata, association, topology, _ = _write_network_quotient_cli_fixture(tmp_path)
    covariance = cavity / "final_pair_cavity_covariance.csv"
    lines = covariance.read_text().splitlines()
    covariance.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="complete"):
        module.read_scorer_inputs(cavity, metadata, association, topology, "xy")


def test_optional_current_transform_is_composed_on_tree(tmp_path):
    module = load_module()
    cavity, metadata, association, topology, _ = _write_network_quotient_cli_fixture(tmp_path)
    transform_path = cavity / "final_association_transforms.csv"
    header = (
        "receiver_robot", "sender_robot", "valid", "rows", "cols",
        *(f"value_{index}" for index in range(12)),
    )
    matrix01 = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 0.0]])
    matrix12 = np.array([[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    with transform_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow((0, 1, 1, 3, 4, *matrix01.reshape(-1)))
        writer.writerow((1, 2, 1, 3, 4, *matrix12.reshape(-1)))
    loaded = module.read_scorer_inputs(cavity, metadata, association, topology, "xy")
    assert loaded.current_transforms is not None
    assert loaded.current_transforms[2][0] == pytest.approx(np.pi / 2)
    assert loaded.current_transforms[2][1] == pytest.approx(np.array([3.0, 3.0]))


@pytest.mark.parametrize(
    "relative_path",
    (
        "ground_truth/association.csv",
        "groundtruth/association.csv",
        "inputs/truth_edges.csv",
        "inputs/gt_poses.csv",
        "inputs/ate/association.csv",
        "inputs/rmse_associations.csv",
    ),
)
def test_scorer_rejects_ground_truth_association_path_before_open(
    tmp_path, relative_path
):
    module = load_module()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("robot_id,local_slot_id,category\n")
    ground_truth = tmp_path / relative_path
    with pytest.raises(ValueError, match="ground-truth"):
        module.run_cli([
            "--metadata", str(metadata), "--association", str(ground_truth),
            "--candidate-manifest", str(tmp_path / "out.json"),
            "--build-candidate-manifest", "single-column",
            "--edit-robot-pair", "0", "1", "--fixed-copy", "0:0",
        ])


def test_scorer_path_filter_accepts_canonical_research_paths():
    module = load_module()
    for path in (
        "results/dba_distributed_cavity_ablation_20260716/07/"
        "final_association_weights.csv",
        "association_result/object_results/kitti360_4r_public_default/"
        "object_pgo/local_object_metadata.csv",
    ):
        module._reject_ground_truth_path(path, "input")


def test_manifest_builder_rejects_ground_truth_frozen_alternative_before_open(
    tmp_path
):
    module = load_module()
    _, metadata, association, _, _ = _write_network_quotient_cli_fixture(tmp_path)
    forbidden = tmp_path / "ground_truth" / "association.csv"
    with pytest.raises(ValueError, match="ground-truth"):
        module.run_cli([
            "--metadata", str(metadata), "--association", str(association),
            "--candidate-manifest", str(tmp_path / "out.json"),
            "--build-candidate-manifest", "symmetric-difference-permutations",
            "--edit-robot-pair", "1", "2",
            "--frozen-alternative-association", str(forbidden),
        ])


@pytest.mark.parametrize(
    "relative_path",
    (
        "mytruth/association.csv",
        "inputs/gtposes.csv",
        "inputs/groundtruthposes.csv",
        "inputs/ateposes.csv",
        "inputs/rmseassociation.csv",
    ),
)
def test_direct_association_reader_rejects_concatenated_truth_paths_before_open(
    tmp_path, relative_path
):
    module = load_module()
    forbidden = tmp_path / relative_path
    with pytest.raises(ValueError, match="ground-truth"):
        module.read_association_edges(forbidden, {})


def test_direct_manifest_builder_guards_association_before_metadata_open(tmp_path):
    module = load_module()
    with pytest.raises(ValueError, match="ground-truth"):
        module.build_candidate_manifest(
            "single-column",
            tmp_path / "missing_metadata.csv",
            tmp_path / "mytruth" / "association.csv",
            (0, 1),
            fixed_copy="0:0",
        )


@pytest.mark.parametrize(
    "reader",
    ("cavity", "scorer", "transform"),
)
def test_direct_cavity_readers_guard_truth_directory_before_any_read(
    tmp_path, reader
):
    module = load_module()
    forbidden = tmp_path / "groundtruthposes"
    with pytest.raises(ValueError, match="ground-truth"):
        if reader == "cavity":
            module._read_cavity_states(forbidden, {}, "xy")
        elif reader == "scorer":
            module.read_scorer_inputs(
                forbidden,
                tmp_path / "missing_metadata.csv",
                tmp_path / "missing_association.csv",
                tmp_path / "missing_topology.csv",
                "xy",
            )
        else:
            module.read_optional_current_transforms(
                forbidden, (), (0,), "xy"
            )


@pytest.mark.parametrize(
    ("reader", "relative_path"),
    (
        ("topology", "ground_truth/gt_poses.csv"),
        ("manifest", "ground_truth/gt_associations.json"),
    ),
)
def test_direct_protocol_readers_guard_truth_paths_before_open(
    tmp_path, reader, relative_path
):
    module = load_module()
    forbidden = tmp_path / relative_path
    with pytest.raises(ValueError, match="ground-truth"):
        if reader == "topology":
            module.read_topology(forbidden, (0, 1))
        else:
            module.read_candidate_manifest(forbidden)


def _scorer_inputs_fixture(module):
    problem = _three_robot_fixture(module)
    metadata = {
        (copy.robot, copy.object_id): copy.semantic_class
        for copy in problem.partition.copies
    }
    return problem, list(module.canonical_partition_forest(problem.partition)), metadata


def test_scorer_inputs_snapshots_and_normalizes_all_public_state():
    module = load_module()
    problem, edges, metadata = _scorer_inputs_fixture(module)
    topology = [(1, 2), (1, 0)]
    translations = {robot: np.array([robot, -robot], dtype=float) for robot in (0, 1, 2)}
    transforms = {robot: (0.1 * robot, translations[robot]) for robot in translations}

    inputs = module.ScorerInputs(
        problem, edges, topology, metadata, transforms
    )
    edges.clear()
    topology.clear()
    metadata.clear()
    translations[1][0] = 999.0

    assert inputs.incumbent_edges == module.canonical_partition_forest(
        problem.partition
    )
    assert inputs.topology == ((0, 1), (1, 2))
    assert len(inputs.metadata_by_key) == problem.partition.copy_count
    assert inputs.current_transforms[1][1][0] == 1.0
    with pytest.raises(TypeError):
        inputs.metadata_by_key[(9, 9)] = 10
    with pytest.raises(ValueError):
        inputs.current_transforms[1][1][0] = 5.0


def test_scorer_inputs_rejects_malformed_cross_field_state():
    module = load_module()
    problem, edges, metadata = _scorer_inputs_fixture(module)
    with pytest.raises(ValueError, match="NetworkQuotientProblem"):
        module.ScorerInputs(object(), edges, ((0, 1), (1, 2)), metadata, None)
    with pytest.raises(ValueError, match="reconstruct"):
        module.ScorerInputs(problem, edges[:-1], ((0, 1), (1, 2)), metadata, None)
    with pytest.raises(ValueError, match="tree"):
        module.ScorerInputs(problem, edges, ((0, 1), (0, 2), (1, 2)), metadata, None)
    missing = dict(metadata)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="metadata support"):
        module.ScorerInputs(problem, edges, ((0, 1), (1, 2)), missing, None)
    wrong = dict(metadata)
    wrong[next(iter(wrong))] += 1
    with pytest.raises(ValueError, match="semantic"):
        module.ScorerInputs(problem, edges, ((0, 1), (1, 2)), wrong, None)
    poisoned = dict(metadata)
    poisoned[(True, 999)] = 10
    with pytest.raises(ValueError, match="non-bool integral"):
        module.ScorerInputs(problem, edges, ((0, 1), (1, 2)), poisoned, None)
    with pytest.raises(ValueError, match="current transforms"):
        module.ScorerInputs(
            problem, edges, ((0, 1), (1, 2)), metadata,
            {0: (0.0, np.zeros(2))},
        )


def test_candidate_application_normalizes_and_validates_public_state():
    module = load_module()
    problem = _three_robot_fixture(module)
    edges = list(reversed(module.canonical_partition_forest(problem.partition)))
    application = module.CandidateApplication(
        True, "ok", problem.partition, edges
    )
    edges.clear()
    assert application.edges == module.canonical_partition_forest(problem.partition)
    with pytest.raises(ValueError, match="bool"):
        module.CandidateApplication(1, "ok", problem.partition, application.edges)
    with pytest.raises(ValueError, match="status"):
        module.CandidateApplication(False, "ok", problem.partition, application.edges)
    with pytest.raises(ValueError, match="NetworkPartition"):
        module.CandidateApplication(False, "bad", object(), application.edges)
    with pytest.raises(ValueError, match="reconstruct"):
        module.CandidateApplication(False, "bad", problem.partition, application.edges[:-1])


def test_permutation_choice_generator_is_lazy(monkeypatch):
    module = load_module()
    events = []

    def observed_permutations(values):
        values = tuple(values)
        events.append(("start", values))
        events.append(("yield", values, 0))
        yield values
        events.append(("yield", values, 1))
        yield tuple(reversed(values))

    monkeypatch.setattr(module.itertools, "permutations", observed_permutations)
    choices = module._iter_permutation_choices(
        (((), ((0, 0), (0, 1))), ((), ((1, 0), (1, 1))))
    )
    first = next(choices)
    assert first == (((0, 0), (0, 1)), ((1, 0), (1, 1)))
    assert events == [
        ("start", ((0, 0), (0, 1))),
        ("yield", ((0, 0), (0, 1)), 0),
        ("start", ((1, 0), (1, 1))),
        ("yield", ((1, 0), (1, 1)), 0),
    ]


def test_equilibrated_spd_logdet_is_exact_on_ill_scaled_congruence():
    module = load_module()
    correlation = np.array(
        [[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]]
    )
    scale = np.diag([1e-9, 1.0, 1e9])
    hessian = scale @ correlation @ scale
    expected = float(np.linalg.slogdet(correlation)[1])
    assert module._equilibrated_spd_logdet(hessian) == pytest.approx(
        expected, abs=1e-12
    )
    with pytest.raises(ValueError, match="positive finite diagonal"):
        module._equilibrated_spd_logdet(np.diag([1.0, 0.0]))

