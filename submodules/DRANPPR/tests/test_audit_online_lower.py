from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from audit_online_lower import replay_fixed_calibration
from test_audit_online_noise import initial_frame, next_frame
from audit_online_noise import replay
from audit_residual_evidence import final_estimate_rows


def test_fixed_calibration_reconstruction_matches_original_online_states():
    packets = initial_frame()
    for index in range(1, 6):
        packets.extend(next_frame(index, 2. - .08 * index))
    original = replay(packets, 3., 5., methods=('fixed', 'profiled_diagonal'))
    robots, pose_times = replay_fixed_calibration(
        packets, original['calibration'], 3., 5., ('fixed', 'profiled_diagonal'))
    for method, local in robots.items():
        reconstructed = final_estimate_rows(local, pose_times, 0)
        expected = original['final_estimates'][method]
        assert len(reconstructed) == len(expected)
        for a, b in zip(reconstructed, expected):
            np.testing.assert_allclose([a['x'], a['y']], [b['x'], b['y']], atol=1e-10)


def test_fixed_calibration_reconstruction_applies_future_q_without_compounding():
    packets = initial_frame()
    for index in range(1, 8):
        packets.extend(next_frame(index, 2. - .08 * index))
    def calibrate(robot, method):
        return dict(method=method, robot=robot.robot, success=True,
                    at_numerical_boundary=False, bearing_variance=.001,
                    range_intercept_variance=.003, range_distance_squared_coefficient=.0001,
                    odometry_variance_scales=[.5, .03, 1.4])
    methods = ('fixed', 'joint')
    original = replay(packets, 3., 7., methods=methods, calibrator=calibrate)
    robots, times = replay_fixed_calibration(packets, original['calibration'], 3., 7., methods)
    for method, local in robots.items():
        expected = original['final_estimates'][method]
        actual = final_estimate_rows(local, times, 0)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose([a['x'], a['y']], [b['x'], b['y']], atol=1e-10)
    np.testing.assert_allclose(robots['joint'][1].odometry_sigma,
                               robots['fixed'][1].odometry_sigma * np.sqrt([.5, .03, 1.4]))
