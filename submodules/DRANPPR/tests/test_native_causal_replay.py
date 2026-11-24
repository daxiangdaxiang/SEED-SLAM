"""Subprocess contracts for the exact-Jacobian causal replay; no Python GTSAM."""
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def pose(t, index, delta=None, robot=1):
    return dict(kind="pose", robot=robot, timestamp_s=t, available_time_s=t,
                support_max_time_s=t, pose_index=index,
                from_pose_index=index-1 if index else None, delta=delta,
                duration_s=1 if index else 0)


def observation(t, index, row, distance, bearing=0, subject=6, robot=1):
    return dict(kind="observation", robot=robot, timestamp_s=t, available_time_s=t,
                support_max_time_s=t, pose_index=index, source_row=row,
                landmark_subject=subject, range_m=distance, bearing_rad=bearing)


def packets():
    return [pose(0, 0), observation(0, 0, 0, 2),
            observation(0, 0, 1, 2, math.pi/2, 7),
            pose(1, 1, [1, 0, 0]), observation(1, 1, 2, 1),
            observation(1, 1, 3, math.sqrt(5), math.atan2(2, -1), 7),
            pose(2, 2, [0, 1, .1]),
            observation(2, 2, 4, math.sqrt(2), -.1-math.pi/4)]


class NativeCausalReplay(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ.get("DRANPPR_NATIVE_CAUSAL_BINARY",
                                         "/tmp/dranppr_backend_build/bin/native-causal-replay"))
        self.assertTrue(self.binary.is_file(), "build native-causal-replay first")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0

    def run_replay(self, data, mode="native", duration=3, success=True):
        self.count += 1
        path = self.root / f"events{self.count}.jsonl"
        path.write_text("".join(json.dumps(p)+"\n" for p in data))
        output = self.root / f"run{self.count}"
        run = subprocess.run([str(self.binary), str(path), str(output), mode, str(duration)],
                             capture_output=True, text=True, timeout=15)
        if not success:
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse((output / "summary.json").exists())
            return run.stderr
        self.assertEqual(run.returncode, 0, run.stderr)
        result = {"summary": json.loads((output / "summary.json").read_text())}
        for name in ("updates", "causal_estimates", "final_estimates"):
            with (output / f"{name}.csv").open() as stream:
                result[name] = list(csv.DictReader(stream))
        return result

    def test_identical_noise_free_motion_and_physical_factor_counts(self):
        for mode in ("native", "native_cached", "isam2_exact"):
            result = self.run_replay(packets(), mode)
            last = result["causal_estimates"][-1]
            for key, expected in zip(("x", "y", "yaw"), (1, 1, .1)):
                self.assertAlmostEqual(float(last[key]), expected, places=8)
            self.assertEqual([int(r["new_factors"]) for r in result["updates"]], [2, 3, 2])
            self.assertEqual(len(result["final_estimates"]), 5)
            self.assertLess(result["summary"]["physical_gaussian_cost"], 1e-18)
            self.assertFalse(result["summary"]["truth_used"])
            self.assertFalse(result["summary"]["communication"])

    def test_interleaved_robots_are_timestamp_atomic(self):
        data = [pose(0, 0), pose(0, 0, robot=2), observation(0, 0, 0, 2),
                observation(0, 0, 0, 3, robot=2)]
        for mode in ("native", "native_cached", "isam2_exact"):
            result = self.run_replay(data, mode)
            self.assertEqual(result["summary"]["frames"], 2)
            self.assertEqual(result["summary"]["packets"], 4)
            self.assertEqual([int(r["new_factors"]) for r in result["updates"]], [1, 1])

    def test_future_mutation_cannot_change_prefix(self):
        changed = copy.deepcopy(packets())
        changed[-1]["range_m"] = 4
        for mode in ("native", "native_cached", "isam2_exact"):
            original = self.run_replay(packets(), mode)
            mutated = self.run_replay(changed, mode)
            prefix = self.run_replay(changed, mode, duration=1)
            self.assertEqual(original["causal_estimates"][:2], mutated["causal_estimates"][:2])
            self.assertEqual(original["causal_estimates"][:2], prefix["causal_estimates"])

    def test_root_without_observations_is_supported(self):
        for mode in ("native", "native_cached", "isam2_exact"):
            result = self.run_replay([pose(0, 0), pose(1, 1, [1, 0, 0])], mode)
            self.assertEqual(result["summary"]["frames"], 2)
            self.assertAlmostEqual(float(result["causal_estimates"][-1]["x"]), 1)

    def test_native_descent_and_timing_accounting(self):
        data = packets()
        data[-1]["range_m"] = 1.8
        result = self.run_replay(data)
        for row in result["updates"]:
            self.assertLessEqual(float(row["cost_after"]), float(row["cost_before"])+1e-12)
            self.assertLessEqual(int(row["accepted_steps"]), 1)
            for key, value in row.items():
                if key.endswith("_ms") or key.endswith("backlog_s"):
                    self.assertGreaterEqual(float(value), 0, key)
            self.assertLessEqual(float(row["private_factorization_ms"]), float(row["response_ms"]))
            self.assertLessEqual(float(row["publication_linearization_ms"]), float(row["publication_ms"]))
            self.assertGreaterEqual(float(row["total_ms"]), float(row["solve_ms"]))
        self.assertEqual(int(result["updates"][-1]["physical_factors"]), 7)
        self.assertEqual(int(result["updates"][-1]["history_values"]), 5)

    def test_invalid_streams_fail_explicitly(self):
        bad = []
        data = packets(); data[-1]["source_row"] = 0; bad.append((data, "duplicate"))
        data = packets(); data[-1]["support_max_time_s"] = 3; bad.append((data, "noncausal"))
        data = packets(); data[-2]["from_pose_index"] = 0; bad.append((data, "parent"))
        data = packets(); data[-1]["range_m"] = -1; bad.append((data, "range"))
        data = packets(); data.append(pose(1, 3, [0, 0, 0])); bad.append((data, "time decreased"))
        for data, message in bad:
            for mode in ("native", "native_cached", "isam2_exact"):
                self.assertIn(message, self.run_replay(data, mode, success=False))

    def test_cached_publication_is_not_overwritten_by_old_bases(self):
        data = packets()
        data[-1]["range_m"] = 1.8
        prefix = self.run_replay(data, "native_cached")
        full = self.run_replay(data+[pose(3, 3, [.5, 0, 0])], "native_cached", duration=4)
        self.assertEqual(prefix["causal_estimates"], full["causal_estimates"][:3])
        self.assertGreater(abs(float(prefix["causal_estimates"][-1]["x"])-1), .01)
        rows = full["updates"]
        self.assertGreater(sum(int(r["model_refresh_reused"]) for r in rows), 0)
        self.assertGreater(sum(int(r["model_refresh_factors"]) for r in rows), 0)
        for row in rows:
            self.assertLessEqual(float(row["cost_after"]), float(row["cost_before"])+1e-12)
            self.assertGreaterEqual(float(row["refresh_ms"]), float(row["refresh_linearization_ms"]))


if __name__ == "__main__":
    unittest.main()
