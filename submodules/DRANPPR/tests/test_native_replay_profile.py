import importlib.util
from pathlib import Path
import sys
import unittest


class NativeReplayProfile(unittest.TestCase):
    def test_nested_timers_are_not_added_to_service_cost(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "experiments/audit_native_replay.py"
        self.assertTrue(script.is_file(), "implement native replay audit")
        sys.path.insert(0, str(root / "experiments"))
        spec = importlib.util.spec_from_file_location("native_replay_audit", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [dict(robot="1", elapsed_s=str(i), total_ms="10", initialization_ms="1" if i == 0 else "0",
                     solve_ms="7", insert_ms="2", response_ms="4", private_factorization_ms="3",
                     history_values=str(i+1), serial_backlog_s=".1", robot_backlog_s=".01",
                     accepted_steps="1", cost_before="2", cost_after="1",
                     model_refresh_factors="2", model_refresh_reused="8") for i in range(4)]
        summary, growth = module.profile(rows)
        self.assertAlmostEqual(summary["measured_work_s"], .041)
        self.assertAlmostEqual(summary["components_s"]["response_ms"], .016)
        self.assertAlmostEqual(summary["components_s"]["private_factorization_ms"], .012)
        self.assertEqual(summary["descent_violations"], 0)
        self.assertEqual(summary["accepted_steps"], 4)
        self.assertEqual(summary["model_refresh_factors"], 8)
        self.assertAlmostEqual(summary["model_refresh_fraction"], .2)
        self.assertEqual(growth[-1]["max_history_values"], 4)
        rows[-1]["cost_after"] = "3"
        self.assertEqual(module.profile(rows)[0]["descent_violations"], 1)


if __name__ == "__main__":
    unittest.main()
