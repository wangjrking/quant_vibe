import unittest

from prepare_v12_research_training_launch import prepare_launch


class PrepareV12ResearchTrainingLaunchTest(unittest.TestCase):
    def _plan(self):
        return {
            "experiments": [
                {
                    "horizon": "3d",
                    "variant": "stability_first",
                    "output_table": "table_3d_research",
                    "command": "python run_parallel_expanding2010_folds.py --output-table table_3d_research",
                },
                {
                    "horizon": "1d",
                    "variant": "stability_first",
                    "output_table": "table_1d_research",
                    "command": "python run_parallel_expanding2010_folds.py --output-table table_1d_research",
                },
            ],
        }

    def test_blocked_gate_returns_no_training_commands(self):
        gate = {
            "ok": False,
            "status": "not_ready_waiting_for_user_or_supervisor_approval",
            "errors": ["authorization packet is not approved"],
        }

        result = prepare_launch(plan=self._plan(), execution_gate=gate)

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["commands"], [])
        self.assertIn("execution gate is not ready", result["errors"])

    def test_ready_gate_returns_plan_commands_without_execution(self):
        gate = {
            "ok": True,
            "status": "ready_for_research_only_training_execution",
            "errors": [],
        }

        result = prepare_launch(plan=self._plan(), execution_gate=gate)

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready_commands_prepared_not_executed")
        self.assertEqual(len(result["commands"]), 2)
        self.assertTrue(result["boundaries"]["no_training_executed_by_launcher"])


if __name__ == "__main__":
    unittest.main()
