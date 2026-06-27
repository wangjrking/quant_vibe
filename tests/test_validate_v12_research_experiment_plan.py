import unittest

from validate_v12_research_experiment_plan import validate_experiment_plan


class ValidateV12ResearchExperimentPlanTest(unittest.TestCase):
    def _valid_plan(self):
        return {
            "approval_status": "research_plan_not_approved_for_execution",
            "experiments": [
                {
                    "horizon": "3d",
                    "label": "executable_3d_open_return",
                    "requires_training_authorization": True,
                    "output_table": "stock_predict_data_model_agent_v12_3d_research",
                    "output_dir": "D:/work/quant/quant_mcp/quant/data_file/reports/model_agent_v12/run",
                    "command": (
                        "D:/work/quant/quant_mcp/.venv/Scripts/python.exe "
                        "D:/work/quant/quant_mcp/quant/main/run_parallel_expanding2010_folds.py "
                        "--data-file-url D:/work/quant/quant_mcp/quant/data_file "
                        "--first-test 20240604 "
                        "--final-test 20260623 "
                        "--label executable_3d_open_return "
                        "--model-type reg "
                        "--train-years 5 "
                        "--prediction-output-mode independent "
                        "--output-table stock_predict_data_model_agent_v12_3d_research "
                        "--output-dir D:/work/quant/quant_mcp/quant/data_file/reports/model_agent_v12/run "
                        "--experiment-name v12_3d_test "
                        "--skip-existing"
                    ),
                }
            ],
            "skipped_validation_only": ["10d"],
            "boundaries": {
                "no_training_executed_by_plan_generation": True,
                "no_prediction_generated_by_plan_generation": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }

    def test_valid_research_plan_passes_static_safety_checks(self):
        result = validate_experiment_plan(self._valid_plan())

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["experiments_checked"], 1)
        self.assertEqual(result["window_summaries"][0]["last_test_end"], "20260623")
        self.assertGreater(result["window_summaries"][0]["folds"], 0)

    def test_non_research_or_formal_output_table_is_rejected(self):
        plan = self._valid_plan()
        plan["experiments"][0]["output_table"] = "stock_predict_data_model_agent_3d_formal"

        result = validate_experiment_plan(plan)

        self.assertFalse(result["ok"])
        self.assertIn("3d output_table must contain _research", result["errors"])
        self.assertIn("3d output_table must not contain formal", result["errors"])

    def test_dangerous_command_terms_are_rejected(self):
        plan = self._valid_plan()
        plan["experiments"][0]["command"] += " --production-manifest config/prod.json --backtest"

        result = validate_experiment_plan(plan)

        self.assertFalse(result["ok"])
        self.assertIn("3d command contains forbidden term: production-manifest", result["errors"])
        self.assertIn("3d command contains forbidden term: backtest", result["errors"])

    def test_missing_boundary_is_rejected(self):
        plan = self._valid_plan()
        plan["boundaries"]["no_signal"] = False

        result = validate_experiment_plan(plan)

        self.assertFalse(result["ok"])
        self.assertIn("boundary no_signal must be true", result["errors"])

    def test_non_positive_train_years_is_rejected(self):
        plan = self._valid_plan()
        plan["experiments"][0]["command"] = plan["experiments"][0]["command"].replace(
            "--train-years 5",
            "--train-years 0",
        )

        result = validate_experiment_plan(plan)

        self.assertFalse(result["ok"])
        self.assertIn("3d command --train-years must be positive", result["errors"])

    def test_unknown_run_parallel_argument_is_rejected(self):
        plan = self._valid_plan()
        plan["experiments"][0]["command"] += " --not-a-real-argument 1"

        result = validate_experiment_plan(plan)

        self.assertFalse(result["ok"])
        self.assertIn("3d command does not parse with run_parallel_expanding2010_folds.py", result["errors"])

    def test_final_test_before_expected_latest_date_is_rejected(self):
        plan = self._valid_plan()
        plan["experiments"][0]["command"] = plan["experiments"][0]["command"].replace(
            "--final-test 20260623",
            "--final-test 20260612",
        )

        result = validate_experiment_plan(plan, expected_final_test="20260623")

        self.assertFalse(result["ok"])
        self.assertIn("3d final-test must be 20260623", result["errors"])


if __name__ == "__main__":
    unittest.main()
