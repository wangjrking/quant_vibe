import unittest

from build_v12_research_experiment_plan import build_experiment_plan


class BuildV12ResearchExperimentPlanTest(unittest.TestCase):
    def test_builds_training_experiments_only_for_authorized_required_items(self):
        runbook = {
            "standard_chain": {
                "feature_input": "D:/work/quant/quant_mcp/quant/data_file/production_factor_parts/",
                "label_input": "D:/work/quant/quant_mcp/quant/data_file/prediction_label_parts/",
                "prediction_db": "D:/work/quant/quant_mcp/quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
            },
            "entrypoints": {
                "parallel_folds": "D:/work/quant/quant_mcp/quant/main/run_parallel_expanding2010_folds.py",
            },
            "v12_execution_order": [
                {
                    "horizon": "3d",
                    "label": "executable_3d_open_return",
                    "priority": "P1",
                    "job_type": "retrain_or_feature_search",
                    "failed_constraint": "full_daily_rank_ic_delta_floor",
                    "requires_training_authorization": True,
                    "suggested_feature_top_n": [120, 160, 200],
                    "suggested_max_depth": [2, 3],
                    "suggested_learning_rate": [0.003, 0.005],
                    "suggested_n_estimators": [4000, 6000],
                    "suggested_reg_lambda": [3, 6, 10],
                    "suggested_reg_alpha": [0, 0.5],
                    "initial_max_workers": 1,
                },
                {
                    "horizon": "10d",
                    "label": "executable_10d_open_return",
                    "priority": "P1",
                    "job_type": "audit_and_strategy_research_validation",
                    "failed_constraint": "",
                    "requires_training_authorization": False,
                },
            ],
        }
        status = {
            "horizons": {
                "3d": {
                    "weighted_score": 0.007,
                    "latest_mature_label_date": "20260609",
                },
                "10d": {
                    "weighted_score": 0.018,
                    "latest_mature_label_date": "20260528",
                },
            }
        }

        plan = build_experiment_plan(runbook=runbook, status=status, run_id="v12_20260624")

        self.assertEqual([item["horizon"] for item in plan["experiments"]], ["3d", "3d"])
        self.assertEqual(plan["skipped_validation_only"], ["10d"])
        first = plan["experiments"][0]
        self.assertTrue(first["requires_training_authorization"])
        self.assertIn("_research", first["output_table"])
        self.assertIn("--label executable_3d_open_return", first["command"])
        self.assertIn("--feature-source production_split", first["command"])
        self.assertIn("--output-table", first["command"])
        self.assertTrue(plan["boundaries"]["no_training_executed_by_plan_generation"])


if __name__ == "__main__":
    unittest.main()
