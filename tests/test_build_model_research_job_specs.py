import unittest

from build_model_research_job_specs import build_job_specs


class BuildModelResearchJobSpecsTest(unittest.TestCase):
    def test_failed_queue_item_generates_training_job_spec_with_saved_model_requirement(self):
        queue_payload = {
            "queue": [
                {
                    "queue_rank": 1,
                    "horizon": "3d",
                    "label": "executable_3d_open_return",
                    "status": "failed_objective_gate",
                    "next_action_type": "retrain_or_feature_search",
                    "failed_constraints": "full_daily_rank_ic_delta_floor",
                    "priority": "P1",
                },
                {
                    "queue_rank": 2,
                    "horizon": "10d",
                    "label": "executable_10d_open_return",
                    "status": "passed_objective_gate",
                    "next_action_type": "audit_and_strategy_research_validation",
                    "failed_constraints": "",
                    "priority": "P1",
                },
            ]
        }
        objective_config = {
            "standard_chain": {
                "feature_input": "quant/data_file/production_factor_parts/",
                "label_input": "quant/data_file/prediction_label_parts/",
                "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
            },
            "horizon_objectives": {
                "3d": {
                    "optimization_style": "rankic_constrained_topn_objective",
                    "additional_constraints": {"full_daily_rank_ic_delta_floor": -0.002},
                },
                "10d": {
                    "optimization_style": "preserve_v11_constrained_blend_then_validate",
                    "additional_constraints": {"full_top5_delta_floor": 0.0},
                },
            },
        }

        specs = build_job_specs(queue_payload, objective_config, run_id="v12_20260624")

        self.assertEqual([item["horizon"] for item in specs], ["3d", "10d"])
        self.assertEqual(specs[0]["execution_status"], "not_started_requires_authorization")
        self.assertTrue(specs[0]["requires_training_authorization"])
        self.assertTrue(specs[0]["model_artifacts_required"]["fold_model_files"])
        self.assertEqual(specs[0]["feature_input"], "quant/data_file/production_factor_parts/")
        self.assertEqual(specs[0]["stop_conditions"][0], "production_feature_or_label_input_missing")
        self.assertEqual(specs[1]["execution_status"], "validation_only_not_training")
        self.assertFalse(specs[1]["requires_training_authorization"])


if __name__ == "__main__":
    unittest.main()
