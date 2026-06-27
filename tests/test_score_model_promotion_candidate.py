import unittest

from score_model_promotion_candidate import evaluate_candidate


class ScoreModelPromotionCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "standard_chain": {
                "feature_input": "quant/data_file/production_factor_parts/",
                "label_input": "quant/data_file/prediction_label_parts/",
                "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
            },
            "hard_constraints": {
                "governance": {
                    "required_candidate_approval_status": "research_only_not_approved_for_l4_or_l5",
                },
                "coverage": {
                    "required_min_trade_date": "20240604",
                    "latest_day_rows_must_match_production_factor_rows": True,
                    "null_pred_prob_must_equal": 0,
                    "duplicate_key_groups_must_equal": 0,
                },
                "reproducibility": {
                    "saved_model_files_required_for_train_candidates": True,
                    "model_params_required_for_train_candidates": True,
                    "selected_features_required_for_train_candidates": True,
                    "training_window_required_for_train_candidates": True,
                    "evaluation_report_required": True,
                    "candidate_manifest_required": True,
                },
                "quality": {
                    "common": {
                        "full_top5_delta_floor": 0.0,
                        "recent63_top5_delta_floor": 0.0,
                        "recent20_top5_delta_floor": 0.0,
                        "positive_top5_periods_floor": 3,
                    },
                    "by_label": {
                        "executable_10d_open_return": {
                            "full_rank_ic_delta_floor": -0.0005,
                            "min_period_top5_delta_floor": 0.0,
                        }
                    },
                },
            },
            "soft_constraints": {
                "recent_effectiveness": ["recent63_top1_delta", "recent63_top5_delta"],
                "full_window_stability": ["full_rank_ic_delta"],
                "engineering": ["prefer_simpler_formula"],
            },
            "promotion_states": {
                "research_only": "research_only_not_approved_for_l4_or_l5",
                "promotable_formal_candidate": "approved_for_l4_candidate_only",
            },
        }

    def test_candidate_passes_all_hard_constraints(self):
        candidate = {
            "label": "executable_10d_open_return",
            "asset": "research_10d_risk_balanced_v5_shrink",
            "table": "some_table",
            "approval_status": "research_only_not_approved_for_l4_or_l5",
            "feature_input": "quant/data_file/production_factor_parts/",
            "label_input": "quant/data_file/prediction_label_parts/",
            "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
            "forbidden_inputs_present": False,
            "min_trade_date": "20240604",
            "latest_trade_date": "20260624",
            "expected_latest_trade_date": "20260624",
            "latest_day_rows": 5512,
            "expected_latest_day_rows": 5512,
            "null_pred_prob": 0,
            "duplicate_key_groups": 0,
            "is_train_candidate": True,
            "saved_model_files_present": True,
            "model_params_present": True,
            "selected_features_present": True,
            "training_window_present": True,
            "evaluation_report_present": True,
            "candidate_manifest_present": True,
            "full_top5_delta": 0.014,
            "recent63_top5_delta": 0.058,
            "recent20_top5_delta": 0.124,
            "positive_top5_periods": 4,
            "full_rank_ic_delta": 0.00043,
            "min_period_top5_delta": 0.0015,
            "recent63_top1_delta": 0.147,
            "prefer_simpler_formula": "yes",
        }

        result = evaluate_candidate(candidate, self.config)

        self.assertTrue(result["hard_constraint_passed"])
        self.assertEqual(result["promotion_decision"], "promotable_formal_candidate")
        self.assertEqual(result["target_approval_status"], "approved_for_l4_candidate_only")
        self.assertEqual(result["failed_hard_constraints"], [])
        self.assertGreaterEqual(len(result["soft_reminders"]), 3)

    def test_candidate_fails_when_quality_and_reproducibility_break(self):
        candidate = {
            "label": "executable_10d_open_return",
            "asset": "bad_candidate",
            "table": "some_table",
            "approval_status": "research_only_not_approved_for_l4_or_l5",
            "feature_input": "quant/data_file/production_factor_parts/",
            "label_input": "quant/data_file/prediction_label_parts/",
            "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
            "forbidden_inputs_present": False,
            "min_trade_date": "20240604",
            "latest_trade_date": "20260624",
            "expected_latest_trade_date": "20260624",
            "latest_day_rows": 5512,
            "expected_latest_day_rows": 5512,
            "null_pred_prob": 0,
            "duplicate_key_groups": 0,
            "is_train_candidate": True,
            "saved_model_files_present": False,
            "model_params_present": True,
            "selected_features_present": True,
            "training_window_present": True,
            "evaluation_report_present": True,
            "candidate_manifest_present": True,
            "full_top5_delta": -0.001,
            "recent63_top5_delta": 0.01,
            "recent20_top5_delta": 0.01,
            "positive_top5_periods": 2,
            "full_rank_ic_delta": -0.002,
            "min_period_top5_delta": -0.001,
        }

        result = evaluate_candidate(candidate, self.config)

        self.assertFalse(result["hard_constraint_passed"])
        self.assertEqual(result["promotion_decision"], "research_only")
        self.assertIn("saved_model_files_present", result["failed_hard_constraints"])
        self.assertIn("full_top5_delta", result["failed_hard_constraints"])
        self.assertIn("positive_top5_periods", result["failed_hard_constraints"])
        self.assertIn("full_rank_ic_delta", result["failed_hard_constraints"])


if __name__ == "__main__":
    unittest.main()
