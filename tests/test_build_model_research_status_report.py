import unittest

from build_model_research_status_report import build_status_report


class BuildModelResearchStatusReportTest(unittest.TestCase):
    def test_builds_per_horizon_status_with_formula_validation(self):
        queue = {
            "queue": [
                {
                    "horizon": "3d",
                    "label": "executable_3d_open_return",
                    "priority": "P1",
                    "status": "failed_objective_gate",
                    "next_action_type": "retrain_or_feature_search",
                    "weighted_score": 0.007,
                    "failed_constraints": "full_daily_rank_ic_delta_floor",
                    "current_best_research_table": "research_3d",
                    "requires_training_authorization": True,
                    "queue_rank": 1,
                },
                {
                    "horizon": "10d",
                    "label": "executable_10d_open_return",
                    "priority": "P1",
                    "status": "passed_objective_gate",
                    "next_action_type": "audit_and_strategy_research_validation",
                    "weighted_score": 0.018,
                    "failed_constraints": "",
                    "current_best_research_table": "research_10d_blend",
                    "requires_training_authorization": False,
                    "queue_rank": 4,
                },
            ]
        }
        snapshot = {
            "formal_baseline": {
                "3d": "formal_3d",
                "10d": "formal_10d",
            },
            "coverage": [
                {
                    "horizon": "3d",
                    "rows": 100,
                    "min_trade_date": "20240604",
                    "max_trade_date": "20260623",
                    "null_pred_prob": 0,
                    "duplicate_key_groups": 0,
                },
                {
                    "horizon": "10d",
                    "rows": 100,
                    "min_trade_date": "20240604",
                    "max_trade_date": "20260623",
                    "null_pred_prob": 0,
                    "duplicate_key_groups": 0,
                },
            ],
        }
        readiness = {
            "factor_asset": {"latest_trade_date": "20260623", "latest_day_rows": 5513},
            "label_asset": {
                "mature_label_dates": {
                    "executable_3d_open_return": {"latest_non_null_date": "20260609"},
                    "executable_10d_open_return": {"latest_non_null_date": "20260528"},
                }
            },
            "readiness_decision": {
                "training_authorization_required_for": ["3d"],
                "validation_only_for": ["10d"],
            },
        }
        formula_validation = {
            "ok": True,
            "status": "formula_score_asset_ready_for_model_side_audit",
            "target_table": "research_10d_blend",
        }

        report = build_status_report(
            queue=queue,
            snapshot=snapshot,
            readiness=readiness,
            formula_validations={"10d": formula_validation},
        )

        self.assertEqual(report["summary"]["feature_latest_trade_date"], "20260623")
        self.assertEqual(report["summary"]["next_training_order"], ["3d"])
        self.assertEqual(report["horizons"]["3d"]["next_step"], "需要 research-only 训练授权后重训/特征搜索")
        self.assertEqual(report["horizons"]["3d"]["formal_baseline_table"], "formal_3d")
        self.assertEqual(report["horizons"]["10d"]["acceptance_status"], "formula_score_asset_ready_for_model_side_audit")
        self.assertEqual(report["horizons"]["10d"]["next_step"], "可交审计做 research-only 公式型评分资产复核")


if __name__ == "__main__":
    unittest.main()
