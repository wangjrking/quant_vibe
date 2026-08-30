import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_four_year_candidate_priority import evaluate_priority


class EvaluateFourYearCandidatePriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "config_id": "test_policy",
            "generated_at": "2026-06-28T16:20:00+08:00",
            "default_rules": {
                "discussion_requires": [
                    "promotable_formal_candidate",
                    "full_top5_delta_positive",
                    "recent63_top5_delta_positive",
                    "recent20_top5_delta_positive",
                    "recent63_abs_top5_positive",
                    "recent20_abs_top5_positive",
                ],
                "clear_replacement_requires": [
                    "current_full_rank_ic_delta_gte_replaced",
                    "current_full_top5_delta_gte_replaced",
                ],
                "mixed_case_status": "candidate_discussion_only_requires_formal_explanation",
                "clear_case_status": "candidate_discussion_clear_replacement_case",
            },
            "labels": {
                "executable_1d_open_return": {
                    "selection_axis": "full_window_and_recent_front_rank_balance",
                    "note": "1D test note",
                },
                "executable_10d_open_return": {
                    "selection_axis": "front_rank_priority_with_recent_window_guard",
                    "note": "10D test note",
                },
            },
            "boundaries": {"no_training": True},
        }

    def test_clear_replacement_case(self):
        candidate_status = {
            "current_best_candidates": {
                "executable_1d_open_return": {
                    "asset": "asset_1d",
                    "table": "table_1d",
                    "decision": "promotable_formal_candidate",
                    "full_rank_ic_delta": 0.02,
                    "full_top5_delta": 0.03,
                    "recent63_top5_delta": 0.01,
                    "recent20_top5_delta": 0.02,
                    "replaced_full_rank_ic_delta": 0.01,
                    "replaced_full_top5_delta": 0.02,
                }
            }
        }
        comparison = {
            "labels": {
                "1d": {
                    "recent63_candidate_abs": {"top5": 0.01},
                    "recent20_candidate_abs": {"top5": 0.02, "rank_ic": 0.01},
                }
            }
        }
        stability = {"rows": [{"label": "executable_1d_open_return", "stability_score": 1.0, "optimize_priority": "low"}]}

        review = evaluate_priority(candidate_status, comparison, stability, self.policy)
        row = review["rows"][0]

        self.assertTrue(row["discussion_passed"])
        self.assertTrue(row["clear_replacement"])
        self.assertEqual(row["priority_status"], "candidate_discussion_clear_replacement_case")
        self.assertTrue(row["formal_precheck_ready"])
        self.assertFalse(row["requires_formal_explanation"])

    def test_mixed_case_requires_formal_explanation(self):
        candidate_status = {
            "current_best_candidates": {
                "executable_10d_open_return": {
                    "asset": "asset_10d",
                    "table": "table_10d",
                    "decision": "promotable_formal_candidate",
                    "full_rank_ic_delta": 0.001,
                    "full_top5_delta": 0.005,
                    "recent63_top5_delta": 0.04,
                    "recent20_top5_delta": 0.10,
                    "replaced_full_rank_ic_delta": 0.002,
                    "replaced_full_top5_delta": 0.004,
                }
            }
        }
        comparison = {
            "labels": {
                "10d": {
                    "recent63_candidate_abs": {"top5": 0.04},
                    "recent20_candidate_abs": {"top5": 0.05, "rank_ic": -0.001},
                }
            }
        }
        stability = {"rows": [{"label": "executable_10d_open_return", "stability_score": 5.0, "optimize_priority": "low"}]}

        review = evaluate_priority(candidate_status, comparison, stability, self.policy)
        row = review["rows"][0]

        self.assertTrue(row["discussion_passed"])
        self.assertFalse(row["clear_replacement"])
        self.assertEqual(row["priority_status"], "candidate_discussion_only_requires_formal_explanation")
        self.assertFalse(row["formal_precheck_ready"])
        self.assertTrue(row["requires_formal_explanation"])
        self.assertIn("RankIC", row["explanation"])


if __name__ == "__main__":
    unittest.main()
