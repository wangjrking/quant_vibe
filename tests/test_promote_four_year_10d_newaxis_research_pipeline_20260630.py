from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import promote_four_year_10d_newaxis_research_pipeline_20260630 as target


class PromoteFourYear10DNewaxisResearchPipeline20260630Tests(unittest.TestCase):
    def test_build_refresh_args_uses_package_report_outputs(self):
        package_report_dir = Path("D:/tmp/package_report")
        bestset_report_dir = Path("D:/tmp/bestset_report")

        args = target.build_refresh_args(
            package_report_dir=package_report_dir,
            bestset_report_dir=bestset_report_dir,
        )

        self.assertEqual(args["report_dir"], str(bestset_report_dir))
        self.assertEqual(args["candidate_json"], str(package_report_dir / "promotion_candidate.json"))
        self.assertEqual(args["gate_json"], str(package_report_dir / "promotion_gate_result.json"))
        self.assertEqual(args["summary_json"], str(package_report_dir / "newaxis_candidate_summary.json"))
        self.assertEqual(
            args["eval_json"],
            str(
                package_report_dir
                / "standard_eval"
                / "executable_10d_open_return_four_year_newaxis_train_candidate_20260630_eval_summary.json"
            ),
        )

    def test_should_refresh_bestset_requires_gate_and_replacement_rule(self):
        candidate = {
            "current_baseline_delta": {
                "full_rank_ic_delta": 0.001,
                "full_top5_delta": 0.002,
                "recent63_rank_ic_delta": 0.0,
                "recent63_top5_delta": 0.003,
                "recent20_rank_ic_delta": 0.0,
                "recent20_top5_delta": 0.004,
            }
        }
        gate = {"result": {"hard_constraint_passed": True}}
        self.assertTrue(target.should_refresh_bestset(candidate, gate))

        bad_gate = {"result": {"hard_constraint_passed": False}}
        self.assertFalse(target.should_refresh_bestset(candidate, bad_gate))

        bad_candidate = {
            "current_baseline_delta": {
                "full_rank_ic_delta": 0.001,
                "full_top5_delta": 0.002,
                "recent63_rank_ic_delta": -0.001,
                "recent63_top5_delta": 0.003,
                "recent20_rank_ic_delta": 0.0,
                "recent20_top5_delta": 0.004,
            }
        }
        self.assertFalse(target.should_refresh_bestset(bad_candidate, gate))


if __name__ == "__main__":
    unittest.main()
