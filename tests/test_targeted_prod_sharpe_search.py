import json
import tempfile
import unittest
from pathlib import Path

from targeted_prod_sharpe_search import (
    build_juejin_command,
    build_param_grid,
    expand_followup_grid,
    normalize_indicator,
    pick_qualified_candidates,
    resolve_juejin_python,
    select_seed_rows,
)


class TargetedProdSharpeSearchTests(unittest.TestCase):
    def test_build_param_grid_stays_within_prod_micro_tune_scope(self):
        rows = build_param_grid()

        self.assertTrue(rows)
        self.assertTrue(all(row["top_k"] == 1 for row in rows))
        self.assertTrue(all(row["max_positions"] == 1 for row in rows))
        self.assertEqual(sorted({row["holding_days"] for row in rows}), [4, 5, 6])
        self.assertEqual(
            sorted({row["score_exit_entry_ratio"] for row in rows}),
            [0.88, 0.9, 0.92, 0.93, 0.95, 0.97, 1.0],
        )
        self.assertEqual(sorted({row["min_holding_days_before_score_exit"] for row in rows}), [1, 2, 3])

    def test_pick_qualified_candidates_prefers_sharpe_and_applies_goal_thresholds(self):
        rows = [
            {"name": "low_sharpe", "pnl_ratio_annual": 3.5, "sharp_ratio": 1.9},
            {"name": "qualified_but_lower", "pnl_ratio_annual": 3.2, "sharp_ratio": 2.01},
            {"name": "qualified_best", "pnl_ratio_annual": 4.1, "sharp_ratio": 2.2},
        ]

        winners = pick_qualified_candidates(rows)

        self.assertEqual([row["name"] for row in winners], ["qualified_best", "qualified_but_lower"])

    def test_expand_followup_grid_keeps_prod_shape_for_liquidity_and_quantile_rounds(self):
        base_rows = [
            {
                "top_k": 1,
                "max_positions": 1,
                "holding_days": 5,
                "score_exit_entry_ratio": 0.95,
                "min_holding_days_before_score_exit": 2,
                "min_amount": 800000.0,
                "min_turnover_rate": 2.0,
                "max_total_mv": 1000000.0,
                "min_pred_quantile": 0.95,
            }
        ]

        liquidity_rows = expand_followup_grid(base_rows, stage="liquidity")
        quantile_rows = expand_followup_grid(base_rows, stage="quantile")

        self.assertTrue(liquidity_rows)
        self.assertTrue(all(row["top_k"] == 1 for row in liquidity_rows))
        self.assertTrue(all(row["max_positions"] == 1 for row in liquidity_rows))
        self.assertEqual(sorted({row["min_amount"] for row in liquidity_rows}), [800000.0, 1000000.0, 1200000.0])
        self.assertEqual(sorted({row["min_turnover_rate"] for row in liquidity_rows}), [2.0, 3.0, 4.0])
        self.assertEqual(sorted({row["max_total_mv"] for row in liquidity_rows}), [800000.0, 1000000.0, 1200000.0])
        self.assertEqual(sorted({row["min_pred_quantile"] for row in quantile_rows}), [0.95, 0.96, 0.97, 0.98])

    def test_expand_followup_grid_supports_stability_stage(self):
        base_rows = [
            {
                "top_k": 1,
                "max_positions": 1,
                "holding_days": 5,
                "score_exit_entry_ratio": 0.9,
                "min_holding_days_before_score_exit": 2,
                "min_amount": 1000000.0,
                "min_turnover_rate": 2.0,
                "max_total_mv": 800000.0,
                "min_pred_quantile": 0.95,
            }
        ]

        stability_rows = expand_followup_grid(base_rows, stage="stability")

        self.assertTrue(stability_rows)
        self.assertTrue(all(row["top_k"] == 1 for row in stability_rows))
        self.assertEqual(sorted({row["max_atr_ratio"] for row in stability_rows}, key=lambda value: (value is not None, value)), [None, 0.06, 0.08, 0.1])
        self.assertEqual(sorted({row["stop_loss_pct"] for row in stability_rows}), [0.05, 0.06, 0.08])
        self.assertEqual(sorted({row["score_continue_entry_ratio"] for row in stability_rows}), [0.95, 1.0, 1.05])

    def test_resolve_juejin_python_prefers_reproduction_workspace_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reproduction_path = Path(tmpdir) / "reproduction_v1_0.json"
            reproduction_path.write_text(
                json.dumps(
                    {
                        "workspace": {
                            "python": "C:/Users/wangj/.conda/envs/my_quant/python.exe",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            python_path = resolve_juejin_python(reproduction_path)

        self.assertEqual(
            python_path,
            Path("C:/Users/wangj/.conda/envs/my_quant/python.exe"),
        )

    def test_normalize_indicator_can_parse_datetime_payload_string(self):
        payload = (
            "{'pnl_ratio': 10.1, 'pnl_ratio_annual': 4.8, 'sharp_ratio': 2.05, "
            "'created_at': datetime.datetime(2026, 6, 17, 12, 13, 7)}"
        )

        indicator = normalize_indicator(payload)

        self.assertEqual(indicator["pnl_ratio"], 10.1)
        self.assertEqual(indicator["pnl_ratio_annual"], 4.8)
        self.assertEqual(indicator["sharp_ratio"], 2.05)

    def test_select_seed_rows_skips_turnover_only_duplicates(self):
        rows = [
            {
                "holding_days": 5,
                "score_exit_entry_ratio": 0.93,
                "min_holding_days_before_score_exit": 2,
                "min_amount": 1000000.0,
                "min_turnover_rate": 2.0,
                "max_total_mv": 800000.0,
                "min_pred_quantile": 0.95,
                "pnl_ratio_annual": 13.16,
                "sharp_ratio": 1.84,
            },
            {
                "holding_days": 5,
                "score_exit_entry_ratio": 0.93,
                "min_holding_days_before_score_exit": 2,
                "min_amount": 1000000.0,
                "min_turnover_rate": 3.0,
                "max_total_mv": 800000.0,
                "min_pred_quantile": 0.95,
                "pnl_ratio_annual": 13.16,
                "sharp_ratio": 1.84,
            },
            {
                "holding_days": 5,
                "score_exit_entry_ratio": 0.95,
                "min_holding_days_before_score_exit": 2,
                "min_amount": 800000.0,
                "min_turnover_rate": 2.0,
                "max_total_mv": 800000.0,
                "min_pred_quantile": 0.95,
                "pnl_ratio_annual": 12.98,
                "sharp_ratio": 1.79,
            },
        ]

        picked = select_seed_rows(rows, limit=3)

        self.assertEqual(len(picked), 2)
        self.assertEqual(picked[0]["min_turnover_rate"], 2.0)

    def test_build_juejin_command_includes_optional_risk_controls(self):
        params = {
            "top_k": 1,
            "max_positions": 1,
            "holding_days": 5,
            "score_exit_entry_ratio": 0.9,
            "min_holding_days_before_score_exit": 2,
            "score_continue_entry_ratio": 1.05,
            "stop_loss_pct": 0.06,
        }

        command = build_juejin_command(Path("signals.csv"), Path("run.log"), params)

        self.assertIn("--stop-loss-pct", command)
        self.assertIn("0.06", command)
        self.assertIn("--score-continue-entry-ratio", command)
        self.assertIn("1.05", command)
        self.assertIn("--score-exit-entry-ratio", command)
        self.assertIn("0.9", command)


if __name__ == "__main__":
    unittest.main()
