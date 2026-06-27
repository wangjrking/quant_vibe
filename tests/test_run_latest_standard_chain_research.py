import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_latest_standard_chain_research import build_commands, main, preflight_readiness


class RunLatestStandardChainResearchTests(unittest.TestCase):
    def test_preflight_readiness_collects_prediction_and_fusion_blockers(self):
        with patch(
            "run_latest_standard_chain_research.resolve_prediction_inputs",
            side_effect=RuntimeError("stale prediction manifest for min_trade_date=20260618: 5d=20260612"),
        ), patch(
            "run_latest_standard_chain_research.resolve_fusion_assets",
            side_effect=RuntimeError("stale fusion manifest for min_trade_date=20260618: fusion=20260616"),
        ):
            preflight = preflight_readiness("20260618")

        self.assertFalse(preflight["ready"])
        self.assertEqual([row["name"] for row in preflight["checks"]], [
            "prediction_inputs",
            "fusion_assets",
        ])
        self.assertEqual(preflight["checks"][0]["status"], "blocked")
        self.assertEqual(preflight["checks"][1]["status"], "stale_rebuild_required")

    def test_build_commands_wires_fusion_then_tuning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fusion_dir = root / "fusion"
            report_dir = root / "report"

            commands = build_commands(
                python_executable=Path("C:/Python/python.exe"),
                fusion_output_dir=fusion_dir,
                tune_report_dir=report_dir,
                min_trade_date="20260618",
                limit=24,
            )

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0]["name"], "build_research_fusion_library")
        self.assertEqual(commands[1]["name"], "tune_standard_chain_full_investment")
        self.assertEqual(commands[0]["command"][0], "C:/Python/python.exe")
        self.assertIn("--output-dir", commands[0]["command"])
        self.assertIn(str(fusion_dir), commands[0]["command"])
        self.assertIn("--min-trade-date", commands[0]["command"])
        self.assertIn("20260618", commands[0]["command"])
        self.assertIn("--report-dir", commands[1]["command"])
        self.assertIn(str(report_dir), commands[1]["command"])
        self.assertIn("--min-standard-trade-date", commands[1]["command"])
        self.assertIn("--end-date", commands[1]["command"])
        self.assertIn("20260618", commands[1]["command"])
        self.assertIn("--limit", commands[1]["command"])
        self.assertIn("24", commands[1]["command"])

    def test_main_dry_run_returns_planned_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fusion_dir = root / "fusion"
            report_dir = root / "report"

            with patch(
                "run_latest_standard_chain_research.resolve_prediction_inputs",
                return_value={"table_3d": "t3", "table_5d": "t5", "table_10d": "t10"},
            ), patch(
                "run_latest_standard_chain_research.resolve_fusion_assets",
                return_value=[{"asset": "fusion_mincons", "table": "combo_rank_min_consensus"}],
            ):
                summary = main(
                    [
                        "--dry-run",
                        "--python",
                        "C:/Python/python.exe",
                        "--fusion-output-dir",
                        str(fusion_dir),
                        "--tune-report-dir",
                        str(report_dir),
                        "--min-trade-date",
                        "20260618",
                        "--limit",
                        "12",
                    ]
                )

        self.assertEqual(summary["status"], "dry_run")
        self.assertTrue(summary["preflight"]["ready"])
        self.assertEqual([step["name"] for step in summary["steps"]], [
            "build_research_fusion_library",
            "tune_standard_chain_full_investment",
        ])
        self.assertEqual(summary["steps"][0]["command"][0], "C:/Python/python.exe")

    def test_main_executes_commands_in_order(self):
        executed = []

        def _fake_run(command, cwd, check):
            executed.append({"command": list(command), "cwd": cwd, "check": check})

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fusion_dir = root / "fusion"
            report_dir = root / "report"

            with patch(
                "run_latest_standard_chain_research.resolve_prediction_inputs",
                return_value={"table_3d": "t3", "table_5d": "t5", "table_10d": "t10"},
            ), patch(
                "run_latest_standard_chain_research.resolve_fusion_assets",
                return_value=[{"asset": "fusion_mincons", "table": "combo_rank_min_consensus"}],
            ), patch("run_latest_standard_chain_research.subprocess.run", side_effect=_fake_run):
                summary = main(
                    [
                        "--python",
                        "C:/Python/python.exe",
                        "--fusion-output-dir",
                        str(fusion_dir),
                        "--tune-report-dir",
                        str(report_dir),
                        "--min-trade-date",
                        "20260618",
                        "--limit",
                        "8",
                    ]
                )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual([step["name"] for step in summary["steps"]], [
            "build_research_fusion_library",
            "tune_standard_chain_full_investment",
        ])
        self.assertEqual(len(executed), 2)
        self.assertTrue(executed[0]["command"][1].endswith("build_research_fusion_library.py"))
        self.assertTrue(executed[1]["command"][1].endswith("tune_standard_chain_full_investment.py"))
        self.assertTrue(all(row["check"] for row in executed))

    def test_main_raises_before_subprocess_when_preflight_blocked(self):
        with patch(
            "run_latest_standard_chain_research.resolve_prediction_inputs",
            side_effect=RuntimeError("stale prediction manifest for min_trade_date=20260618: 5d=20260612"),
        ), patch(
            "run_latest_standard_chain_research.resolve_fusion_assets",
            side_effect=RuntimeError("stale fusion manifest for min_trade_date=20260618: fusion=20260616"),
        ), patch("run_latest_standard_chain_research.subprocess.run") as run_mock:
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                main(["--min-trade-date", "20260618", "--limit", "0"])

        run_mock.assert_not_called()

    def test_main_executes_when_only_existing_fusion_is_stale(self):
        executed = []

        def _fake_run(command, cwd, check):
            executed.append({"command": list(command), "cwd": cwd, "check": check})

        with patch(
            "run_latest_standard_chain_research.resolve_prediction_inputs",
            return_value={"table_3d": "t3", "table_5d": "t5", "table_10d": "t10"},
        ), patch(
            "run_latest_standard_chain_research.resolve_fusion_assets",
            side_effect=RuntimeError("stale fusion manifest for min_trade_date=20260618: fusion=20260616"),
        ), patch("run_latest_standard_chain_research.subprocess.run", side_effect=_fake_run):
            summary = main(["--min-trade-date", "20260618", "--limit", "0"])

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["preflight"]["checks"][0]["status"], "ok")
        self.assertEqual(summary["preflight"]["checks"][1]["status"], "stale_rebuild_required")
        self.assertEqual(len(executed), 2)


if __name__ == "__main__":
    unittest.main()
