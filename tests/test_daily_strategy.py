import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from daily_strategy import (
    DataState,
    StageResult,
    should_update_factors,
    should_update_predictions,
    run_daily,
)


class DailyStrategyTests(unittest.TestCase):
    def test_factor_update_only_when_source_is_newer(self):
        self.assertTrue(should_update_factors(DataState("20260604", "20260603", "20260603")))
        self.assertFalse(should_update_factors(DataState("20260604", "20260604", "20260603")))
        self.assertFalse(should_update_factors(DataState(None, "20260604", "20260603")))

    def test_prediction_update_when_missing_old_or_forced(self):
        self.assertTrue(should_update_predictions(DataState("20260604", "20260604", "20260603")))
        self.assertTrue(should_update_predictions(DataState("20260604", "20260604", "20260604"), force=True))
        self.assertFalse(should_update_predictions(DataState("20260604", "20260604", "20260604")))

    def test_run_daily_skips_expensive_stages_when_dates_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                project_dir=".",
                data_dir=temp_dir,
                python="python",
                source_command=None,
                force_prediction=False,
                top_k=3,
                min_pred=0.01,
                max_atr_ratio=0.05,
                output=None,
            )
            state = DataState("20260604", "20260604", "20260604")
            calls = []

            def fake_run_selection(*_args, **_kwargs):
                calls.append("selection")
                return StageResult("daily_selection", "ok", "done")

            with patch("daily_strategy.load_data_state_for_label", return_value=state), patch(
                "daily_strategy.run_command"
            ) as run_command, patch("daily_strategy.run_selection", fake_run_selection):
                summary = run_daily(args)

            run_command.assert_not_called()
            self.assertEqual(calls, ["selection"])
            self.assertEqual(summary["results"][0]["name"], "factor_update")
            self.assertEqual(summary["results"][0]["status"], "skipped")
            self.assertEqual(summary["results"][1]["name"], "prediction_update")
            self.assertEqual(summary["results"][1]["status"], "skipped")

    def test_run_daily_updates_factor_then_prediction_when_source_advances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                project_dir=".",
                data_dir=temp_dir,
                python="python",
                source_command=None,
                force_prediction=False,
                top_k=3,
                min_pred=0.01,
                max_atr_ratio=0.05,
                output=str(Path(temp_dir) / "selection.csv"),
            )
            states = [
                DataState("20260604", "20260603", "20260603"),
                DataState("20260604", "20260604", "20260603"),
                DataState("20260604", "20260604", "20260604"),
                DataState("20260604", "20260604", "20260604"),
            ]
            commands = []

            def fake_run_command(command, *_args):
                commands.append(command)
                return StageResult("stage", "ok", "done")

            with patch("daily_strategy.load_data_state_for_label", side_effect=states), patch(
                "daily_strategy.run_command", fake_run_command
            ), patch("daily_strategy.run_selection", return_value=StageResult("daily_selection", "ok", "done")):
                summary = run_daily(args)

            self.assertEqual(commands[0], ["python", "run_incremental_cdb_update.py"])
            self.assertEqual(commands[1], ["python", "run_repair_factor_types.py"])
            self.assertEqual(commands[2][:4], ["python", "run_pdb_update.py", "--label", "10d_yield_rate"])
            self.assertEqual(summary["results"][-1]["name"], "daily_selection")


if __name__ == "__main__":
    unittest.main()
