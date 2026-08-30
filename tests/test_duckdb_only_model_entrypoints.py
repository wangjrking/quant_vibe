import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DuckDBOnlyModelEntrypointsTests(unittest.TestCase):
    def test_run_pdb_update_rejects_legacy_cli_modes(self):
        run_pdb_update = importlib.import_module("run_pdb_update")
        with self.assertRaises(SystemExit):
            run_pdb_update.parse_args(["--feature-source", "legacy_mixed"])
        with self.assertRaises(SystemExit):
            run_pdb_update.parse_args(["--prediction-output-mode", "legacy_odb"])

    def test_run_mlp_pdb_update_rejects_legacy_cli_modes(self):
        fake_mlp_module = ModuleType("mlp_model_module")
        fake_mlp_module.build_prediction_frame = lambda *args, **kwargs: None
        fake_mlp_module.predict_with_mlp = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"mlp_model_module": fake_mlp_module}):
            run_mlp_pdb_update = importlib.import_module("run_mlp_pdb_update")
            with self.assertRaises(SystemExit):
                run_mlp_pdb_update.parse_args(["--feature-source", "legacy_mixed"])
            with self.assertRaises(SystemExit):
                run_mlp_pdb_update.parse_args(["--prediction-output-mode", "legacy_odb"])

    def test_run_light_pdb_update_rejects_legacy_prediction_mode(self):
        run_light_pdb_update = importlib.import_module("run_light_pdb_update")
        with self.assertRaises(SystemExit):
            run_light_pdb_update.parse_args(
                [
                    "--output-table",
                    "pred_x",
                    "--prediction-output-mode",
                    "legacy_odb",
                ]
            )

    def test_run_parallel_expanding2010_folds_rejects_legacy_cli_modes(self):
        run_parallel_expanding2010_folds = importlib.import_module("run_parallel_expanding2010_folds")
        base_args = [
            "--data-file-url",
            "data_file",
            "--label",
            "executable_10d_open_return",
            "--model-type",
            "reg",
            "--output-table",
            "pred_x",
            "--output-dir",
            "out_dir",
            "--experiment-name",
            "unit",
        ]
        with self.assertRaises(SystemExit):
            run_parallel_expanding2010_folds.parse_args(base_args + ["--feature-source", "legacy_mixed"])
        with self.assertRaises(SystemExit):
            run_parallel_expanding2010_folds.parse_args(base_args + ["--prediction-output-mode", "legacy_odb"])


if __name__ == "__main__":
    unittest.main()
