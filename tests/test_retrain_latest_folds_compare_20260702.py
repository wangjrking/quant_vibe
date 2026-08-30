import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retrain_latest_folds_compare_20260702 as target


class RetrainLatestFoldsCompare20260702Tests(unittest.TestCase):
    def test_resolve_saved_model_path_supports_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            model_path.write_text("{}", encoding="utf-8")
            metadata = {"model_path": str(model_path)}

            resolved = target.resolve_saved_model_path(metadata)

        self.assertEqual(resolved, model_path)

    def test_resolve_saved_model_path_supports_legacy_quant_main_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            quant_main = workspace / "quant" / "main"
            quant_main.mkdir(parents=True, exist_ok=True)
            report_dir = workspace / "quant" / "data_file" / "reports" / "demo" / "models"
            report_dir.mkdir(parents=True, exist_ok=True)
            model_path = report_dir / "model_fold01.json"
            model_path.write_text("{}", encoding="utf-8")
            metadata_path = report_dir / "model_fold01_metadata.json"
            metadata_payload = {
                "model_path": r"..\data_file\reports\demo\models\model_fold01.json",
                "metadata_path": str(metadata_path),
            }
            metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False), encoding="utf-8")

            original_root = target.ROOT
            original_script_dir = target.SCRIPT_DIR
            try:
                target.ROOT = workspace
                target.SCRIPT_DIR = quant_main
                resolved = target.resolve_saved_model_path(metadata_payload)
            finally:
                target.ROOT = original_root
                target.SCRIPT_DIR = original_script_dir

        self.assertEqual(resolved, model_path)


if __name__ == "__main__":
    unittest.main()
