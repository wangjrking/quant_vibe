import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research_3d_fixed4y_latestfold_retrain_v1_20260628 as target


class Research3dFixed4yLatestfoldRetrainV1Tests(unittest.TestCase):
    def test_build_command_uses_correct_3d_embargo(self):
        candidate = target.candidates()[0]
        command = target.build_command(target.LABEL_CONFIG, candidate, fold=16)
        idx = command.index("--embargo-days")
        self.assertEqual(command[idx + 1], "3")

    def test_build_command_uses_standard_chain_split_features(self):
        candidate = target.candidates()[0]
        command = target.build_command(target.LABEL_CONFIG, candidate, fold=16)
        self.assertIn("--use-light-factor-data", command)
        idx = command.index("--feature-source")
        self.assertEqual(command[idx + 1], "production_split")

    def test_fold_prediction_paths_are_unique_per_fold(self):
        candidate = target.candidates()[0]
        fold16 = target.fold_prediction_path(candidate, 16)
        fold17 = target.fold_prediction_path(candidate, 17)
        self.assertNotEqual(fold16, fold17)
        self.assertTrue(str(fold16).endswith("fold16.parquet"))
        self.assertTrue(str(fold17).endswith("fold17.parquet"))

    def test_build_command_targets_single_fold_specific_output(self):
        candidate = target.candidates()[0]
        command = target.build_command(target.LABEL_CONFIG, candidate, fold=16)
        start_idx = command.index("--start-fold")
        end_idx = command.index("--end-fold")
        path_idx = command.index("--prediction-output-path")
        self.assertEqual(command[start_idx + 1], "16")
        self.assertEqual(command[end_idx + 1], "16")
        self.assertTrue(command[path_idx + 1].endswith("fold16.parquet"))


if __name__ == "__main__":
    unittest.main()
