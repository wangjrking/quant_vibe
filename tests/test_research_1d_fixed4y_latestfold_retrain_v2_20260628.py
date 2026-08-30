import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research_1d_fixed4y_latestfold_retrain_v2_20260628 as target


class Research1dFixed4yLatestfoldRetrainV2Tests(unittest.TestCase):
    def test_build_command_uses_correct_1d_embargo(self):
        candidate = target.candidates()[0]
        command = target.build_command(target.LABEL_CONFIG, candidate, fold=15)
        idx = command.index("--embargo-days")
        self.assertEqual(command[idx + 1], "1")

    def test_build_command_uses_standard_chain_split_features(self):
        candidate = target.candidates()[0]
        command = target.build_command(target.LABEL_CONFIG, candidate, fold=15)
        self.assertIn("--use-light-factor-data", command)
        idx = command.index("--feature-source")
        self.assertEqual(command[idx + 1], "production_split")

    def test_fold_prediction_paths_are_unique_per_fold(self):
        candidate = target.candidates()[0]
        fold15 = target.fold_prediction_path(candidate, 15)
        fold16 = target.fold_prediction_path(candidate, 16)
        self.assertNotEqual(fold15, fold16)
        self.assertTrue(str(fold15).endswith("fold15.parquet"))
        self.assertTrue(str(fold16).endswith("fold16.parquet"))

    def test_build_command_targets_single_fold_specific_output(self):
        candidate = target.candidates()[0]
        command = target.build_command(target.LABEL_CONFIG, candidate, fold=15)
        start_idx = command.index("--start-fold")
        end_idx = command.index("--end-fold")
        path_idx = command.index("--prediction-output-path")
        self.assertEqual(command[start_idx + 1], "15")
        self.assertEqual(command[end_idx + 1], "15")
        self.assertTrue(command[path_idx + 1].endswith("fold15.parquet"))


if __name__ == "__main__":
    unittest.main()
