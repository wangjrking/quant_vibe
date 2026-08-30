import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_four_year_3d_hybrid_refine_candidate_package_20260629 as target


class BuildFourYear3dHybridRefineCandidatePackage20260629Tests(unittest.TestCase):
    def test_candidate_package_keeps_research_only_boundary(self):
        self.assertEqual(target.APPROVAL_STATUS, "research_only_not_approved_for_l4_or_l5")

    def test_candidate_package_targets_refined_candidate_table(self):
        self.assertIn("hybrid_refine_candidate", target.TABLE)
        self.assertIn("clear_replacement_gate", target.BASELINE_TABLE)


if __name__ == "__main__":
    unittest.main()
