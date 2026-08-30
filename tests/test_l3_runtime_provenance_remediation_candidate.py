import unittest

from l3_runtime_provenance_remediation_candidate import (
    build_runtime_remediation_manifest,
    discover_local_python_candidates,
    validate_self_contained_runtime,
)


def clean_evidence():
    root = "D:/runtime/standalone-python"
    return {
        "sys_executable": root + "/python.exe",
        "sys_prefix": root,
        "sys_base_prefix": root,
        "stdlib_path": root + "/Lib",
        "unittest_path": root + "/Lib/unittest/__init__.py",
        "sys_path": [root, root + "/Lib", root + "/Lib/site-packages"],
        "packages": {
            name: {"version": "fixture", "file": root + "/Lib/site-packages/" + name + "/__init__.py"}
            for name in ("duckdb", "pyarrow", "xgboost")
        },
    }


class RuntimeProvenanceRemediationTests(unittest.TestCase):
    def test_clean_standalone_fixture_passes(self):
        evidence = clean_evidence()
        result = validate_self_contained_runtime(
            evidence, runtime_root="D:/runtime/standalone-python",
            expected_executable="D:/runtime/standalone-python/python.exe",
        )
        self.assertEqual(result["status"], "passed")

    def test_conda_base_prefix_fails_closed(self):
        evidence = clean_evidence()
        evidence["sys_base_prefix"] = "C:/Users/wangj/.conda/envs/my_quant"
        result = validate_self_contained_runtime(evidence, runtime_root="D:/runtime/standalone-python")
        self.assertEqual(result["status"], "failed_closed")
        self.assertIn("sys_base_prefix_conda_provenance", result["errors"])

    def test_external_sys_path_and_package_fail_closed(self):
        evidence = clean_evidence()
        evidence["sys_path"].append("C:/Users/wangj/.conda/envs/my_quant/Lib")
        evidence["packages"]["xgboost"]["file"] = "C:/Users/wangj/.conda/envs/my_quant/xgboost.py"
        result = validate_self_contained_runtime(evidence, runtime_root="D:/runtime/standalone-python")
        self.assertEqual(result["status"], "failed_closed")
        self.assertIn("sys_path_outside_runtime_root", result["errors"])
        self.assertIn("xgboost_provenance_mismatch", result["errors"])

    def test_discovery_does_not_authorize_execution(self):
        candidates = discover_local_python_candidates(["D:/path/that/does/not/exist"])
        self.assertEqual(candidates, [])

    def test_manifest_preserves_no_action_flags(self):
        evidence = clean_evidence()
        validation = validate_self_contained_runtime(evidence, runtime_root="D:/runtime/standalone-python")
        manifest = build_runtime_remediation_manifest(
            current_evidence=evidence, discovered_candidates=[], runtime_validation=validation
        )
        self.assertFalse(manifest["candidate_runtime_created"])
        self.assertFalse(manifest["download_performed"])
        self.assertFalse(manifest["business_assets_written"])
        self.assertFalse(manifest["probe_started"])
        self.assertFalse(manifest["allow_probe_only"])
        self.assertTrue(manifest["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
