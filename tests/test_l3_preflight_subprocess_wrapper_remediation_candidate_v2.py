import json
import subprocess
import unittest
from unittest import mock

from l3_preflight_subprocess_wrapper_remediation_candidate_v2 import (
    run_structured_preflight_command,
)


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class PreflightSubprocessWrapperV2Tests(unittest.TestCase):
    @mock.patch("l3_preflight_subprocess_wrapper_remediation_candidate_v2.subprocess.run")
    def test_zero_exit_stderr_is_diagnostic_only(self, run):
        run.return_value = Completed(stdout=json.dumps({"passed": True}), stderr="PowerShell warning")
        result = run_structured_preflight_command(["python", "-c", "probe"])
        self.assertTrue(result["passed"])
        self.assertTrue(result["evidence"]["stderr_present"])
        run.assert_called_once_with(["python", "-c", "probe"], capture_output=True, text=True, check=False)

    @mock.patch("l3_preflight_subprocess_wrapper_remediation_candidate_v2.subprocess.run")
    def test_nonzero_exit_fails_closed_even_with_json_stdout(self, run):
        run.return_value = Completed(returncode=1, stdout=json.dumps({"passed": True}), stderr="error")
        result = run_structured_preflight_command(["python", "-c", "probe"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "child_returncode_nonzero")

    @mock.patch("l3_preflight_subprocess_wrapper_remediation_candidate_v2.subprocess.run")
    def test_invalid_stdout_fails_closed(self, run):
        run.return_value = Completed(stdout="not-json", stderr="warning")
        result = run_structured_preflight_command(["python", "-c", "probe"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "structured_stdout_invalid_json")

    @mock.patch("l3_preflight_subprocess_wrapper_remediation_candidate_v2.subprocess.run")
    def test_structured_failure_fails_closed(self, run):
        run.return_value = Completed(stdout=json.dumps({"passed": False, "reason": "gate"}))
        result = run_structured_preflight_command(["python", "-c", "probe"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "structured_preflight_failed")

    @mock.patch("l3_preflight_subprocess_wrapper_remediation_candidate_v2.subprocess.run")
    def test_empty_stdout_fails_closed(self, run):
        run.return_value = Completed(stderr="warning")
        result = run_structured_preflight_command(["python", "-c", "probe"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "structured_stdout_missing")

    @mock.patch("l3_preflight_subprocess_wrapper_remediation_candidate_v2.subprocess.run")
    def test_json_array_stdout_fails_closed(self, run):
        run.return_value = Completed(stdout=json.dumps([{"passed": True}]))
        result = run_structured_preflight_command(["python", "-c", "probe"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "structured_stdout_not_object")


if __name__ == "__main__":
    unittest.main()
