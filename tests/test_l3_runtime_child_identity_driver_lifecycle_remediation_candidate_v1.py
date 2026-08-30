import json
import subprocess
import unittest
from unittest.mock import patch

from l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1 import (
    run_driver_lifecycle,
)


RUNTIME = "D:/runtime/standalone-python/python.exe"
DRIVER = "D:/fixture/driver.py"
PAYLOAD = {"status": "child_identity_accepted", "pid": 101, "ppid": 202}


class FakeProcess:
    pid = 444
    returncode = 0

    def communicate(self, timeout=None):
        return json.dumps(PAYLOAD), "diagnostic native stderr"

    def poll(self):
        return self.returncode


class LifecycleTests(unittest.TestCase):
    def test_zero_exit_and_stderr_are_accepted(self):
        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            return_value=FakeProcess(),
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER)
        self.assertTrue(result["passed"])
        self.assertEqual(result["evidence"]["returncode"], 0)
        self.assertTrue(result["evidence"]["stderr_present"])

    def test_missing_returncode_is_fail_closed(self):
        class UnknownExitProcess(FakeProcess):
            returncode = None

            def poll(self):
                return None

        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            return_value=UnknownExitProcess(),
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "driver_returncode_unavailable")

    def test_nonzero_exit_is_fail_closed(self):
        class FailedProcess(FakeProcess):
            returncode = 7

        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            return_value=FailedProcess(),
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "driver_returncode_nonzero")

    def test_timeout_with_cleanup_is_classified(self):
        class TimeoutProcess(FakeProcess):
            returncode = -9

            def __init__(self):
                self.killed = False

            def communicate(self, timeout=None):
                if not self.killed:
                    raise subprocess.TimeoutExpired(DRIVER, timeout)
                return "", "terminated"

            def kill(self):
                self.killed = True

        process = TimeoutProcess()
        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            return_value=process,
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER, timeout_seconds=0.1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "driver_timeout")
        self.assertEqual(result["evidence"]["cleanup_status"], "killed_and_collected")

    def test_timeout_without_deterministic_exit_is_fail_closed(self):
        class UncleanTimeoutProcess(FakeProcess):
            returncode = None

            def __init__(self):
                self.killed = False

            def communicate(self, timeout=None):
                if not self.killed:
                    raise subprocess.TimeoutExpired(DRIVER, timeout)
                return "", "terminated"

            def kill(self):
                self.killed = True

            def poll(self):
                return None

        process = UncleanTimeoutProcess()
        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            return_value=process,
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER, timeout_seconds=0.1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "driver_timeout_cleanup_incomplete")

    def test_invalid_stdout_is_fail_closed(self):
        class InvalidOutputProcess(FakeProcess):
            def communicate(self, timeout=None):
                return "not-json", ""

        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            return_value=InvalidOutputProcess(),
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "driver_structured_stdout_invalid_json")

    def test_start_error_is_classified(self):
        with patch(
            "l3_runtime_child_identity_driver_lifecycle_remediation_candidate_v1.subprocess.Popen",
            side_effect=FileNotFoundError("missing runtime"),
        ):
            result = run_driver_lifecycle(RUNTIME, DRIVER)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "driver_start_file_not_found")


if __name__ == "__main__":
    unittest.main()
