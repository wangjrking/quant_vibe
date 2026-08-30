import json
import unittest
from unittest.mock import patch

from l3_runtime_child_identity_capture_remediation_candidate_v2 import (
    capture_child_runtime_identity,
    interpret_child_identity_result,
)


RUNTIME = "D:/runtime/standalone-python/python.exe"
ROOT = "D:/runtime/standalone-python"


def clean_payload():
    return {
        "pid": 101,
        "ppid": 202,
        "sys_executable": RUNTIME,
        "sys_prefix": ROOT,
        "sys_base_prefix": ROOT,
        "stdlib": ROOT + "/Lib",
        "unittest": ROOT + "/Lib/unittest/__init__.py",
        "sys_path": [ROOT, ROOT + "/Lib"],
    }


class FakeProcess:
    pid = 444
    returncode = 0

    def communicate(self, timeout=None):
        return json.dumps(clean_payload()), "diagnostic native stderr"

    def kill(self):
        raise AssertionError("kill must not be called for a completed child")


class RuntimeChildIdentityTests(unittest.TestCase):
    def test_valid_stdout_with_stderr_is_accepted(self):
        result = interpret_child_identity_result(
            returncode=0,
            stdout=json.dumps(clean_payload()),
            stderr="diagnostic native stderr",
            expected_executable=RUNTIME,
            expected_runtime_root=ROOT,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["evidence"]["stderr_present"])

    def test_nonzero_exit_is_fail_closed(self):
        result = interpret_child_identity_result(
            returncode=1,
            stdout=json.dumps(clean_payload()),
            stderr="error",
            expected_executable=RUNTIME,
            expected_runtime_root=ROOT,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "child_returncode_nonzero")

    def test_invalid_stdout_is_fail_closed(self):
        result = interpret_child_identity_result(
            returncode=0,
            stdout="not-json",
            stderr="",
            expected_executable=RUNTIME,
            expected_runtime_root=ROOT,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "structured_stdout_invalid_json")

    def test_identity_mismatch_is_fail_closed(self):
        payload = clean_payload()
        payload["sys_base_prefix"] = "C:/Users/wangj/.conda/envs/my_quant"
        result = interpret_child_identity_result(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
            expected_executable=RUNTIME,
            expected_runtime_root=ROOT,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "child_base_prefix_mismatch")

    def test_capture_uses_explicit_stream_contract(self):
        with patch(
            "l3_runtime_child_identity_capture_remediation_candidate_v2.subprocess.Popen",
            return_value=FakeProcess(),
        ) as popen:
            result = capture_child_runtime_identity(RUNTIME, expected_runtime_root=ROOT)
        self.assertTrue(result["passed"])
        self.assertEqual(popen.call_args.kwargs["text"], True)
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(popen.call_args.kwargs["errors"], "strict")

    def test_timeout_is_fail_closed_and_kills_child(self):
        class TimeoutProcess(FakeProcess):
            def communicate(self, timeout=None):
                if timeout is not None:
                    from subprocess import TimeoutExpired

                    raise TimeoutExpired([RUNTIME], timeout)
                return "", ""

            def kill(self):
                self.killed = True

        process = TimeoutProcess()
        with patch(
            "l3_runtime_child_identity_capture_remediation_candidate_v2.subprocess.Popen",
            return_value=process,
        ):
            result = capture_child_runtime_identity(RUNTIME, expected_runtime_root=ROOT)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "child_identity_timeout")
        self.assertTrue(process.killed)


if __name__ == "__main__":
    unittest.main()

