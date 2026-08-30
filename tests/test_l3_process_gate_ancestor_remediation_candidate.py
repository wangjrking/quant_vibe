import unittest

from l3_process_gate_ancestor_remediation_candidate import classify_runtime_shell_ancestor


RUNTIME = r"D:\work\quant\quant_mcp\runtime_candidates\my_quant_copy_20260804\python.exe"
SCRIPT = r"D:\work\quant\quant_mcp\quant\main\deliver_l3_target_date_duckdb_mainline.py"
RUN_ID = "incremental-trading-signal-20260806-L3-target-date-incremental-feature-delivery-compliant-runtime-r1"
WORKSPACE = r"D:\work\quant\quant_mcp\workspace"
REPORT = r"D:\work\quant\quant_mcp\report"


def row(pid, ppid, name, exe, command, create_time):
    return {"pid": pid, "ppid": ppid, "name": name, "exe": exe, "cmdline": command, "create_time": create_time}


def records(runtime=RUNTIME, shell_command=None, current_exe=RUNTIME):
    shell_command = shell_command or (
        f"powershell.exe -Command & '{runtime}' '{SCRIPT}' --mode precheck "
        f"--workflow-run-id {RUN_ID} --workspace-dir '{WORKSPACE}' --report-dir '{REPORT}'"
    )
    return [
        row(100, 1, "powershell.exe", r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", shell_command, 10.0),
        row(200, 100, "python.exe", runtime, f"{runtime} {SCRIPT} --mode precheck --workflow-run-id {RUN_ID} --workspace-dir {WORKSPACE} --report-dir {REPORT}", 11.0),
        row(300, 200, "python.exe", current_exe, f"{current_exe} {SCRIPT} --mode precheck --workflow-run-id {RUN_ID} --workspace-dir {WORKSPACE} --report-dir {REPORT}", 12.0),
    ]


class RuntimeShellAncestorCandidateTests(unittest.TestCase):
    def test_exact_runtime_shell_ancestor_is_allowed(self):
        result = classify_runtime_shell_ancestor(records(), current_pid=300, approved_runtime=RUNTIME, script_path=SCRIPT, workflow_run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["process_role"], "approved_runtime_shell_launcher_ancestor")

    def test_conda_current_process_is_blocked(self):
        result = classify_runtime_shell_ancestor(records(current_exe=r"C:\Users\wangj\.conda\envs\my_quant\python.exe"), current_pid=300, approved_runtime=RUNTIME, script_path=SCRIPT, workflow_run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT)
        self.assertFalse(result["passed"])

    def test_shell_not_in_current_lineage_is_blocked(self):
        rows = records() + [row(400, 1, "powershell.exe", r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", "powershell.exe", 13.0)]
        result = classify_runtime_shell_ancestor(rows, current_pid=300, approved_runtime=RUNTIME, script_path=SCRIPT, workflow_run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT)
        self.assertTrue(result["passed"])

    def test_shell_missing_command_role_is_blocked(self):
        result = classify_runtime_shell_ancestor(records(shell_command="powershell.exe -Command Get-Date"), current_pid=300, approved_runtime=RUNTIME, script_path=SCRIPT, workflow_run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT)
        self.assertFalse(result["passed"])

    def test_missing_create_time_is_blocked(self):
        rows = records()
        rows[0]["create_time"] = None
        result = classify_runtime_shell_ancestor(rows, current_pid=300, approved_runtime=RUNTIME, script_path=SCRIPT, workflow_run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT)
        self.assertFalse(result["passed"])

    def test_duplicate_pid_is_blocked(self):
        rows = records() + [row(300, 200, "python.exe", RUNTIME, "duplicate", 12.1)]
        result = classify_runtime_shell_ancestor(rows, current_pid=300, approved_runtime=RUNTIME, script_path=SCRIPT, workflow_run_id=RUN_ID, workspace=WORKSPACE, report_dir=REPORT)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
