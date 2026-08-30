"""Runtime-only parent/child provenance probe; never opens business assets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import unittest


PACKAGE_NAMES = ("duckdb", "pyarrow", "xgboost")


def collect(role: str) -> dict[str, object]:
    packages: dict[str, dict[str, str]] = {}
    for name in PACKAGE_NAMES:
        module = __import__(name)
        packages[name] = {
            "version": str(getattr(module, "__version__", "unknown")),
            "file": str(getattr(module, "__file__", "")).replace("\\", "/"),
        }
    return {
        "role": role,
        "sys_executable": sys.executable.replace("\\", "/"),
        "sys_prefix": sys.prefix.replace("\\", "/"),
        "sys_base_prefix": sys.base_prefix.replace("\\", "/"),
        "stdlib_path": sysconfig.get_paths().get("stdlib", "").replace("\\", "/"),
        "unittest_path": unittest.__file__.replace("\\", "/"),
        "sys_path": [str(value).replace("\\", "/") for value in sys.path],
        "packages": packages,
    }


def main() -> None:
    if os.environ.get("L3_RUNTIME_CHILD") == "1":
        print(json.dumps(collect("child"), sort_keys=True))
        return
    parent = collect("parent")
    child = subprocess.run(
        [sys.executable, "-I", __file__],
        cwd=sys.prefix,
        env={**os.environ, "L3_RUNTIME_CHILD": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    child_payload = json.loads(child.stdout) if child.returncode == 0 else None
    print(
        json.dumps(
            {
                "probe_type": "runtime_provenance_parent_child",
                "parent": parent,
                "child_returncode": child.returncode,
                "child": child_payload,
                "child_stderr": child.stderr,
                "same_executable": bool(child_payload and child_payload.get("sys_executable") == parent["sys_executable"]),
                "business_inputs_read": False,
                "production_paths_opened": False,
                "probe_scope": "runtime_only",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
