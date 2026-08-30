"""Fail-closed checks for the governed ``quant/main`` source layout.

This tool reads filenames and policy JSON only. It never imports project
modules or touches data, model, strategy, signal, or runtime assets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATE_TOKEN = re.compile(r"(?:^|_)20\d{6}(?:_|\.py$)")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_policy(main_dir: Path) -> dict[str, Any]:
    path = main_dir / "config" / "source_layout_policy_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_layout(main_dir: Path, policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    root_python = sorted(path.name for path in main_dir.glob("*.py"))
    limit = int(policy["root_python_file_limit"])
    if len(root_python) > limit:
        errors.append(f"root python file limit exceeded: {len(root_python)} > {limit}")

    for name in policy["required_root_entrypoints"]:
        if not (main_dir / name).is_file():
            errors.append(f"missing required root entrypoint: {name}")

    for relative in policy["required_directories"]:
        if not (main_dir / relative).is_dir():
            errors.append(f"missing required directory: {relative}")

    allowed_dated = set(policy["allowed_root_dated_python"])
    dated = {name for name in root_python if DATE_TOKEN.search(name)}
    unexpected_dated = sorted(dated - allowed_dated)
    if unexpected_dated:
        errors.append("unexpected dated root scripts: " + ", ".join(unexpected_dated))

    if policy.get("root_research_prefix_exception_only"):
        unexpected_research = sorted(
            name for name in root_python if name.startswith("research_") and name not in allowed_dated
        )
        if unexpected_research:
            errors.append("unexpected root research scripts: " + ", ".join(unexpected_research))
    return errors


def run_check(root: Path) -> list[str]:
    main_dir = root / "quant" / "main"
    if not main_dir.is_dir():
        return [f"missing quant/main: {main_dir}"]
    try:
        policy = load_policy(main_dir)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return [f"invalid source layout policy: {exc}"]
    return validate_layout(main_dir, policy)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the quant/main source layout.")
    parser.add_argument("--project-root", default=str(project_root()))
    args = parser.parse_args()
    errors = run_check(Path(args.project_root).resolve())
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS check_source_layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
