from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
from typing import Any

from rolling_train_module import build_rolling_windows
from run_parallel_expanding2010_folds import parse_args as parse_run_parallel_args


REQUIRED_BOUNDARIES = [
    "no_training_executed_by_plan_generation",
    "no_prediction_generated_by_plan_generation",
    "no_production_manifest_change",
    "no_signal",
    "no_backtest",
]

FORBIDDEN_COMMAND_TERMS = [
    "production-manifest",
    "production_tasks",
    "approved_for_l5",
    "formal",
    "backtest",
    "signal",
    "odb.db",
    "stock_factor_data.parquet",
    "standard_factor_by_date_parts",
]


def _run_parallel_argv(command: str) -> list[str]:
    parts = shlex.split(command, posix=False)
    for index, part in enumerate(parts):
        if str(part).replace("\\", "/").endswith("run_parallel_expanding2010_folds.py"):
            return parts[index + 1 :]
    return []


def validate_experiment_plan(plan: dict[str, Any], *, expected_final_test: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    window_summaries: list[dict[str, Any]] = []

    if plan.get("approval_status") != "research_plan_not_approved_for_execution":
        errors.append("approval_status must be research_plan_not_approved_for_execution")

    boundaries = plan.get("boundaries", {})
    for key in REQUIRED_BOUNDARIES:
        if boundaries.get(key) is not True:
            errors.append(f"boundary {key} must be true")

    experiments = plan.get("experiments", [])
    if not experiments:
        warnings.append("experiments is empty")

    for index, item in enumerate(experiments, start=1):
        horizon = str(item.get("horizon") or f"experiment_{index}")
        if item.get("requires_training_authorization") is not True:
            errors.append(f"{horizon} requires_training_authorization must be true")

        output_table = str(item.get("output_table") or "")
        if "_research" not in output_table:
            errors.append(f"{horizon} output_table must contain _research")
        if "formal" in output_table:
            errors.append(f"{horizon} output_table must not contain formal")
        if "production" in output_table:
            errors.append(f"{horizon} output_table must not contain production")

        output_dir = str(item.get("output_dir") or "")
        if "reports" not in output_dir.replace("\\", "/"):
            errors.append(f"{horizon} output_dir must be under reports")

        command = str(item.get("command") or "")
        if not command:
            errors.append(f"{horizon} command is required")
        if "--output-table" not in command:
            errors.append(f"{horizon} command must include --output-table")
        if output_table and output_table not in command:
            errors.append(f"{horizon} command must reference output_table")
        if "--skip-existing" not in command:
            errors.append(f"{horizon} command must include --skip-existing")
        if "--execute" in command:
            errors.append(f"{horizon} command must not include --execute in plan-only output")

        command_lower = command.lower()
        for term in FORBIDDEN_COMMAND_TERMS:
            if term.lower() in command_lower:
                errors.append(f"{horizon} command contains forbidden term: {term}")

        parts = shlex.split(command, posix=False)
        if "--train-years" not in parts:
            errors.append(f"{horizon} command must include --train-years")
        else:
            idx = parts.index("--train-years")
            value = parts[idx + 1] if idx + 1 < len(parts) else ""
            try:
                if int(value) <= 0:
                    errors.append(f"{horizon} command --train-years must be positive")
            except ValueError:
                errors.append(f"{horizon} command --train-years must be an integer")

        run_parallel_argv = _run_parallel_argv(command)
        if not run_parallel_argv:
            errors.append(f"{horizon} command must call run_parallel_expanding2010_folds.py")
        else:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    parsed = parse_run_parallel_args(run_parallel_argv)
            except SystemExit:
                errors.append(f"{horizon} command does not parse with run_parallel_expanding2010_folds.py")
            else:
                if expected_final_test and str(parsed.final_test) != str(expected_final_test):
                    errors.append(f"{horizon} final-test must be {expected_final_test}")
                try:
                    windows = build_rolling_windows(
                        data_start=parsed.data_start,
                        first_test=parsed.first_test,
                        final_test=parsed.final_test,
                        train_years=parsed.train_years,
                        test_months=parsed.test_months,
                        step_months=parsed.step_months,
                        embargo_days=parsed.embargo_days,
                        train_mode="expanding",
                    )
                except Exception as exc:
                    errors.append(f"{horizon} rolling windows cannot be built: {exc}")
                else:
                    if not windows:
                        errors.append(f"{horizon} rolling windows must not be empty")
                    else:
                        window_summaries.append(
                            {
                                "horizon": horizon,
                                "folds": len(windows),
                                "first_test_start": windows[0].test_start,
                                "last_test_end": windows[-1].test_end,
                                "first_train_start": windows[0].train_start,
                                "last_train_end": windows[-1].train_end,
                            }
                        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "experiments_checked": len(experiments),
        "window_summaries": window_summaries,
        "skipped_validation_only": plan.get("skipped_validation_only", []),
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate V12 research experiment plan safety.")
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--expected-final-test")
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
    result = validate_experiment_plan(plan, expected_final_test=args.expected_final_test)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
