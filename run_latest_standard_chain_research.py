from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_research_fusion_library import resolve_prediction_inputs
from tune_standard_chain_full_investment import resolve_fusion_assets


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT.parent / "data_file" / "reports"


def _default_run_root() -> Path:
    return REPORTS_DIR / "strategy_agent_model_application_20260620" / "latest_standard_chain_research"


def parse_args(argv=None):
    default_root = _default_run_root()
    parser = argparse.ArgumentParser(
        description="Build the latest fusion library from standard-chain manifests, then run standard-chain tuning."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fusion-output-dir", default=str(default_root / "fusion"))
    parser.add_argument("--tune-report-dir", default=str(default_root / "tuning"))
    parser.add_argument("--min-trade-date", default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_commands(
    *,
    python_executable: Path,
    fusion_output_dir: Path,
    tune_report_dir: Path,
    min_trade_date: str | None,
    limit: int,
) -> list[dict]:
    python_value = python_executable.as_posix()
    build_command = [
        python_value,
        str(ROOT / "build_research_fusion_library.py"),
        "--output-dir",
        str(fusion_output_dir),
    ]
    tune_command = [
        python_value,
        str(ROOT / "tune_standard_chain_full_investment.py"),
        "--report-dir",
        str(tune_report_dir),
        "--limit",
        str(limit),
    ]
    if min_trade_date not in (None, ""):
        build_command.extend(["--min-trade-date", str(min_trade_date)])
        tune_command.extend(
            [
                "--min-standard-trade-date",
                str(min_trade_date),
                "--end-date",
                str(min_trade_date),
            ]
        )
    return [
        {"name": "build_research_fusion_library", "command": build_command},
        {"name": "tune_standard_chain_full_investment", "command": tune_command},
    ]


def preflight_readiness(min_trade_date: str | None) -> dict:
    checks = []

    try:
        prediction_inputs = resolve_prediction_inputs(
            min_trade_date=(None if min_trade_date in (None, "") else str(min_trade_date))
        )
        checks.append(
            {
                "name": "prediction_inputs",
                "status": "ok",
                "detail": {
                    "table_3d": prediction_inputs.get("table_3d"),
                    "table_5d": prediction_inputs.get("table_5d"),
                    "table_10d": prediction_inputs.get("table_10d"),
                    "max_trade_date_3d": prediction_inputs.get("max_trade_date_3d"),
                    "max_trade_date_5d": prediction_inputs.get("max_trade_date_5d"),
                    "max_trade_date_10d": prediction_inputs.get("max_trade_date_10d"),
                },
            }
        )
    except Exception as exc:
        checks.append({"name": "prediction_inputs", "status": "blocked", "detail": str(exc)})

    try:
        fusion_assets = resolve_fusion_assets(
            min_trade_date=(None if min_trade_date in (None, "") else str(min_trade_date))
        )
        checks.append(
            {
                "name": "fusion_assets",
                "status": "ok",
                "detail": {
                    "assets": [
                        {
                            "asset": row.get("asset"),
                            "table": row.get("table"),
                            "max_trade_date": row.get("max_trade_date"),
                        }
                        for row in fusion_assets
                    ]
                },
            }
        )
    except Exception as exc:
        detail = str(exc)
        status = "stale_rebuild_required" if "stale fusion manifest" in detail else "blocked"
        checks.append({"name": "fusion_assets", "status": status, "detail": detail})

    return {
        "ready": not any(row["status"] == "blocked" for row in checks),
        "checks": checks,
    }


def main(argv=None):
    args = parse_args(argv)
    min_trade_date = None if args.min_trade_date in (None, "") else str(args.min_trade_date)
    preflight = preflight_readiness(min_trade_date)
    commands = build_commands(
        python_executable=Path(args.python),
        fusion_output_dir=Path(args.fusion_output_dir),
        tune_report_dir=Path(args.tune_report_dir),
        min_trade_date=min_trade_date,
        limit=int(args.limit),
    )
    summary = {
        "status": "dry_run" if args.dry_run else "ok",
        "preflight": preflight,
        "steps": [{"name": row["name"], "command": row["command"]} for row in commands],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    if not preflight["ready"]:
        raise RuntimeError(f"preflight failed: {json.dumps(preflight, ensure_ascii=False)}")

    for row in commands:
        subprocess.run(row["command"], cwd=str(ROOT), check=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
