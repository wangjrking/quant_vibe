from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import build_four_year_research_frontier_status_v20_20260630 as build_frontier
import package_four_year_10d_newaxis_candidate_20260630 as package_candidate
import refresh_four_year_latest_bestset_status_v16_20260630 as refresh_bestset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DEFAULT_EXPERIMENT_DIR = package_candidate.DEFAULT_EXPERIMENT_DIR
DEFAULT_ASSET = package_candidate.DEFAULT_ASSET
DEFAULT_BESTSET_REPORT_DIR = refresh_bestset.REPORT_DIR
DEFAULT_FRONTIER_REPORT_DIR = build_frontier.REPORT_DIR
CN_TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(CN_TZ).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a completed 10D four-year new-axis experiment and, if it passes replacement rules, refresh research bestset/frontier."
    )
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--output-table", default="")
    parser.add_argument("--package-report-dir", default="")
    parser.add_argument("--bestset-report-dir", default=str(DEFAULT_BESTSET_REPORT_DIR))
    parser.add_argument("--frontier-report-dir", default=str(DEFAULT_FRONTIER_REPORT_DIR))
    return parser.parse_args(argv)


def standard_eval_dir(package_report_dir: Path) -> Path:
    return package_report_dir / "standard_eval"


def build_refresh_args(*, package_report_dir: Path, bestset_report_dir: Path) -> dict[str, str]:
    eval_dir = standard_eval_dir(package_report_dir)
    return {
        "report_dir": str(bestset_report_dir),
        "candidate_json": str(package_report_dir / "promotion_candidate.json"),
        "gate_json": str(package_report_dir / "promotion_gate_result.json"),
        "summary_json": str(package_report_dir / "newaxis_candidate_summary.json"),
        "eval_json": str(
            eval_dir / "executable_10d_open_return_four_year_newaxis_train_candidate_20260630_eval_summary.json"
        ),
        "candidate_annual_csv": str(
            eval_dir / "executable_10d_open_return_four_year_newaxis_train_candidate_20260630_annual_eval.csv"
        ),
        "candidate_monthly_control_csv": str(
            eval_dir
            / "executable_10d_open_return_four_year_newaxis_train_candidate_20260630_vs_four_year_control_10d_monthly_delta.csv"
        ),
        "control_annual_csv": str(refresh_bestset.CONTROL_10D_ANNUAL),
        "prior_status": str(refresh_bestset.PRIOR_STATUS),
    }


def build_frontier_args(
    *,
    package_report_dir: Path,
    bestset_report_dir: Path,
    frontier_report_dir: Path,
) -> dict[str, str]:
    return {
        "report_dir": str(frontier_report_dir),
        "bestset_json": str(bestset_report_dir / "latest_bestset_status_v16.json"),
        "previous_frontier_json": str(build_frontier.FRONTIER_V19),
        "newaxis_summary_json": str(package_report_dir / "newaxis_candidate_summary.json"),
    }


def should_refresh_bestset(candidate_payload: dict[str, Any], gate_payload: dict[str, Any]) -> bool:
    return refresh_bestset.should_replace_10d_bestset(
        candidate_payload["current_baseline_delta"],
        gate_passed=bool(gate_payload["result"]["hard_constraint_passed"]),
    )


def write_pipeline_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    package_report_dir = Path(args.package_report_dir) if args.package_report_dir else experiment_dir
    bestset_report_dir = Path(args.bestset_report_dir)
    frontier_report_dir = Path(args.frontier_report_dir)

    package_argv = [
        "--experiment-dir",
        str(experiment_dir),
        "--asset",
        str(args.asset),
        "--report-dir",
        str(package_report_dir),
    ]
    if args.output_table:
        package_argv.extend(["--output-table", str(args.output_table)])

    package_candidate.main(package_argv)

    candidate_payload = json.loads((package_report_dir / "promotion_candidate.json").read_text(encoding="utf-8"))
    gate_payload = json.loads((package_report_dir / "promotion_gate_result.json").read_text(encoding="utf-8"))

    pipeline_status: dict[str, Any] = {
        "generated_at": now_iso(),
        "scope": "research_only_four_year_10d_newaxis_promotion_pipeline",
        "experiment_dir": str(experiment_dir),
        "package_report_dir": str(package_report_dir),
        "asset": str(args.asset),
        "gate_hard_constraint_passed": bool(gate_payload["result"]["hard_constraint_passed"]),
        "replacement_rule_passed": should_refresh_bestset(candidate_payload, gate_payload),
        "bestset_refreshed": False,
        "frontier_refreshed": False,
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    if pipeline_status["replacement_rule_passed"]:
        refresh_args = build_refresh_args(
            package_report_dir=package_report_dir,
            bestset_report_dir=bestset_report_dir,
        )
        refresh_bestset.main(
            [
                "--report-dir",
                refresh_args["report_dir"],
                "--prior-status",
                refresh_args["prior_status"],
                "--candidate-json",
                refresh_args["candidate_json"],
                "--gate-json",
                refresh_args["gate_json"],
                "--summary-json",
                refresh_args["summary_json"],
                "--eval-json",
                refresh_args["eval_json"],
                "--candidate-annual-csv",
                refresh_args["candidate_annual_csv"],
                "--candidate-monthly-control-csv",
                refresh_args["candidate_monthly_control_csv"],
                "--control-annual-csv",
                refresh_args["control_annual_csv"],
            ]
        )
        pipeline_status["bestset_refreshed"] = True

        frontier_args = build_frontier_args(
            package_report_dir=package_report_dir,
            bestset_report_dir=bestset_report_dir,
            frontier_report_dir=frontier_report_dir,
        )
        build_frontier.main(
            [
                "--report-dir",
                frontier_args["report_dir"],
                "--bestset-json",
                frontier_args["bestset_json"],
                "--previous-frontier-json",
                frontier_args["previous_frontier_json"],
                "--newaxis-summary-json",
                frontier_args["newaxis_summary_json"],
            ]
        )
        pipeline_status["frontier_refreshed"] = True

    write_pipeline_status(package_report_dir / "promotion_pipeline_status.json", pipeline_status)
    print(json.dumps(pipeline_status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
