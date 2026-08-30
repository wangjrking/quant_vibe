from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import package_four_year_10d_newaxis_candidate_20260630 as package_candidate
import promote_four_year_10d_newaxis_research_pipeline_20260630 as promote_pipeline


CN_TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(CN_TZ).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try to promote a 10D four-year new-axis experiment. If folds are incomplete, emit a status snapshot instead of failing."
    )
    parser.add_argument("--experiment-dir", default=str(promote_pipeline.DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--asset", default=promote_pipeline.DEFAULT_ASSET)
    parser.add_argument("--output-table", default="")
    parser.add_argument("--package-report-dir", default="")
    parser.add_argument("--bestset-report-dir", default=str(promote_pipeline.DEFAULT_BESTSET_REPORT_DIR))
    parser.add_argument("--frontier-report-dir", default=str(promote_pipeline.DEFAULT_FRONTIER_REPORT_DIR))
    return parser.parse_args(argv)


def decide_action(status: dict[str, Any]) -> dict[str, Any]:
    if bool(status["is_complete"]):
        return {"action": "run_promotion_pipeline", "reason": "experiment_complete"}
    return {"action": "wait_for_more_folds", "reason": "experiment_incomplete"}


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    package_report_dir = Path(args.package_report_dir) if args.package_report_dir else experiment_dir
    status = package_candidate.check_experiment_complete(experiment_dir)
    decision = decide_action(status)

    payload = {
        "generated_at": now_iso(),
        "experiment_dir": str(experiment_dir),
        "package_report_dir": str(package_report_dir),
        "decision": decision,
        "fold_status": status,
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    if decision["action"] == "run_promotion_pipeline":
        promote_pipeline.main(
            [
                "--experiment-dir",
                str(experiment_dir),
                "--asset",
                str(args.asset),
                "--package-report-dir",
                str(package_report_dir),
                "--bestset-report-dir",
                str(args.bestset_report_dir),
                "--frontier-report-dir",
                str(args.frontier_report_dir),
                *(["--output-table", str(args.output_table)] if args.output_table else []),
            ]
        )
        payload["promotion_triggered"] = True
    else:
        payload["promotion_triggered"] = False

    write_status(package_report_dir / "try_promotion_status.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
