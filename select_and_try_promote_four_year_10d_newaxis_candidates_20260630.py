from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import package_four_year_10d_newaxis_candidate_20260630 as package_candidate
import try_promote_four_year_10d_newaxis_research_pipeline_20260630 as try_pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_10d_new_axis_plan_20260629" / "selection_status_20260630"
CN_TZ = timezone(timedelta(hours=8))

DEFAULT_EXPERIMENTS = [
    (
        "fs40",
        DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_new_axis_plan_20260629"
        / "research_training"
        / "four_year_10d_newaxis_v1_20260629_fixed4y_fs40_d3_l8_alpha01_topn_balance_fs40_d3_lr0p004_n5000",
    ),
    (
        "fs80",
        DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_new_axis_plan_20260629"
        / "research_training"
        / "four_year_10d_newaxis_v1_20260629_fixed4y_fs80_d2_l12_alpha05_stability_fs80_d2_lr0p003_n7000",
    ),
]
DEFAULT_ASSET_BY_NAME = {
    "fs40": "research_10d_four_year_newaxis_fs40_d3_l8_alpha01_topn_balance_candidate_20260630",
    "fs80": "research_10d_four_year_newaxis_fs80_d2_l12_alpha05_stability_candidate_20260630",
}


def now_iso() -> str:
    return datetime.now(CN_TZ).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the most actionable 10D four-year new-axis experiment and try the research-only promotion pipeline on it."
    )
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    return parser.parse_args(argv)


def completion_ratio(status: dict[str, Any]) -> float:
    expected = len(status.get("expected_folds", []))
    if expected <= 0:
        return 0.0
    missing = len(status.get("missing_folds", []))
    bad = len(status.get("bad_status_folds", []))
    missing_files = len(status.get("missing_prediction_files", []))
    penalty = max(missing, bad, missing_files)
    return max(0.0, float(expected - penalty) / float(expected))


def choose_best_experiment(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
        status = item["status"]
        missing_files = len(status.get("missing_prediction_files", []))
        return (
            1 if bool(status.get("is_complete")) else 0,
            -missing_files,
            completion_ratio(status),
        )

    return max(candidates, key=sort_key)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for name, experiment_dir in DEFAULT_EXPERIMENTS:
        status = package_candidate.check_experiment_complete(experiment_dir)
        candidates.append(
            {
                "name": name,
                "experiment_dir": str(experiment_dir),
                "status": status,
                "completion_ratio": completion_ratio(status),
            }
        )

    selected = choose_best_experiment(candidates)
    selected_dir = Path(selected["experiment_dir"])
    selected_asset = DEFAULT_ASSET_BY_NAME[selected["name"]]
    try_pipeline.main(["--experiment-dir", str(selected_dir), "--asset", selected_asset])

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_newaxis_multi_experiment_selection",
        "candidates": candidates,
        "selected": {
            "name": selected["name"],
            "asset": selected_asset,
            "experiment_dir": selected["experiment_dir"],
            "completion_ratio": selected["completion_ratio"],
            "is_complete": bool(selected["status"].get("is_complete")),
        },
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (report_dir / "selection_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
