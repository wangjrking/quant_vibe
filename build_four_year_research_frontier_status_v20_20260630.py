from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v20_20260630"

BESTSET_V16 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v16_20260630"
    / "latest_bestset_status_v16.json"
)
FRONTIER_V19 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v19_20260629"
    / "research_frontier_status_v19.json"
)
NEWAXIS_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_new_axis_plan_20260629"
    / "research_training"
    / "four_year_10d_newaxis_v1_20260629_fixed4y_fs80_d2_l12_alpha05_stability_fs80_d2_lr0p003_n7000"
    / "newaxis_candidate_summary.json"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the four-year research frontier status from a refreshed bestset snapshot and a chosen 10D new-axis summary."
    )
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--bestset-json", default=str(BESTSET_V16))
    parser.add_argument("--previous-frontier-json", default=str(FRONTIER_V19))
    parser.add_argument("--newaxis-summary-json", default=str(NEWAXIS_SUMMARY))
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bestset_record(bestset: dict[str, Any], label_key: str) -> dict[str, Any]:
    for row in bestset["best_set"]:
        if row["label_key"] == label_key:
            return row
    raise KeyError(f"label_key not found: {label_key}")


def assess_newaxis_status(summary: dict[str, Any], *, evidence_path: str) -> dict[str, Any]:
    gate = summary["gate_result"]
    direct = summary["full_delta_vs_current_bestset"]
    if bool(gate["hard_constraint_passed"]):
        return {
            "decision": "newaxis_train_candidate_promoted_into_current_bestset",
            "reason": "newaxis training candidate passed four-year gate and replaced the prior 10d bestset",
            "asset": summary["asset"],
            "table": summary["table"],
            "full_rank_ic_delta_vs_prior_bestset": direct["rank_ic"],
            "full_top5_delta_vs_prior_bestset": direct["top5"],
            "recent63_top5_delta_vs_prior_bestset": summary["recent63_delta_vs_current_bestset"]["top5"],
            "recent20_top5_delta_vs_prior_bestset": summary["recent20_delta_vs_current_bestset"]["top5"],
            "evidence": evidence_path,
        }
    return {
        "decision": "newaxis_train_candidate_not_promoted",
        "reason": "newaxis training candidate did not pass four-year gate",
        "asset": summary.get("asset"),
        "table": summary.get("table"),
        "evidence": evidence_path,
    }


def choose_next_focus(status: dict[str, Any]) -> dict[str, str]:
    if status["decision"] == "newaxis_train_candidate_promoted_into_current_bestset":
        return {
            "priority_label": "10d",
            "action": "continue_newaxis_model_family_and_compare_alternative_param_axes",
            "reason": "10d newaxis model family is now the active frontier and should be optimized further under the same four-year gate",
        }
    return {
        "priority_label": "10d",
        "action": "continue_previous_frontier_until_newaxis_candidate_passes",
        "reason": "newaxis candidate has not yet replaced the current 10d bestset",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    bestset_json = Path(args.bestset_json)
    previous_frontier_json = Path(args.previous_frontier_json)
    newaxis_summary_json = Path(args.newaxis_summary_json)

    report_dir.mkdir(parents=True, exist_ok=True)
    bestset = load_json(bestset_json)
    frontier_v19 = load_json(previous_frontier_json)
    newaxis_summary = load_json(newaxis_summary_json)
    newaxis_status = assess_newaxis_status(newaxis_summary, evidence_path=str(newaxis_summary_json))

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v20",
        "bestset_snapshot_path": str(bestset_json),
        "previous_frontier_path": str(previous_frontier_json),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "10d_candidate_pool_scan_v1": frontier_v19["recent_progress"]["10d_candidate_pool_scan_v1"],
            "10d_sparse_triple_gate_v6": frontier_v19["recent_progress"]["10d_sparse_triple_gate_v6"],
            "10d_newaxis_train_candidate": newaxis_status,
        },
        "next_focus": choose_next_focus(newaxis_status),
        "boundaries": {
            "research_only": True,
            "no_training_this_report": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    json_path = report_dir / "research_frontier_status_v20.json"
    md_path = report_dir / "research_frontier_status_v20.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 四年观察期研究前沿状态 v20",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 10D 新轴候选状态：`{newaxis_status['decision']}`",
        f"- 当前 10D active frontier：`{summary['current_bestset']['10d']['asset']}`",
        f"- 下一步动作：`{summary['next_focus']['action']}`",
        "",
        "## 边界",
        "",
        "- 本报告只更新 research 状态。",
        "- 未训练模型。",
        "- 未修改 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
