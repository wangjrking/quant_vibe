from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v17_20260629"

BESTSET_V15 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v15_20260629"
    / "latest_bestset_status_v15.json"
)
FRONTIER_V16 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v16_20260629"
    / "research_frontier_status_v16.json"
)
SOURCE_FRONT_GATE_V5 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_source_front_gate_v5_20260629"
    / "source_front_gate_v5_summary.json"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bestset_record(bestset: dict[str, Any], label_key: str) -> dict[str, Any]:
    for row in bestset["best_set"]:
        if row["label_key"] == label_key:
            return row
    raise KeyError(f"label_key not found: {label_key}")


def assess_10d_source_front_gate_v5(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary["best"]
    pass_hard_count = int(summary["pass_hard_count"])
    decision = str(summary["decision"])
    if pass_hard_count > 0 and decision.endswith("candidate_materialized"):
        return {
            "decision": "candidate_ready_for_bestset_review",
            "reason": "source_front_gate_v5 produced at least one hard-pass candidate; compare against current 10d bestset before promotion",
            "candidate_table": summary["target_table"],
            "pass_hard_count": pass_hard_count,
            "best_condition": best["condition"],
            "evidence": str(SOURCE_FRONT_GATE_V5),
        }
    return {
        "decision": "reject_axis_keep_current_bestset",
        "reason": "source_front_gate_v5 did not satisfy four-year hard constraints; no research table was materialized",
        "candidate_table": summary["target_table"],
        "pass_hard_count": pass_hard_count,
        "best_condition": best["condition"],
        "best_failed_constraints": {
            "full_rank_ic_delta_vs_base": best["full_rank_ic_delta_vs_base"],
            "recent20_top5_delta_vs_base": best["recent20_top5_delta_vs_base"],
            "recent20_rank_ic_delta_vs_base": best["recent20_rank_ic_delta_vs_base"],
            "pass_hard": best["pass_hard"],
        },
        "evidence": str(SOURCE_FRONT_GATE_V5),
    }


def choose_next_focus(source_front_gate_status: dict[str, Any]) -> dict[str, str]:
    if source_front_gate_status["decision"] == "candidate_ready_for_bestset_review":
        return {
            "priority_label": "10d",
            "action": "review_source_front_gate_v5_candidate_against_current_bestset",
            "reason": "a new hard-pass 10d source-front candidate exists and must be compared before further search",
        }
    return {
        "priority_label": "10d",
        "action": "start_new_10d_axis_with_different_source_or_structure",
        "reason": "source_front_gate_v5 failed hard constraints; continue 10d optimization but exclude this gate family as a promotion candidate",
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bestset = load_json(BESTSET_V15)
    frontier_v16 = load_json(FRONTIER_V16)
    source_front_gate_v5 = load_json(SOURCE_FRONT_GATE_V5)

    source_front_status = assess_10d_source_front_gate_v5(source_front_gate_v5)
    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v17",
        "bestset_snapshot_path": str(BESTSET_V15),
        "previous_frontier_path": str(FRONTIER_V16),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "10d_source_front_gate_v5": source_front_status,
            "previous_v16_next_focus": frontier_v16["next_focus"],
        },
        "next_focus": choose_next_focus(source_front_status),
        "boundaries": {
            "research_only": True,
            "no_training_this_report": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    json_path = REPORT_DIR / "research_frontier_status_v17.json"
    md_path = REPORT_DIR / "research_frontier_status_v17.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 四年观察期研究前沿状态 v17",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 10D `source_front_gate_v5` 未通过四年硬约束，未物化新 research 表。",
        "- 当前 10D research bestset 仍保持 `research_10d_four_year_active_recent_blend_gate_v3_20260629`。",
        "- 下一步继续优化 10D，但排除 `source_front_gate_v5` 这一门控族作为晋升候选。",
        "",
        "## 失败约束摘要",
        "",
        f"- `pass_hard_count`: {source_front_status['pass_hard_count']}",
        f"- `candidate_table`: {source_front_status['candidate_table']}",
        f"- `best_condition`: `{source_front_status['best_condition']}`",
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
