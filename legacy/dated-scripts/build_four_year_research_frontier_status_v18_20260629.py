from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v18_20260629"

BESTSET_V15 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v15_20260629"
    / "latest_bestset_status_v15.json"
)
FRONTIER_V17 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v17_20260629"
    / "research_frontier_status_v17.json"
)
CANDIDATE_POOL_SCAN = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_candidate_pool_scan_v1_20260629"
    / "candidate_pool_scan_summary.json"
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


def assess_10d_candidate_pool(scan: dict[str, Any]) -> dict[str, Any]:
    best = scan["best"]
    if int(scan["hard_pass_count"]) > 0:
        return {
            "decision": "candidate_pool_has_hard_pass",
            "reason": "at least one existing four-year 10d table passed hard constraints; review before new source search",
            "best_table": best["table"],
            "hard_pass_count": int(scan["hard_pass_count"]),
            "best_deltas_vs_base": {
                "full_rank_ic_delta": best["full_rank_ic_delta_vs_base"],
                "full_top5_delta": best["full_top5_delta_vs_base"],
                "recent63_top5_delta": best["recent63_top5_delta_vs_base"],
                "recent20_top5_delta": best["recent20_top5_delta_vs_base"],
                "min_month_top5_delta": best["min_month_top5_delta_vs_base"],
            },
            "evidence": str(CANDIDATE_POOL_SCAN),
        }
    return {
        "decision": "reject_existing_four_year_candidate_pool",
        "reason": "no scanned existing four-year 10d table passed hard constraints against current bestset",
        "scanned_count": int(scan["scanned_count"]),
        "hard_pass_count": int(scan["hard_pass_count"]),
        "best_table": best["table"],
        "best_deltas_vs_base": {
            "full_rank_ic_delta": best["full_rank_ic_delta_vs_base"],
            "full_top5_delta": best["full_top5_delta_vs_base"],
            "recent63_top5_delta": best["recent63_top5_delta_vs_base"],
            "recent20_top5_delta": best["recent20_top5_delta_vs_base"],
            "min_month_top5_delta": best["min_month_top5_delta_vs_base"],
        },
        "evidence": str(CANDIDATE_POOL_SCAN),
    }


def choose_next_focus(candidate_pool_status: dict[str, Any]) -> dict[str, str]:
    if candidate_pool_status["decision"] == "candidate_pool_has_hard_pass":
        return {
            "priority_label": "10d",
            "action": "review_existing_hard_pass_candidate_for_bestset_refresh",
            "reason": "existing candidate pool contains a hard-pass table",
        }
    return {
        "priority_label": "10d",
        "action": "start_new_10d_formula_or_model_axis",
        "reason": "both source_front_gate_v5 and existing four-year 10d candidate pool failed hard constraints",
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bestset = load_json(BESTSET_V15)
    frontier_v17 = load_json(FRONTIER_V17)
    pool_scan = load_json(CANDIDATE_POOL_SCAN)
    pool_status = assess_10d_candidate_pool(pool_scan)

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v18",
        "bestset_snapshot_path": str(BESTSET_V15),
        "previous_frontier_path": str(FRONTIER_V17),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "10d_source_front_gate_v5": frontier_v17["recent_progress"]["10d_source_front_gate_v5"],
            "10d_candidate_pool_scan_v1": pool_status,
        },
        "next_focus": choose_next_focus(pool_status),
        "boundaries": {
            "research_only": True,
            "no_training_this_report": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    json_path = REPORT_DIR / "research_frontier_status_v18.json"
    md_path = REPORT_DIR / "research_frontier_status_v18.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 四年观察期研究前沿状态 v18",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 10D 既有四年候选池扫描未发现硬约束通过资产。",
        "- 当前 10D research bestset 仍保持 `research_10d_four_year_active_recent_blend_gate_v3_20260629`。",
        "- 下一步需要换新的 10D 公式或模型轴，而不是继续复用现有四年候选池。",
        "",
        "## 候选池扫描摘要",
        "",
        f"- 扫描数量：{pool_status.get('scanned_count', pool_scan['scanned_count'])}",
        f"- 硬约束通过数量：{pool_status['hard_pass_count']}",
        f"- 最优未通过表：`{pool_status['best_table']}`",
        f"- 证据：`{CANDIDATE_POOL_SCAN}`",
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
