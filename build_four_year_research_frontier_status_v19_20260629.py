from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v19_20260629"

BESTSET_V15 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v15_20260629"
    / "latest_bestset_status_v15.json"
)
FRONTIER_V18 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v18_20260629"
    / "research_frontier_status_v18.json"
)
SPARSE_TRIPLE_V6 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_sparse_triple_gate_v6_20260629"
    / "sparse_triple_gate_v6_summary.json"
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


def assess_sparse_triple_v6(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary["best"]
    if int(summary["pass_hard_count"]) > 0:
        return {
            "decision": "sparse_triple_gate_has_hard_pass_candidate",
            "reason": "at least one sparse triple gate candidate passed hard constraints and needs review",
            "pass_hard_count": int(summary["pass_hard_count"]),
            "best_condition": best["condition"],
            "best_deltas_vs_base": {
                "full_rank_ic_delta": best["full_rank_ic_delta_vs_base"],
                "full_top1_delta": best["full_top1_delta_vs_base"],
                "full_top5_delta": best["full_top5_delta_vs_base"],
                "recent63_top5_delta": best["recent63_top5_delta_vs_base"],
                "recent20_top5_delta": best["recent20_top5_delta_vs_base"],
                "min_month_top5_delta": best["min_month_top5_delta_vs_base"],
            },
            "evidence": str(SPARSE_TRIPLE_V6),
        }
    return {
        "decision": "reject_sparse_triple_gate_v6_keep_current_bestset",
        "reason": "sparse triple gate did not pass hard constraints; best row still fails at least one required metric",
        "scan_rows": int(summary["scan_rows"]),
        "pass_hard_count": int(summary["pass_hard_count"]),
        "best_condition": best["condition"],
        "best_failed_or_binding_metrics": {
            "full_rank_ic_delta": best["full_rank_ic_delta_vs_base"],
            "full_top1_delta": best["full_top1_delta_vs_base"],
            "full_top5_delta": best["full_top5_delta_vs_base"],
            "recent63_top5_delta": best["recent63_top5_delta_vs_base"],
            "recent20_top5_delta": best["recent20_top5_delta_vs_base"],
            "min_month_top5_delta": best["min_month_top5_delta_vs_base"],
        },
        "evidence": str(SPARSE_TRIPLE_V6),
    }


def choose_next_focus(status: dict[str, Any]) -> dict[str, str]:
    if status["decision"] == "sparse_triple_gate_has_hard_pass_candidate":
        return {
            "priority_label": "10d",
            "action": "review_sparse_triple_gate_candidate_before_any_table_materialization",
            "reason": "a hard-pass sparse gate exists, but production rules require review before any asset promotion",
        }
    return {
        "priority_label": "10d",
        "action": "switch_from_score_reuse_to_new_model_or_feature_axis",
        "reason": "existing score-reuse families have failed hard constraints under the current four-year gate",
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bestset = load_json(BESTSET_V15)
    frontier_v18 = load_json(FRONTIER_V18)
    sparse_v6 = load_json(SPARSE_TRIPLE_V6)
    sparse_status = assess_sparse_triple_v6(sparse_v6)
    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v19",
        "bestset_snapshot_path": str(BESTSET_V15),
        "previous_frontier_path": str(FRONTIER_V18),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "10d_candidate_pool_scan_v1": frontier_v18["recent_progress"]["10d_candidate_pool_scan_v1"],
            "10d_sparse_triple_gate_v6": sparse_status,
        },
        "next_focus": choose_next_focus(sparse_status),
        "boundaries": {
            "research_only": True,
            "no_training_this_report": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    json_path = REPORT_DIR / "research_frontier_status_v19.json"
    md_path = REPORT_DIR / "research_frontier_status_v19.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 四年观察期研究前沿状态 v19",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 10D sparse triple gate v6 未通过四年硬约束。",
        "- 当前 10D research bestset 仍保持 `research_10d_four_year_active_recent_blend_gate_v3_20260629`。",
        "- 下一步应从评分复用转向新的模型或特征结构轴。",
        "",
        "## v6 摘要",
        "",
        f"- 扫描组合数：{sparse_status.get('scan_rows', sparse_v6['scan_rows'])}",
        f"- 硬约束通过数量：{sparse_status['pass_hard_count']}",
        f"- 最优条件：`{sparse_status['best_condition']}`",
        f"- 证据：`{SPARSE_TRIPLE_V6}`",
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
