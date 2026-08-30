from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v16_20260629"

BESTSET_V15 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v15_20260629"
    / "latest_bestset_status_v15.json"
)
FIVE_D_V4 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_front_rank_state_gate_v4_20260629"
    / "promotion_candidate.json"
)
FIVE_D_V4_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_front_rank_state_gate_v4_20260629"
    / "front_rank_state_gate_v4_summary.json"
)
FIVE_D_V4_GATE = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_front_rank_state_gate_v4_20260629"
    / "promotion_gate_result.json"
)
ONE_D_FIXED4Y_V3 = (
    DATA_DIR
    / "reports"
    / "model_agent_1d_fixed4y_latestfold_retrain_v3_20260628"
    / "latestfold_retrain_v3_summary.json"
)
DEFAULT_NEXT_PRIORITY_LABEL = "10d"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bestset_record(bestset: dict[str, Any], label_key: str) -> dict[str, Any]:
    for row in bestset["best_set"]:
        if row["label_key"] == label_key:
            return row
    raise KeyError(f"label_key not found: {label_key}")


def assess_5d_progress(
    candidate: dict[str, Any],
    summary: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    delta = candidate["current_baseline_delta"]
    gate_passed = bool(gate["result"]["hard_constraint_passed"])
    min_month_top5_delta_vs_base = float(summary["best"]["min_month_top5_delta_vs_base"])
    direct_ok = (
        gate_passed
        and float(delta["full_rank_ic_delta"]) > 0.0
        and float(delta["full_top5_delta"]) > 0.0
        and float(delta["recent63_top5_delta"]) >= 0.0
        and float(delta["recent20_top5_delta"]) >= 0.0
        and min_month_top5_delta_vs_base >= 0.0
    )
    if direct_ok:
        return {
            "decision": "updated_bestset",
            "new_asset": candidate["asset"],
            "reason": "front_rank_state_gate_v4 achieves positive direct deltas vs prior 5d bestset and keeps monthly top5 tail nonnegative vs prior bestset",
            "next_priority": "new_formula_search_on_new_bestset",
            "direct_delta_vs_bestset": delta,
            "active_days": int(summary["best"]["active_days"]),
            "evidence": [
                str(FIVE_D_V4),
                str(FIVE_D_V4_SUMMARY),
                str(FIVE_D_V4_GATE),
            ],
        }
    return {
        "decision": "keep_current_bestset",
        "reason": "front_rank_state_gate_v4 did not satisfy refresh rule",
        "next_priority": "recheck_candidate_inputs",
        "direct_delta_vs_bestset": delta,
        "evidence": [
            str(FIVE_D_V4),
            str(FIVE_D_V4_SUMMARY),
            str(FIVE_D_V4_GATE),
        ],
    }


def assess_1d_status(fixed4y_v3: dict[str, Any]) -> dict[str, Any]:
    best_candidate = fixed4y_v3["best_candidate"]
    current_bestset = next(
        row for row in fixed4y_v3["baselines"] if row["candidate"] == "current_four_year_bestset"
    )
    return {
        "decision": "keep_current_bestset",
        "reason": "latest_fixed4y_retrain_window_underperforms_current_bestset",
        "next_priority": "pause_fixed4y_retrain_line",
        "objective_delta_vs_bestset": float(best_candidate["objective"]) - float(current_bestset["objective"]),
        "top5_delta_vs_bestset": float(best_candidate["top5"]) - float(current_bestset["top5"]),
        "rank_ic_delta_vs_bestset": float(best_candidate["rank_ic"]) - float(current_bestset["rank_ic"]),
        "best_candidate": best_candidate["candidate"],
        "evidence": str(ONE_D_FIXED4Y_V3),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bestset = load_json(BESTSET_V15)
    five_d_v4 = load_json(FIVE_D_V4)
    five_d_v4_summary = load_json(FIVE_D_V4_SUMMARY)
    five_d_v4_gate = load_json(FIVE_D_V4_GATE)
    one_d_fixed4y_v3 = load_json(ONE_D_FIXED4Y_V3)

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v16",
        "bestset_snapshot_path": str(BESTSET_V15),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "10d": {
                "decision": "keep_current_bestset",
                "new_asset": bestset_record(bestset, "10d")["asset"],
                "reason": "10d bestset already refreshed earlier; current work turn focuses on 5d refresh",
                "evidence": str(BESTSET_V15),
            },
            "5d": assess_5d_progress(five_d_v4, five_d_v4_summary, five_d_v4_gate),
            "1d": assess_1d_status(one_d_fixed4y_v3),
        },
        "next_focus": {
            "priority_label": DEFAULT_NEXT_PRIORITY_LABEL,
            "reason": "5d bestset has now been refreshed; 10d remains the highest-value next search axis for further monthly-tail compression without giving back current full/recent gains",
            "action": "design a new 10d formula search on top of active_recent_blend_gate_v3 rather than returning to rejected fixed4y legacy source families",
        },
        "boundaries": {
            "research_only": True,
            "no_training_this_report": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    json_path = REPORT_DIR / "research_frontier_status_v16.json"
    md_path = REPORT_DIR / "research_frontier_status_v16.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 四年观察期研究前沿状态 v16",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 5D 当前 research bestset 已更新为 `{summary['current_bestset']['5d']['asset']}`。",
        f"- 10D 当前保留 `{summary['current_bestset']['10d']['asset']}`，继续作为下一轮主攻方向。",
        f"- 1D 当前保留 `{summary['current_bestset']['1d']['asset']}`，`fixed4y latestfold retrain v3` 仍明显弱于现有四年 bestset。",
        "",
        "## 下一步",
        "",
        "- 下一优先标签切到 10D。",
        "- 方向不是回到旧 fixed4y 家族，而是在 `active_recent_blend_gate_v3` 之上继续压缩月度尾部风险。",
        "",
        "## 边界",
        "",
        "- 本报告只更新 research 状态判断。",
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
