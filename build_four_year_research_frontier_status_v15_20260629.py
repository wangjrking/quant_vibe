from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v15_20260629"

BESTSET_V14 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v14_20260629"
    / "latest_bestset_status_v14.json"
)
FIVE_D_V2 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_lightweight_blend_v2_20260629"
    / "promotion_candidate.json"
)
FIVE_D_CLEAR_V2 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_bestset_clear_gate_v2_20260629"
    / "bestset_clear_gate_v2_summary.json"
)
FIVE_D_ACTIVE_CLEAR_HYBRID_EVAL = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_active_clear_gate_hybrid_20260628"
    / "standard_eval"
    / "executable_5d_open_return_four_year_active_clear_gate_hybrid_20260628_eval_summary.json"
)
ONE_D_FIXED4Y_V3 = (
    DATA_DIR
    / "reports"
    / "model_agent_1d_fixed4y_latestfold_retrain_v3_20260628"
    / "latestfold_retrain_v3_summary.json"
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


def assess_5d_status(
    lightweight_v2: dict[str, Any],
    clear_gate_v2: dict[str, Any],
    current_bestset_5d: dict[str, Any],
    active_clear_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta = lightweight_v2.get("current_baseline_delta", {})
    direct_ok = (
        float(delta.get("full_rank_ic_delta", 0.0)) > 0.0
        and float(delta.get("full_top5_delta", 0.0)) > 0.0
        and float(delta.get("recent63_top5_delta", 0.0)) > 0.0
        and float(delta.get("recent20_top5_delta", 0.0)) > 0.0
    )
    if direct_ok:
        return {
            "decision": "replace_with_lightweight_blend_v2",
            "reason": "direct_deltas_vs_current_bestset_all_positive",
            "next_priority": "refresh_bestset_snapshot",
            "direct_delta_vs_bestset": delta,
            "evidence": str(FIVE_D_V2),
        }

    closest_miss: dict[str, Any] | None = None
    if active_clear_eval:
        full = active_clear_eval.get("full", {})
        recent = active_clear_eval.get("recent_windows", {})
        recent63 = recent.get("recent63", {})
        recent20 = recent.get("recent20", {})
        closest_miss = {
            "asset": "research_5d_four_year_active_clear_gate_hybrid_20260628",
            "direct_delta_vs_bestset": {
                "full_rank_ic_delta": float(full.get("rank_ic", 0.0)) - float(current_bestset_5d["full_rank_ic"]),
                "full_top5_delta": float(full.get("top5", 0.0)) - float(current_bestset_5d["full_top5"]),
                "recent63_top5_delta": float(recent63.get("top5", 0.0)) - float(current_bestset_5d["recent63_abs_top5"]),
                "recent20_top5_delta": float(recent20.get("top5", 0.0)) - float(current_bestset_5d["recent20_abs_top5"]),
            },
            "evidence": str(FIVE_D_ACTIVE_CLEAR_HYBRID_EVAL),
            "status": "near_miss_full_rank_ic_below_current_bestset",
        }

    return {
        "decision": "keep_current_bestset",
        "reason": clear_gate_v2.get("decision", "no_directly_better_candidate_found"),
        "next_priority": "new_formula_search",
        "direct_delta_vs_bestset": delta,
        "clear_gate_scan_rows": int(clear_gate_v2.get("scan_rows", 0)),
        "clear_gate_pass_hard_count": int(clear_gate_v2.get("pass_hard_count", 0)),
        "closest_near_miss": closest_miss,
        "evidence": [
            str(FIVE_D_V2),
            str(FIVE_D_CLEAR_V2),
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

    bestset = load_json(BESTSET_V14)
    five_d_v2 = load_json(FIVE_D_V2)
    five_d_clear_v2 = load_json(FIVE_D_CLEAR_V2)
    five_d_active_clear_eval = load_json(FIVE_D_ACTIVE_CLEAR_HYBRID_EVAL)
    one_d_fixed4y_v3 = load_json(ONE_D_FIXED4Y_V3)

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v15",
        "bestset_snapshot_path": str(BESTSET_V14),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "10d": {
                "decision": "updated_bestset",
                "new_asset": bestset_record(bestset, "10d")["asset"],
                "reason": "direct positive deltas vs prior 10d bestset on full and recent windows",
                "evidence": str(BESTSET_V14),
            },
            "5d": assess_5d_status(
                five_d_v2,
                five_d_clear_v2,
                bestset_record(bestset, "5d"),
                five_d_active_clear_eval,
            ),
            "1d": assess_1d_status(one_d_fixed4y_v3),
        },
        "next_focus": {
            "priority_label": "5d",
            "reason": "10d already improved this turn; 5d remains high-priority but no existing candidate directly dominates current bestset; 1d fixed4y retrain line is currently ineffective",
            "action": "design a new 5d formula search around current bestset rather than reusing current direct candidates or fixed4y 1d retrain settings",
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

    json_path = REPORT_DIR / "research_frontier_status_v15.json"
    md_path = REPORT_DIR / "research_frontier_status_v15.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 四年观察期研究前沿状态 v15",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 10D 当前 research bestset 已更新为 `{summary['current_bestset']['10d']['asset']}`。",
        f"- 5D 暂时继续保留 `{summary['current_bestset']['5d']['asset']}`，现有 `lightweight_blend_v2` 和 `bestset_clear_gate_v2` 都没有形成可直接替换依据。",
        f"- 1D 暂时继续保留 `{summary['current_bestset']['1d']['asset']}`，`fixed4y latestfold retrain v3` 明显弱于当前四年 bestset。",
        "",
        "## 直接证据",
        "",
        f"- 10D bestset 刷新：`{BESTSET_V14}`",
        f"- 5D lightweight v2：`{FIVE_D_V2}`",
        f"- 5D clear gate v2：`{FIVE_D_CLEAR_V2}`",
        f"- 1D fixed4y retrain v3：`{ONE_D_FIXED4Y_V3}`",
        "",
        "## 下一步",
        "",
        "- 下一优先标签仍是 5D。",
        "- 方向不是重复现有候选，而是围绕当前 5D bestset 重新设计新的 score formula 搜索。",
        "- 1D fixed4y 重训线先不继续放大投入，除非更换目标函数或特征口径。",
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
