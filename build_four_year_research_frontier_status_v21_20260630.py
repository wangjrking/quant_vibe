from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v21_20260630"

BESTSET_V17 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v17_20260630"
    / "latest_bestset_status_v17.json"
)
FRONTIER_V20 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v20_20260630"
    / "research_frontier_status_v20.json"
)
ONE_D_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_1d_balanced_dualgate_scan_20260630"
    / "balanced_dualgate_summary.json"
)
FIVE_D_BESTSET_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_current_bestset_direct_guard_v1_20260630"
    / "current_bestset_direct_guard_v1_summary.json"
)
FIVE_D_DIRECT_SCAN = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_v4_direct_scan_tmp_20260630"
    / "v4_direct_scan.csv"
)
TEN_D_SATURATION_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_v7_source_saturation_scan_20260630"
    / "source_saturation_summary.json"
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


def assess_1d(summary: dict[str, Any], *, evidence_path: str) -> dict[str, Any]:
    best = summary["best"]
    return {
        "decision": summary["decision"],
        "reason": "1D balanced dualgate 本轮没有形成新的 hard-pass 候选，当前仍保留既有四年 bestset。",
        "scan_rows": int(summary["scan_rows"]),
        "pass_hard_count": int(summary["pass_hard_count"]),
        "best_condition": best["condition"],
        "best_union_days": int(best["union_days"]),
        "full_rank_ic_delta_vs_base": float(best["full_rank_ic_delta_vs_base"]),
        "full_top5_delta_vs_base": float(best["full_top5_delta_vs_base"]),
        "recent63_top5_delta_vs_base": float(best["recent63_top5_delta_vs_base"]),
        "recent20_top5_delta_vs_base": float(best["recent20_top5_delta_vs_base"]),
        "evidence": evidence_path,
    }


def assess_5d_direct_scan(scan_path: Path, bestset_summary: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(scan_path)
    improving_full = df[
        (df["full_rank_ic_delta_vs_base"] > 0.0) | (df["full_top5_delta_vs_base"] > 0.0)
    ].copy()
    recent_top5_positive = df[
        (df["recent63_top5_delta_vs_base"] > 0.0)
        | (df["recent20_top5_delta_vs_base"] > 0.0)
    ].copy()
    improving_full_and_recent_top5 = improving_full[
        (improving_full["recent63_top5_delta_vs_base"] > 0.0)
        | (improving_full["recent20_top5_delta_vs_base"] > 0.0)
    ].copy()
    best_full = improving_full.sort_values("objective", ascending=False).iloc[0].to_dict()
    decision = "same_source_direct_scan_saturated_no_recent_gain"
    reason = (
        "5D 当前这轮同源直扫只得到全窗口微增量，"
        "但没有任何一条同时做到“全窗口优于当前 bestset”且“recent20/recent63 Top5 也更优”。"
    )
    if not improving_full_and_recent_top5.empty:
        decision = "same_source_direct_scan_has_full_and_recent_top5_gain_candidate"
        reason = "5D 同源直扫中存在相对当前 bestset 同时改善全窗口与 recent20/recent63 Top5 的候选。"
    return {
        "decision": decision,
        "reason": reason,
        "scan_rows": int(len(df)),
        "improving_full_count": int(len(improving_full)),
        "recent_top5_positive_count": int(len(recent_top5_positive)),
        "improving_full_and_recent_top5_count": int(len(improving_full_and_recent_top5)),
        "current_bestset_asset": bestset_summary["promotion_gate_result"]["asset"],
        "current_bestset_condition": bestset_summary["condition"],
        "current_bestset_active_days": int(bestset_summary["active_days"]),
        "best_full_candidate": {
            "condition": best_full["condition"],
            "active_days": int(best_full["active_days"]),
            "full_rank_ic_delta_vs_base": float(best_full["full_rank_ic_delta_vs_base"]),
            "full_top5_delta_vs_base": float(best_full["full_top5_delta_vs_base"]),
            "recent63_top5_delta_vs_base": float(best_full["recent63_top5_delta_vs_base"]),
            "recent20_top5_delta_vs_base": float(best_full["recent20_top5_delta_vs_base"]),
            "objective": float(best_full["objective"]),
        },
        "evidence": str(scan_path),
    }


def choose_next_focus(one_d: dict[str, Any], five_d: dict[str, Any], ten_d: dict[str, Any]) -> dict[str, str]:
    if five_d["decision"] == "same_source_direct_scan_saturated_no_recent_gain":
        return {
            "priority_label": "5d",
            "action": "switch_5d_to_new_training_family_or_cross_source_candidate_search",
            "reason": "5D 当前 bestset 仍是高优先级，但同源公式微调已经不再带来近期窗口增量。",
        }
    if one_d["pass_hard_count"] == 0:
        return {
            "priority_label": "1d",
            "action": "move_1d_from_score_splice_to_fixed4y_retrain_family",
            "reason": "1D 评分拼接路线没有新 pass，应转入新的 fixed4y 训练族实验。",
        }
    return {
        "priority_label": "10d",
        "action": "keep_10d_v7_and_only_do_tail_risk_compression",
        "reason": "10D 当前 frontier 仍有效，但同源扫描已接近饱和，不宜继续做同类型微调。",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rp = payload["recent_progress"]
    lines = [
        "# 四年观察期研究前沿状态 v21",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 1D：`{rp['1d_balanced_dualgate']['decision']}`",
        f"- 5D：`{rp['5d_same_source_direct_scan']['decision']}`",
        f"- 10D：`{rp['10d_same_source_saturation']['decision']}`",
        f"- 下一步主攻：`{payload['next_focus']['priority_label']}` / `{payload['next_focus']['action']}`",
        "",
        "## 关键事实",
        "",
        f"- 1D 扫描行数：{rp['1d_balanced_dualgate']['scan_rows']}，hard pass：{rp['1d_balanced_dualgate']['pass_hard_count']}",
        f"- 5D 直扫总行数：{rp['5d_same_source_direct_scan']['scan_rows']}，相对当前 bestset 有全窗口微增量的候选：{rp['5d_same_source_direct_scan']['improving_full_count']}，近期 Top5 为正的候选：{rp['5d_same_source_direct_scan']['recent_top5_positive_count']}，但两者同时成立的候选：{rp['5d_same_source_direct_scan']['improving_full_and_recent_top5_count']}",
        f"- 10D 同源扫描结论：`{rp['10d_same_source_saturation']['decision']}`",
        "",
        "## 边界",
        "",
        "- 本报告只更新 research 状态。",
        "- 未训练模型。",
        "- 未修改 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bestset = load_json(BESTSET_V17)
    frontier_v20 = load_json(FRONTIER_V20)
    one_d_summary = load_json(ONE_D_SUMMARY)
    five_d_bestset_summary = load_json(FIVE_D_BESTSET_SUMMARY)
    ten_d_summary = load_json(TEN_D_SATURATION_SUMMARY)

    one_d_status = assess_1d(one_d_summary, evidence_path=str(ONE_D_SUMMARY))
    five_d_status = assess_5d_direct_scan(FIVE_D_DIRECT_SCAN, five_d_bestset_summary)
    next_focus = choose_next_focus(one_d_status, five_d_status, ten_d_summary)

    payload = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v21",
        "bestset_snapshot_path": str(BESTSET_V17),
        "previous_frontier_path": str(FRONTIER_V20),
        "current_bestset": {
            "1d": bestset_record(bestset, "1d"),
            "3d": bestset_record(bestset, "3d"),
            "5d": bestset_record(bestset, "5d"),
            "10d": bestset_record(bestset, "10d"),
        },
        "recent_progress": {
            "1d_balanced_dualgate": one_d_status,
            "5d_current_bestset_direct_guard_v1": {
                "decision": five_d_bestset_summary["promotion_gate_result"]["promotion_decision"],
                "asset": five_d_bestset_summary["promotion_gate_result"]["asset"],
                "condition": five_d_bestset_summary["condition"],
                "full_rank_ic_delta_vs_control": float(five_d_bestset_summary["full_delta_vs_control"]["rank_ic"]),
                "full_top5_delta_vs_control": float(five_d_bestset_summary["full_delta_vs_control"]["top5"]),
                "recent63_top5_delta_vs_control": float(five_d_bestset_summary["recent63_delta_vs_control"]["top5"]),
                "recent20_top5_delta_vs_control": float(five_d_bestset_summary["recent20_delta_vs_control"]["top5"]),
                "full_rank_ic_delta_vs_base": float(five_d_bestset_summary["full_delta_vs_base"]["rank_ic"]),
                "full_top5_delta_vs_base": float(five_d_bestset_summary["full_delta_vs_base"]["top5"]),
                "evidence": str(FIVE_D_BESTSET_SUMMARY),
            },
            "5d_same_source_direct_scan": five_d_status,
            "10d_same_source_saturation": ten_d_summary,
            "10d_newaxis_train_candidate": frontier_v20["recent_progress"]["10d_newaxis_train_candidate"],
        },
        "next_focus": next_focus,
        "boundaries": {
            "research_only": True,
            "no_training_this_report": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    json_path = REPORT_DIR / "research_frontier_status_v21.json"
    md_path = REPORT_DIR / "research_frontier_status_v21.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
