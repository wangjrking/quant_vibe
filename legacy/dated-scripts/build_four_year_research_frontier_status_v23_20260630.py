from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v23_20260630"

FRONTIER_V22 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v22_20260630"
    / "research_frontier_status_v22.json"
)
W2_FAMILY_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_top5pct_w2_featurefamily_scan_20260630"
    / "featurefamily_scan_summary.csv"
)
W2_FAMILY_DELTAS = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_top5pct_w2_featurefamily_scan_20260630"
    / "featurefamily_scan_deltas.csv"
)
W2_FS40_STRUCTURE_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_top5pct_w2_fs40_structure_scan_20260630"
    / "structure_scan_summary.csv"
)
W2_FS40_STRUCTURE_DELTAS = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_top5pct_w2_fs40_structure_scan_20260630"
    / "structure_scan_deltas.csv"
)
W3_STRUCTURE_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_top5pct_w3_structure_scan_20260630"
    / "structure_scan_summary.csv"
)
W3_STRUCTURE_DELTAS = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_top5pct_w3_structure_scan_20260630"
    / "structure_scan_deltas.csv"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rp = payload["recent_progress"]
    lines = [
        "# 四年观察期研究前沿状态 v23",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 5D `top5pct_w2` 特征族：`{rp['5d_top5pct_w2_featurefamily']['decision']}`",
        f"- 5D `top5pct_w2 + fs40` 结构扫参：`{rp['5d_top5pct_w2_fs40_structure']['decision']}`",
        f"- 5D `top5pct_w3` 结构扫参：`{rp['5d_top5pct_w3_structure']['decision']}`",
        f"- 下一步主攻：`{payload['next_focus']['priority_label']}` / `{payload['next_focus']['action']}`",
        "",
        "## 5D 新增事实",
        "",
        f"- `top5pct_w2` 分支最优仍落后当前 5D bestset，最佳结构候选相对当前 bestset `delta_objective={rp['5d_top5pct_w2_fs40_structure']['best_delta_objective_vs_current']:.6f}`",
        f"- `top5pct_w3` 分支结构扫参也未形成替代，最佳结构候选相对当前 bestset `delta_objective={rp['5d_top5pct_w3_structure']['best_delta_objective_vs_current']:.6f}`",
        f"- 因此 5D 当前已完成：同源直扫、库存扩扫、互补性检查、`top5pct_w2` 邻域、`top5pct_w3` 邻域五层排查",
        "",
        "## 边界",
        "",
        "- 本报告仅更新 research 状态。",
        "- 未切 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def extract_best_delta(delta_df: pd.DataFrame, baseline: str) -> dict[str, float | str]:
    subset = delta_df[delta_df["baseline"] == baseline].copy()
    subset = subset.sort_values("delta_objective", ascending=False).reset_index(drop=True)
    best = subset.iloc[0]
    return {
        "asset": str(best["asset"]),
        "delta_rank_ic": float(best["delta_rank_ic"]),
        "delta_top1": float(best["delta_top1"]),
        "delta_top5": float(best["delta_top5"]),
        "delta_objective": float(best["delta_objective"]),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frontier_v22 = load_json(FRONTIER_V22)

    w2_family_summary = load_csv(W2_FAMILY_SUMMARY)
    w2_family_deltas = load_csv(W2_FAMILY_DELTAS)
    w2_fs40_summary = load_csv(W2_FS40_STRUCTURE_SUMMARY)
    w2_fs40_deltas = load_csv(W2_FS40_STRUCTURE_DELTAS)
    w3_structure_summary = load_csv(W3_STRUCTURE_SUMMARY)
    w3_structure_deltas = load_csv(W3_STRUCTURE_DELTAS)

    recent_progress = dict(frontier_v22["recent_progress"])

    best_w2_family_vs_current = extract_best_delta(w2_family_deltas, "research5d_current")
    best_w2_fs40_vs_current = extract_best_delta(w2_fs40_deltas, "research5d_current")
    best_w3_structure_vs_current = extract_best_delta(w3_structure_deltas, "research5d_current")

    recent_progress["5d_top5pct_w2_featurefamily"] = {
        "decision": "top5pct_w2_featurefamily_no_candidate_beats_current_bestset",
        "best_asset": best_w2_family_vs_current["asset"],
        "best_delta_rank_ic_vs_current": best_w2_family_vs_current["delta_rank_ic"],
        "best_delta_top5_vs_current": best_w2_family_vs_current["delta_top5"],
        "best_delta_objective_vs_current": best_w2_family_vs_current["delta_objective"],
        "candidate_count": int(len(w2_family_summary[w2_family_summary["asset"].isin([
            "rebuild_fold1_fs64", "featurecount_fs40", "featurecount_fs80", "featurecount_fs120", "legacy_fs160_fold8"
        ])])),
        "evidence": str(W2_FAMILY_SUMMARY),
    }
    recent_progress["5d_top5pct_w2_fs40_structure"] = {
        "decision": "top5pct_w2_fs40_structure_saturated_no_current_bestset_replacement",
        "best_asset": best_w2_fs40_vs_current["asset"],
        "best_delta_rank_ic_vs_current": best_w2_fs40_vs_current["delta_rank_ic"],
        "best_delta_top5_vs_current": best_w2_fs40_vs_current["delta_top5"],
        "best_delta_objective_vs_current": best_w2_fs40_vs_current["delta_objective"],
        "candidate_count": int(len(w2_fs40_summary[w2_fs40_summary["asset"].isin([
            "base_fs40", "d2_l12_a03", "d2_l8_a01", "d3_l12_a03", "d3_l8_a01_lr004_n5000", "d4_l8_a02"
        ])])),
        "evidence": str(W2_FS40_STRUCTURE_SUMMARY),
    }
    recent_progress["5d_top5pct_w3_structure"] = {
        "decision": "top5pct_w3_structure_saturated_no_current_bestset_replacement",
        "best_asset": best_w3_structure_vs_current["asset"],
        "best_delta_rank_ic_vs_current": best_w3_structure_vs_current["delta_rank_ic"],
        "best_delta_top5_vs_current": best_w3_structure_vs_current["delta_top5"],
        "best_delta_objective_vs_current": best_w3_structure_vs_current["delta_objective"],
        "candidate_count": int(len(w3_structure_summary[w3_structure_summary["asset"].isin([
            "base_top5pct_w3", "d2_l4_a0", "d2_l8_a01", "d3_l8_a01", "d3_l12_a03", "d4_l4_a0", "d4_l8_a02_lr004_n5000"
        ])])),
        "evidence": str(W3_STRUCTURE_SUMMARY),
    }

    payload = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v23",
        "previous_frontier_path": str(FRONTIER_V22),
        "current_bestset": frontier_v22["current_bestset"],
        "recent_progress": recent_progress,
        "next_focus": {
            "priority_label": "10d",
            "action": "shift_to_10d_next_structure_or_source_axis",
            "reason": (
                "5D 已连续完成同源直扫、库存扩扫、互补性检查、top5pct_w2 邻域结构/特征扫描、"
                "top5pct_w3 邻域结构扫描，均未形成对 current_bestset_direct_guard_v1 的直接替代。"
                "继续在 5D 当前 top-quantile 加权邻域细抠的性价比已显著下降；"
                "10D 当前稳定性分数最高且仍处于 high optimize priority，更适合作为下一主攻标签。"
            ),
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

    (REPORT_DIR / "research_frontier_status_v23.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(REPORT_DIR / "research_frontier_status_v23.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
