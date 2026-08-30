from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v22_20260630"

FRONTIER_V21 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v21_20260630"
    / "research_frontier_status_v21.json"
)
FIVE_D_BROAD_SCAN = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_broad_asset_inventory_20260630"
    / "broad_asset_inventory_summary.json"
)
FIVE_D_COMPLEMENTARITY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_candidate_complementarity_20260630"
    / "candidate_complementarity_summary.json"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rp = payload["recent_progress"]
    lines = [
        "# 四年观察期研究前沿状态 v22",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 1D：`{rp['1d_balanced_dualgate']['decision']}`",
        f"- 5D 同源直扫：`{rp['5d_same_source_direct_scan']['decision']}`",
        f"- 5D 广域库存：`{rp['5d_broad_asset_inventory']['decision']}`",
        f"- 5D 互补性：`{rp['5d_candidate_complementarity']['decision']}`",
        f"- 10D：`{rp['10d_same_source_saturation']['decision']}`",
        f"- 下一步主攻：`{payload['next_focus']['priority_label']}` / `{payload['next_focus']['action']}`",
        "",
        "## 5D 新增事实",
        "",
        f"- 5D 广域四年资产扫描：`{rp['5d_broad_asset_inventory']['candidate_count']}` 个资产，`replaceable_count={rp['5d_broad_asset_inventory']['replaceable_count']}`",
        f"- 5D 互补性判断：`{rp['5d_candidate_complementarity']['decision_reason']}`",
        "",
        "## 边界",
        "",
        "- 本报告仅更新 research 状态。",
        "- 未训练模型。",
        "- 未修改 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frontier_v21 = load_json(FRONTIER_V21)
    broad_scan = load_json(FIVE_D_BROAD_SCAN)
    complementarity = load_json(FIVE_D_COMPLEMENTARITY)

    recent_progress = dict(frontier_v21["recent_progress"])
    recent_progress["5d_broad_asset_inventory"] = {
        "decision": "broad_inventory_no_replaceable_candidate",
        "candidate_count": int(broad_scan["candidate_count"]),
        "replaceable_count": int(broad_scan["replaceable_count"]),
        "best_candidate_dir": broad_scan["best_by_full_top5"]["candidate_dir"],
        "best_full_top5_delta": float(broad_scan["best_by_full_top5"]["full_top5_delta"]),
        "best_full_rank_ic_delta": float(broad_scan["best_by_full_top5"]["full_rank_ic_delta"]),
        "best_recent63_top5_delta": float(broad_scan["best_by_full_top5"]["recent63_top5_delta"]),
        "best_recent20_top5_delta": float(broad_scan["best_by_full_top5"]["recent20_top5_delta"]),
        "evidence": str(FIVE_D_BROAD_SCAN),
    }
    recent_progress["5d_candidate_complementarity"] = {
        "decision": complementarity["decision"]["next_route"],
        "decision_reason": complementarity["decision"]["reason"],
        "best_candidate": complementarity["decision"]["best_candidate"]["candidate"],
        "best_recent63_top5_delta": float(complementarity["decision"]["best_candidate"]["recent63_top5_delta"]),
        "best_recent20_top5_delta": float(complementarity["decision"]["best_candidate"]["recent20_top5_delta"]),
        "best_salvage_top5_day_ratio": float(complementarity["decision"]["best_candidate"]["salvage_top5_day_ratio"]),
        "evidence": str(FIVE_D_COMPLEMENTARITY),
    }

    payload = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v22",
        "previous_frontier_path": str(FRONTIER_V21),
        "current_bestset": frontier_v21["current_bestset"],
        "recent_progress": recent_progress,
        "next_focus": {
            "priority_label": "5d",
            "action": "switch_5d_to_new_training_family",
            "reason": (
                "5D 同源直扫、广域库存扫描和候选互补性分析都没有支持继续做库存替换或跨源融合；"
                "下一步应进入新的训练族。"
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
    (REPORT_DIR / "research_frontier_status_v22.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(REPORT_DIR / "research_frontier_status_v22.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
