from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_research_frontier_status_v24_20260630"

FRONTIER_V23 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v23_20260630"
    / "research_frontier_status_v23.json"
)
FS40_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_newaxis_fs40_candidate_20260630"
    / "newaxis_candidate_summary.json"
)
FS80_SUMMARY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_newaxis_fs80_candidate_20260630"
    / "newaxis_candidate_summary.json"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rp = payload["recent_progress"]
    fs40 = rp["10d_newaxis_fs40_complete"]
    fs80 = rp["10d_newaxis_fs80_complete"]
    lines = [
        "# 四年观察期研究前沿状态 v24",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 10D `fs40`：`{fs40['decision']}`",
        f"- 10D `fs80`：`{fs80['decision']}`",
        f"- 下一步主攻：`{payload['next_focus']['priority_label']}` / `{payload['next_focus']['action']}`",
        "",
        "## 10D 本轮新增事实",
        "",
        f"- `fs40` 已完成全 17 折、四年全窗口评估，但 `full_rank_ic_delta={fs40['full_rank_ic_delta_vs_control']:+.6f}`、`full_top5_delta={fs40['full_top5_delta_vs_control']:+.6f}`，硬门槛未过。",
        f"- `fs80` 已完成全 17 折、四年全窗口评估，`full_top5_delta={fs80['full_top5_delta_vs_control']:+.6f}` 为正，但 `full_rank_ic_delta={fs80['full_rank_ic_delta_vs_control']:+.6f}` 仍为负，因此仍不能进入生产候选讨论。",
        f"- `fs80` 相对当前 10D bestset 的近期表现分化：`recent20_top5_delta={fs80['recent20_top5_delta_vs_current']:+.6f}`，但 `recent63_top5_delta={fs80['recent63_top5_delta_vs_current']:+.6f}`。",
        "",
        "## 判断",
        "",
        "- 10D 的 `fs40` 与 `fs80` 两条新结构轴都已完成完整验证，但都没有通过四年硬门槛。",
        "- 因此当前 10D bestset 维持不变，下一步不该回头重复这两条分支，而应继续推进剩余更宽特征轴或新的来源轴。",
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


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frontier_v23 = load_json(FRONTIER_V23)
    fs40 = load_json(FS40_SUMMARY)
    fs80 = load_json(FS80_SUMMARY)

    recent_progress = dict(frontier_v23["recent_progress"])
    recent_progress["10d_newaxis_fs40_complete"] = {
        "decision": "full_candidate_failed_four_year_hard_gate",
        "asset": fs40["asset"],
        "table": fs40["table"],
        "hard_constraint_passed": bool(fs40["gate_result"]["hard_constraint_passed"]),
        "failed_hard_constraints": list(fs40["gate_result"]["failed_hard_constraints"]),
        "full_rank_ic_delta_vs_control": float(fs40["full_delta_vs_control"]["rank_ic"]),
        "full_top5_delta_vs_control": float(fs40["full_delta_vs_control"]["top5"]),
        "full_rank_ic_delta_vs_current": float(fs40["full_delta_vs_current_bestset"]["rank_ic"]),
        "full_top5_delta_vs_current": float(fs40["full_delta_vs_current_bestset"]["top5"]),
        "recent63_top5_delta_vs_current": float(fs40["recent63_delta_vs_current_bestset"]["top5"]),
        "recent20_top5_delta_vs_current": float(fs40["recent20_delta_vs_current_bestset"]["top5"]),
        "latest_trade_date": str(fs40["quality"]["latest_trade_date"]),
        "trade_days": int(fs40["quality"]["trade_days"]),
        "evidence": str(FS40_SUMMARY),
    }
    recent_progress["10d_newaxis_fs80_complete"] = {
        "decision": "full_candidate_top5_improved_but_rankic_hard_gate_failed",
        "asset": fs80["asset"],
        "table": fs80["table"],
        "hard_constraint_passed": bool(fs80["gate_result"]["hard_constraint_passed"]),
        "failed_hard_constraints": list(fs80["gate_result"]["failed_hard_constraints"]),
        "full_rank_ic_delta_vs_control": float(fs80["full_delta_vs_control"]["rank_ic"]),
        "full_top5_delta_vs_control": float(fs80["full_delta_vs_control"]["top5"]),
        "full_rank_ic_delta_vs_current": float(fs80["full_delta_vs_current_bestset"]["rank_ic"]),
        "full_top5_delta_vs_current": float(fs80["full_delta_vs_current_bestset"]["top5"]),
        "recent63_top5_delta_vs_current": float(fs80["recent63_delta_vs_current_bestset"]["top5"]),
        "recent20_top5_delta_vs_current": float(fs80["recent20_delta_vs_current_bestset"]["top5"]),
        "latest_trade_date": str(fs80["quality"]["latest_trade_date"]),
        "trade_days": int(fs80["quality"]["trade_days"]),
        "evidence": str(FS80_SUMMARY),
    }

    payload = {
        "generated_at": now_iso(),
        "scope": "four_year_research_frontier_status_v24",
        "previous_frontier_path": str(FRONTIER_V23),
        "current_bestset": frontier_v23["current_bestset"],
        "recent_progress": recent_progress,
        "next_focus": {
            "priority_label": "10d",
            "action": "start_fs120_or_new_source_axis_after_fs40_fs80_failures",
            "reason": (
                "10D 新结构轴里的 fs40 与 fs80 都已完成全 17 折和四年全窗口评估。"
                "其中 fs40 在 Full RankIC 和 Full Top5 上都未打赢 control/current bestset，"
                "fs80 虽然 Full Top5 为正增量，但 Full RankIC 仍明显为负，未过四年硬门槛。"
                "因此 10D 当前 bestset 保持不变，下一步应继续推进剩余更宽特征轴 fs120，"
                "或转向新的来源轴，而不是重复 fs40/fs80 这两个已证伪分支。"
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

    (REPORT_DIR / "research_frontier_status_v24.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(REPORT_DIR / "research_frontier_status_v24.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
