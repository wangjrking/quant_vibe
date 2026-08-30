from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_candidate_status_v2_20260628"

SPECS = {
    "executable_1d_open_return": {
        "candidate_json": DATA_DIR
        / "reports"
        / "model_agent_four_year_1d_splice_condblend_20260628"
        / "promotion_candidate.json",
        "replaced_asset": "research_1d_four_year_splice_condblend_20260628",
        "replaced_table": "stock_predict_data_model_agent_four_year_splice_condblend_20260628_executable_1d_open_return_research",
        "note": "1D 本轮未替换，继续沿用四年 splice_condblend。",
    },
    "executable_3d_open_return": {
        "candidate_json": DATA_DIR
        / "reports"
        / "model_agent_four_year_3d_clear_replacement_gate_20260628"
        / "promotion_candidate.json",
        "replaced_asset": "research_3d_four_year_clear_replacement_gate_20260628",
        "replaced_table": "stock_predict_data_model_agent_four_year_3d_clear_replacement_gate_20260628_executable_3d_open_return_research",
        "note": "3D 本轮未替换，继续沿用 clear_replacement_gate。",
    },
    "executable_5d_open_return": {
        "candidate_json": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_lightweight_blend_v1_20260628"
        / "promotion_candidate.json",
        "replaced_asset": "research_5d_four_year_clear_replacement_gate_20260628",
        "replaced_table": "stock_predict_data_model_agent_four_year_5d_clear_replacement_gate_20260628_executable_5d_open_return_research",
        "note": "5D 新切到 lightweight_blend_v1；相对 control 更强，相对旧 5D canonical 为 Top5 更强、RankIC 略弱。",
    },
    "executable_10d_open_return": {
        "candidate_json": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_lightweight_blend_v1_20260628"
        / "promotion_candidate.json",
        "replaced_asset": "research_10d_four_year_clear_replacement_time_splice_20260628",
        "replaced_table": "stock_predict_data_model_agent_four_year_10d_clear_replacement_time_splice_20260628_executable_10d_open_return_research",
        "note": "10D 新切到 lightweight_blend_v1；相对 control 和旧 10D canonical 都有正增。",
    },
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_candidate(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(label: str, spec: dict) -> dict:
    candidate = load_candidate(spec["candidate_json"])
    current_delta = candidate.get("current_baseline_delta", {})
    return {
        "label": label,
        "asset": candidate["asset"],
        "table": candidate["table"],
        "decision": "promotable_formal_candidate",
        "target_approval_status": "approved_for_l4_candidate_only_after_four_year_observation",
        "full_rank_ic_delta_vs_control": candidate["full_rank_ic_delta"],
        "full_top5_delta_vs_control": candidate["full_top5_delta"],
        "recent63_top5_delta_vs_control": candidate["recent63_top5_delta"],
        "recent20_top5_delta_vs_control": candidate["recent20_top5_delta"],
        "full_rank_ic_delta_vs_previous_current": current_delta.get("full_rank_ic_delta"),
        "full_top5_delta_vs_previous_current": current_delta.get("full_top5_delta"),
        "recent63_rank_ic_delta_vs_previous_current": current_delta.get("recent63_rank_ic_delta"),
        "recent63_top5_delta_vs_previous_current": current_delta.get("recent63_top5_delta"),
        "recent20_rank_ic_delta_vs_previous_current": current_delta.get("recent20_rank_ic_delta"),
        "recent20_top5_delta_vs_previous_current": current_delta.get("recent20_top5_delta"),
        "replaced_asset": spec["replaced_asset"],
        "replaced_table": spec["replaced_table"],
        "candidate_json": str(spec["candidate_json"]),
        "note": spec["note"],
    }


def write_markdown(rows: list[dict]) -> None:
    lines = [
        "# 四年候选状态 v2",
        "",
        f"生成时间：{now_iso()}",
        "",
        "## 当前结论",
        "",
        "- 本轮在不训练模型、不改 formal/production manifest 的前提下，新增两版 research-only 候选：5D lightweight_blend_v1、10D lightweight_blend_v1。",
        "- 1D、3D 维持上一版四年最佳候选不变。",
        "- 5D 新候选相对旧 5D canonical：Top5 更强，RankIC 略弱，属于可进入生产候选讨论但不构成单调支配。",
        "- 10D 新候选相对旧 10D canonical：Full/Recent Top5 与 Full RankIC 均为正增，是目前 10D 四年观察期下的新 best set。",
        "",
        "## 标签摘要",
        "",
        "| 标签 | 当前候选 | Full RankIC Δ vs control | Full Top5 Δ vs control | Recent63 Top5 Δ vs control | Recent20 Top5 Δ vs control | 相对上一版 current 的判断 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["full_top5_delta_vs_previous_current"] is None:
            current_note = "沿用"
        else:
            current_note = (
                f"RankIC {row['full_rank_ic_delta_vs_previous_current']:+.6f}, "
                f"Top5 {row['full_top5_delta_vs_previous_current']:+.6f}"
            )
        lines.append(
            f"| {row['label']} | `{row['asset']}` | "
            f"{row['full_rank_ic_delta_vs_control']:.6f} | {row['full_top5_delta_vs_control']:.6f} | "
            f"{row['recent63_top5_delta_vs_control']:.6f} | {row['recent20_top5_delta_vs_control']:.6f} | {current_note} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本次只更新四年研究候选状态，不训练模型。",
            "- 不改 formal manifest，不改 production manifest。",
            "- 不生成信号，不跑回测。",
        ]
    )
    (REPORT_DIR / "four_year_candidate_status_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [build_row(label, spec) for label, spec in SPECS.items()]
    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_candidate_status_v2_after_5d_10d_lightweight_blend",
        "current_best_candidates": {row["label"]: row for row in rows},
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "four_year_candidate_status_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
