from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_latest_bestset_status_v14_20260629"
PRIOR_STATUS = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v13_20260629"
    / "latest_bestset_status_v13.json"
)
NEW_10D_CANDIDATE = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_active_recent_blend_gate_v3_20260629"
    / "promotion_candidate.json"
)
NEW_10D_GATE = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_active_recent_blend_gate_v3_20260629"
    / "promotion_gate_result.json"
)
NEW_10D_ASSET = "research_10d_four_year_active_recent_blend_gate_v3_20260629"
NEW_10D_TABLE = "stock_predict_data_model_agent_four_year_10d_active_recent_blend_gate_v3_20260629_executable_10d_open_return_research"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def should_replace_10d_bestset(delta: dict[str, float]) -> bool:
    return (
        float(delta["full_rank_ic_delta"]) > 0.0
        and float(delta["full_top5_delta"]) > 0.0
        and float(delta["recent63_rank_ic_delta"]) > 0.0
        and float(delta["recent63_top5_delta"]) > 0.0
        and float(delta["recent20_rank_ic_delta"]) > 0.0
        and float(delta["recent20_top5_delta"]) > 0.0
    )


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR_STATUS.read_text(encoding="utf-8"))
    candidate = json.loads(NEW_10D_CANDIDATE.read_text(encoding="utf-8"))
    gate = json.loads(NEW_10D_GATE.read_text(encoding="utf-8"))

    current_delta = candidate["current_baseline_delta"]
    if not should_replace_10d_bestset(current_delta):
        raise ValueError("new 10d candidate does not dominate current 10d bestset on required direct deltas")
    if not gate["result"]["hard_constraint_passed"]:
        raise ValueError("new 10d candidate did not pass hard constraints")

    rows = []
    details = prior["details"]
    replaced = False
    for row in prior["best_set"]:
        if row["label_key"] == "10d":
            new_row = dict(row)
            new_row["asset"] = NEW_10D_ASSET
            new_row["table"] = NEW_10D_TABLE
            new_row["full_rank_ic_delta"] = candidate["full_rank_ic_delta"]
            new_row["full_top1_delta"] = candidate["full_top1_delta"]
            new_row["full_top5_delta"] = candidate["full_top5_delta"]
            new_row["full_top10_delta"] = candidate["full_top10_delta"]
            new_row["recent63_rank_ic_delta"] = candidate["recent63_rank_ic_delta"]
            new_row["recent63_top1_delta"] = candidate["recent63_top1_delta"]
            new_row["recent63_top5_delta"] = candidate["recent63_top5_delta"]
            new_row["recent20_rank_ic_delta"] = candidate["recent20_rank_ic_delta"]
            new_row["recent20_top1_delta"] = candidate["recent20_top1_delta"]
            new_row["recent20_top5_delta"] = candidate["recent20_top5_delta"]
            new_row["source_note"] = "10D 当前 bestset 更新为 active_recent_blend_gate_v3；它相对旧 active_recent_gate_hybrid_v2 的 Full/Recent RankIC 与 Top5 都为正改进。"
            new_row["next_action"] = "10D 后续应在 blend_gate_v3 之上继续压缩负月份，同时尽量保住当前对 Full/Recent TopN 的领先。"
            new_row["direct_delta_vs_prior_bestset"] = current_delta
            rows.append(new_row)
            details["10d"] = {
                "record": new_row,
                "promotion_candidate_json": str(NEW_10D_CANDIDATE),
                "promotion_gate_result_json": str(NEW_10D_GATE),
            }
            replaced = True
        else:
            rows.append(row)
    if not replaced:
        raise ValueError("10d row not found in prior bestset snapshot")

    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (priority_order[item["optimize_priority"]], -item["stability_score"]))

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_latest_bestset_status_v14",
        "best_set": rows,
        "details": details,
        "change_log": {
            "replaced_label": "10d",
            "old_asset": "research_10d_four_year_active_recent_gate_hybrid_v2_20260628",
            "new_asset": NEW_10D_ASSET,
            "replacement_rule": "direct positive deltas vs prior 10d bestset on full rank_ic, full top5, recent63/20 rank_ic, recent63/20 top5",
        },
        "boundaries": prior["boundaries"],
    }
    (REPORT_DIR / "latest_bestset_status_v14.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 四年观察期 latest bestset 状态 v14",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 本次只更新 10D bestset。",
        "- 新 10D bestset 已替换旧 active_recent_gate_hybrid_v2。",
        f"- 新资产：`{NEW_10D_ASSET}`",
        "- 替换依据：相对旧 10D bestset 的 Full RankIC、Full Top5、Recent63/20 RankIC、Recent63/20 Top5 均为正。",
        "",
        "## 边界",
        "",
        "- 仍为 research-only。",
        "- 未训练模型。",
        "- 未改 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
    ]
    (REPORT_DIR / "latest_bestset_status_v14.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
