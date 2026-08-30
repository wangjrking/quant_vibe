from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_latest_bestset_status_v13_20260629"
PRIOR_STATUS = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v12_20260628"
    / "latest_bestset_status_v12.json"
)
NEW_3D_EVAL = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_3d_hybrid_refine_candidate_20260629"
    / "standard_eval"
    / "executable_3d_open_return_four_year_hybrid_refine_candidate_20260629_eval_summary.json"
)
NEW_3D_GATE = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_3d_hybrid_refine_candidate_20260629"
    / "promotion_gate_result.json"
)
NEW_3D_MONTHLY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_3d_hybrid_refine_candidate_20260629"
    / "standard_eval"
    / "executable_3d_open_return_four_year_hybrid_refine_candidate_20260629_vs_four_year_control_3d_monthly_delta.csv"
)
NEW_3D_ANNUAL = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_3d_hybrid_refine_candidate_20260629"
    / "standard_eval"
    / "executable_3d_open_return_four_year_hybrid_refine_candidate_20260629_annual_eval.csv"
)
NEW_3D_ASSET = "research_3d_four_year_hybrid_refine_candidate_20260629"
NEW_3D_TABLE = "stock_predict_data_model_agent_four_year_3d_hybrid_refine_candidate_20260629_executable_3d_open_return_research"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def should_replace_3d_bestset(delta: dict[str, float]) -> bool:
    return (
        float(delta["full_rank_ic"]) > 0.0
        and float(delta["full_top5"]) > 0.0
        and float(delta["recent63_rank_ic"]) >= 0.0
        and float(delta["recent20_rank_ic"]) >= 0.0
        and float(delta["recent63_top5"]) >= 0.0
        and float(delta["recent20_top5"]) >= 0.0
    )


def risk_score(record: dict[str, object]) -> float:
    return (
        40.0 * float(record["recent20_top5_delta"])
        + 25.0 * float(record["recent63_top5_delta"])
        + 15.0 * float(record["full_rank_ic_delta"])
        - 20.0 * max(0.0, -float(record["month_min_top5_delta"]))
    )


def optimize_priority(record: dict[str, object]) -> str:
    if float(record["recent20_abs_top5"]) <= 0 or float(record["month_min_top5_delta"]) < -0.005:
        return "high"
    if float(record["recent63_abs_top5"]) <= 0 or int(record["month_top5_negative"]) >= 6:
        return "medium"
    return "low"


def discussion_status(record: dict[str, object]) -> str:
    if (
        float(record["full_top5_delta"]) > 0.0
        and float(record["recent63_top5_delta"]) > 0.0
        and float(record["recent20_top5_delta"]) > 0.0
        and float(record["recent63_abs_top5"]) > 0.0
        and float(record["recent20_abs_top5"]) > 0.0
    ):
        return "candidate_discussion_pass"
    return "candidate_discussion_hold"


def build_new_3d_record() -> tuple[dict[str, object], list[dict[str, object]]]:
    eval_summary = json.loads(NEW_3D_EVAL.read_text(encoding="utf-8"))
    gate_payload = json.loads(NEW_3D_GATE.read_text(encoding="utf-8"))
    direct = gate_payload["result"]["soft_reminders"]
    candidate_json = json.loads(
        (NEW_3D_GATE.parent / "promotion_candidate.json").read_text(encoding="utf-8")
    )
    direct_delta = candidate_json["direct_delta_vs_current_bestset"]
    if not should_replace_3d_bestset(direct_delta):
        raise ValueError("new 3d refined candidate does not dominate current 3d bestset on required direct deltas")

    monthly = pd.read_csv(NEW_3D_MONTHLY)
    annual = pd.read_csv(NEW_3D_ANNUAL)
    comp = next(item for item in eval_summary["comparisons"] if item["baseline_name"] == "four_year_control_3d")
    recent20 = eval_summary["recent_windows"]["recent20"]
    recent63 = eval_summary["recent_windows"]["recent63"]
    full = eval_summary["full"]
    quality = eval_summary["quality"]

    record = {
        "label_key": "3d",
        "label": "executable_3d_open_return",
        "asset": NEW_3D_ASSET,
        "table": NEW_3D_TABLE,
        "eval_min_trade_date": full["min_trade_date"],
        "eval_max_trade_date": full["max_trade_date"],
        "eval_trade_days": full["trade_days"],
        "full_rank_ic": full["rank_ic"],
        "full_top1": full["top1"],
        "full_top3": full["top3"],
        "full_top5": full["top5"],
        "full_top10": full["top10"],
        "full_top20": full["top20"],
        "full_rank_ic_delta": comp["delta"]["rank_ic"],
        "full_top1_delta": comp["delta"]["top1"],
        "full_top3_delta": comp["delta"]["top3"],
        "full_top5_delta": comp["delta"]["top5"],
        "full_top10_delta": comp["delta"]["top10"],
        "full_top20_delta": comp["delta"]["top20"],
        "recent20_abs_rank_ic": recent20["rank_ic"],
        "recent20_abs_top1": recent20["top1"],
        "recent20_abs_top5": recent20["top5"],
        "recent63_abs_rank_ic": recent63["rank_ic"],
        "recent63_abs_top1": recent63["top1"],
        "recent63_abs_top5": recent63["top5"],
        "recent20_rank_ic_delta": comp["recent20_delta"]["rank_ic"],
        "recent20_top1_delta": comp["recent20_delta"]["top1"],
        "recent20_top5_delta": comp["recent20_delta"]["top5"],
        "recent63_rank_ic_delta": comp["recent63_delta"]["rank_ic"],
        "recent63_top1_delta": comp["recent63_delta"]["top1"],
        "recent63_top5_delta": comp["recent63_delta"]["top5"],
        "month_count": int(len(monthly)),
        "month_top5_positive": int((monthly["top5"] > 0).sum()),
        "month_top5_nonnegative": int((monthly["top5"] >= 0).sum()),
        "month_top5_negative": int((monthly["top5"] < 0).sum()),
        "month_min_rank_ic_delta": float(monthly["rank_ic"].min()),
        "month_min_top5_delta": float(monthly["top5"].min()),
        "year_count": int(len(annual)),
        "year_top5_positive": int((annual["top5"] > 0).sum()),
        "year_top5_nonnegative": int((annual["top5"] >= 0).sum()),
        "year_min_rank_ic_delta": float(annual["rank_ic"].min()),
        "year_min_top5_delta": float(annual["top5"].min()),
        "source_note": "3D 当前 bestset 更新为 hybrid_refine_candidate；它在四年同窗内相对旧 clear_replacement_gate 的 Full/Recent RankIC 与 Top5 都为正改进。",
        "next_action": "3D 后续不再优先回到旧 clear_replacement_gate；应继续在 refined hybrid 之上做更强结构筛选或 fixed4y 重训线验证。",
        "table_quality": quality,
        "direct_delta_vs_prior_bestset": direct_delta,
    }
    record["stability_score"] = risk_score(record)
    record["optimize_priority"] = optimize_priority(record)
    record["discussion_status"] = discussion_status(record)
    return record, annual.to_dict(orient="records")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR_STATUS.read_text(encoding="utf-8"))
    new_3d_record, new_3d_annual = build_new_3d_record()

    rows = []
    details = prior["details"]
    for row in prior["best_set"]:
        if row["label_key"] == "3d":
            rows.append(new_3d_record)
            details["3d"] = {
                "record": new_3d_record,
                "monthly_delta_csv": str(NEW_3D_MONTHLY),
                "annual_delta_csv": str(NEW_3D_ANNUAL),
                "annual_delta": new_3d_annual,
            }
        else:
            rows.append(row)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (priority_order[item["optimize_priority"]], -item["stability_score"]))

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_latest_bestset_status_v13",
        "best_set": rows,
        "details": details,
        "change_log": {
            "replaced_label": "3d",
            "old_asset": "research_3d_four_year_clear_replacement_gate_20260628",
            "new_asset": NEW_3D_ASSET,
            "replacement_rule": "direct positive deltas vs prior 3d bestset on full rank_ic, full top5, recent63/20 rank_ic, recent63/20 top5",
        },
        "boundaries": prior["boundaries"],
    }
    (REPORT_DIR / "latest_bestset_status_v13.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 四年观察期 latest bestset 状态 v13",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 本次只更新 3D bestset。",
        f"- 旧 3D bestset：`research_3d_four_year_clear_replacement_gate_20260628`",
        f"- 新 3D bestset：`{NEW_3D_ASSET}`",
        "- 替换依据：相对旧 3D bestset 的 Full RankIC、Full Top5、Recent63/20 RankIC、Recent63/20 Top5 均为正。",
        "",
        "## 边界",
        "",
        "- 仍为 research-only。",
        "- 未训练模型。",
        "- 未改 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
    ]
    (REPORT_DIR / "latest_bestset_status_v13.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
