from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_latest_bestset_status_v15_20260629"
PRIOR_STATUS = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v14_20260629"
    / "latest_bestset_status_v14.json"
)
NEW_5D_DIR = DATA_DIR / "reports" / "model_agent_four_year_5d_front_rank_state_gate_v4_20260629"
NEW_5D_CANDIDATE = NEW_5D_DIR / "promotion_candidate.json"
NEW_5D_GATE = NEW_5D_DIR / "promotion_gate_result.json"
NEW_5D_SUMMARY = NEW_5D_DIR / "front_rank_state_gate_v4_summary.json"
NEW_5D_EVAL = (
    NEW_5D_DIR
    / "standard_eval"
    / "executable_5d_open_return_four_year_front_rank_state_gate_v4_20260629_eval_summary.json"
)
NEW_5D_ANNUAL = (
    NEW_5D_DIR
    / "standard_eval"
    / "executable_5d_open_return_four_year_front_rank_state_gate_v4_20260629_annual_eval.csv"
)
CONTROL_5D_ANNUAL = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_control_assets_20260627"
    / "standard_eval"
    / "executable_5d_open_return_four_year_control_annual_eval.csv"
)
NEW_5D_ASSET = "research_5d_four_year_front_rank_state_gate_v4_20260629"
NEW_5D_TABLE = (
    "stock_predict_data_model_agent_four_year_5d_front_rank_state_gate_v4_20260629"
    "_executable_5d_open_return_research"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def should_replace_5d_bestset(
    delta: dict[str, float],
    *,
    min_month_top5_delta_vs_base: float,
    gate_passed: bool,
) -> bool:
    return (
        gate_passed
        and float(delta["full_rank_ic_delta"]) > 0.0
        and float(delta["full_top5_delta"]) > 0.0
        and float(delta["recent63_top5_delta"]) >= 0.0
        and float(delta["recent20_top5_delta"]) >= 0.0
        and float(min_month_top5_delta_vs_base) >= 0.0
    )


def comparison_block(eval_summary: dict[str, Any], baseline_name: str) -> dict[str, Any]:
    for item in eval_summary["comparisons"]:
        if item["baseline_name"] == baseline_name:
            return item
    raise KeyError(f"baseline_name not found: {baseline_name}")


def annual_delta_stats(candidate_rows: list[dict[str, str]], control_rows: list[dict[str, str]]) -> dict[str, Any]:
    control_by_period = {int(row["period"]): row for row in control_rows}
    deltas: list[dict[str, float | int]] = []
    for row in candidate_rows:
        period = int(row["period"])
        control = control_by_period[period]
        deltas.append(
            {
                "period": period,
                "rank_ic": float(row["rank_ic"]) - float(control["rank_ic"]),
                "top1": float(row["top1"]) - float(control["top1"]),
                "top3": float(row["top3"]) - float(control["top3"]),
                "top5": float(row["top5"]) - float(control["top5"]),
                "top10": float(row["top10"]) - float(control["top10"]),
                "top20": float(row["top20"]) - float(control["top20"]),
                "trade_days": int(row["trade_days"]),
            }
        )
    top5_values = [float(row["top5"]) for row in deltas]
    rank_ic_values = [float(row["rank_ic"]) for row in deltas]
    return {
        "annual_delta": deltas,
        "year_count": len(deltas),
        "year_top5_positive": sum(value > 0.0 for value in top5_values),
        "year_top5_nonnegative": sum(value >= 0.0 for value in top5_values),
        "year_min_rank_ic_delta": min(rank_ic_values),
        "year_min_top5_delta": min(top5_values),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prior = load_json(PRIOR_STATUS)
    candidate = load_json(NEW_5D_CANDIDATE)
    gate = load_json(NEW_5D_GATE)
    summary = load_json(NEW_5D_SUMMARY)
    eval_summary = load_json(NEW_5D_EVAL)
    candidate_annual = load_csv_rows(NEW_5D_ANNUAL)
    control_annual = load_csv_rows(CONTROL_5D_ANNUAL)

    current_delta = candidate["current_baseline_delta"]
    gate_passed = bool(gate["result"]["hard_constraint_passed"])
    min_month_top5_delta_vs_base = float(summary["best"]["min_month_top5_delta_vs_base"])
    if not should_replace_5d_bestset(
        current_delta,
        min_month_top5_delta_vs_base=min_month_top5_delta_vs_base,
        gate_passed=gate_passed,
    ):
        raise ValueError("new 5d candidate does not satisfy bestset replacement rule")

    control_cmp = comparison_block(eval_summary, "four_year_control_5d")
    base_cmp = comparison_block(eval_summary, "four_year_lightweight_blend_v1_5d")
    annual_stats = annual_delta_stats(candidate_annual, control_annual)

    rows = []
    details = prior["details"]
    replaced = False
    for row in prior["best_set"]:
        if row["label_key"] == "5d":
            new_row = dict(row)
            new_row["asset"] = NEW_5D_ASSET
            new_row["table"] = NEW_5D_TABLE
            new_row["eval_min_trade_date"] = eval_summary["full"]["min_trade_date"]
            new_row["eval_max_trade_date"] = eval_summary["full"]["max_trade_date"]
            new_row["eval_trade_days"] = int(eval_summary["full"]["trade_days"])
            new_row["full_rank_ic"] = float(eval_summary["full"]["rank_ic"])
            new_row["full_top1"] = float(eval_summary["full"]["top1"])
            new_row["full_top3"] = float(eval_summary["full"]["top3"])
            new_row["full_top5"] = float(eval_summary["full"]["top5"])
            new_row["full_top10"] = float(eval_summary["full"]["top10"])
            new_row["full_top20"] = float(eval_summary["full"]["top20"])
            new_row["full_rank_ic_delta"] = float(control_cmp["delta"]["rank_ic"])
            new_row["full_top1_delta"] = float(control_cmp["delta"]["top1"])
            new_row["full_top3_delta"] = float(control_cmp["delta"]["top3"])
            new_row["full_top5_delta"] = float(control_cmp["delta"]["top5"])
            new_row["full_top10_delta"] = float(control_cmp["delta"]["top10"])
            new_row["full_top20_delta"] = float(control_cmp["delta"]["top20"])
            new_row["recent20_abs_rank_ic"] = float(eval_summary["recent_windows"]["recent20"]["rank_ic"])
            new_row["recent20_abs_top1"] = float(eval_summary["recent_windows"]["recent20"]["top1"])
            new_row["recent20_abs_top5"] = float(eval_summary["recent_windows"]["recent20"]["top5"])
            new_row["recent63_abs_rank_ic"] = float(eval_summary["recent_windows"]["recent63"]["rank_ic"])
            new_row["recent63_abs_top1"] = float(eval_summary["recent_windows"]["recent63"]["top1"])
            new_row["recent63_abs_top5"] = float(eval_summary["recent_windows"]["recent63"]["top5"])
            new_row["recent20_rank_ic_delta"] = float(control_cmp["recent20_delta"]["rank_ic"])
            new_row["recent20_top1_delta"] = float(control_cmp["recent20_delta"]["top1"])
            new_row["recent20_top5_delta"] = float(control_cmp["recent20_delta"]["top5"])
            new_row["recent63_rank_ic_delta"] = float(control_cmp["recent63_delta"]["rank_ic"])
            new_row["recent63_top1_delta"] = float(control_cmp["recent63_delta"]["top1"])
            new_row["recent63_top5_delta"] = float(control_cmp["recent63_delta"]["top5"])
            new_row["month_count"] = int(summary["best"]["month_count_vs_control"])
            new_row["month_top5_positive"] = int(summary["best"]["positive_top5_months_vs_control"])
            new_row["month_top5_nonnegative"] = int(summary["best"]["nonnegative_top5_months_vs_control"])
            new_row["month_top5_negative"] = (
                int(summary["best"]["month_count_vs_control"])
                - int(summary["best"]["nonnegative_top5_months_vs_control"])
            )
            new_row["month_min_rank_ic_delta"] = float(candidate["min_period_rank_ic_delta"])
            new_row["month_min_top5_delta"] = float(candidate["min_period_top5_delta"])
            new_row["year_count"] = annual_stats["year_count"]
            new_row["year_top5_positive"] = annual_stats["year_top5_positive"]
            new_row["year_top5_nonnegative"] = annual_stats["year_top5_nonnegative"]
            new_row["year_min_rank_ic_delta"] = annual_stats["year_min_rank_ic_delta"]
            new_row["year_min_top5_delta"] = annual_stats["year_min_top5_delta"]
            new_row["source_note"] = (
                "5D 当前 bestset 更新为 front_rank_state_gate_v4；"
                "相对 lightweight_blend_v1 保持 Full RankIC 和 Top5 正改进，"
                "且月度 Top5 对基线非负。"
            )
            new_row["next_action"] = (
                "5D 后续应围绕 front_rank_state_gate_v4 继续做新公式搜索，"
                "重点继续抬 Full RankIC，并控制月度尾部。"
            )
            new_row["table_quality"] = dict(candidate["evidence"])  # placeholder overwritten below
            new_row["table_quality"] = {
                "row_count": int(eval_summary["quality"]["row_count"]),
                "min_trade_date": eval_summary["quality"]["min_trade_date"],
                "max_trade_date": eval_summary["quality"]["max_trade_date"],
                "trade_days": int(eval_summary["quality"]["trade_days"]),
                "null_pred_prob": int(eval_summary["quality"]["null_pred_prob"]),
                "duplicate_key_groups": int(eval_summary["quality"]["duplicate_key_groups"]),
                "latest_trade_date": eval_summary["quality"]["latest_trade_date"],
                "latest_day_rows": int(eval_summary["quality"]["latest_day_rows"]),
                "latest_day_stocks": int(eval_summary["quality"]["latest_day_stocks"]),
            }
            new_row["stability_score"] = float(row["stability_score"])
            new_row["optimize_priority"] = "high"
            new_row["discussion_status"] = "candidate_discussion_pass"
            new_row["direct_delta_vs_prior_bestset"] = {
                "full_rank_ic_delta": float(base_cmp["delta"]["rank_ic"]),
                "full_top5_delta": float(base_cmp["delta"]["top5"]),
                "recent63_rank_ic_delta": float(base_cmp["recent63_delta"]["rank_ic"]),
                "recent63_top5_delta": float(base_cmp["recent63_delta"]["top5"]),
                "recent20_rank_ic_delta": float(base_cmp["recent20_delta"]["rank_ic"]),
                "recent20_top5_delta": float(base_cmp["recent20_delta"]["top5"]),
            }
            rows.append(new_row)
            details["5d"] = {
                "record": new_row,
                "promotion_candidate_json": str(NEW_5D_CANDIDATE),
                "promotion_gate_result_json": str(NEW_5D_GATE),
                "monthly_delta_csv": str(
                    NEW_5D_DIR
                    / "standard_eval"
                    / "executable_5d_open_return_four_year_front_rank_state_gate_v4_20260629_vs_four_year_control_5d_monthly_delta.csv"
                ),
                "annual_delta_csv": str(NEW_5D_ANNUAL),
                "annual_delta": annual_stats["annual_delta"],
            }
            replaced = True
        else:
            rows.append(row)

    if not replaced:
        raise ValueError("5d row not found in prior bestset snapshot")

    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (priority_order[item["optimize_priority"]], -item["stability_score"]))

    summary_json = {
        "generated_at": now_iso(),
        "scope": "four_year_latest_bestset_status_v15",
        "best_set": rows,
        "details": details,
        "change_log": {
            "replaced_label": "5d",
            "old_asset": "research_5d_four_year_lightweight_blend_v1_20260628",
            "new_asset": NEW_5D_ASSET,
            "replacement_rule": (
                "positive full_rank_ic_delta/full_top5_delta vs prior 5d bestset, "
                "nonnegative recent63/20 top5 delta, nonnegative monthly top5 tail vs prior bestset"
            ),
        },
        "boundaries": prior["boundaries"],
    }
    (REPORT_DIR / "latest_bestset_status_v15.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 四年观察期 latest bestset 状态 v15",
        "",
        f"生成时间：{summary_json['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 本次只更新 5D bestset。",
        f"- 新 5D bestset：`{NEW_5D_ASSET}`。",
        "- 替换依据：相对旧 5D bestset 的 Full RankIC / Full Top5 为正，Recent63/20 Top5 非负，且月度 Top5 对旧 bestset 非负。",
        "",
        "## 边界",
        "",
        "- 仍为 research-only。",
        "- 未训练模型。",
        "- 未改 formal / production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
    ]
    (REPORT_DIR / "latest_bestset_status_v15.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
