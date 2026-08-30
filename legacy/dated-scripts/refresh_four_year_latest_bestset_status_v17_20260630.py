from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_latest_bestset_status_v17_20260630"
PRIOR_STATUS = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v16_20260630"
    / "latest_bestset_status_v16.json"
)
CONTROL_DIR = DATA_DIR / "reports" / "model_agent_four_year_control_assets_20260627" / "standard_eval"

CFG = {
    "label_key": "5d",
    "asset": "research_5d_four_year_current_bestset_direct_guard_v1_20260630",
    "table": "stock_predict_data_model_agent_four_year_5d_current_bestset_direct_guard_v1_20260630_executable_5d_open_return_research",
    "report_dir": DATA_DIR / "reports" / "model_agent_four_year_5d_current_bestset_direct_guard_v1_20260630",
    "summary_file": "current_bestset_direct_guard_v1_summary.json",
    "eval_file": (
        "standard_eval"
        "/executable_5d_open_return_four_year_current_bestset_direct_guard_v1_20260630_eval_summary.json"
    ),
    "annual_file": (
        "standard_eval"
        "/executable_5d_open_return_four_year_current_bestset_direct_guard_v1_20260630_annual_eval.csv"
    ),
    "monthly_control_file": (
        "standard_eval"
        "/executable_5d_open_return_four_year_current_bestset_direct_guard_v1_20260630_vs_four_year_control_monthly_delta.csv"
    ),
    "promotion_candidate_file": "promotion_candidate.json",
    "promotion_gate_file": "promotion_gate_result.json",
    "control_annual_file": "executable_5d_open_return_four_year_control_annual_eval.csv",
    "control_baseline_name": "four_year_control_5d",
    "prior_baseline_name": "front_rank_state_gate_v4_5d",
    "replacement_rule": (
        "full rank_ic/top1/top5 nonnegative vs prior 5d bestset, recent63/20 rank_ic/top5 nonnegative, and monthly top5 tail nonnegative"
    ),
    "source_note": "5D 当前 bestset 更新为 current_bestset_direct_guard_v1；相对上一版 5D bestset，Full RankIC、Top1、Top5 均为非负改进，月度 Top5 尾部不恶化，近期窗口持平。",
    "next_action": "5D 后续应继续围绕 current_bestset_direct_guard_v1 寻找能提升 recent63/recent20 的增量线；当前这版属于更稳的微调替换，不是大幅改观版。",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


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


def monthly_delta_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    top5_values = [float(row["top5"]) for row in rows]
    rank_ic_values = [float(row["rank_ic"]) for row in rows]
    return {
        "month_count": len(rows),
        "month_top5_positive": sum(value > 0.0 for value in top5_values),
        "month_top5_nonnegative": sum(value >= 0.0 for value in top5_values),
        "month_top5_negative": sum(value < 0.0 for value in top5_values),
        "month_min_rank_ic_delta": min(rank_ic_values),
        "month_min_top5_delta": min(top5_values),
    }


def gate_passed(gate_payload: dict[str, Any]) -> bool:
    if "hard_constraint_passed" in gate_payload:
        return bool(gate_payload["hard_constraint_passed"])
    return bool(gate_payload.get("result", {}).get("hard_constraint_passed"))


def should_replace(prior_delta: dict[str, float], summary: dict[str, Any], gate_ok: bool) -> bool:
    if not gate_ok:
        return False
    base_cmp = summary["materialize_report"]["comparisons"]["vs_current_bestset"]["delta"]
    return (
        float(prior_delta["full_rank_ic_delta"]) >= 0.0
        and float(base_cmp["top1"]) >= 0.0
        and float(prior_delta["full_top5_delta"]) >= 0.0
        and float(prior_delta["recent63_rank_ic_delta"]) >= 0.0
        and float(prior_delta["recent63_top5_delta"]) >= 0.0
        and float(prior_delta["recent20_rank_ic_delta"]) >= 0.0
        and float(prior_delta["recent20_top5_delta"]) >= 0.0
        and float(summary["month_stats_vs_base"]["min_top5_delta"]) >= 0.0
    )


def update_row(
    row: dict[str, Any],
    *,
    summary: dict[str, Any],
    eval_summary: dict[str, Any],
    annual_stats: dict[str, Any],
    monthly_stats: dict[str, Any],
) -> dict[str, Any]:
    control_cmp = comparison_block(eval_summary, CFG["control_baseline_name"])
    prior_cmp = comparison_block(eval_summary, CFG["prior_baseline_name"])
    quality = summary["materialize_report"]["table_quality"]
    out = dict(row)
    out["asset"] = CFG["asset"]
    out["table"] = CFG["table"]
    out["eval_min_trade_date"] = quality["min_trade_date"]
    out["eval_max_trade_date"] = quality["max_trade_date"]
    out["eval_trade_days"] = int(quality["trade_days"])
    out["full_rank_ic"] = float(eval_summary["full"]["rank_ic"])
    out["full_top1"] = float(eval_summary["full"]["top1"])
    out["full_top3"] = float(eval_summary["full"]["top3"])
    out["full_top5"] = float(eval_summary["full"]["top5"])
    out["full_top10"] = float(eval_summary["full"]["top10"])
    out["full_top20"] = float(eval_summary["full"]["top20"])
    out["full_rank_ic_delta"] = float(control_cmp["delta"]["rank_ic"])
    out["full_top1_delta"] = float(control_cmp["delta"]["top1"])
    out["full_top3_delta"] = float(control_cmp["delta"]["top3"])
    out["full_top5_delta"] = float(control_cmp["delta"]["top5"])
    out["full_top10_delta"] = float(control_cmp["delta"]["top10"])
    out["full_top20_delta"] = float(control_cmp["delta"]["top20"])
    out["recent20_abs_rank_ic"] = float(eval_summary["recent_windows"]["recent20"]["rank_ic"])
    out["recent20_abs_top1"] = float(eval_summary["recent_windows"]["recent20"]["top1"])
    out["recent20_abs_top5"] = float(eval_summary["recent_windows"]["recent20"]["top5"])
    out["recent63_abs_rank_ic"] = float(eval_summary["recent_windows"]["recent63"]["rank_ic"])
    out["recent63_abs_top1"] = float(eval_summary["recent_windows"]["recent63"]["top1"])
    out["recent63_abs_top5"] = float(eval_summary["recent_windows"]["recent63"]["top5"])
    out["recent20_rank_ic_delta"] = float(control_cmp["recent20_delta"]["rank_ic"])
    out["recent20_top1_delta"] = float(control_cmp["recent20_delta"]["top1"])
    out["recent20_top5_delta"] = float(control_cmp["recent20_delta"]["top5"])
    out["recent63_rank_ic_delta"] = float(control_cmp["recent63_delta"]["rank_ic"])
    out["recent63_top1_delta"] = float(control_cmp["recent63_delta"]["top1"])
    out["recent63_top5_delta"] = float(control_cmp["recent63_delta"]["top5"])
    out["month_count"] = monthly_stats["month_count"]
    out["month_top5_positive"] = monthly_stats["month_top5_positive"]
    out["month_top5_nonnegative"] = monthly_stats["month_top5_nonnegative"]
    out["month_top5_negative"] = monthly_stats["month_top5_negative"]
    out["month_min_rank_ic_delta"] = monthly_stats["month_min_rank_ic_delta"]
    out["month_min_top5_delta"] = monthly_stats["month_min_top5_delta"]
    out["year_count"] = annual_stats["year_count"]
    out["year_top5_positive"] = annual_stats["year_top5_positive"]
    out["year_top5_nonnegative"] = annual_stats["year_top5_nonnegative"]
    out["year_min_rank_ic_delta"] = annual_stats["year_min_rank_ic_delta"]
    out["year_min_top5_delta"] = annual_stats["year_min_top5_delta"]
    out["source_note"] = CFG["source_note"]
    out["next_action"] = CFG["next_action"]
    out["table_quality"] = {
        "row_count": int(quality["row_count"]),
        "min_trade_date": quality["min_trade_date"],
        "max_trade_date": quality["max_trade_date"],
        "trade_days": int(quality["trade_days"]),
        "null_pred_prob": int(quality["null_pred_prob"]),
        "duplicate_key_groups": int(quality["duplicate_key_groups"]),
        "latest_trade_date": quality["latest_trade_date"],
        "latest_day_rows": int(quality["latest_day_rows"]),
        "latest_day_stocks": int(quality["latest_day_stocks"]),
    }
    out["optimize_priority"] = "high"
    out["discussion_status"] = "candidate_discussion_pass"
    out["direct_delta_vs_prior_bestset"] = {
        "full_rank_ic_delta": float(prior_cmp["delta"]["rank_ic"]),
        "full_top5_delta": float(prior_cmp["delta"]["top5"]),
        "recent63_rank_ic_delta": float(prior_cmp["recent63_delta"]["rank_ic"]),
        "recent63_top5_delta": float(prior_cmp["recent63_delta"]["top5"]),
        "recent20_rank_ic_delta": float(prior_cmp["recent20_delta"]["rank_ic"]),
        "recent20_top5_delta": float(prior_cmp["recent20_delta"]["top5"]),
    }
    return out


def write_markdown(path: Path, generated_at: str, replaced_labels: list[str]) -> None:
    lines = [
        "# 四年观察最新 bestset 状态 v17",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 当前结论",
        "",
        "- 本次只更新 5D bestset。",
        f"- 新 5D 资产：`{CFG['asset']}`",
        "- 替换依据：相对上一版 5D bestset，Full RankIC、Top1、Top5 非负改进；Recent63/Recent20 不退化；月度 Top5 尾部非负。",
        "",
        "## 边界",
        "",
        "- 仍为 research-only。",
        "- 未训练模型。",
        "- 未改 formal manifest。",
        "- 未改 production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
        "",
        f"已替换标签：{', '.join(replaced_labels) if replaced_labels else '无'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prior = load_json(PRIOR_STATUS)
    details = prior["details"]
    report_dir = CFG["report_dir"]
    summary = load_json(report_dir / CFG["summary_file"])
    gate = load_json(report_dir / CFG["promotion_gate_file"])
    eval_summary = load_json(report_dir / CFG["eval_file"])
    annual_rows = load_csv_rows(report_dir / CFG["annual_file"])
    monthly_rows = load_csv_rows(report_dir / CFG["monthly_control_file"])
    control_annual = load_csv_rows(CONTROL_DIR / CFG["control_annual_file"])
    prior_delta = load_json(report_dir / CFG["promotion_candidate_file"])["current_baseline_delta"]
    gate_ok = gate_passed(gate)
    if not should_replace(prior_delta, summary, gate_ok):
        raise ValueError("5d candidate does not satisfy bestset replacement rule")

    annual_stats = annual_delta_stats(annual_rows, control_annual)
    monthly_stats = monthly_delta_stats(monthly_rows)

    rows = []
    replaced_labels: list[str] = []
    for row in prior["best_set"]:
        if row["label_key"] == CFG["label_key"]:
            new_row = update_row(row, summary=summary, eval_summary=eval_summary, annual_stats=annual_stats, monthly_stats=monthly_stats)
            rows.append(new_row)
            details[CFG["label_key"]] = {
                "record": new_row,
                "promotion_candidate_json": str(report_dir / CFG["promotion_candidate_file"]),
                "promotion_gate_result_json": str(report_dir / CFG["promotion_gate_file"]),
                "eval_summary_json": str(report_dir / CFG["eval_file"]),
                "annual_delta_csv": str(report_dir / CFG["annual_file"]),
                "monthly_delta_csv": str(report_dir / CFG["monthly_control_file"]),
                "annual_delta": annual_stats["annual_delta"],
                "summary_json": str(report_dir / CFG["summary_file"]),
            }
            replaced_labels.append(CFG["label_key"])
        else:
            rows.append(row)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (priority_order[item["optimize_priority"]], -float(item["stability_score"])))

    summary_json = {
        "generated_at": now_iso(),
        "scope": "four_year_latest_bestset_status_v17",
        "best_set": rows,
        "details": details,
        "change_log": {
            "replaced_labels": replaced_labels,
            "label_changes": {
                CFG["label_key"]: {
                    "old_asset": next(item["asset"] for item in prior["best_set"] if item["label_key"] == CFG["label_key"]),
                    "new_asset": CFG["asset"],
                    "replacement_rule": CFG["replacement_rule"],
                }
            },
        },
        "boundaries": prior["boundaries"],
    }
    (REPORT_DIR / "latest_bestset_status_v17.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(REPORT_DIR / "latest_bestset_status_v17.md", summary_json["generated_at"], replaced_labels)
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
