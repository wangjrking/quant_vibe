from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_latest_bestset_status_v16_20260630"
PRIOR_STATUS = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v15_20260629"
    / "latest_bestset_status_v15.json"
)
CONTROL_DIR = DATA_DIR / "reports" / "model_agent_four_year_control_assets_20260627" / "standard_eval"
CONTROL_10D_ANNUAL = CONTROL_DIR / "executable_10d_open_return_four_year_control_annual_eval.csv"

UPDATE_CONFIG = {
    "3d": {
        "asset": "research_3d_four_year_recent_safe_microblend_v1_20260630",
        "table": "stock_predict_data_model_agent_four_year_3d_recent_safe_microblend_v1_20260630_executable_3d_open_return_research",
        "report_dir": DATA_DIR / "reports" / "model_agent_four_year_3d_recent_safe_microblend_v1_20260630",
        "summary_file": "recent_safe_microblend_v1_summary.json",
        "eval_file": (
            "standard_eval"
            "/executable_3d_open_return_four_year_recent_safe_microblend_v1_20260630_eval_summary.json"
        ),
        "annual_file": (
            "standard_eval"
            "/executable_3d_open_return_four_year_recent_safe_microblend_v1_20260630_annual_eval.csv"
        ),
        "monthly_control_file": (
            "standard_eval"
            "/executable_3d_open_return_four_year_recent_safe_microblend_v1_20260630_vs_four_year_control_monthly_delta.csv"
        ),
        "promotion_candidate_file": "promotion_candidate.json",
        "promotion_gate_file": "promotion_gate_result.json",
        "control_annual_file": "executable_3d_open_return_four_year_control_annual_eval.csv",
        "control_baseline_name": "four_year_control_3d",
        "prior_baseline_name": "current_bestset_3d",
        "replacement_rule": (
            "full rank_ic/top5 and recent63/20 rank_ic/top5 must be nonnegative vs prior 3d bestset, "
            "with nonnegative monthly top5 tail vs prior bestset"
        ),
        "source_note": "3D 当前 bestset 更新为 recent_safe_microblend_v1；相对上一版 3D bestset，"
        "Full/Recent63/Recent20 的 RankIC 与 Top5 均非负改善，且月度 Top5 尾部不再恶化。",
        "next_action": "3D 后续可继续围绕 recent_safe_microblend_v1 做更强的结构筛选，但当前提升幅度较小，"
        "应优先继续寻找更高信噪比的增量来源。",
    },
    "10d": {
        "asset": "research_10d_four_year_v3_direct_guard_v7_20260630",
        "table": "stock_predict_data_model_agent_four_year_10d_v3_direct_guard_v7_20260630_executable_10d_open_return_research",
        "report_dir": DATA_DIR / "reports" / "model_agent_four_year_10d_v3_direct_guard_v7_20260630",
        "summary_file": "v3_direct_guard_v7_summary.json",
        "eval_file": (
            "standard_eval"
            "/executable_10d_open_return_four_year_v3_direct_guard_v7_20260630_eval_summary.json"
        ),
        "annual_file": (
            "standard_eval"
            "/executable_10d_open_return_four_year_v3_direct_guard_v7_20260630_annual_eval.csv"
        ),
        "monthly_control_file": (
            "standard_eval"
            "/executable_10d_open_return_four_year_v3_direct_guard_v7_20260630_vs_four_year_control_monthly_delta.csv"
        ),
        "promotion_candidate_file": "promotion_candidate.json",
        "promotion_gate_file": "promotion_gate_result.json",
        "control_annual_file": "executable_10d_open_return_four_year_control_annual_eval.csv",
        "control_baseline_name": "four_year_control_10d",
        "prior_baseline_name": "active_recent_blend_gate_v3_10d",
        "replacement_rule": (
            "full rank_ic/top5 positive and recent63/20 rank_ic/top5 nonnegative vs prior 10d bestset"
        ),
        "source_note": "10D 当前 bestset 更新为 v3_direct_guard_v7；相对上一版 10D bestset，"
        "Full RankIC/Top5 继续抬升，Recent63 Top5 也保持正向改善。",
        "next_action": "10D 后续应继续在 v3_direct_guard_v7 之上压缩负月尾部，同时观察 recent20 的绝对排序质量。"
    },
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


def should_replace_10d_bestset(prior_delta: dict[str, float], *, gate_passed: bool) -> bool:
    if not gate_passed:
        return False
    return (
        float(prior_delta["full_rank_ic_delta"]) > 0.0
        and float(prior_delta["full_top5_delta"]) > 0.0
        and float(prior_delta["recent63_rank_ic_delta"]) >= 0.0
        and float(prior_delta["recent63_top5_delta"]) >= 0.0
        and float(prior_delta["recent20_rank_ic_delta"]) >= 0.0
        and float(prior_delta["recent20_top5_delta"]) >= 0.0
    )


def should_replace(label_key: str, prior_delta: dict[str, float], summary: dict[str, Any], gate_ok: bool) -> bool:
    if not gate_ok:
        return False
    if label_key == "3d":
        return (
            float(prior_delta["full_rank_ic_delta"]) >= 0.0
            and float(prior_delta["full_top5_delta"]) >= 0.0
            and float(prior_delta["recent63_rank_ic_delta"]) >= 0.0
            and float(prior_delta["recent63_top5_delta"]) >= 0.0
            and float(prior_delta["recent20_rank_ic_delta"]) >= 0.0
            and float(prior_delta["recent20_top5_delta"]) >= 0.0
            and float(summary["month_stats_vs_base"]["min_top5_delta"]) >= 0.0
        )
    if label_key == "10d":
        return should_replace_10d_bestset(prior_delta, gate_passed=gate_ok)
    raise ValueError(f"unsupported label_key: {label_key}")


def update_row(
    row: dict[str, Any],
    *,
    label_key: str,
    cfg: dict[str, Any],
    summary: dict[str, Any],
    eval_summary: dict[str, Any],
    annual_stats: dict[str, Any],
    monthly_stats: dict[str, Any],
) -> dict[str, Any]:
    control_cmp = comparison_block(eval_summary, cfg["control_baseline_name"])
    prior_cmp = comparison_block(eval_summary, cfg["prior_baseline_name"])
    quality = summary["materialize_report"]["table_quality"]
    out = dict(row)
    out["asset"] = cfg["asset"]
    out["table"] = cfg["table"]
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
    out["source_note"] = cfg["source_note"]
    out["next_action"] = cfg["next_action"]
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
    if label_key == "3d":
        out["stability_score"] = max(float(row["stability_score"]), 2.70)
    return out


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prior = load_json(PRIOR_STATUS)
    rows = []
    details = prior["details"]
    replaced_labels: list[str] = []

    updates: dict[str, dict[str, Any]] = {}
    for label_key, cfg in UPDATE_CONFIG.items():
        report_dir = cfg["report_dir"]
        summary = load_json(report_dir / cfg["summary_file"])
        gate = load_json(report_dir / cfg["promotion_gate_file"])
        eval_summary = load_json(report_dir / cfg["eval_file"])
        annual_rows = load_csv_rows(report_dir / cfg["annual_file"])
        monthly_rows = load_csv_rows(report_dir / cfg["monthly_control_file"])
        control_annual = load_csv_rows(CONTROL_DIR / cfg["control_annual_file"])
        prior_delta = load_json(report_dir / cfg["promotion_candidate_file"])["current_baseline_delta"]
        gate_ok = gate_passed(gate)
        if not should_replace(label_key, prior_delta, summary, gate_ok):
            raise ValueError(f"{label_key} candidate does not satisfy bestset replacement rule")
        annual_stats = annual_delta_stats(annual_rows, control_annual)
        monthly_stats = monthly_delta_stats(monthly_rows)
        updates[label_key] = {
            "cfg": cfg,
            "summary": summary,
            "eval_summary": eval_summary,
            "annual_stats": annual_stats,
            "monthly_stats": monthly_stats,
            "gate": gate,
        }

    for row in prior["best_set"]:
        label_key = row["label_key"]
        if label_key in updates:
            payload = updates[label_key]
            new_row = update_row(
                row,
                label_key=label_key,
                cfg=payload["cfg"],
                summary=payload["summary"],
                eval_summary=payload["eval_summary"],
                annual_stats=payload["annual_stats"],
                monthly_stats=payload["monthly_stats"],
            )
            rows.append(new_row)
            details[label_key] = {
                "record": new_row,
                "promotion_candidate_json": str(payload["cfg"]["report_dir"] / payload["cfg"]["promotion_candidate_file"]),
                "promotion_gate_result_json": str(payload["cfg"]["report_dir"] / payload["cfg"]["promotion_gate_file"]),
                "eval_summary_json": str(payload["cfg"]["report_dir"] / payload["cfg"]["eval_file"]),
                "annual_delta_csv": str(payload["cfg"]["report_dir"] / payload["cfg"]["annual_file"]),
                "monthly_delta_csv": str(payload["cfg"]["report_dir"] / payload["cfg"]["monthly_control_file"]),
                "annual_delta": payload["annual_stats"]["annual_delta"],
                "summary_json": str(payload["cfg"]["report_dir"] / payload["cfg"]["summary_file"]),
            }
            replaced_labels.append(label_key)
        else:
            rows.append(row)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (priority_order[item["optimize_priority"]], -float(item["stability_score"])))

    summary_json = {
        "generated_at": now_iso(),
        "scope": "four_year_latest_bestset_status_v16",
        "best_set": rows,
        "details": details,
        "change_log": {
            "replaced_labels": replaced_labels,
            "label_changes": {
                label_key: {
                    "old_asset": next(item["asset"] for item in prior["best_set"] if item["label_key"] == label_key),
                    "new_asset": UPDATE_CONFIG[label_key]["asset"],
                    "replacement_rule": UPDATE_CONFIG[label_key]["replacement_rule"],
                }
                for label_key in replaced_labels
            },
        },
        "boundaries": prior["boundaries"],
    }
    (REPORT_DIR / "latest_bestset_status_v16.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 四年观察期 latest bestset 状态 v16",
        "",
        f"生成时间：{summary_json['generated_at']}",
        "",
        "## 当前结论",
        "",
        "- 本次更新了 `3D` 与 `10D` 两个 bestset。",
        f"- 新 3D 资产：`{UPDATE_CONFIG['3d']['asset']}`",
        f"- 新 10D 资产：`{UPDATE_CONFIG['10d']['asset']}`",
        "- 1D 与 5D 保持上一版不变。",
        "",
        "## 替换依据",
        "",
        "- 3D：相对上一版 3D bestset，Full/Recent63/Recent20 的 RankIC 与 Top5 均非负改善，且月度 Top5 尾部不恶化。",
        "- 10D：相对上一版 10D bestset，Full RankIC/Top5 继续正向改善，Recent63/20 的 RankIC 与 Top5 不退化。",
        "",
        "## 边界",
        "",
        "- 仍为 research-only。",
        "- 未训练模型。",
        "- 未改 formal manifest。",
        "- 未改 production manifest。",
        "- 未生成信号。",
        "- 未跑回测。",
    ]
    (REPORT_DIR / "latest_bestset_status_v16.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
