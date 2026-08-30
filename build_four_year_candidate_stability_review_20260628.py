from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_candidate_stability_review_20260628"

SPECS = {
    "1d": {
        "label": "executable_1d_open_return",
        "asset": "research_1d_four_year_splice_condblend_20260628",
        "eval_summary": DATA_DIR
        / "reports"
        / "model_agent_four_year_1d_splice_condblend_20260628"
        / "standard_eval"
        / "executable_1d_open_return_four_year_splice_condblend_20260628_eval_summary.json",
        "monthly_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_1d_splice_condblend_20260628"
        / "standard_eval"
        / "executable_1d_open_return_four_year_splice_condblend_20260628_vs_four_year_control_1d_monthly_delta.csv",
        "daily_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_1d_splice_condblend_20260628"
        / "standard_eval"
        / "executable_1d_open_return_four_year_splice_condblend_20260628_vs_four_year_control_1d_daily.csv",
    },
    "3d": {
        "label": "executable_3d_open_return",
        "asset": "research_3d_four_year_clear_replacement_gate_20260628",
        "eval_summary": DATA_DIR
        / "reports"
        / "model_agent_four_year_3d_clear_replacement_gate_20260628"
        / "standard_eval"
        / "executable_3d_open_return_four_year_clear_replacement_gate_20260628_eval_summary.json",
        "monthly_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_3d_clear_replacement_gate_20260628"
        / "standard_eval"
        / "executable_3d_open_return_four_year_clear_replacement_gate_20260628_vs_four_year_control_monthly_delta.csv",
        "daily_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_3d_clear_replacement_gate_20260628"
        / "standard_eval"
        / "executable_3d_open_return_four_year_clear_replacement_gate_20260628_vs_four_year_control_daily.csv",
    },
    "5d": {
        "label": "executable_5d_open_return",
        "asset": "research_5d_four_year_clear_replacement_gate_20260628",
        "eval_summary": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_clear_replacement_gate_20260628"
        / "standard_eval"
        / "executable_5d_open_return_four_year_clear_replacement_gate_20260628_eval_summary.json",
        "monthly_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_clear_replacement_gate_20260628"
        / "standard_eval"
        / "executable_5d_open_return_four_year_clear_replacement_gate_20260628_vs_four_year_control_monthly_delta.csv",
        "daily_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_clear_replacement_gate_20260628"
        / "standard_eval"
        / "executable_5d_open_return_four_year_clear_replacement_gate_20260628_vs_four_year_control_daily.csv",
    },
    "10d": {
        "label": "executable_10d_open_return",
        "asset": "research_10d_four_year_clear_replacement_time_splice_20260628",
        "eval_summary": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_clear_replacement_time_splice_20260628"
        / "standard_eval"
        / "executable_10d_open_return_four_year_clear_replacement_time_splice_20260628_eval_summary.json",
        "monthly_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_clear_replacement_time_splice_20260628"
        / "standard_eval"
        / "executable_10d_open_return_four_year_clear_replacement_time_splice_20260628_vs_four_year_control_10d_monthly_delta.csv",
        "daily_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_clear_replacement_time_splice_20260628"
        / "standard_eval"
        / "executable_10d_open_return_four_year_clear_replacement_time_splice_20260628_vs_four_year_control_10d_daily.csv",
    },
}

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def annual_delta_table(daily_delta_path: Path) -> pd.DataFrame:
    daily = pd.read_csv(daily_delta_path)
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    delta_cols = [f"{metric}_delta" for metric in METRICS]
    grouped = daily.groupby("year", as_index=False)[delta_cols].mean()
    grouped["trade_days"] = daily.groupby("year").size().values
    return grouped.rename(columns={f"{metric}_delta": metric for metric in METRICS})


def risk_score(
    *,
    full_rank_ic_delta: float,
    recent63_top5_delta: float,
    recent20_top5_delta: float,
    min_month_top5_delta: float,
) -> float:
    return (
        40.0 * recent20_top5_delta
        + 25.0 * recent63_top5_delta
        + 15.0 * full_rank_ic_delta
        - 20.0 * max(0.0, -min_month_top5_delta)
    )


def summarize_one(key: str, spec: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary = json.loads(Path(spec["eval_summary"]).read_text(encoding="utf-8"))
    monthly = pd.read_csv(spec["monthly_delta"])
    annual = annual_delta_table(Path(spec["daily_delta"]))
    comp = summary["comparisons"][0]
    recent20_abs = summary["recent_windows"]["recent20"]
    recent63_abs = summary["recent_windows"]["recent63"]
    record = {
        "label_key": key,
        "label": spec["label"],
        "asset": spec["asset"],
        "table": summary["table"],
        "eval_days": summary["full"]["trade_days"],
        "full_rank_ic_delta": comp["delta"]["rank_ic"],
        "full_top1_delta": comp["delta"]["top1"],
        "full_top5_delta": comp["delta"]["top5"],
        "recent63_rank_ic_delta": comp["recent63_delta"]["rank_ic"],
        "recent63_top5_delta": comp["recent63_delta"]["top5"],
        "recent20_rank_ic_delta": comp["recent20_delta"]["rank_ic"],
        "recent20_top5_delta": comp["recent20_delta"]["top5"],
        "recent63_abs_rank_ic": recent63_abs["rank_ic"],
        "recent63_abs_top5": recent63_abs["top5"],
        "recent20_abs_rank_ic": recent20_abs["rank_ic"],
        "recent20_abs_top5": recent20_abs["top5"],
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
    }
    record["stability_score"] = risk_score(
        full_rank_ic_delta=record["full_rank_ic_delta"],
        recent63_top5_delta=record["recent63_top5_delta"],
        recent20_top5_delta=record["recent20_top5_delta"],
        min_month_top5_delta=record["month_min_top5_delta"],
    )
    return record, monthly, annual


def priority_label(record: dict[str, object]) -> str:
    if float(record["recent20_abs_top5"]) <= 0 or float(record["month_min_top5_delta"]) < -0.005:
        return "high"
    if float(record["recent63_abs_top5"]) <= 0 or int(record["month_top5_negative"]) >= 6:
        return "medium"
    return "low"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    detail_payload: dict[str, object] = {}
    for key, spec in SPECS.items():
        record, monthly, annual = summarize_one(key, spec)
        record["optimize_priority"] = priority_label(record)
        rows.append(record)
        detail_payload[key] = {
            "record": record,
            "monthly_delta_path": str(spec["monthly_delta"]),
            "annual_delta": annual.to_dict(orient="records"),
        }

    priority_order = ["high", "medium", "low"]
    summary_df = pd.DataFrame(rows)
    summary_df["optimize_priority"] = pd.Categorical(
        summary_df["optimize_priority"],
        categories=priority_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values(["optimize_priority", "stability_score"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(REPORT_DIR / "candidate_stability_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_current_best_candidate_stability_review",
        "rows": summary_df.to_dict(orient="records"),
        "details": detail_payload,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "candidate_stability_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top = summary_df.iloc[0].to_dict()
    lines = [
        "# 四年主候选稳定性复核",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前判断",
        "",
        f"- 当前稳定性得分最高候选：`{top['label_key']}` / `{top['asset']}`",
        f"- 当前最高优化优先级：`{summary_df.iloc[0]['optimize_priority']}`。该字段表示后续仍需重点优化或复核的方向，不等于资产强弱排序。",
        "",
        "## 各标签结论",
        "",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.extend(
            [
                f"### {row['label_key']}",
                "",
                f"- 资产：`{row['asset']}`",
                f"- 优先级：`{row['optimize_priority']}`",
                f"- Full RankIC Delta：`{row['full_rank_ic_delta']:.6f}`",
                f"- Full Top5 Delta：`{row['full_top5_delta']:.6f}`",
                f"- Recent63 Top5 Delta：`{row['recent63_top5_delta']:.6f}`",
                f"- Recent20 Top5 Delta：`{row['recent20_top5_delta']:.6f}`",
                f"- Recent63 绝对 Top5：`{row['recent63_abs_top5']:.6f}`",
                f"- Recent20 绝对 Top5：`{row['recent20_abs_top5']:.6f}`",
                f"- 月度 Top5 负值月份数：`{row['month_top5_negative']}`",
                f"- 最差月度 Top5 Delta：`{row['month_min_top5_delta']:.6f}`",
                f"- 最差年度 Top5 Delta：`{row['year_min_top5_delta']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 边界",
            "",
            "- 本次只做四年主候选稳定性复核和优先级判断。",
            "- 未训练模型。",
            "- 未修改 formal / production manifest。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    (REPORT_DIR / "candidate_stability_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
