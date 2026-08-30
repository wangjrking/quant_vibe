from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_candidate_stability_review_v11_20260628"

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
        / "executable_1d_open_return_four_year_splice_condblend_20260628_daily_eval.csv",
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
        / "executable_3d_open_return_four_year_clear_replacement_gate_20260628_daily_eval.csv",
    },
    "5d": {
        "label": "executable_5d_open_return",
        "asset": "research_5d_four_year_lightweight_blend_v1_20260628",
        "eval_summary": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_lightweight_blend_v1_20260628"
        / "standard_eval"
        / "executable_5d_open_return_four_year_lightweight_blend_v1_20260628_eval_summary.json",
        "monthly_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_lightweight_blend_v1_20260628"
        / "standard_eval"
        / "executable_5d_open_return_four_year_lightweight_blend_v1_20260628_vs_four_year_control_5d_monthly_delta.csv",
        "daily_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_5d_lightweight_blend_v1_20260628"
        / "standard_eval"
        / "executable_5d_open_return_four_year_lightweight_blend_v1_20260628_daily_eval.csv",
    },
    "10d": {
        "label": "executable_10d_open_return",
        "asset": "research_10d_four_year_lightweight_blend_v1_20260628",
        "eval_summary": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_lightweight_blend_v1_20260628"
        / "standard_eval"
        / "executable_10d_open_return_four_year_lightweight_blend_v1_20260628_eval_summary.json",
        "monthly_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_lightweight_blend_v1_20260628"
        / "standard_eval"
        / "executable_10d_open_return_four_year_lightweight_blend_v1_20260628_vs_four_year_control_10d_monthly_delta.csv",
        "daily_delta": DATA_DIR
        / "reports"
        / "model_agent_four_year_10d_lightweight_blend_v1_20260628"
        / "standard_eval"
        / "executable_10d_open_return_four_year_lightweight_blend_v1_20260628_daily_eval.csv",
    },
}

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def annual_delta_table(candidate_daily_path: Path, control_daily_path: Path) -> pd.DataFrame:
    cand = pd.read_csv(candidate_daily_path)
    ctrl = pd.read_csv(control_daily_path)
    cand["trade_date"] = cand["trade_date"].astype(str)
    ctrl["trade_date"] = ctrl["trade_date"].astype(str)
    common = sorted(set(cand["trade_date"]).intersection(set(ctrl["trade_date"])))
    cand = cand[cand["trade_date"].isin(common)].sort_values("trade_date").reset_index(drop=True)
    ctrl = ctrl[ctrl["trade_date"].isin(common)].sort_values("trade_date").reset_index(drop=True)
    cand["period"] = cand["trade_date"].str.slice(0, 4)
    ctrl["period"] = ctrl["trade_date"].str.slice(0, 4)
    cand_group = cand.groupby("period", as_index=False)[METRICS].mean()
    ctrl_group = ctrl.groupby("period", as_index=False)[METRICS].mean()
    joined = cand_group.merge(ctrl_group, on="period", suffixes=("_candidate", "_control"))
    out = joined[["period"]].copy()
    for metric in METRICS:
        out[metric] = joined[f"{metric}_candidate"] - joined[f"{metric}_control"]
    out["trade_days"] = cand.groupby("period").size().values
    return out


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
    control_daily = (
        DATA_DIR
        / "reports"
        / "model_agent_four_year_control_assets_20260627"
        / "standard_eval"
        / f"{spec['label']}_four_year_control_daily_eval.csv"
    )
    annual = annual_delta_table(Path(spec["daily_delta"]), control_daily)
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
    summary_df.to_csv(REPORT_DIR / "candidate_stability_summary_v11.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": now_iso(),
        "scope": "four_year_current_best_candidate_stability_review_v11",
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
    (REPORT_DIR / "candidate_stability_summary_v11.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top = summary_df.iloc[0].to_dict()
    lines = [
        "# 四年观察期最新候选稳定性复核 v11",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 当前稳定性得分最高的候选：`{top['label_key']}` / `{top['asset']}`",
        "- 本版统一采用当前真实 bestset：1D `splice_condblend`、3D `clear_replacement_gate`、5D `lightweight_blend_v1`、10D `lightweight_blend_v1`。",
        "- 这些资产仍然是四年观察期下的 research-only 候选，只能用于生产候选讨论，不能直接解释为 formal 或 L5 准入通过。",
        "",
        "## 各标签摘要",
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
            "- 本次只做四年观察期 research-only 候选稳定性复核。",
            "- 未训练模型。",
            "- 未修改 formal / production manifest。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    (REPORT_DIR / "candidate_stability_review_v11.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
