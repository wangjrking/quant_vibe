from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_3d5d10d_failure_localization_20260714"
)
STATUS_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

LABEL_KEYS = ["3d", "5d", "10d"]
TRAIN_END = "20251231"
HOLDOUT_START = "20260101"

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


def _safe_float(value: Any) -> float | None:
    return rb._to_float(value)


def _metrics_from_daily(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "full": rb._metrics(daily),
        "train": rb._metrics(daily[daily["trade_date"] <= TRAIN_END]),
        "holdout": rb._metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": rb._metrics(daily.tail(20)),
        "recent63": rb._metrics(daily.tail(63)),
        "recent126": rb._metrics(daily.tail(126)),
        "annual_stability": rb._period_stability(daily, "year"),
        "monthly_stability": rb._period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
        "train_trade_days": int((daily["trade_date"] <= TRAIN_END).sum()),
        "holdout_trade_days": int((daily["trade_date"] >= HOLDOUT_START).sum()),
    }


def _period_frame(daily: pd.DataFrame, period_col: str) -> pd.DataFrame:
    metric_cols = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    grouped = daily.groupby(period_col, sort=True)[metric_cols].mean().reset_index()
    grouped = grouped.rename(columns={period_col: "period"})
    grouped["trade_days"] = daily.groupby(period_col, sort=True).size().values
    return grouped


def _worst_periods(periods: pd.DataFrame, metric: str, n: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in periods.sort_values(metric, ascending=True).head(n).to_dict("records"):
        rows.append(
            {
                "period": str(item["period"]),
                "trade_days": int(item["trade_days"]),
                metric: _safe_float(item[metric]),
                "rank_ic": _safe_float(item.get("rank_ic")),
                "top1": _safe_float(item.get("top1")),
                "top3": _safe_float(item.get("top3")),
                "top5": _safe_float(item.get("top5")),
                "top10": _safe_float(item.get("top10")),
                "top20": _safe_float(item.get("top20")),
            }
        )
    return rows


def _failed_candidate_digest() -> dict[str, Any]:
    strict_path = (
        ROOT
        / "quant"
        / "data_file"
        / "reports"
        / "model_agent_four_year_3d5d10d_strict_train_selected_rank_blend_fast_20260714"
        / "strict_train_selected_3d5d10d_fast_report.json"
    )
    size_path = (
        ROOT
        / "quant"
        / "data_file"
        / "reports"
        / "model_agent_four_year_10d_size_liquidity_condition_fast_20260714"
        / "size_liquidity_condition_10d_report.json"
    )
    out: dict[str, Any] = {}
    if strict_path.exists():
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        out["strict_train_selected_rank_blend_fast"] = {
            key: {
                "selected_weights": value["selected_weights"],
                "full_top5_delta": value["deltas"]["full"]["top5"],
                "holdout_top5_delta": value["deltas"]["holdout"]["top5"],
                "recent63_top5_delta": value["deltas"]["recent63"]["top5"],
                "recent20_top5_delta": value["deltas"]["recent20"]["top5"],
                "failed_reasons": value["gate"]["failed_reasons"],
            }
            for key, value in strict["summaries"].items()
        }
    if size_path.exists():
        size = json.loads(size_path.read_text(encoding="utf-8"))
        summary = size["summary"]
        out["size_liquidity_condition_10d_fast"] = {
            "selected": summary["selected"],
            "full_top5_delta": summary["deltas"]["full"]["top5"],
            "holdout_top5_delta": summary["deltas"]["holdout"]["top5"],
            "recent63_top5_delta": summary["deltas"]["recent63"]["top5"],
            "recent20_top5_delta": summary["deltas"]["recent20"]["top5"],
            "failed_reasons": summary["gate"]["failed_reasons"],
        }
    return out


def _diagnose_label(base: pd.DataFrame, label_key: str) -> dict[str, Any]:
    label_col = f"label_{label_key}"
    pred_col = f"pred_{label_key}_rank"
    frame = base[base[label_col].notna()].copy()
    daily = rb._daily_eval(frame, pred_col, label_col)
    metrics = _metrics_from_daily(daily)
    annual = _period_frame(daily, "year")
    monthly = _period_frame(daily, "month")

    annual_csv = REPORT_DIR / f"{label_key}_formal_annual_metrics.csv"
    monthly_csv = REPORT_DIR / f"{label_key}_formal_monthly_metrics.csv"
    daily_csv = REPORT_DIR / f"{label_key}_formal_daily_metrics.csv"
    annual.to_csv(annual_csv, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_csv, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    negative_months = monthly[monthly["top5"] <= 0].copy()
    weak_recent = daily.tail(63).copy()
    return {
        "label": rb.LABELS[label_key],
        "eval_min_trade_date": metrics["eval_min_trade_date"],
        "eval_max_trade_date": metrics["eval_max_trade_date"],
        "eval_trade_days": metrics["eval_trade_days"],
        "metrics": metrics,
        "annual_metrics_csv": str(annual_csv),
        "monthly_metrics_csv": str(monthly_csv),
        "daily_metrics_csv": str(daily_csv),
        "negative_top5_month_count": int(len(negative_months)),
        "negative_top5_month_ratio": _safe_float(len(negative_months) / len(monthly)) if len(monthly) else None,
        "worst_top5_months": _worst_periods(monthly, "top5", n=8),
        "worst_rank_ic_months": _worst_periods(monthly, "rank_ic", n=8),
        "worst_top5_years": _worst_periods(annual, "top5", n=5),
        "worst_rank_ic_years": _worst_periods(annual, "rank_ic", n=5),
        "recent63_negative_top5_days": int((weak_recent["top5"] <= 0).sum()),
        "recent63_negative_top5_day_ratio": _safe_float((weak_recent["top5"] <= 0).mean()) if len(weak_recent) else None,
        "recent63_min_top5_day": _safe_float(weak_recent["top5"].min()) if len(weak_recent) else None,
        "recent63_worst_top5_trade_date": str(weak_recent.sort_values("top5").head(1).iloc[0]["trade_date"])
        if len(weak_recent)
        else None,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D/5D/10D 四年失败定位报告（20260714）",
        "",
        "## 结论",
        "",
        "- 当前 3D/5D/10D 的 active formal 本身在四年全窗口仍是正向模型，但简单融合或简单条件加分主要在训练期改善，不能稳定外推到 2026 holdout 和 Recent20/63。",
        "- 失败不是数据覆盖问题；本报告只读 formal 预测和成熟标签，未训练、未写预测资产、未改 manifest。",
        "- 后续不建议继续扩大简单权重/bonus 网格，下一步应转向严格滚动的条件模型或 residual learner，并把 2026 holdout 与 Recent20/63 作为硬验证窗口。",
        "",
        "## Formal 基线弱点定位",
        "",
        "| 标签 | 可评价区间 | 交易日 | Full RankIC | Full Top5 | Recent63 Top5 | Recent20 Top5 | Top5 负月数 | 最差 Top5 月 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in LABEL_KEYS:
        item = payload["formal_diagnostics"][key]
        metrics = item["metrics"]
        worst_month = item["worst_top5_months"][0] if item["worst_top5_months"] else {}
        lines.append(
            "| {key} | {start}-{end} | {days} | {rank_ic:.6f} | {top5:.6f} | {r63:.6f} | {r20:.6f} | {neg_months} | {worst} ({worst_val:.6f}) |".format(
                key=key,
                start=item["eval_min_trade_date"],
                end=item["eval_max_trade_date"],
                days=item["eval_trade_days"],
                rank_ic=metrics["full"]["rank_ic"] or 0.0,
                top5=metrics["full"]["top5"] or 0.0,
                r63=metrics["recent63"]["top5"] or 0.0,
                r20=metrics["recent20"]["top5"] or 0.0,
                neg_months=item["negative_top5_month_count"],
                worst=worst_month.get("period", "-"),
                worst_val=worst_month.get("top5") or 0.0,
            )
        )

    lines.extend(
        [
            "",
            "## 已验证失败方向",
            "",
        ]
    )
    failed = payload["failed_candidate_digest"]
    strict = failed.get("strict_train_selected_rank_blend_fast", {})
    if strict:
        lines.append("### 严格训练窗口 rank-blend")
        lines.append("")
        lines.append("| 标签 | 选中权重 | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for key, item in strict.items():
            lines.append(
                "| {key} | `{weights}` | {full:.6f} | {holdout:.6f} | {r63:.6f} | {r20:.6f} |".format(
                    key=key,
                    weights=item["selected_weights"],
                    full=item["full_top5_delta"] or 0.0,
                    holdout=item["holdout_top5_delta"] or 0.0,
                    r63=item["recent63_top5_delta"] or 0.0,
                    r20=item["recent20_top5_delta"] or 0.0,
                )
            )
        lines.append("")
    if "size_liquidity_condition_10d_fast" in failed:
        item = failed["size_liquidity_condition_10d_fast"]
        lines.extend(
            [
                "### 10D size/liquidity 条件加分",
                "",
                f"- 选中条件：`{item['selected']}`",
                f"- Full Top5 Δ：`{item['full_top5_delta']:.6f}`",
                f"- Holdout Top5 Δ：`{item['holdout_top5_delta']:.6f}`",
                f"- Recent63 Top5 Δ：`{item['recent63_top5_delta']:.6f}`",
                f"- Recent20 Top5 Δ：`{item['recent20_top5_delta']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步研究方向",
            "",
            "1. 对 3D/5D/10D 暂停简单 rank-blend 和单因子 bonus 扩网格。",
            "2. 优先做严格 train/holdout 分离的 conditional learner：训练期只学哪些截面/股票状态应该降权或保留，holdout/recent 只验证。",
            "3. 10D 的近期 formal Top5 绝对值较强，任何候选都必须先证明不破坏 Recent20/63；否则不进入生产候选讨论。",
            "4. 3D/5D 的下一轮重点不是追 Full RankIC，而是提高 Holdout/Recent Top5 的非负稳定性。",
            "",
            "## 边界",
            "",
            "- research-only 诊断。",
            "- 未训练模型。",
            "- 未写 L4 预测资产。",
            "- 未修改 formal manifest / approved_for_l5 / production manifest。",
            "- 未生成信号，未跑回测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_status_and_frontier(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["failure_localization_3d5d10d"] = {
            "decision": "diagnostic_only_no_candidate_promoted",
            "summary": str(report_json),
            "review_md": str(report_md),
            "reason": "Formal baselines and failed scans show simple blends/bonuses do not generalize to 2026 holdout or Recent20/63 for 3D/5D/10D.",
        }
        status["next_research_priority"] = (
            "For 3D/5D/10D, move to strict train/holdout separated conditional or residual learners; "
            "stop expanding simple rank-blend and simple single-feature bonus grids."
        )
        status["latest_failure_localization"] = str(report_json)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_failure_localization"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "headline": payload["headline"],
        }
        frontier["recommendations"] = [
            "Do not promote 3D/5D/10D simple score-bonus or rank-blend variants based on training-window gains alone.",
            "Do not expand simple rank-blend or single-feature bonus grids further without a new hypothesis.",
            "Prioritize strict rolling or train-holdout separated conditional/residual learners for 3D/5D/10D.",
            "Keep strict train-selected 1D as the current audited research candidate-only best; production promotion still requires separate user authorization and formal audit.",
        ]
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    diagnostics = {key: _diagnose_label(base, key) for key in LABEL_KEYS}
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_failure_localization",
        "observation_window": {"start": rb.START_DATE, "end": "label_maturity_by_horizon"},
        "headline": (
            "3D/5D/10D active formal models are positive over the four-year window, "
            "but simple blend/bonus variants fail because train-window Top5 gains do not generalize to 2026 holdout/recent windows."
        ),
        "formal_diagnostics": diagnostics,
        "failed_candidate_digest": _failed_candidate_digest(),
        "recommendation": {
            "stop": [
                "simple rank-blend grid expansion",
                "single-feature fixed bonus grid expansion",
            ],
            "next": [
                "strict train-holdout separated conditional learner",
                "residual learner focused on holdout/recent Top5 preservation",
                "regime or bucket diagnostics before any new score formula",
            ],
        },
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_asset_write": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "failure_localization_report.json"
    report_md = REPORT_DIR / "failure_localization_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status_and_frontier(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
