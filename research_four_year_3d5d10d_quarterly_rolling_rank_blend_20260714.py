from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_quarterly_rolling_rank_blend_20260714"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


LABEL_KEYS = ["3d", "5d", "10d"]
KEYS = ["1d", "3d", "5d", "10d"]
SELECTION_DAYS = 252
MIN_SELECTION_DAYS = 63
HOLDOUT_START = "20260101"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _weight_score(frame: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.Series:
    return (
        weights[0] * frame["pred_1d_rank"]
        + weights[1] * frame["pred_3d_rank"]
        + weights[2] * frame["pred_5d_rank"]
        + weights[3] * frame["pred_10d_rank"]
    )


def _candidate_grid(label_key: str) -> list[tuple[float, float, float, float]]:
    candidates: set[tuple[float, float, float, float]] = set()
    candidates.add(rb.BASELINE_WEIGHTS[label_key])
    base_index = KEYS.index(label_key)
    steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 0.90, 0.95, 1.0]
    for aux_key in KEYS:
        if aux_key == label_key:
            continue
        aux_index = KEYS.index(aux_key)
        for target_weight in steps:
            weights = [0.0, 0.0, 0.0, 0.0]
            weights[base_index] = float(target_weight)
            weights[aux_index] = float(1.0 - target_weight)
            candidates.add(tuple(round(v, 4) for v in weights))
    return sorted(candidates)


def _daily_for_weights(frame: pd.DataFrame, label_key: str, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    work = frame[frame[label_col].notna()].copy()
    work["candidate_score"] = _weight_score(work, weights)
    return rb._daily_eval(work, "candidate_score", label_col)


def _selection_objective(daily: pd.DataFrame) -> float:
    if len(daily) < MIN_SELECTION_DAYS:
        return -999.0
    metrics = rb._metrics(daily)
    monthly = rb._period_stability(daily, "month")
    return float(
        (metrics.get("top5") or -1.0)
        + 0.35 * (metrics.get("top10") or -1.0)
        + 0.05 * (metrics.get("rank_ic") or -1.0)
        + 0.01 * (monthly.get("top5_positive_period_ratio") or 0.0)
    )


def _quarter_starts(trade_dates: list[str]) -> list[str]:
    starts: list[str] = []
    last_quarter = None
    for date in trade_dates:
        month = int(date[4:6])
        quarter = (date[:4], (month - 1) // 3 + 1)
        if quarter != last_quarter:
            starts.append(date)
            last_quarter = quarter
    return starts


def _select_weights_for_quarter(
    base: pd.DataFrame,
    label_key: str,
    quarter_start: str,
    grid: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    label_col = f"label_{label_key}"
    eligible_dates = sorted(base.loc[(base["trade_date"] < quarter_start) & base[label_col].notna(), "trade_date"].unique())
    if len(eligible_dates) < MIN_SELECTION_DAYS:
        weights = rb.BASELINE_WEIGHTS[label_key]
        return {
            "quarter_start": quarter_start,
            "weights": dict(zip(["w1", "w3", "w5", "w10"], weights)),
            "selection_trade_days": int(len(eligible_dates)),
            "selection_min_trade_date": eligible_dates[0] if eligible_dates else None,
            "selection_max_trade_date": eligible_dates[-1] if eligible_dates else None,
            "objective": None,
            "selection_reason": "insufficient_history_use_baseline",
        }

    selection_dates = set(eligible_dates[-SELECTION_DAYS:])
    selection = base[base["trade_date"].isin(selection_dates)].copy()
    best: dict[str, Any] | None = None
    for weights in grid:
        daily = _daily_for_weights(selection, label_key, weights)
        objective = _selection_objective(daily)
        metrics = rb._metrics(daily)
        row = {
            "weights_tuple": weights,
            "weights": dict(zip(["w1", "w3", "w5", "w10"], weights)),
            "objective": objective,
            "top5": metrics.get("top5"),
            "top10": metrics.get("top10"),
            "rank_ic": metrics.get("rank_ic"),
        }
        if best is None or (
            row["objective"],
            row["top5"] or -999.0,
            row["rank_ic"] or -999.0,
        ) > (
            best["objective"],
            best["top5"] or -999.0,
            best["rank_ic"] or -999.0,
        ):
            best = row
    assert best is not None
    return {
        "quarter_start": quarter_start,
        "weights": best["weights"],
        "selection_trade_days": int(len(selection_dates)),
        "selection_min_trade_date": min(selection_dates),
        "selection_max_trade_date": max(selection_dates),
        "objective": _safe_float(best["objective"]),
        "selection_top5": _safe_float(best["top5"]),
        "selection_top10": _safe_float(best["top10"]),
        "selection_rank_ic": _safe_float(best["rank_ic"]),
        "selection_reason": "rolling_prior_window",
    }


def _apply_quarterly_weights(base: pd.DataFrame, label_key: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    label_col = f"label_{label_key}"
    frame = base[base[label_col].notna()].copy()
    trade_dates = sorted(frame["trade_date"].unique())
    quarters = _quarter_starts(trade_dates)
    grid = _candidate_grid(label_key)
    traces: list[dict[str, Any]] = []
    pieces: list[pd.DataFrame] = []
    for idx, start in enumerate(quarters):
        end = quarters[idx + 1] if idx + 1 < len(quarters) else "99999999"
        trace = _select_weights_for_quarter(base, label_key, start, grid)
        traces.append(trace)
        weights = tuple(trace["weights"][key] for key in ["w1", "w3", "w5", "w10"])
        part = frame[(frame["trade_date"] >= start) & (frame["trade_date"] < end)].copy()
        if part.empty:
            continue
        part["candidate_score"] = _weight_score(part, weights)
        part["selected_quarter_start"] = start
        pieces.append(part)
    if not pieces:
        raise RuntimeError(f"no quarterly pieces for {label_key}")
    scored = pd.concat(pieces, ignore_index=True)
    daily = rb._daily_eval(scored, "candidate_score", label_col)
    return daily, traces


def _metrics_pack(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "full": rb._metrics(daily),
        "holdout": rb._metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": rb._metrics(daily.tail(20)),
        "recent63": rb._metrics(daily.tail(63)),
        "recent126": rb._metrics(daily.tail(126)),
        "annual_stability": rb._period_stability(daily, "year"),
        "monthly_stability": rb._period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], block: str, metric: str) -> float | None:
    cv = candidate[block].get(metric)
    bv = baseline[block].get(metric)
    if cv is None or bv is None:
        return None
    return float(cv - bv)


def _delta_pack(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    blocks = ["full", "holdout", "recent20", "recent63", "recent126"]
    metrics = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    return {block: {metric: _delta(candidate, baseline, block, metric) for metric in metrics} for block in blocks}


def _improvement_gate(deltas: dict[str, dict[str, float | None]], candidate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    full = deltas["full"]
    holdout = deltas["holdout"]
    recent63 = deltas["recent63"]
    recent20 = deltas["recent20"]
    if candidate["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (full.get("rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for metric in ["top1", "top3", "top5", "top10", "top20"]:
        if (full.get(metric) or 0.0) <= 0.0:
            reasons.append(f"full_{metric}_delta_non_positive")
    if (holdout.get("top5") or 0.0) < 0.0:
        reasons.append("holdout_top5_delta_negative")
    if (recent63.get("top5") or 0.0) < 0.0:
        reasons.append("recent63_top5_delta_negative")
    if (recent20.get("top5") or 0.0) < 0.0:
        reasons.append("recent20_top5_delta_negative")
    return {"passed": not reasons, "failed_reasons": reasons}


def _update_status(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    passed = {k: v for k, v in payload["summaries"].items() if v["improvement_gate"]["passed"]}
    decision = (
        "quarterly_rolling_rank_blend_passed_for_some_labels_candidate_discussion_required"
        if passed
        else "no_3d_5d_10d_candidate_passed_quarterly_rolling_rank_blend_scan"
    )
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["quarterly_rolling_rank_blend_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": "Quarterly weights are selected only from prior mature-label history and applied to the next quarter.",
        }
        if passed:
            status["pending_stronger_research_candidate"] = {
                "source": str(report_json),
                "labels": sorted(passed.keys()),
                "note": "Research-only pass; formal/L5/production changes require separate user authorization and audit.",
            }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_quarterly_rolling_rank_blend_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D / 5D / 10D 季度滚动 Rank Blend 研究",
        "",
        "## 结论",
        "",
    ]
    passed = [k for k, v in payload["summaries"].items() if v["improvement_gate"]["passed"]]
    if passed:
        lines.append(f"通过 research-only 候选门槛的标签：`{', '.join(passed)}`。")
    else:
        lines.append("本轮没有 3D / 5D / 10D 标签通过四年观察与近期窗口门槛，不能进入生产候选讨论。")
    lines.extend(
        [
            "",
            "## 方法",
            "",
            f"- 每个季度只使用该季度之前最多 `{SELECTION_DAYS}` 个成熟标签交易日选择权重。",
            "- 候选权重只在当前 horizon 与一个辅助 horizon 之间组合，避免复杂多参数过拟合。",
            "- 本轮只输出 research-only 评估报告，不写 formal manifest，不生成信号，不跑回测。",
            "",
            "## 指标摘要",
            "",
            "| 标签 | 通过 | Full RankIC Δ | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label_key, item in payload["summaries"].items():
        d = item["deltas"]
        lines.append(
            "| {label} | {passed} | {fr:.6f} | {ft5:.6f} | {ht5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key,
                passed="是" if item["improvement_gate"]["passed"] else "否",
                fr=d["full"].get("rank_ic") or 0.0,
                ft5=d["full"].get("top5") or 0.0,
                ht5=d["holdout"].get("top5") or 0.0,
                r63=d["recent63"].get("top5") or 0.0,
                r20=d["recent20"].get("top5") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- research-only。",
            "- 未训练生产模型。",
            "- 未修改 formal manifest。",
            "- 未修改 `approved_for_l5`。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    summaries: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        baseline_daily = _daily_for_weights(base, label_key, rb.BASELINE_WEIGHTS[label_key])
        baseline_metrics = _metrics_pack(baseline_daily)
        candidate_daily, traces = _apply_quarterly_weights(base, label_key)
        candidate_metrics = _metrics_pack(candidate_daily)
        deltas = _delta_pack(candidate_metrics, baseline_metrics)
        gate = _improvement_gate(deltas, candidate_metrics)
        for trace in traces:
            trace_rows.append({"label_key": label_key, **trace})
        summaries[label_key] = {
            "label": rb.LABELS[label_key],
            "baseline_current_formal": baseline_metrics,
            "candidate_metrics": candidate_metrics,
            "deltas": deltas,
            "improvement_gate": gate,
            "quarter_count": len(traces),
            "quarter_weight_trace": traces,
        }
    trace_csv = REPORT_DIR / "quarterly_weight_trace.csv"
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False, encoding="utf-8")
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_quarterly_rolling_rank_blend",
        "observation_start": rb.START_DATE,
        "selection_days": SELECTION_DAYS,
        "min_selection_days": MIN_SELECTION_DAYS,
        "holdout_start": HOLDOUT_START,
        "summaries": summaries,
        "trace_csv": str(trace_csv),
        "boundaries": {
            "research_only": True,
            "no_production_training": True,
            "no_prediction_asset_write": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "quarterly_rolling_rank_blend_report.json"
    report_md = REPORT_DIR / "quarterly_rolling_rank_blend_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
