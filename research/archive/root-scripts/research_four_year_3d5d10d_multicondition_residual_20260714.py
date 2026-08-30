from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_3d5d10d_multicondition_residual_20260714"
)
FEATURE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
STATUS_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

LABEL_KEYS = ["3d", "5d", "10d"]
TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
FEATURES = ["amount", "total_mv", "turnover_rate", "atr_qfq"]

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "penalty_low_amount_high_atr",
        "terms": [
            {"feature": "amount", "side": "low", "threshold": 0.20, "alpha": -0.04},
            {"feature": "atr_qfq", "side": "high", "threshold": 0.80, "alpha": -0.02},
        ],
    },
    {
        "name": "penalty_low_amount_low_mv",
        "terms": [
            {"feature": "amount", "side": "low", "threshold": 0.20, "alpha": -0.04},
            {"feature": "total_mv", "side": "low", "threshold": 0.25, "alpha": -0.02},
        ],
    },
    {
        "name": "penalty_low_liquidity_low_turnover",
        "terms": [
            {"feature": "amount", "side": "low", "threshold": 0.25, "alpha": -0.03},
            {"feature": "turnover_rate", "side": "low", "threshold": 0.25, "alpha": -0.02},
        ],
    },
    {
        "name": "boost_high_amount_low_atr",
        "terms": [
            {"feature": "amount", "side": "high", "threshold": 0.80, "alpha": 0.03},
            {"feature": "atr_qfq", "side": "low", "threshold": 0.25, "alpha": 0.02},
        ],
    },
    {
        "name": "boost_high_amount_high_turnover",
        "terms": [
            {"feature": "amount", "side": "high", "threshold": 0.80, "alpha": 0.03},
            {"feature": "turnover_rate", "side": "high", "threshold": 0.75, "alpha": 0.02},
        ],
    },
    {
        "name": "boost_large_mv_low_atr",
        "terms": [
            {"feature": "total_mv", "side": "high", "threshold": 0.75, "alpha": 0.025},
            {"feature": "atr_qfq", "side": "low", "threshold": 0.25, "alpha": 0.02},
        ],
    },
    {
        "name": "risk_trim_low_amount_low_mv_high_atr",
        "terms": [
            {"feature": "amount", "side": "low", "threshold": 0.25, "alpha": -0.03},
            {"feature": "total_mv", "side": "low", "threshold": 0.25, "alpha": -0.02},
            {"feature": "atr_qfq", "side": "high", "threshold": 0.80, "alpha": -0.02},
        ],
    },
    {
        "name": "quality_boost_high_amount_large_mv_low_atr",
        "terms": [
            {"feature": "amount", "side": "high", "threshold": 0.75, "alpha": 0.025},
            {"feature": "total_mv", "side": "high", "threshold": 0.75, "alpha": 0.02},
            {"feature": "atr_qfq", "side": "low", "threshold": 0.25, "alpha": 0.02},
        ],
    },
]


def _load_feature_slice() -> pd.DataFrame:
    cols = ", ".join(["trade_date", "stock_code", *FEATURES])
    with duckdb.connect(str(FEATURE_DB), read_only=True) as con:
        return con.execute(
            f"""
            SELECT {cols}
            FROM {FEATURE_TABLE}
            WHERE trade_date >= ?
              AND stock_code NOT LIKE '%.BJ'
            ORDER BY trade_date, stock_code
            """,
            [rb.START_DATE],
        ).fetchdf()


def _prepare_base() -> pd.DataFrame:
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    features = _load_feature_slice()
    frame = base.merge(features, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    for col in FEATURES:
        frame[f"{col}_rank"] = frame.groupby("trade_date")[col].rank(method="average", pct=True)
    return frame


def _apply_candidate(frame: pd.DataFrame, label_key: str, candidate: dict[str, Any]) -> pd.Series:
    score = frame[f"pred_{label_key}_rank"].copy()
    for term in candidate["terms"]:
        rank_col = f"{term['feature']}_rank"
        values = frame[rank_col].fillna(0.5)
        if term["side"] == "low":
            mask = values <= float(term["threshold"])
        else:
            mask = values >= float(term["threshold"])
        score = score + float(term["alpha"]) * mask.astype(float)
    return score


def _daily_for_candidate(base: pd.DataFrame, label_key: str, candidate: dict[str, Any] | None) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    frame = base[base[label_col].notna()].copy()
    if candidate is None:
        score_col = f"pred_{label_key}_rank"
    else:
        score_col = "candidate_score"
        frame[score_col] = _apply_candidate(frame, label_key, candidate)
    return rb._daily_eval(frame, score_col, label_col)


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
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


def _delta_block(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
        out[key] = rb._to_float((candidate[block].get(key) or 0.0) - (baseline[block].get(key) or 0.0))
    return out


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = _delta_block(candidate, baseline, "full")
    holdout = _delta_block(candidate, baseline, "holdout")
    recent63 = _delta_block(candidate, baseline, "recent63")
    recent20 = _delta_block(candidate, baseline, "recent20")
    if candidate["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (full["rank_ic"] or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for key in ["top1", "top3", "top5", "top10", "top20"]:
        if (full[key] or 0.0) <= 0.0:
            reasons.append(f"full_{key}_delta_non_positive")
    if (holdout["top5"] or 0.0) <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if (recent63["top5"] or 0.0) <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if (recent20["top5"] or 0.0) <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    return not reasons, reasons


def _train_objective(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    train = _delta_block(candidate, baseline, "train")
    return float(
        45.0 * (train["top5"] or 0.0)
        + 10.0 * (train["top1"] or 0.0)
        + 5.0 * (train["rank_ic"] or 0.0)
        + 4.0 * (train["top20"] or 0.0)
    )


def _evaluate_label(base: pd.DataFrame, label_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = _metrics(_daily_for_candidate(base, label_key, None))
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_key: tuple[float, float] | None = None
    for candidate in CANDIDATES:
        metrics = _metrics(_daily_for_candidate(base, label_key, candidate))
        gate_passed, failures = _gate(metrics, baseline)
        full = _delta_block(metrics, baseline, "full")
        train = _delta_block(metrics, baseline, "train")
        holdout = _delta_block(metrics, baseline, "holdout")
        recent63 = _delta_block(metrics, baseline, "recent63")
        recent20 = _delta_block(metrics, baseline, "recent20")
        objective = _train_objective(metrics, baseline)
        row = {
            "label_key": label_key,
            "candidate_name": candidate["name"],
            "train_objective": objective,
            "gate_passed": gate_passed,
            "failed_reasons": ";".join(failures),
            "train_top5_delta": train["top5"],
            "train_rank_ic_delta": train["rank_ic"],
            "full_rank_ic_delta": full["rank_ic"],
            "full_top1_delta": full["top1"],
            "full_top3_delta": full["top3"],
            "full_top5_delta": full["top5"],
            "full_top10_delta": full["top10"],
            "full_top20_delta": full["top20"],
            "holdout_top5_delta": holdout["top5"],
            "recent63_top5_delta": recent63["top5"],
            "recent20_top5_delta": recent20["top5"],
        }
        rows.append(row)
        key = (objective, train["top5"] or -999.0)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected = {
                "label": rb.LABELS[label_key],
                "selection_rule": "Only train-window objective over 20220606-20251231 is used for multi-condition residual selection.",
                "selected_candidate": candidate,
                "train_objective": objective,
                "deltas": {
                    "full": full,
                    "train": train,
                    "holdout": holdout,
                    "recent63": recent63,
                    "recent20": recent20,
                },
                "candidate_metrics": metrics,
                "baseline_metrics": baseline,
                "gate": {"passed": gate_passed, "failed_reasons": failures},
            }
    assert selected is not None
    return selected, rows


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D/5D/10D 多条件 Residual 扫描（20260714）",
        "",
        "## 结论",
        "",
        "本轮只做 research-only 多条件 residual 调节。训练窗口选择候选，holdout / recent 只验证；未训练生产模型，未写预测资产，未改 manifest。",
        "",
        "| 标签 | 训练期选中候选 | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ | 是否过门 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in LABEL_KEYS:
        item = payload["summaries"][key]
        lines.append(
            "| {key} | `{name}` | {full:.6f} | {holdout:.6f} | {r63:.6f} | {r20:.6f} | {passed} |".format(
                key=key,
                name=item["selected_candidate"]["name"],
                full=item["deltas"]["full"]["top5"] or 0.0,
                holdout=item["deltas"]["holdout"]["top5"] or 0.0,
                r63=item["deltas"]["recent63"]["top5"] or 0.0,
                r20=item["deltas"]["recent20"]["top5"] or 0.0,
                passed="是" if item["gate"]["passed"] else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- research-only。",
            "- 未训练生产模型。",
            "- 未写 L4 formal 预测资产。",
            "- 未修改 `approved_for_l5` manifest。",
            "- 未生成信号，未跑回测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_status_and_frontier(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    passed = {k: v for k, v in payload["summaries"].items() if v["gate"]["passed"]}
    decision = (
        "multicondition_residual_passed_for_some_labels_candidate_discussion_required"
        if passed
        else "no_3d_5d_10d_candidate_passed_multicondition_residual_scan"
    )
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["multicondition_residual_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": "Multi-condition residual candidates were selected only on train window and validated on holdout/recent.",
        }
        if passed:
            status["pending_stronger_research_candidate"] = {
                "source": str(report_json),
                "labels": sorted(passed.keys()),
                "note": "Research-only pass; formal/L5/production changes require separate user authorization and audit.",
            }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_multicondition_residual_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = _prepare_base()
    summaries: dict[str, Any] = {}
    scan_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        summary, rows = _evaluate_label(base, label_key)
        summaries[label_key] = summary
        scan_rows.extend(rows)
    scan_csv = REPORT_DIR / "multicondition_residual_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_multicondition_residual",
        "feature_db": str(FEATURE_DB),
        "feature_table": FEATURE_TABLE,
        "features": FEATURES,
        "candidates": CANDIDATES,
        "train_window": {"start": rb.START_DATE, "end": TRAIN_END},
        "holdout_window": {"start": HOLDOUT_START, "end": "mature_label_cutoff_by_horizon"},
        "scan_csv": str(scan_csv),
        "summaries": summaries,
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
    report_json = REPORT_DIR / "multicondition_residual_report.json"
    report_md = REPORT_DIR / "multicondition_residual_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status_and_frontier(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
