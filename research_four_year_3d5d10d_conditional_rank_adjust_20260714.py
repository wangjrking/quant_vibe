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
    / "model_agent_four_year_3d5d10d_conditional_rank_adjust_20260714"
)
FEATURE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
STATUS_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

LABEL_KEYS = ["3d", "5d", "10d"]
TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
FEATURES = [
    "amount",
    "total_mv",
    "turnover_rate",
    "atr_qfq",
]
ALPHAS = [-0.04, -0.02, 0.02, 0.04]

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


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


def _candidate_daily(base: pd.DataFrame, label_key: str, feature: str, alpha: float) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    pred_rank_col = f"pred_{label_key}_rank"
    feature_rank_col = f"{feature}_rank"
    frame = base[base[label_col].notna()].copy()
    frame["candidate_score"] = frame[pred_rank_col] + alpha * (frame[feature_rank_col].fillna(0.5) - 0.5)
    return rb._daily_eval(frame, "candidate_score", label_col)


def _baseline_daily(base: pd.DataFrame, label_key: str) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    pred_rank_col = f"pred_{label_key}_rank"
    frame = base[base[label_col].notna()].copy()
    return rb._daily_eval(frame, pred_rank_col, label_col)


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
        35.0 * (train["top5"] or 0.0)
        + 10.0 * (train["top1"] or 0.0)
        + 4.0 * (train["rank_ic"] or 0.0)
        - 5.0 * max(0.0, -(train["top20"] or 0.0))
    )


def _evaluate_label(base: pd.DataFrame, label_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = _metrics(_baseline_daily(base, label_key))
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_summary: dict[str, Any] | None = None

    for feature in FEATURES:
        for alpha in ALPHAS:
            candidate = _metrics(_candidate_daily(base, label_key, feature, alpha))
            gate_passed, failures = _gate(candidate, baseline)
            train_objective = _train_objective(candidate, baseline)
            full_delta = _delta_block(candidate, baseline, "full")
            train_delta = _delta_block(candidate, baseline, "train")
            holdout_delta = _delta_block(candidate, baseline, "holdout")
            recent63_delta = _delta_block(candidate, baseline, "recent63")
            recent20_delta = _delta_block(candidate, baseline, "recent20")
            row = {
                "label_key": label_key,
                "feature": feature,
                "alpha": alpha,
                "train_objective": train_objective,
                "gate_passed": gate_passed,
                "failed_reasons": ";".join(failures),
                "train_top5_delta": train_delta["top5"],
                "train_rank_ic_delta": train_delta["rank_ic"],
                "full_rank_ic_delta": full_delta["rank_ic"],
                "full_top1_delta": full_delta["top1"],
                "full_top3_delta": full_delta["top3"],
                "full_top5_delta": full_delta["top5"],
                "full_top10_delta": full_delta["top10"],
                "full_top20_delta": full_delta["top20"],
                "holdout_top5_delta": holdout_delta["top5"],
                "recent63_top5_delta": recent63_delta["top5"],
                "recent20_top5_delta": recent20_delta["top5"],
            }
            rows.append(row)
            key = (train_objective, train_delta["top5"] or -999.0)
            if best_key is None or key > best_key:
                best_key = key
                best_summary = {
                    "label": rb.LABELS[label_key],
                    "selection_rule": "Only train-window objective over 20220606-20251231 is used for feature/alpha selection.",
                    "selected_feature": feature,
                    "selected_alpha": alpha,
                    "train_objective": train_objective,
                    "deltas": {
                        "full": full_delta,
                        "train": train_delta,
                        "holdout": holdout_delta,
                        "recent63": recent63_delta,
                        "recent20": recent20_delta,
                    },
                    "candidate_metrics": candidate,
                    "baseline_metrics": baseline,
                    "gate": {"passed": gate_passed, "failed_reasons": failures},
                }

    assert best_summary is not None
    return best_summary, rows


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D/5D/10D 条件 Rank 调节扫描（20260714）",
        "",
        "## 结论",
        "",
        "本轮是 research-only 小规模条件调节：用训练窗口选择一个 L3 条件特征和调节强度，再在 holdout / recent 上验证。没有训练生产模型，没有写预测资产，没有改 formal manifest。",
        "",
        "| 标签 | 选中特征 | alpha | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ | 是否过门 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in LABEL_KEYS:
        item = payload["summaries"][key]
        lines.append(
            "| {key} | `{feature}` | {alpha:.3f} | {full:.6f} | {holdout:.6f} | {r63:.6f} | {r20:.6f} | {passed} |".format(
                key=key,
                feature=item["selected_feature"],
                alpha=item["selected_alpha"],
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
            "## 判断",
            "",
            "- 该方向如果仍不能同时守住 holdout / Recent20 / Recent63，说明单一条件调节不足，需要进入更严格的多条件 residual learner。",
            "- 若某个标签通过门槛，也只能进入 L4 生产候选讨论，不能直接发布 formal / L5。",
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
        "conditional_rank_adjust_passed_for_some_labels_candidate_discussion_required"
        if passed
        else "no_3d_5d_10d_candidate_passed_conditional_rank_adjust_scan"
    )
    reason = (
        "At least one label passed the four-year model-side gate in research-only conditional rank adjustment."
        if passed
        else "Train-selected single-feature rank adjustment did not satisfy all full/holdout/recent TopN gates."
    )
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["conditional_rank_adjust_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": reason,
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
        frontier["latest_conditional_rank_adjust_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        if not passed:
            frontier["recommendations"] = [
                "Do not promote 3D/5D/10D simple score-bonus, rank-blend, or single-feature rank-adjust variants based on training-window gains alone.",
                "Move next to multi-condition residual learners with strict train/holdout separation.",
                "Keep strict train-selected 1D as the current audited research candidate-only best; production promotion still requires separate user authorization and formal audit.",
            ]
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

    scan_csv = REPORT_DIR / "conditional_rank_adjust_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_conditional_rank_adjust",
        "feature_db": str(FEATURE_DB),
        "feature_table": FEATURE_TABLE,
        "features": FEATURES,
        "alphas": ALPHAS,
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
    report_json = REPORT_DIR / "conditional_rank_adjust_report.json"
    report_md = REPORT_DIR / "conditional_rank_adjust_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status_and_frontier(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
