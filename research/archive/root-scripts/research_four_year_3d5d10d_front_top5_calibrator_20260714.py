from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_front_top5_calibrator_20260714"
MODEL_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "front_top5_calibrator_3d5d10d_20260714"
FEATURE_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


LABEL_KEYS = ["3d", "5d", "10d"]
FIT_END = "20250630"
TAIL_START = "20250701"
TAIL_END = "20251231"
HOLDOUT_START = "20260101"
POOL_THRESHOLDS = [0.90, 0.95, 0.97]
BETAS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]
FEATURES = [
    "amount",
    "total_mv",
    "circ_mv",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "atr_qfq",
    "macd_qfq",
    "macdsignal_qfq",
    "macdhist_qfq",
    "kdj_qfq",
    "kdj_k_qfq",
    "kdj_d_qfq",
    "alpha158_std20",
    "alpha158_std60",
    "vol",
]
MODEL_FEATURES = [
    "pred_1d_rank",
    "pred_3d_rank",
    "pred_5d_rank",
    "pred_10d_rank",
    "pred_1d_gap",
    "pred_3d_gap",
    "pred_5d_gap",
    "pred_10d_gap",
    "rank_consensus_mean",
    "rank_consensus_min",
    "rank_consensus_std",
    *[f"{col}_rank" for col in FEATURES],
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


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
    rank_cols = [f"pred_{key}_rank" for key in ["1d", "3d", "5d", "10d"]]
    for key in ["1d", "3d", "5d", "10d"]:
        others = [col for col in rank_cols if col != f"pred_{key}_rank"]
        frame[f"pred_{key}_gap"] = frame[f"pred_{key}_rank"] - frame[others].mean(axis=1)
    frame["rank_consensus_mean"] = frame[rank_cols].mean(axis=1)
    frame["rank_consensus_min"] = frame[rank_cols].min(axis=1)
    frame["rank_consensus_std"] = frame[rank_cols].std(axis=1)
    for col in FEATURES:
        frame[f"{col}_rank"] = frame.groupby("trade_date")[col].rank(method="average", pct=True)
    for col in MODEL_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.5).astype("float32")
    return frame


def _add_daily_top5_flag(frame: pd.DataFrame, label_key: str) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    work = frame[frame[label_col].notna()].copy()
    work["_label_rank_desc"] = work.groupby("trade_date")[label_col].rank(method="first", ascending=False)
    work["target_top5"] = (work["_label_rank_desc"] <= 5).astype("int8")
    return work


def _train_classifier(train_pool: pd.DataFrame, label_key: str) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    positives = int(train_pool["target_top5"].sum())
    negatives = int(len(train_pool) - positives)
    scale = max(1.0, negatives / max(1, positives))
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=220,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=30,
        subsample=0.78,
        colsample_bytree=0.85,
        reg_lambda=14.0,
        scale_pos_weight=scale,
        tree_method="hist",
        n_jobs=4,
        random_state=20260714 + {"3d": 3, "5d": 5, "10d": 10}[label_key],
        eval_metric="logloss",
    )
    model.fit(train_pool[MODEL_FEATURES], train_pool["target_top5"])
    info = {
        "train_rows": int(len(train_pool)),
        "positive_rows": positives,
        "negative_rows": negatives,
        "scale_pos_weight": float(scale),
        "feature_count": len(MODEL_FEATURES),
        "fit_min_trade_date": str(train_pool["trade_date"].min()) if len(train_pool) else None,
        "fit_max_trade_date": str(train_pool["trade_date"].max()) if len(train_pool) else None,
        "model_params": model.get_params(),
    }
    return model, info


def _daily_eval(frame: pd.DataFrame, label_key: str, score_col: str) -> pd.DataFrame:
    return rb._daily_eval(frame, score_col, f"label_{label_key}")


def _metrics_pack(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "full": rb._metrics(daily),
        "fit": rb._metrics(daily[daily["trade_date"] <= FIT_END]),
        "tail": rb._metrics(daily[(daily["trade_date"] >= TAIL_START) & (daily["trade_date"] <= TAIL_END)]),
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


def _candidate_daily(label_frame: pd.DataFrame, label_key: str, proba: np.ndarray, beta: float, pool_threshold: float) -> pd.DataFrame:
    work = label_frame.copy()
    base_col = f"pred_{label_key}_rank"
    work["_front_proba"] = proba.astype("float32")
    work["_front_proba_rank"] = work.groupby("trade_date")["_front_proba"].rank(method="average", pct=True)
    mask = work[base_col] >= pool_threshold
    work["candidate_score"] = work[base_col]
    work.loc[mask, "candidate_score"] = work.loc[mask, base_col] + beta * (work.loc[mask, "_front_proba_rank"] - 0.5)
    return _daily_eval(work, label_key, "candidate_score")


def _delta_block(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float | None]:
    metrics = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    return {
        metric: _safe_float((candidate[block].get(metric) or 0.0) - (baseline[block].get(metric) or 0.0))
        for metric in metrics
    }


def _deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    return {block: _delta_block(candidate, baseline, block) for block in ["full", "fit", "tail", "holdout", "recent20", "recent63", "recent126"]}


def _gate(deltas: dict[str, dict[str, float | None]], candidate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if candidate["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    full = deltas["full"]
    holdout = deltas["holdout"]
    recent63 = deltas["recent63"]
    recent20 = deltas["recent20"]
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


def _tail_objective(deltas: dict[str, dict[str, float | None]]) -> float:
    tail = deltas["tail"]
    full = deltas["full"]
    return float(
        1.2 * (tail.get("top1") or -1.0)
        + 1.0 * (tail.get("top3") or -1.0)
        + 1.0 * (tail.get("top5") or -1.0)
        + 0.2 * (full.get("top5") or -1.0)
        + 0.02 * (full.get("rank_ic") or -1.0)
    )


def _run_label(base: pd.DataFrame, label_key: str) -> dict[str, Any]:
    label_frame = _add_daily_top5_flag(base, label_key)
    baseline_daily = _daily_eval(label_frame, label_key, f"pred_{label_key}_rank")
    baseline_metrics = _metrics_pack(baseline_daily)
    train_all = label_frame[(label_frame["trade_date"] <= FIT_END) & (label_frame[f"pred_{label_key}_rank"] >= min(POOL_THRESHOLDS))]
    model, train_info = _train_classifier(train_all, label_key)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"front_top5_calibrator_{label_key}_research_20260714.json"
    model.save_model(str(model_path))
    proba = model.predict_proba(label_frame[MODEL_FEATURES])[:, 1]
    scan_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for pool_threshold in POOL_THRESHOLDS:
        for beta in BETAS:
            daily = _candidate_daily(label_frame, label_key, proba, beta, pool_threshold)
            metrics = _metrics_pack(daily)
            deltas = _deltas(metrics, baseline_metrics)
            gate = _gate(deltas, metrics)
            row = {
                "label_key": label_key,
                "pool_threshold": pool_threshold,
                "beta": beta,
                "metrics": metrics,
                "deltas": deltas,
                "gate": gate,
                "tail_objective": _tail_objective(deltas),
            }
            scan_rows.append(
                {
                    "label_key": label_key,
                    "pool_threshold": pool_threshold,
                    "beta": beta,
                    "gate_passed": gate["passed"],
                    "tail_objective": row["tail_objective"],
                    "full_rank_ic_delta": deltas["full"].get("rank_ic"),
                    "full_top1_delta": deltas["full"].get("top1"),
                    "full_top3_delta": deltas["full"].get("top3"),
                    "full_top5_delta": deltas["full"].get("top5"),
                    "full_top10_delta": deltas["full"].get("top10"),
                    "full_top20_delta": deltas["full"].get("top20"),
                    "tail_top5_delta": deltas["tail"].get("top5"),
                    "holdout_top5_delta": deltas["holdout"].get("top5"),
                    "recent63_top5_delta": deltas["recent63"].get("top5"),
                    "recent20_top5_delta": deltas["recent20"].get("top5"),
                    "failed_reasons": ";".join(gate["failed_reasons"]),
                }
            )
            if selected is None or (
                gate["passed"],
                row["tail_objective"],
                deltas["full"].get("top5") or -999.0,
                deltas["recent63"].get("top5") or -999.0,
            ) > (
                selected["gate"]["passed"],
                selected["tail_objective"],
                selected["deltas"]["full"].get("top5") or -999.0,
                selected["deltas"]["recent63"].get("top5") or -999.0,
            ):
                selected = row
    assert selected is not None
    return {
        "label": rb.LABELS[label_key],
        "model_path": str(model_path),
        "train_info": train_info,
        "model_features": MODEL_FEATURES,
        "baseline_current_formal": baseline_metrics,
        "selected": selected,
        "scan_rows": scan_rows,
    }


def _update_status(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    passed = {k: v for k, v in payload["summaries"].items() if v["selected"]["gate"]["passed"]}
    decision = (
        "front_top5_calibrator_passed_for_some_labels_candidate_discussion_required"
        if passed
        else "no_3d_5d_10d_candidate_passed_front_top5_calibrator_scan"
    )
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["front_top5_calibrator_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": "Front-pool top5 classifier calibrates only high formal-score names and is selected by tail guard.",
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
        frontier["latest_front_top5_calibrator_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D / 5D / 10D 前排 Top5 校准研究",
        "",
        "## 结论",
        "",
    ]
    passed = [k for k, v in payload["summaries"].items() if v["selected"]["gate"]["passed"]]
    if passed:
        lines.append(f"通过 research-only 候选门槛的标签：`{', '.join(passed)}`。")
    else:
        lines.append("本轮没有 3D / 5D / 10D 标签通过四年观察与近期窗口门槛，不能进入生产候选讨论。")
    lines.extend(
        [
            "",
            "## 方法",
            "",
            f"- 用 `<= {FIT_END}` 的成熟标签训练前排 Top5 二分类校准器。",
            f"- 只对 formal 分数排名超过 `{POOL_THRESHOLDS}` 的前排池做小幅重排。",
            f"- 用 `{TAIL_START}-{TAIL_END}` tail guard 选择 pool threshold 和 beta，再用 `{HOLDOUT_START}` 之后验证。",
            "- 本轮只输出 research-only 评估和模型文件，不写预测资产表，不修改 formal manifest。",
            "",
            "## 指标摘要",
            "",
            "| 标签 | 通过 | pool | beta | Full RankIC Δ | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label_key, item in payload["summaries"].items():
        selected = item["selected"]
        d = selected["deltas"]
        lines.append(
            "| {label} | {passed} | {pool:.2f} | {beta:.2f} | {fr:.6f} | {ft5:.6f} | {ht5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key,
                passed="是" if selected["gate"]["passed"] else "否",
                pool=selected["pool_threshold"],
                beta=selected["beta"],
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
            "- 未写 formal L4 预测资产。",
            "- 未修改 formal manifest。",
            "- 未修改 `approved_for_l5`。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base = _prepare_base()
    summaries: dict[str, Any] = {}
    scan_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        summary = _run_label(base, label_key)
        scan_rows.extend(summary.pop("scan_rows"))
        summaries[label_key] = summary
    scan_csv = REPORT_DIR / "front_top5_calibrator_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8")
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_front_top5_calibrator",
        "feature_db": str(FEATURE_DB),
        "feature_table": FEATURE_TABLE,
        "fit_window": {"start": rb.START_DATE, "end": FIT_END},
        "tail_guard_window": {"start": TAIL_START, "end": TAIL_END},
        "holdout_window": {"start": HOLDOUT_START, "end": "mature_label_cutoff_by_horizon"},
        "pool_thresholds": POOL_THRESHOLDS,
        "betas": BETAS,
        "scan_csv": str(scan_csv),
        "model_dir": str(MODEL_DIR),
        "summaries": summaries,
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
    report_json = REPORT_DIR / "front_top5_calibrator_report.json"
    report_md = REPORT_DIR / "front_top5_calibrator_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
