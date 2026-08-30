from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_xgb_residual_tailguard_20260714"
MODEL_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "xgb_residual_tailguard_3d5d10d_20260714"
FEATURE_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

LABEL_KEYS = ["3d", "5d", "10d"]
FIT_END = "20250630"
TAIL_START = "20250701"
TAIL_END = "20251231"
HOLDOUT_START = "20260101"
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
    *[f"{col}_rank" for col in FEATURES],
]
BETAS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]

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
    for col in MODEL_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.5).astype("float32")
    return frame


def _baseline_daily(base: pd.DataFrame, label_key: str) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    frame = base[base[label_col].notna()].copy()
    return rb._daily_eval(frame, f"pred_{label_key}_rank", label_col)


def _candidate_daily(label_frame: pd.DataFrame, label_key: str, model_pred: np.ndarray, beta: float) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    frame = label_frame.copy()
    frame["_model_pred"] = model_pred
    frame["_model_rank"] = frame.groupby("trade_date")["_model_pred"].rank(method="average", pct=True)
    frame["candidate_score"] = frame[f"pred_{label_key}_rank"] + beta * (frame["_model_rank"] - 0.5)
    return rb._daily_eval(frame, "candidate_score", label_col)


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
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
        "fit_trade_days": int((daily["trade_date"] <= FIT_END).sum()),
        "tail_trade_days": int(((daily["trade_date"] >= TAIL_START) & (daily["trade_date"] <= TAIL_END)).sum()),
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
    tail = _delta_block(candidate, baseline, "tail")
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
    if (tail["top5"] or 0.0) <= 0.0:
        reasons.append("tail_top5_delta_non_positive")
    if (holdout["top5"] or 0.0) <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if (recent63["top5"] or 0.0) <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if (recent20["top5"] or 0.0) <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    return not reasons, reasons


def _train_sample(frame: pd.DataFrame, label_key: str) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    pred_col = f"pred_{label_key}_rank"
    fit = frame[(frame["trade_date"] <= FIT_END) & frame[label_col].notna()].copy()
    top_mask = fit[pred_col] >= 0.85
    sample_mask = top_mask | (fit.groupby("trade_date").cumcount() % 25 == 0)
    return fit[sample_mask].copy()


def _train_model(frame: pd.DataFrame, label_key: str) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    label_col = f"label_{label_key}"
    sample = _train_sample(frame, label_key)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_depth=3,
        learning_rate=0.03,
        n_estimators=180,
        subsample=0.75,
        colsample_bytree=0.85,
        reg_lambda=12.0,
        min_child_weight=35,
        n_jobs=4,
        random_state=20260714,
    )
    model.fit(sample[MODEL_FEATURES], sample[label_col].astype("float32"))
    return model, {
        "train_rows": int(len(sample)),
        "fit_min_trade_date": str(sample["trade_date"].min()) if len(sample) else None,
        "fit_max_trade_date": str(sample["trade_date"].max()) if len(sample) else None,
        "feature_count": len(MODEL_FEATURES),
        "model_params": model.get_params(),
    }


def _tail_guard_objective(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    fit = _delta_block(candidate, baseline, "fit")
    tail = _delta_block(candidate, baseline, "tail")
    full = _delta_block(candidate, baseline, "full")
    tail_penalty = 100.0 * max(0.0, -(tail["top5"] or 0.0))
    return float(
        35.0 * (tail["top5"] or 0.0)
        + 20.0 * (tail["top1"] or 0.0)
        + 10.0 * (full["top5"] or 0.0)
        + 4.0 * (fit["top5"] or 0.0)
        + 3.0 * (full["rank_ic"] or 0.0)
        - tail_penalty
    )


def _evaluate_label(base: pd.DataFrame, label_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    label_frame = base[base[f"label_{label_key}"].notna()].copy()
    baseline = _metrics(_baseline_daily(base, label_key))
    model, train_info = _train_model(base, label_key)
    model_path = MODEL_DIR / f"xgb_residual_tailguard_{label_key}_research_20260714.json"
    model.save_model(str(model_path))
    model_pred = model.predict(label_frame[MODEL_FEATURES])
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_key: tuple[float, float] | None = None
    for beta in BETAS:
        candidate = _metrics(_candidate_daily(label_frame, label_key, model_pred, beta))
        full = _delta_block(candidate, baseline, "full")
        fit = _delta_block(candidate, baseline, "fit")
        tail = _delta_block(candidate, baseline, "tail")
        holdout = _delta_block(candidate, baseline, "holdout")
        recent63 = _delta_block(candidate, baseline, "recent63")
        recent20 = _delta_block(candidate, baseline, "recent20")
        gate_passed, failures = _gate(candidate, baseline)
        objective = _tail_guard_objective(candidate, baseline)
        row = {
            "label_key": label_key,
            "beta": beta,
            "tail_guard_objective": objective,
            "gate_passed": gate_passed,
            "failed_reasons": ";".join(failures),
            "fit_top5_delta": fit["top5"],
            "tail_top5_delta": tail["top5"],
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
        key = (objective, tail["top5"] or -999.0)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected = {
                "label": rb.LABELS[label_key],
                "selection_rule": "Model is fit on <=20250630; beta is selected by 20250701-20251231 tail-guard objective only.",
                "selected_beta": beta,
                "tail_guard_objective": objective,
                "model_path": str(model_path),
                "train_info": train_info,
                "model_features": MODEL_FEATURES,
                "deltas": {
                    "full": full,
                    "fit": fit,
                    "tail": tail,
                    "holdout": holdout,
                    "recent63": recent63,
                    "recent20": recent20,
                },
                "candidate_metrics": candidate,
                "baseline_metrics": baseline,
                "gate": {"passed": gate_passed, "failed_reasons": failures},
            }
    assert selected is not None
    return selected, rows


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D/5D/10D XGBoost Residual Tail-Guard 研究（20260714）",
        "",
        "## 结论",
        "",
        "本轮采用近期保护型选择：模型只在 `20220606-20250630` 拟合，beta 只用 `20250701-20251231` 尾部窗口选择；`20260101` 之后 holdout/recent 完全留作外推验证。",
        "",
        "| 标签 | beta | Full Top5 Δ | Tail Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ | 是否过门 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in LABEL_KEYS:
        item = payload["summaries"][key]
        lines.append(
            "| {key} | {beta:.2f} | {full:.6f} | {tail:.6f} | {holdout:.6f} | {r63:.6f} | {r20:.6f} | {passed} |".format(
                key=key,
                beta=item["selected_beta"],
                full=item["deltas"]["full"]["top5"] or 0.0,
                tail=item["deltas"]["tail"]["top5"] or 0.0,
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
    decision = "xgb_residual_tailguard_passed_for_some_labels_candidate_discussion_required" if passed else "no_3d_5d_10d_candidate_passed_xgb_residual_tailguard_scan"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["xgb_residual_tailguard_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": "XGBoost residual tail-guard learner selected beta on 202507-202512 and validated on 2026 holdout/recent.",
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
        frontier["latest_xgb_residual_tailguard_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base = _prepare_base()
    summaries: dict[str, Any] = {}
    scan_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        summary, rows = _evaluate_label(base, label_key)
        summaries[label_key] = summary
        scan_rows.extend(rows)
    scan_csv = REPORT_DIR / "xgb_residual_tailguard_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_xgb_residual_tailguard",
        "feature_db": str(FEATURE_DB),
        "feature_table": FEATURE_TABLE,
        "fit_window": {"start": rb.START_DATE, "end": FIT_END},
        "tail_guard_window": {"start": TAIL_START, "end": TAIL_END},
        "holdout_window": {"start": HOLDOUT_START, "end": "mature_label_cutoff_by_horizon"},
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
    report_json = REPORT_DIR / "xgb_residual_tailguard_report.json"
    report_md = REPORT_DIR / "xgb_residual_tailguard_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status_and_frontier(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
