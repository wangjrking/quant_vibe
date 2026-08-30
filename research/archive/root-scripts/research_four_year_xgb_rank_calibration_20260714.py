from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import research_four_year_top_consensus_holdout_20260714 as base


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_xgb_rank_calibration_20260714"
)
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_xgb_rank_calibration_20260714.duckdb"
)
MODEL_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "models"
    / "xgb_rank_calibration_20260714"
)

TARGET_LABELS = ["3d", "5d", "10d"]
TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
MAX_TRAIN_ROWS = 700_000


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    rank_cols = ["pred_1d_rank", "pred_3d_rank", "pred_5d_rank", "pred_10d_rank"]
    out["rank_mean"] = out[rank_cols].mean(axis=1)
    out["rank_min"] = out[rank_cols].min(axis=1)
    out["rank_max"] = out[rank_cols].max(axis=1)
    out["rank_spread"] = out["rank_max"] - out["rank_min"]
    out["rank_top2_mean"] = np.sort(out[rank_cols].to_numpy(dtype=np.float64), axis=1)[:, -2:].mean(axis=1)
    out["rank_bottom2_mean"] = np.sort(out[rank_cols].to_numpy(dtype=np.float64), axis=1)[:, :2].mean(axis=1)
    out["r1_r3_min"] = np.minimum(out["pred_1d_rank"], out["pred_3d_rank"])
    out["r3_r5_min"] = np.minimum(out["pred_3d_rank"], out["pred_5d_rank"])
    out["r5_r10_min"] = np.minimum(out["pred_5d_rank"], out["pred_10d_rank"])
    out["r1_r10_min"] = np.minimum(out["pred_1d_rank"], out["pred_10d_rank"])
    out["r3_r5_prod"] = out["pred_3d_rank"] * out["pred_5d_rank"]
    out["r5_r10_prod"] = out["pred_5d_rank"] * out["pred_10d_rank"]
    out["r1_r10_prod"] = out["pred_1d_rank"] * out["pred_10d_rank"]
    feature_cols = [
        "pred_1d_rank",
        "pred_3d_rank",
        "pred_5d_rank",
        "pred_10d_rank",
        "rank_mean",
        "rank_min",
        "rank_max",
        "rank_spread",
        "rank_top2_mean",
        "rank_bottom2_mean",
        "r1_r3_min",
        "r3_r5_min",
        "r5_r10_min",
        "r1_r10_min",
        "r3_r5_prod",
        "r5_r10_prod",
        "r1_r10_prod",
    ]
    return out, feature_cols


def _fit_model(train: pd.DataFrame, label_col: str, feature_cols: list[str]) -> XGBRegressor:
    work = train[train[label_col].notna()].copy()
    if len(work) > MAX_TRAIN_ROWS:
        work = work.sample(n=MAX_TRAIN_ROWS, random_state=20260714)
    model = XGBRegressor(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.75,
        colsample_bytree=0.85,
        min_child_weight=30,
        reg_lambda=12.0,
        reg_alpha=0.2,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=20260714,
    )
    model.fit(work[feature_cols].to_numpy(dtype=np.float32), work[label_col].to_numpy(dtype=np.float32))
    return model


def _evaluate_score(frame: pd.DataFrame, score_col: str, label_key: str) -> dict[str, Any]:
    daily = base._daily_eval(frame, score_col, f"label_{label_key}")
    return base._summarize_daily(daily)


def _improvement_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    passed, reasons = base._improvement_gate(candidate, baseline)
    delta = base._delta(candidate, baseline)
    if (delta.get("holdout_rank_ic") or 0.0) < -0.0015:
        reasons.append("holdout_rank_ic_delta_below_-0_0015")
    if (delta.get("recent63_rank_ic") or 0.0) < -0.003:
        reasons.append("recent63_rank_ic_delta_below_-0_003")
    return not reasons, reasons


def _materialize(full_frame: pd.DataFrame, label_key: str, score_col: str) -> dict[str, Any]:
    table = f"stock_predict_data_model_agent_four_year_xgb_rank_calibration_20260714_executable_{label_key}_open_return_research"
    out = full_frame[["trade_date", "stock_code", score_col]].copy()
    out = out.rename(columns={score_col: "pred_prob"})
    out["score_source"] = "xgb_rank_calibration_20260714_research"
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_frame", out)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM candidate_frame")
        latest_date = con.execute(f"SELECT max(trade_date) FROM {table}").fetchone()[0]
        quality = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(distinct trade_date) AS trade_days,
                count(distinct stock_code) AS stock_count,
                sum(case when pred_prob is null then 1 else 0 end) AS null_pred_prob,
                sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows,
                (
                    SELECT count(*)
                    FROM (
                        SELECT trade_date, stock_code, count(*) c
                        FROM {table}
                        GROUP BY 1,2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups,
                sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
                count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {table}
            """,
            [latest_date, latest_date],
        ).fetchdf().iloc[0].to_dict()
        digest = hashlib.sha256()
        for trade_date, stock_code, pred_prob in con.execute(
            f"SELECT trade_date, stock_code, pred_prob FROM {table} ORDER BY trade_date, stock_code"
        ).fetchall():
            digest.update(f"{trade_date}|{stock_code}|{float(pred_prob):.17g}\n".encode("utf-8"))
    return {
        "db_path": str(EXPERIMENT_DB),
        "table": table,
        "quality": {key: _to_float(value) for key, value in quality.items()},
        "pred_prob_sha256": digest.hexdigest(),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labeled = base._add_rank_columns(base._read_scores(with_labels=True))
    full = base._add_rank_columns(base._read_scores(with_labels=False))
    labeled, feature_cols = _feature_frame(labeled)
    full, _ = _feature_frame(full)

    reports: dict[str, Any] = {}
    for label_key in TARGET_LABELS:
        label_col = f"label_{label_key}"
        baseline_frame = labeled[labeled[label_col].notna()].copy()
        baseline_frame["baseline_score"] = baseline_frame[f"pred_{label_key}_rank"]
        baseline = _evaluate_score(baseline_frame, "baseline_score", label_key)

        train = labeled[(labeled["trade_date"] <= TRAIN_END) & labeled[label_col].notna()].copy()
        model = _fit_model(train, label_col, feature_cols)
        model_path = MODEL_DIR / f"xgb_rank_calibration_{label_key}_20260714.json"
        model.save_model(str(model_path))

        scored = baseline_frame.copy()
        raw_pred = model.predict(scored[feature_cols].to_numpy(dtype=np.float32))
        scored["xgb_raw_score"] = raw_pred
        scored["xgb_score"] = scored.groupby("trade_date")["xgb_raw_score"].rank(method="average", pct=True)
        candidate = _evaluate_score(scored, "xgb_score", label_key)
        delta = base._delta(candidate, baseline)
        passed, failures = _improvement_gate(candidate, baseline)

        full_scored = full.copy()
        full_scored["xgb_raw_score"] = model.predict(full_scored[feature_cols].to_numpy(dtype=np.float32))
        full_scored[f"xgb_{label_key}_score"] = full_scored.groupby("trade_date")["xgb_raw_score"].rank(method="average", pct=True)
        materialized = _materialize(full_scored, label_key, f"xgb_{label_key}_score")

        booster = model.get_booster()
        importance = booster.get_score(importance_type="gain")
        reports[label_key] = {
            "label": base.LABELS[label_key],
            "train_window": {"min_trade_date": base.START_DATE, "max_trade_date": TRAIN_END},
            "holdout_window": {"min_trade_date": HOLDOUT_START, "max_trade_date": "label_maturity_by_horizon"},
            "train_rows_used": int(min(len(train), MAX_TRAIN_ROWS)),
            "model_config": {
                "type": "XGBRegressor",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "subsample": model.subsample,
                "colsample_bytree": model.colsample_bytree,
                "min_child_weight": model.min_child_weight,
                "reg_lambda": model.reg_lambda,
                "reg_alpha": model.reg_alpha,
                "random_state": model.random_state,
            },
            "feature_cols": feature_cols,
            "model_file": str(model_path),
            "feature_importance_gain": {k: _to_float(v) for k, v in importance.items()},
            "baseline": baseline,
            "candidate": candidate,
            "delta_vs_baseline": delta,
            "improvement_gate": {"passed": passed, "failed_reasons": failures},
            "materialized_candidate": materialized,
        }

    summary = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_four_year_xgb_rank_calibration_3d5d10d",
        "observation_start": base.START_DATE,
        "labels": reports,
        "candidate_db": str(EXPERIMENT_DB),
        "model_dir": str(MODEL_DIR),
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
            "label_maturity_respected_for_evaluation": True,
        },
    }
    report_json = REPORT_DIR / "xgb_rank_calibration_summary.json"
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 3D/5D/10D XGBoost 排名校准实验",
        "",
        "## 结论",
        "",
        "本轮为 research-only 实验，使用当前 formal L4 四个 horizon 分数的截面排名及交互项训练轻量 XGBoost 校准器。",
        "",
        "| 标签 | 通过改进门 | Full RankIC Δ | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label_key, item in reports.items():
        delta = item["delta_vs_baseline"]
        gate = item["improvement_gate"]
        lines.append(
            "| {label} | {passed} | {ric:.6f} | {f5:.6f} | {h5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key.upper(),
                passed="通过" if gate["passed"] else "未通过",
                ric=delta.get("full_rank_ic") or 0.0,
                f5=delta.get("full_top5") or 0.0,
                h5=delta.get("holdout_top5") or 0.0,
                r63=delta.get("recent63_top5") or 0.0,
                r20=delta.get("recent20_top5") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 实验资产在模型实验区。",
            "- 未修改 formal manifest。",
            "- 未生成 `approved_for_l5` 资产。",
            "- 未生成策略信号，未跑回测。",
            "",
            "## 证据",
            "",
            f"- JSON：`{report_json}`",
            f"- 实验库：`{EXPERIMENT_DB}`",
        ]
    )
    report_md = REPORT_DIR / "xgb_rank_calibration_summary.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report_json": str(report_json), "candidate_db": str(EXPERIMENT_DB)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
