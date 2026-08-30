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
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "model_agent_four_year_3d_wide_feature_screen_xgb_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_3d_quarterly_feature_tophit_20260714 as base  # noqa: E402
from adjustment_semantics import (  # noqa: E402
    NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS,
    NAKED_MARKET_PRICE_COLUMNS,
)
from model_asset_route import (  # noqa: E402
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_model_label_duckdb_path,
    resolve_model_label_duckdb_table,
)

START_DATE = "20220606"
TRAIN_END = base.TRAIN_END
LABEL_NAME = base.LABEL_NAME
LABEL_COL = base.LABEL_COL
TOP_K = base.TOP_K
SAMPLE_MOD = 50
SELECTED_FEATURE_COUNT = 50
MAX_TRAIN_ROWS = 260_000
MAX_POS_ROWS = 70_000

NUMERIC_TYPES = {
    "DOUBLE",
    "FLOAT",
    "REAL",
    "BIGINT",
    "INTEGER",
    "HUGEINT",
    "UBIGINT",
    "UINTEGER",
    "SMALLINT",
    "TINYINT",
    "DECIMAL",
}


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _candidate_numeric_features(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    banned = set(NAKED_MARKET_PRICE_COLUMNS) | set(NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS)
    banned.update({"trade_date", "stock_code"})
    cols: list[str] = []
    for row in con.execute(f"PRAGMA table_info({_quote(table)})").fetchall():
        name = str(row[1])
        typ = str(row[2]).upper().split("(")[0]
        if typ not in NUMERIC_TYPES:
            continue
        if name in banned:
            continue
        cols.append(name)
    return cols


def _screen_features() -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    feature_db = resolve_model_feature_duckdb_path(DATA, require_exists=True)
    feature_table = resolve_model_feature_duckdb_table(DATA)
    label_db = resolve_model_label_duckdb_path(DATA, require_exists=True)
    label_table = resolve_model_label_duckdb_table(DATA)
    if not feature_table or not label_table:
        raise RuntimeError("active L3 feature/label DuckDB route is missing")

    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{feature_db.as_posix()}' AS feat (READ_ONLY)")
        con.execute(f"ATTACH '{label_db.as_posix()}' AS lab (READ_ONLY)")
        all_features = _candidate_numeric_features(con, f"feat.{_quote(feature_table)}")
        select_cols = ",\n                ".join(f"f.{_quote(col)} AS {_quote(col)}" for col in all_features)
        sample = con.execute(
            f"""
            WITH labeled AS (
                SELECT
                    trade_date,
                    stock_code,
                    PERCENT_RANK() OVER (
                        PARTITION BY trade_date
                        ORDER BY {_quote(LABEL_NAME)}
                    ) AS label_rank
                FROM lab.{_quote(label_table)}
                WHERE trade_date >= ?
                  AND {_quote(LABEL_NAME)} IS NOT NULL
                  AND stock_code NOT LIKE '%.BJ'
            )
            SELECT
                f.trade_date,
                f.stock_code,
                l.label_rank,
                {select_cols}
            FROM feat.{_quote(feature_table)} f
            JOIN labeled l USING (trade_date, stock_code)
            WHERE hash(f.stock_code || f.trade_date) % {SAMPLE_MOD} = 0
            ORDER BY f.trade_date, f.stock_code
            """,
            [START_DATE],
        ).fetchdf()

    feature_rows: list[dict[str, Any]] = []
    y = pd.to_numeric(sample["label_rank"], errors="coerce")
    for col in all_features:
        x = pd.to_numeric(sample[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = x.notna() & y.notna()
        valid_count = int(valid.sum())
        if valid_count < 5_000:
            continue
        std = float(x[valid].std())
        if not np.isfinite(std) or std <= 1e-12:
            continue
        corr = float(x[valid].corr(y[valid]))
        if not np.isfinite(corr):
            continue
        top_cut = x[valid].quantile(0.95)
        bottom_cut = x[valid].quantile(0.05)
        top_mean = float(y[valid & (x >= top_cut)].mean())
        bottom_mean = float(y[valid & (x <= bottom_cut)].mean())
        feature_rows.append(
            {
                "feature": col,
                "corr": corr,
                "abs_corr": abs(corr),
                "valid_count": valid_count,
                "null_rate": float(1.0 - valid_count / max(len(sample), 1)),
                "top5_label_rank_mean": top_mean,
                "bottom5_label_rank_mean": bottom_mean,
                "top_bottom_spread": top_mean - bottom_mean,
                "qfq_explicit": col.endswith("_qfq") or not col.startswith("gtja_alpha"),
            }
        )
    screen = pd.DataFrame(feature_rows).sort_values(["abs_corr", "valid_count"], ascending=[False, False])
    selected = screen.head(SELECTED_FEATURE_COUNT)["feature"].astype(str).tolist()
    route_info = {
        "feature_asset": f"{feature_db.as_posix()}::{feature_table}",
        "label_asset": f"{label_db.as_posix()}::{label_table}",
        "candidate_numeric_feature_count": len(all_features),
        "sample_mod": SAMPLE_MOD,
        "sample_rows": int(len(sample)),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "excluded_naked_front_adjusted_columns": sorted(set(NAKED_MARKET_PRICE_COLUMNS) | set(NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS)),
    }
    return selected, screen, route_info


def _load_feature_frame(features: list[str]) -> pd.DataFrame:
    feature_db = resolve_model_feature_duckdb_path(DATA, require_exists=True)
    feature_table = resolve_model_feature_duckdb_table(DATA)
    label_db = resolve_model_label_duckdb_path(DATA, require_exists=True)
    label_table = resolve_model_label_duckdb_table(DATA)
    if not feature_table or not label_table:
        raise RuntimeError("active L3 feature/label DuckDB route is missing")
    select_cols = ",\n                ".join(f"f.{_quote(col)} AS {_quote(col)}" for col in features)
    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{feature_db.as_posix()}' AS feat (READ_ONLY)")
        con.execute(f"ATTACH '{label_db.as_posix()}' AS lab (READ_ONLY)")
        frame = con.execute(
            f"""
            SELECT
                f.trade_date,
                f.stock_code,
                {select_cols},
                l.{_quote(LABEL_NAME)} AS {LABEL_COL}
            FROM feat.{_quote(feature_table)} f
            JOIN lab.{_quote(label_table)} l USING (trade_date, stock_code)
            WHERE f.trade_date >= ?
              AND l.{_quote(LABEL_NAME)} IS NOT NULL
              AND f.stock_code NOT LIKE '%.BJ'
            ORDER BY f.trade_date, f.stock_code
            """,
            [START_DATE],
        ).fetchdf()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["label_rank"] = frame.groupby("trade_date")[LABEL_COL].rank(method="average", pct=True)
    frame["quarter"] = pd.PeriodIndex(pd.to_datetime(frame["trade_date"]), freq="Q").astype(str)
    return frame


def _target(frame: pd.DataFrame, top_pct: float) -> np.ndarray:
    return (frame["label_rank"].to_numpy(dtype=np.float32) >= (1.0 - top_pct)).astype(np.int8)


def _sample_training_rows(frame: pd.DataFrame, top_pct: float, neg_ratio: int) -> pd.DataFrame:
    y = _target(frame, top_pct)
    positives = frame[y == 1]
    negatives = frame[y == 0]
    if positives.empty or negatives.empty:
        return frame.iloc[0:0].copy()
    if len(positives) > MAX_POS_ROWS:
        positives = positives.sample(n=MAX_POS_ROWS, random_state=42)
    n_neg = min(len(negatives), len(positives) * neg_ratio, max(0, MAX_TRAIN_ROWS - len(positives)))
    sampled_neg = negatives.sample(n=n_neg, random_state=42)
    return pd.concat([positives, sampled_neg], ignore_index=True).sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _fit_predict_quarterly_xgb(frame: pd.DataFrame, features: list[str], config: dict[str, Any]) -> pd.DataFrame:
    dates = list(frame["trade_date"].drop_duplicates())
    eval_start = dates[base.INITIAL_TRAIN_DAYS]
    eval_frame = frame[frame["trade_date"] >= eval_start].copy()
    rows: list[pd.DataFrame] = []
    for quarter in list(eval_frame["quarter"].drop_duplicates()):
        test = eval_frame[eval_frame["quarter"] == quarter].copy()
        train = frame[frame["trade_date"] < str(test["trade_date"].min())].copy()
        train_sample = _sample_training_rows(train, float(config["top_pct"]), int(config["neg_ratio"]))
        if train_sample.empty or len(train_sample) < 5_000:
            continue
        x_train = train_sample[features].apply(pd.to_numeric, errors="coerce")
        medians = x_train.median(numeric_only=True).fillna(0.0)
        x_train = x_train.fillna(medians).to_numpy(dtype=np.float32, copy=True)
        y_train = _target(train_sample, float(config["top_pct"]))
        pos = max(float(y_train.sum()), 1.0)
        neg = max(float(len(y_train) - y_train.sum()), 1.0)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            n_estimators=int(config["n_estimators"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            min_child_weight=10,
            reg_alpha=0.0,
            reg_lambda=6.0,
            scale_pos_weight=neg / pos,
            n_jobs=8,
            random_state=42,
        )
        model.fit(x_train, y_train, verbose=False)
        x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
        x_test = x_test.to_numpy(dtype=np.float32, copy=True)
        out = test[["trade_date", "stock_code", LABEL_COL]].copy()
        out["wide_xgb_proba"] = model.predict_proba(x_test)[:, 1]
        out["quarter_model_train_rows"] = int(len(train_sample))
        out["quarter_model_positive_rows"] = int(y_train.sum())
        rows.append(out)
    pred = pd.concat(rows, ignore_index=True)
    pred["wide_xgb_rank"] = pred.groupby("trade_date")["wide_xgb_proba"].rank(method="average", pct=True)
    return pred


def _register_result(payload: dict[str, Any], report_json: Path, report_md: Path, scan_csv: Path, screen_csv: Path) -> None:
    status_path = DATA / "reports" / "model_agent_current_research_candidate_status_20260714" / "current_research_candidate_status_20260714.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.setdefault("not_promoted_research_lines", {})["wide_feature_screen_xgb_3d"] = {
        "label": LABEL_NAME,
        "decision": payload["decision"],
        "selected_config": payload["selected_by_train"]["config"],
        "blend_mode": payload["selected_by_train"]["blend_mode"],
        "blend_weight": payload["selected_by_train"]["blend_weight"],
        "passing_count": payload["passing_count"],
        "failed_reasons": payload["selected_by_train"]["gate"]["failed_reasons"],
        "key_deltas": {
            "full_rank_ic_delta": payload["selected_by_train"]["deltas"]["full"]["rank_ic"],
            "full_top1_delta": payload["selected_by_train"]["deltas"]["full"]["top1"],
            "full_top3_delta": payload["selected_by_train"]["deltas"]["full"]["top3"],
            "full_top5_delta": payload["selected_by_train"]["deltas"]["full"]["top5"],
            "holdout_top5_delta": payload["selected_by_train"]["deltas"]["holdout"]["top5"],
            "recent63_top5_delta": payload["selected_by_train"]["deltas"]["recent63"]["top5"],
            "recent20_top5_delta": payload["selected_by_train"]["deltas"]["recent20"]["top5"],
        },
        "evidence": {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "scan_csv": str(scan_csv),
            "feature_screen_csv": str(screen_csv),
            "script": str(Path(__file__).resolve()),
        },
        "boundary": "research_only; no formal/L5/production/signal/backtest change",
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected_features, screen, route_info = _screen_features()
    screen_csv = REPORT_DIR / "wide_feature_screen_3d.csv"
    screen.to_csv(screen_csv, index=False, encoding="utf-8-sig")

    features = _load_feature_frame(selected_features)
    baseline_frame = base._formal_3d_frame()
    configs = [
        {"top_pct": 0.05, "neg_ratio": 6, "max_depth": 2, "learning_rate": 0.025, "n_estimators": 220, "subsample": 0.80, "colsample_bytree": 0.75},
        {"top_pct": 0.03, "neg_ratio": 8, "max_depth": 2, "learning_rate": 0.020, "n_estimators": 260, "subsample": 0.80, "colsample_bytree": 0.75},
    ]
    blend_weights = [0.01, 0.02, 0.05, 0.10, 0.20]
    blend_modes = ["direct", "inverse"]

    scan_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, Any]]] = []
    for config in configs:
        pred = _fit_predict_quarterly_xgb(features, selected_features, config)
        merged = baseline_frame.merge(pred, on=["trade_date", "stock_code", LABEL_COL], how="inner")
        merged["baseline_score"] = merged["pred_3d_rank"]
        _, baseline_metrics = base._metrics_for_score(merged, "baseline_score")
        for mode in blend_modes:
            feature_rank = merged["wide_xgb_rank"] if mode == "direct" else 1.0 - merged["wide_xgb_rank"]
            for weight in blend_weights:
                merged["candidate_score"] = (1.0 - weight) * merged["pred_3d_rank"] + weight * feature_rank
                _, candidate_metrics = base._metrics_for_score(merged, "candidate_score")
                deltas = {
                    block: base._delta_block(candidate_metrics, baseline_metrics, block)
                    for block in ["full", "train", "holdout", "recent63", "recent20", "recent126"]
                }
                passed, failed_reasons = base._gate(candidate_metrics, baseline_metrics)
                selection_score = deltas["train"]["top5"] + 0.5 * deltas["train"]["top10"] + 0.25 * deltas["train"]["top3"]
                row = {
                    **config,
                    "blend_mode": mode,
                    "blend_weight": weight,
                    "passed": passed,
                    "failed_reasons": ";".join(failed_reasons),
                    "selection_score_train_only": selection_score,
                    "eval_trade_days": candidate_metrics["eval_trade_days"],
                    "eval_min_trade_date": candidate_metrics["eval_min_trade_date"],
                    "eval_max_trade_date": candidate_metrics["eval_max_trade_date"],
                    "full_rank_ic_delta": deltas["full"]["rank_ic"],
                    "full_top1_delta": deltas["full"]["top1"],
                    "full_top3_delta": deltas["full"]["top3"],
                    "full_top5_delta": deltas["full"]["top5"],
                    "full_top10_delta": deltas["full"]["top10"],
                    "full_top20_delta": deltas["full"]["top20"],
                    "holdout_top5_delta": deltas["holdout"]["top5"],
                    "recent63_top5_delta": deltas["recent63"]["top5"],
                    "recent20_top5_delta": deltas["recent20"]["top5"],
                }
                scan_rows.append(row)
                candidates.append(
                    (
                        selection_score,
                        {
                            "label": LABEL_NAME,
                            "config": config,
                            "blend_mode": mode,
                            "blend_weight": weight,
                            "selected_features": selected_features,
                            "baseline_current_formal_same_dates": baseline_metrics,
                            "candidate_metrics": candidate_metrics,
                            "deltas": deltas,
                            "gate": {"passed": passed, "failed_reasons": failed_reasons},
                            "selection_score_train_only": selection_score,
                        },
                    )
                )

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "wide_feature_xgb_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    passing = [payload for _, payload in candidates if payload["gate"]["passed"]]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_wide_feature_screen_xgb",
        "method": "sampled feature screen plus quarterly rolling XGBClassifier for 3D top-hit probability",
        "selection_rule": "Feature screen uses deterministic sample; model config/blend selected only by train-window deltas through 20251231. Holdout/recent windows are validation only.",
        "route_info": route_info,
        "max_train_rows": MAX_TRAIN_ROWS,
        "max_positive_rows": MAX_POS_ROWS,
        "selected_by_train": selected,
        "passing_count": len(passing),
        "best_passing_by_train_rank": passing[0] if passing else None,
        "feature_screen_csv": str(screen_csv),
        "scan_csv": str(scan_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else ("candidate_exists_but_not_train_selected" if passing else "no_3d_candidate_passed_wide_feature_screen_xgb_scan")
        ),
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "wide_feature_xgb_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D 宽特征筛选 XGBoost 研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- 候选数值特征数：`{route_info['candidate_numeric_feature_count']}`",
        f"- 筛选样本行数：`{route_info['sample_rows']}`",
        f"- 选中特征数：`{route_info['selected_feature_count']}`",
        f"- 训练窗口选择配置：`{sel['config']}`",
        f"- blend_mode：`{sel['blend_mode']}`",
        f"- blend_weight：`{sel['blend_weight']}`",
        f"- 是否过门：`{sel['gate']['passed']}`",
        f"- 失败原因：`{'; '.join(sel['gate']['failed_reasons']) if sel['gate']['failed_reasons'] else '-'}`",
        "",
        "## 关键增量",
        "",
        "| 指标 | 增量 |",
        "|---|---:|",
    ]
    for block in ["full", "holdout", "recent63", "recent20"]:
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
            lines.append(f"| {block}.{metric} | {sel['deltas'][block][metric]:.8f} |")
    lines.extend(
        [
            "",
            "## 前复权输入契约治理",
            "",
            "- 特征筛选排除了裸价格和已知裸前复权技术字段。",
            "- 允许进入筛选的 GTJA 字段必须是显式 `_qfq` 命名。",
            "- 本实验未要求 L3 回退字段命名，未使用 legacy 宽表。",
            "",
            "## 边界",
            "",
            "- research-only。",
            "- 未修改 formal manifest。",
            "- 未修改 approved_for_l5。",
            "- 未生成交易信号，未跑策略回测。",
        ]
    )
    report_md = REPORT_DIR / "wide_feature_xgb_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _register_result(payload, report_json, report_md, scan_csv, screen_csv)
    print(json.dumps({"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
