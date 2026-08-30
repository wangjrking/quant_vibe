from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "model_agent_four_year_3d_quarterly_feature_tophit_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases  # noqa: E402
from model_asset_route import (  # noqa: E402
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_model_label_duckdb_path,
    resolve_model_label_duckdb_table,
)

START_DATE = "20220606"
LABEL_COL = "label_3d"
LABEL_NAME = "executable_3d_open_return"
TRAIN_END = "20251231"
INITIAL_TRAIN_DAYS = 20
TOP_K = [1, 3, 5, 10, 20]
RECENT_WINDOWS = [20, 63, 126]

LEGACY_3D_FEATURES = [
    "index_2000_close",
    "index_2000_low",
    "index_2000_high",
    "index_2000_open",
    "amount",
    "turnover_rate_f",
    "index_2000_amount",
    "turnover_rate",
    "total_mv",
    "vol",
    "close",
    "high",
    "open",
    "low",
    "pre_close",
]


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _load_feature_label_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_db = resolve_model_feature_duckdb_path(DATA, require_exists=True)
    feature_table = resolve_model_feature_duckdb_table(DATA)
    label_db = resolve_model_label_duckdb_path(DATA, require_exists=True)
    label_table = resolve_model_label_duckdb_table(DATA)
    if not feature_table or not label_table:
        raise RuntimeError("active L3 feature/label DuckDB route is missing")

    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{feature_db.as_posix()}' AS feat (READ_ONLY)")
        con.execute(f"ATTACH '{label_db.as_posix()}' AS lab (READ_ONLY)")
        available = {str(row[0]) for row in con.execute(f"DESCRIBE feat.{_quote(feature_table)}").fetchall()}
        resolved = resolve_savedmodel_feature_aliases(LEGACY_3D_FEATURES, available)
        if resolved["missing"]:
            raise RuntimeError(f"missing feature columns after qfq alias resolution: {resolved['missing']}")
        query_columns = resolved["query_columns"]
        select_feature_cols = ",\n                ".join(f"f.{_quote(col)} AS {_quote(col)}" for col in query_columns)
        frame = con.execute(
            f"""
            SELECT
                f.trade_date,
                f.stock_code,
                {select_feature_cols},
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

    for requested_name, source_name in resolved["alias_pairs"]:
        frame[requested_name] = frame[source_name]
    frame = frame[["trade_date", "stock_code", *LEGACY_3D_FEATURES, LABEL_COL]].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["label_rank"] = frame.groupby("trade_date")[LABEL_COL].rank(method="average", pct=True)
    frame["quarter"] = pd.PeriodIndex(pd.to_datetime(frame["trade_date"]), freq="Q").astype(str)
    route_info = {
        "feature_asset": f"{feature_db.as_posix()}::{feature_table}",
        "label_asset": f"{label_db.as_posix()}::{label_table}",
        "requested_features": LEGACY_3D_FEATURES,
        "query_columns": query_columns,
        "qfq_alias_pairs": resolved["alias_pairs"],
        "rows": int(len(frame)),
        "min_trade_date": str(frame["trade_date"].min()),
        "max_trade_date": str(frame["trade_date"].max()),
        "trade_days": int(frame["trade_date"].nunique()),
        "stocks": int(frame["stock_code"].nunique()),
    }
    return frame, route_info


def _formal_3d_frame() -> pd.DataFrame:
    frame = rb._add_rank_columns(rb._read_scores(with_labels=True))
    frame = frame[frame[LABEL_COL].notna()][["trade_date", "stock_code", LABEL_COL, "pred_3d_rank"]].copy()
    return frame.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _target(frame: pd.DataFrame, top_pct: float) -> np.ndarray:
    return (frame["label_rank"].to_numpy(dtype=np.float32) >= (1.0 - top_pct)).astype(np.int8)


def _sample_training_rows(frame: pd.DataFrame, top_pct: float, neg_ratio: int) -> pd.DataFrame:
    y = _target(frame, top_pct)
    positives = frame[y == 1]
    negatives = frame[y == 0]
    if positives.empty or negatives.empty:
        return frame.iloc[0:0].copy()
    n_neg = min(len(negatives), len(positives) * neg_ratio)
    sampled_neg = negatives.sample(n=n_neg, random_state=42)
    sampled = pd.concat([positives, sampled_neg], ignore_index=True)
    return sampled.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _fit_predict_quarterly(
    frame: pd.DataFrame,
    *,
    top_pct: float,
    neg_ratio: int,
    alpha: float,
    eta0: float,
    epochs: int,
) -> pd.DataFrame:
    dates = list(frame["trade_date"].drop_duplicates())
    eval_start = dates[INITIAL_TRAIN_DAYS]
    eval_frame = frame[frame["trade_date"] >= eval_start].copy()
    quarters = list(eval_frame["quarter"].drop_duplicates())
    rows: list[pd.DataFrame] = []

    for quarter in quarters:
        test = eval_frame[eval_frame["quarter"] == quarter].copy()
        train = frame[frame["trade_date"] < str(test["trade_date"].min())].copy()
        train_sample = _sample_training_rows(train, top_pct, neg_ratio)
        if train_sample.empty or len(train_sample["label_rank"].dropna()) < 200:
            continue

        x_train = train_sample[LEGACY_3D_FEATURES].apply(pd.to_numeric, errors="coerce")
        medians = x_train.median(numeric_only=True).fillna(0.0)
        x_train = x_train.fillna(medians).to_numpy(dtype=np.float32, copy=True)
        y_train = _target(train_sample, top_pct)
        weights = np.where(y_train == 1, float(neg_ratio), 1.0).astype(np.float32)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.10,
            alpha=alpha,
            learning_rate="constant",
            eta0=eta0,
            max_iter=1,
            tol=None,
            average=True,
            random_state=42,
        )
        classes = np.array([0, 1], dtype=np.int8)
        for _ in range(max(1, epochs)):
            model.partial_fit(x_train, y_train, classes=classes, sample_weight=weights)

        x_test = test[LEGACY_3D_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians)
        x_test = scaler.transform(x_test.to_numpy(dtype=np.float32, copy=True))
        out = test[["trade_date", "stock_code", LABEL_COL]].copy()
        out["feature_tophit_proba"] = model.predict_proba(x_test)[:, 1]
        out["quarter_model_train_rows"] = int(len(train_sample))
        out["quarter_model_positive_rows"] = int(y_train.sum())
        rows.append(out)

    pred = pd.concat(rows, ignore_index=True)
    pred["feature_tophit_rank"] = pred.groupby("trade_date")["feature_tophit_proba"].rank(method="average", pct=True)
    return pred


def _metrics_for_score(frame: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = rb._daily_eval(frame, score_col, LABEL_COL)
    metrics = {
        "full": rb._metrics(daily),
        "train": rb._metrics(daily[daily["trade_date"] <= TRAIN_END]),
        "holdout": rb._metrics(daily[daily["trade_date"] > TRAIN_END]),
        "recent20": rb._metrics(daily.tail(20)),
        "recent63": rb._metrics(daily.tail(63)),
        "recent126": rb._metrics(daily.tail(126)),
        "annual_stability": rb._period_stability(daily, "year"),
        "monthly_stability": rb._period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
    }
    return daily, metrics


def _delta_block(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float]:
    return {
        key: float((candidate[block].get(key) or 0.0) - (baseline[block].get(key) or 0.0))
        for key in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    }


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = _delta_block(candidate, baseline, "full")
    holdout = _delta_block(candidate, baseline, "holdout")
    recent63 = _delta_block(candidate, baseline, "recent63")
    recent20 = _delta_block(candidate, baseline, "recent20")
    if candidate["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if full["rank_ic"] < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for key in ["top1", "top3", "top5", "top10", "top20"]:
        if full[key] <= 0.0:
            reasons.append(f"full_{key}_delta_non_positive")
    if holdout["top5"] <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if recent63["top5"] <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if recent20["top5"] <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    if (candidate["annual_stability"].get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    if (candidate["monthly_stability"].get("top5_positive_period_ratio") or 0.0) < 0.50:
        reasons.append("monthly_top5_positive_ratio_below_0_50")
    return not reasons, reasons


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features, route_info = _load_feature_label_frame()
    baseline_frame = _formal_3d_frame()

    configs = [
        {"top_pct": 0.03, "neg_ratio": 4, "alpha": 1e-5, "eta0": 0.005, "epochs": 3},
        {"top_pct": 0.05, "neg_ratio": 4, "alpha": 1e-5, "eta0": 0.005, "epochs": 3},
        {"top_pct": 0.05, "neg_ratio": 8, "alpha": 5e-5, "eta0": 0.003, "epochs": 3},
    ]
    blend_weights = [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
    blend_modes = ["direct", "inverse"]

    scan_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, Any]]] = []

    for config in configs:
        feature_pred = _fit_predict_quarterly(features, **config)
        merged = baseline_frame.merge(feature_pred, on=["trade_date", "stock_code", LABEL_COL], how="inner")
        merged["baseline_score"] = merged["pred_3d_rank"]
        _, baseline_metrics = _metrics_for_score(merged, "baseline_score")
        for mode in blend_modes:
            feature_rank = merged["feature_tophit_rank"] if mode == "direct" else 1.0 - merged["feature_tophit_rank"]
            for weight in blend_weights:
                merged["candidate_score"] = (1.0 - weight) * merged["pred_3d_rank"] + weight * feature_rank
                _, candidate_metrics = _metrics_for_score(merged, "candidate_score")
                deltas = {
                    "full": _delta_block(candidate_metrics, baseline_metrics, "full"),
                    "train": _delta_block(candidate_metrics, baseline_metrics, "train"),
                    "holdout": _delta_block(candidate_metrics, baseline_metrics, "holdout"),
                    "recent63": _delta_block(candidate_metrics, baseline_metrics, "recent63"),
                    "recent20": _delta_block(candidate_metrics, baseline_metrics, "recent20"),
                    "recent126": _delta_block(candidate_metrics, baseline_metrics, "recent126"),
                }
                passed, failed_reasons = _gate(candidate_metrics, baseline_metrics)
                selection_score = (
                    deltas["train"]["top5"]
                    + 0.5 * deltas["train"]["top10"]
                    + 0.25 * deltas["train"]["top3"]
                )
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
                            "baseline_current_formal_same_dates": baseline_metrics,
                            "candidate_metrics": candidate_metrics,
                            "deltas": deltas,
                            "gate": {"passed": passed, "failed_reasons": failed_reasons},
                            "selection_score_train_only": selection_score,
                        },
                    )
                )

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "quarterly_feature_tophit_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    passing = [payload for _, payload in candidates if payload["gate"]["passed"]]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_quarterly_feature_tophit",
        "method": "quarterly rolling feature-level SGDClassifier for 3D top-hit probability using active L3 features",
        "selection_rule": "Only train-window deltas through 20251231 select config/blend. Holdout/recent windows are validation only.",
        "route_info": route_info,
        "initial_train_days": INITIAL_TRAIN_DAYS,
        "selected_by_train": selected,
        "passing_count": len(passing),
        "best_passing_by_train_rank": passing[0] if passing else None,
        "scan_csv": str(scan_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else (
                "candidate_exists_but_not_train_selected" if passing else "no_3d_candidate_passed_quarterly_feature_tophit_scan"
            )
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
    report_json = REPORT_DIR / "quarterly_feature_tophit_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D quarterly feature top-hit 研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
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
            "## 边界",
            "",
            "- research-only。",
            "- 未修改 formal manifest。",
            "- 未修改 approved_for_l5。",
            "- 未生成交易信号，未跑策略回测。",
        ]
    )
    report_md = REPORT_DIR / "quarterly_feature_tophit_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
