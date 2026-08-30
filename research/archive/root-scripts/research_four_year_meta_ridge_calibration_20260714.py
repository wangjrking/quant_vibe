from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_meta_ridge_calibration_20260714"
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_meta_ridge_calibration_20260714.duckdb"
)
MODEL_DIR = ROOT / "quant" / "data_file" / "experimental_assets" / "model-agent" / "model_params" / "meta_ridge_20260714"
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"

START_DATE = "20220606"
SELECTION_END = "20251231"
HOLDOUT_START = "20260101"
TOP_K = [1, 3, 5, 10, 20]

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}
LABELS = {
    "1d": "executable_1d_open_return",
    "3d": "executable_3d_open_return",
    "5d": "executable_5d_open_return",
    "10d": "executable_10d_open_return",
}


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


def _load_manifest(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(MAIN))
    from prediction_manifest import load_prediction_source_manifest

    return load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)


def _read_scores(with_labels: bool) -> pd.DataFrame:
    manifests = {key: _load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        for key in ["1d", "3d", "5d", "10d"]:
            con.execute(f"ATTACH '{Path(manifests[key]['db_path']).as_posix()}' AS p{key.replace('d','')} (READ_ONLY)")
        if with_labels:
            con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
            label_select = ",\n                " + ",\n                ".join(
                f"labels.{label} AS label_{key}" for key, label in LABELS.items()
            )
            label_join = f"JOIN labels.{LABEL_TABLE} labels USING (trade_date, stock_code)"
        else:
            label_select = ""
            label_join = ""
        return con.execute(
            f"""
            SELECT
                p1.trade_date,
                p1.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d
                {label_select}
            FROM p1.{manifests['1d']['table']} p1
            JOIN p3.{manifests['3d']['table']} p3 USING (trade_date, stock_code)
            JOIN p5.{manifests['5d']['table']} p5 USING (trade_date, stock_code)
            JOIN p10.{manifests['10d']['table']} p10 USING (trade_date, stock_code)
            {label_join}
            WHERE p1.trade_date >= ?
              AND p1.stock_code NOT LIKE '%.BJ'
              AND p1.pred_prob IS NOT NULL
              AND p3.pred_prob IS NOT NULL
              AND p5.pred_prob IS NOT NULL
              AND p10.pred_prob IS NOT NULL
            ORDER BY p1.trade_date, p1.stock_code
            """,
            [START_DATE],
        ).fetchdf()


def _add_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        out[f"{column}_rank"] = out.groupby("trade_date")[column].rank(method="average", pct=True)
    return out


def _feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    r1 = frame["pred_1d_rank"].to_numpy(dtype=np.float64)
    r3 = frame["pred_3d_rank"].to_numpy(dtype=np.float64)
    r5 = frame["pred_5d_rank"].to_numpy(dtype=np.float64)
    r10 = frame["pred_10d_rank"].to_numpy(dtype=np.float64)
    cols = {
        "bias": np.ones_like(r1),
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "r10": r10,
        "r1_r3_min": np.minimum(r1, r3),
        "r1_r5_min": np.minimum(r1, r5),
        "r1_r10_min": np.minimum(r1, r10),
        "r3_r5_min": np.minimum(r3, r5),
        "r5_r10_min": np.minimum(r5, r10),
        "r1_r10_prod": r1 * r10,
        "r3_r5_prod": r3 * r5,
        "r5_r10_prod": r5 * r10,
        "rank_mean": (r1 + r3 + r5 + r10) / 4.0,
        "rank_spread": np.maximum.reduce([r1, r3, r5, r10]) - np.minimum.reduce([r1, r3, r5, r10]),
    }
    names = list(cols)
    matrix = np.column_stack([cols[name] for name in names])
    return matrix, names


def _fit_ridge(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, l2: float) -> np.ndarray:
    sw = np.sqrt(sample_weight).reshape(-1, 1)
    xw = x * sw
    yw = y * sw.ravel()
    xtx = xw.T @ xw
    penalty = np.eye(xtx.shape[0]) * float(l2)
    penalty[0, 0] = 0.0
    xty = xw.T @ yw
    return np.linalg.solve(xtx + penalty, xty)


def _daily_eval(frame: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group[group[label_col].notna()]
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort")
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rows": int(len(ordered)),
            "rank_ic": _to_float(ordered[score_col].corr(ordered[label_col], method="spearman")),
        }
        for top_k in TOP_K:
            row[f"top{top_k}"] = _to_float(ordered.head(top_k)[label_col].mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    return daily


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {"trade_days": 0}
    result: dict[str, Any] = {
        "trade_days": int(len(daily)),
        "rank_ic": _to_float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": _to_float((daily["rank_ic"] > 0).mean()),
    }
    for top_k in TOP_K:
        result[f"top{top_k}"] = _to_float(daily[f"top{top_k}"].mean())
        result[f"top{top_k}_positive_ratio"] = _to_float((daily[f"top{top_k}"] > 0).mean())
    return result


def _period_stability(daily: pd.DataFrame, by: str) -> dict[str, Any]:
    metric_cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    grouped = daily.groupby(by, sort=True)[metric_cols].mean().reset_index() if not daily.empty else pd.DataFrame()
    out: dict[str, Any] = {"periods": int(len(grouped))}
    for column in metric_cols:
        out[f"{column}_positive_period_ratio"] = _to_float((grouped[column] > 0).mean()) if not grouped.empty else None
        out[f"min_{column}"] = _to_float(grouped[column].min()) if not grouped.empty else None
    return out


def _summary(daily: pd.DataFrame) -> dict[str, Any]:
    result = {
        "eval_min_trade_date": str(daily["trade_date"].min()) if not daily.empty else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if not daily.empty else None,
        "eval_trade_days": int(len(daily)),
        "full": _metrics(daily),
        "selection": _metrics(daily[daily["trade_date"] <= SELECTION_END]),
        "holdout": _metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": _metrics(daily.tail(20)),
        "recent63": _metrics(daily.tail(63)),
        "recent126": _metrics(daily.tail(126)),
        "annual_stability": _period_stability(daily, "year"),
        "monthly_stability": _period_stability(daily, "month"),
    }
    passed, reasons = _four_year_gate(result)
    result["four_year_gate"] = {"passed": passed, "failed_reasons": reasons}
    return result


def _four_year_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (summary["full"].get("rank_ic") or 0.0) <= 0.0:
        reasons.append("full_rank_ic_non_positive")
    if (summary["full"].get("top5") or 0.0) <= 0.0:
        reasons.append("full_top5_non_positive")
    if (summary["recent63"].get("top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_non_positive")
    if (summary["recent20"].get("top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_non_positive")
    if (summary["monthly_stability"].get("top5_positive_period_ratio") or 0.0) < 0.5:
        reasons.append("monthly_top5_positive_ratio_below_0_5")
    if (summary["annual_stability"].get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    return not reasons, reasons


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for window in ["full", "selection", "holdout", "recent63", "recent20"]:
        for field in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
            result[f"{window}_{field}"] = _to_float(
                (candidate[window].get(field) or 0.0) - (baseline[window].get(field) or 0.0)
            )
    return result


def _improvement_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    delta = _delta(candidate, baseline)
    if not candidate["four_year_gate"]["passed"]:
        reasons.append("candidate_four_year_gate_failed")
    if (delta.get("full_rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for field in ["full_top5", "selection_top5", "holdout_top5", "recent63_top5", "recent20_top5"]:
        if (delta.get(field) or 0.0) <= 0.0:
            reasons.append(f"{field}_delta_non_positive")
    return not reasons, reasons


def _objective(summary: dict[str, Any]) -> float:
    selection = summary["selection"]
    return (
        (selection.get("top5") or -1.0)
        + 0.4 * (selection.get("top3") or -1.0)
        + 0.2 * (selection.get("top10") or -1.0)
        + 0.03 * (selection.get("rank_ic") or -1.0)
    )


def _train_configs(label_key: str) -> list[dict[str, Any]]:
    return [
        {"l2": l2, "top_weight": tw, "target_rank_weight": trw}
        for l2 in [0.1, 1.0, 10.0, 100.0]
        for tw in [0.0, 5.0, 20.0]
        for trw in [0.8, 0.9]
    ]


def _materialize(full_base: pd.DataFrame, score: np.ndarray, label_key: str, model_record: dict[str, Any]) -> dict[str, Any]:
    table = f"stock_predict_data_model_agent_four_year_meta_ridge_20260714_executable_{label_key}_open_return_research"
    out = full_base[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = pd.Series(score).groupby(full_base["trade_date"]).rank(method="average", pct=True).to_numpy()
    out["score_source"] = "meta_ridge_calibration_20260714_research"
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"meta_ridge_{label_key}_20260714.json"
    model_path.write_text(json.dumps(model_record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "db_path": str(EXPERIMENT_DB),
        "table": table,
        "quality": {key: _to_float(value) for key, value in quality.items()},
        "pred_prob_sha256": digest.hexdigest(),
        "model_params_path": str(model_path),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    label_env = os.environ.get("MODEL_RESEARCH_LABEL_KEYS")
    label_keys = [item.strip() for item in label_env.split(",") if item.strip()] if label_env else ["1d", "3d", "5d", "10d"]
    invalid = sorted(set(label_keys) - set(LABELS))
    if invalid:
        raise ValueError(f"unsupported label keys: {invalid}")

    base = _add_rank_columns(_read_scores(with_labels=True))
    full_base = _add_rank_columns(_read_scores(with_labels=False))
    x_all, feature_names = _feature_matrix(base)
    x_full, _ = _feature_matrix(full_base)

    scan_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for label_key in label_keys:
        label_col = f"label_{label_key}"
        baseline_frame = base[base[label_col].notna()].copy()
        baseline_frame["candidate_score"] = baseline_frame[f"pred_{label_key}_rank"]
        baseline = _summary(_daily_eval(baseline_frame, "candidate_score", label_col))
        train_mask = (base["trade_date"] <= SELECTION_END) & base[label_col].notna()
        y = base.loc[train_mask, label_col].to_numpy(dtype=np.float64)
        x_train = x_all[train_mask.to_numpy()]
        target_rank = base.loc[train_mask, f"pred_{label_key}_rank"].to_numpy(dtype=np.float64)

        candidates: list[dict[str, Any]] = []
        for config in _train_configs(label_key):
            weight = np.ones_like(y)
            weight += float(config["top_weight"]) * (target_rank >= float(config["target_rank_weight"]))
            coef = _fit_ridge(x_train, y, weight, float(config["l2"]))
            score = x_all @ coef
            eval_frame = base[base[label_col].notna()].copy()
            eval_frame["candidate_score"] = score[base[label_col].notna().to_numpy()]
            summary = _summary(_daily_eval(eval_frame, "candidate_score", label_col))
            summary["config"] = config
            summary["coefficients"] = {name: _to_float(value) for name, value in zip(feature_names, coef)}
            summary["objective"] = _objective(summary)
            passed, failures = _improvement_gate(summary, baseline)
            summary["improvement_gate"] = {"passed": passed, "failed_reasons": failures}
            summary["delta_vs_baseline"] = _delta(summary, baseline)
            candidates.append(summary)
            scan_rows.append(
                {
                    "label_key": label_key,
                    **config,
                    "objective": summary["objective"],
                    "four_year_gate_passed": summary["four_year_gate"]["passed"],
                    "improvement_gate_passed": passed,
                    "improvement_gate_failures": ";".join(failures),
                    **{f"delta_{key}": value for key, value in summary["delta_vs_baseline"].items()},
                    "full_rank_ic": summary["full"].get("rank_ic"),
                    "full_top5": summary["full"].get("top5"),
                    "holdout_top5": summary["holdout"].get("top5"),
                    "recent63_top5": summary["recent63"].get("top5"),
                    "recent20_top5": summary["recent20"].get("top5"),
                }
            )

        candidates.sort(key=lambda item: (item["improvement_gate"]["passed"], item["objective"]), reverse=True)
        selected = candidates[0]
        materialized = None
        if selected["improvement_gate"]["passed"]:
            full_score = x_full @ np.array([selected["coefficients"][name] for name in feature_names], dtype=np.float64)
            materialized = _materialize(
                full_base,
                full_score,
                label_key,
                {
                    "label_key": label_key,
                    "label": LABELS[label_key],
                    "model_type": "weighted_ridge_meta_calibrator",
                    "selection_window": {"min_trade_date": START_DATE, "max_trade_date": SELECTION_END},
                    "holdout_window": {"min_trade_date": HOLDOUT_START, "max_trade_date": "label_maturity_by_horizon"},
                    "feature_names": feature_names,
                    "config": selected["config"],
                    "coefficients": selected["coefficients"],
                    "baseline_manifest_inputs": {k: str(v) for k, v in MANIFESTS.items()},
                    "boundaries": {"research_only": True, "not_for_l4_formal_or_l5": True},
                },
            )

        reports[label_key] = {
            "label": LABELS[label_key],
            "baseline": baseline,
            "selected_candidate": selected,
            "eligible_candidate_count": sum(1 for item in candidates if item["improvement_gate"]["passed"]),
            "materialized_candidate": materialized,
        }

    suffix = "_".join(label_keys)
    scan_csv = REPORT_DIR / f"meta_ridge_{suffix}_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "research_only_four_year_meta_ridge_calibration",
        "selection_window": {"min_trade_date": START_DATE, "max_trade_date": SELECTION_END},
        "holdout_window": {"min_trade_date": HOLDOUT_START, "max_trade_date": "label_maturity_by_horizon"},
        "candidate_db": str(EXPERIMENT_DB),
        "model_param_dir": str(MODEL_DIR),
        "labels": reports,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
            "label_maturity_respected_for_evaluation": True,
        },
    }
    report_json = REPORT_DIR / f"meta_ridge_{suffix}_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 四年观察期二层 Ridge 校准研究",
        "",
        "## 边界",
        "",
        "- research-only。",
        "- 只使用当前 active formal L4 分数和成熟标签做二层校准研究。",
        "- 未修改 formal manifest，未生成交易信号，未运行策略回测。",
        "",
        "## 结果",
        "",
        "| 标签 | 合格候选数 | Full RankIC 增量 | Full Top5 增量 | Holdout Top5 增量 | Recent63 Top5 增量 | Recent20 Top5 增量 | 是否落实验表 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label_key, item in reports.items():
        selected = item["selected_candidate"]
        delta = selected["delta_vs_baseline"]
        lines.append(
            f"| {label_key} | {item['eligible_candidate_count']} | {delta['full_rank_ic']:.6f} | {delta['full_top5']:.6f} | {delta['holdout_top5']:.6f} | {delta['recent63_top5']:.6f} | {delta['recent20_top5']:.6f} | {'是' if item['materialized_candidate'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- JSON：`{report_json}`",
            f"- 扫描明细：`{scan_csv}`",
            f"- 参数目录：`{MODEL_DIR}`",
            f"- 实验库：`{EXPERIMENT_DB}`",
        ]
    )
    report_md = REPORT_DIR / f"meta_ridge_{suffix}_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report_json": str(report_json), "scan_csv": str(scan_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
