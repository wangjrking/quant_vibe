from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_rank_blend_all_labels_20260714"
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_rank_blend_all_labels_20260714.duckdb"
)
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
START_DATE = "20220606"
TOP_K = [1, 3, 5, 10, 20]
RECENT_WINDOWS = [20, 63, 126]

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
BASELINE_WEIGHTS = {
    "1d": (1.0, 0.0, 0.0, 0.0),
    "3d": (0.0, 1.0, 0.0, 0.0),
    "5d": (0.0, 0.0, 1.0, 0.0),
    "10d": (0.0, 0.0, 0.0, 1.0),
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _load_manifest(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(MAIN))
    from prediction_manifest import load_prediction_source_manifest

    return load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)


def _read_scores(with_labels: bool) -> pd.DataFrame:
    manifests = {key: _load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        for key, item in manifests.items():
            con.execute(f"ATTACH '{Path(item['db_path']).as_posix()}' AS p{key.replace('d', '')} (READ_ONLY)")
        if with_labels:
            con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
            label_cols = ",\n                ".join(
                f"labels.{label} AS label_{key}" for key, label in LABELS.items()
            )
            label_join = f"JOIN labels.{LABEL_TABLE} labels USING (trade_date, stock_code)"
        else:
            label_cols = ""
            label_join = ""
        select_labels = f",\n                {label_cols}" if label_cols else ""
        return con.execute(
            f"""
            SELECT
                p1.trade_date,
                p1.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d
                {select_labels}
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
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        out[f"{col}_rank"] = out.groupby("trade_date")[col].rank(method="average", pct=True)
    return out


def _daily_eval(frame: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group[group[label_col].notna()]
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rows": int(len(ordered)),
            "rank_ic": _to_float(ordered[score_col].corr(ordered[label_col], method="spearman")),
        }
        for k in TOP_K:
            row[f"top{k}"] = _to_float(ordered.head(k)[label_col].mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    return daily


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "trade_days": int(len(daily)),
        "rank_ic": _to_float(daily["rank_ic"].mean()) if len(daily) else None,
        "rank_ic_positive_ratio": _to_float((daily["rank_ic"] > 0).mean()) if len(daily) else None,
    }
    for k in TOP_K:
        out[f"top{k}"] = _to_float(daily[f"top{k}"].mean()) if len(daily) else None
        out[f"top{k}_positive_ratio"] = _to_float((daily[f"top{k}"] > 0).mean()) if len(daily) else None
    return out


def _period_stability(daily: pd.DataFrame, by: str) -> dict[str, Any]:
    metric_cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    stable_frame = daily[[by, *metric_cols]].copy()
    for col in metric_cols:
        stable_frame[col] = pd.to_numeric(stable_frame[col], errors="coerce")
    grouped = stable_frame.groupby(by, sort=True)[metric_cols].mean().reset_index()
    out: dict[str, Any] = {"periods": int(len(grouped))}
    for col in metric_cols:
        out[f"{col}_positive_period_ratio"] = _to_float((grouped[col] > 0).mean()) if len(grouped) else None
        out[f"min_{col}"] = _to_float(grouped[col].min()) if len(grouped) else None
        worst = grouped.sort_values(col, ascending=True).head(1)
        out[f"worst_{col}_{by}"] = str(worst.iloc[0][by]) if not worst.empty else None
    return out


def _gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = row["full"]
    r63 = row["recent63"]
    r20 = row["recent20"]
    if row["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (full.get("rank_ic") or 0.0) <= 0.0:
        reasons.append("full_rank_ic_non_positive")
    if (full.get("top5") or 0.0) <= 0.0:
        reasons.append("full_top5_non_positive")
    if (r63.get("top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_non_positive")
    if (r20.get("top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_non_positive")
    if (row["monthly_stability"].get("top5_positive_period_ratio") or 0.0) < 0.5:
        reasons.append("monthly_top5_positive_ratio_below_0_5")
    if (row["annual_stability"].get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    return not reasons, reasons


def _weight_grid(label_key: str) -> list[tuple[float, float, float, float]]:
    candidates: set[tuple[float, float, float, float]] = set()
    keys = ["1d", "3d", "5d", "10d"]
    base_index = keys.index(label_key)
    steps = [i / 10 for i in range(0, 11)]

    # Keep the scan interpretable and fast: target horizon blended with one
    # auxiliary horizon, plus a small number of balanced multi-horizon anchors.
    candidates.add(BASELINE_WEIGHTS[label_key])
    for aux_key in keys:
        if aux_key == label_key:
            continue
        aux_index = keys.index(aux_key)
        for target_weight in steps:
            weights = [0.0, 0.0, 0.0, 0.0]
            weights[base_index] = round(target_weight, 1)
            weights[aux_index] = round(1.0 - target_weight, 1)
            candidates.add(tuple(weights))  # type: ignore[arg-type]

    anchors = [
        (0.25, 0.25, 0.25, 0.25),
        (0.4, 0.2, 0.2, 0.2),
        (0.2, 0.4, 0.2, 0.2),
        (0.2, 0.2, 0.4, 0.2),
        (0.2, 0.2, 0.2, 0.4),
        (0.1, 0.3, 0.3, 0.3),
        (0.3, 0.1, 0.3, 0.3),
        (0.3, 0.3, 0.1, 0.3),
        (0.3, 0.3, 0.3, 0.1),
    ]
    candidates.update(anchors)
    return sorted(candidates)


def _evaluate_weights(base: pd.DataFrame, label_key: str, weights: tuple[float, float, float, float]) -> dict[str, Any]:
    w1, w3, w5, w10 = weights
    label_col = f"label_{label_key}"
    frame = base[base[label_col].notna()].copy()
    frame["candidate_score"] = (
        w1 * frame["pred_1d_rank"]
        + w3 * frame["pred_3d_rank"]
        + w5 * frame["pred_5d_rank"]
        + w10 * frame["pred_10d_rank"]
    )
    daily = _daily_eval(frame, "candidate_score", label_col)
    full = _metrics(daily)
    row: dict[str, Any] = {
        "label_key": label_key,
        "label": LABELS[label_key],
        "weights": {"w1": w1, "w3": w3, "w5": w5, "w10": w10},
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
        "full": full,
        "recent20": _metrics(daily.tail(20)),
        "recent63": _metrics(daily.tail(63)),
        "recent126": _metrics(daily.tail(126)),
        "monthly_stability": _period_stability(daily, "month"),
        "annual_stability": _period_stability(daily, "year"),
    }
    passed, failures = _gate(row)
    row["four_year_observation_gate"] = {"passed": passed, "failed_reasons": failures}
    row["objective"] = (
        (full.get("top5") or -1.0)
        + 0.6 * (row["recent63"].get("top5") or -1.0)
        + 0.6 * (row["recent20"].get("top5") or -1.0)
        + 0.05 * (full.get("rank_ic") or -1.0)
    )
    return row


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_rank_ic": _to_float((candidate["full"].get("rank_ic") or 0.0) - (baseline["full"].get("rank_ic") or 0.0)),
        "full_top1": _to_float((candidate["full"].get("top1") or 0.0) - (baseline["full"].get("top1") or 0.0)),
        "full_top3": _to_float((candidate["full"].get("top3") or 0.0) - (baseline["full"].get("top3") or 0.0)),
        "full_top5": _to_float((candidate["full"].get("top5") or 0.0) - (baseline["full"].get("top5") or 0.0)),
        "full_top10": _to_float((candidate["full"].get("top10") or 0.0) - (baseline["full"].get("top10") or 0.0)),
        "full_top20": _to_float((candidate["full"].get("top20") or 0.0) - (baseline["full"].get("top20") or 0.0)),
        "recent63_top5": _to_float((candidate["recent63"].get("top5") or 0.0) - (baseline["recent63"].get("top5") or 0.0)),
        "recent20_top5": _to_float((candidate["recent20"].get("top5") or 0.0) - (baseline["recent20"].get("top5") or 0.0)),
    }


def _improvement_passed(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    delta = _delta(candidate, baseline)
    if not candidate["four_year_observation_gate"]["passed"]:
        reasons.append("candidate_four_year_gate_failed")
    if (delta.get("full_rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    if (delta.get("full_top5") or 0.0) <= 0.0:
        reasons.append("full_top5_delta_non_positive")
    if (delta.get("recent63_top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if (delta.get("recent20_top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    return not reasons, reasons


def _flatten(label_key: str, item: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    delta = _delta(item, baseline)
    improvement_passed, improvement_failures = _improvement_passed(item, baseline)
    return {
        "label_key": label_key,
        **item["weights"],
        "gate_passed": item["four_year_observation_gate"]["passed"],
        "gate_failures": ";".join(item["four_year_observation_gate"]["failed_reasons"]),
        "improvement_passed": improvement_passed,
        "improvement_failures": ";".join(improvement_failures),
        "objective": item["objective"],
        "full_rank_ic": item["full"]["rank_ic"],
        "full_top1": item["full"]["top1"],
        "full_top3": item["full"]["top3"],
        "full_top5": item["full"]["top5"],
        "full_top10": item["full"]["top10"],
        "full_top20": item["full"]["top20"],
        "recent63_top5": item["recent63"]["top5"],
        "recent20_top5": item["recent20"]["top5"],
        **{f"delta_{key}": value for key, value in delta.items()},
    }


def _materialize_candidate(full_base: pd.DataFrame, label_key: str, best: dict[str, Any]) -> dict[str, Any]:
    weights = best["weights"]
    table = f"stock_predict_data_model_agent_four_year_rank_blend_20260714_executable_{label_key}_open_return_research"
    out = full_base[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = (
        weights["w1"] * full_base["pred_1d_rank"]
        + weights["w3"] * full_base["pred_3d_rank"]
        + weights["w5"] * full_base["pred_5d_rank"]
        + weights["w10"] * full_base["pred_10d_rank"]
    )
    out["score_source"] = "rank_blend_all_labels_20260714_research"
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
    return {
        "db_path": str(EXPERIMENT_DB),
        "table": table,
        "quality": {key: (_to_float(value) if hasattr(value, "item") else value) for key, value in quality.items()},
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    requested_labels = os.environ.get("MODEL_RESEARCH_LABEL_KEYS")
    label_keys = (
        [item.strip() for item in requested_labels.split(",") if item.strip()]
        if requested_labels
        else list(LABELS.keys())
    )
    invalid_labels = sorted(set(label_keys) - set(LABELS))
    if invalid_labels:
        raise ValueError(f"unsupported label keys: {invalid_labels}")
    suffix = "_".join(label_keys) if requested_labels else "all"
    base = _add_rank_columns(_read_scores(with_labels=True))
    full_base = _add_rank_columns(_read_scores(with_labels=False))
    summaries: dict[str, Any] = {}
    scan_rows: list[dict[str, Any]] = []
    for label_key in label_keys:
        baseline = _evaluate_weights(base, label_key, BASELINE_WEIGHTS[label_key])
        rows = [_evaluate_weights(base, label_key, weights) for weights in _weight_grid(label_key)]
        rows.sort(
            key=lambda item: (
                bool(_improvement_passed(item, baseline)[0]),
                bool(item["four_year_observation_gate"]["passed"]),
                item["objective"],
            ),
            reverse=True,
        )
        best = rows[0]
        improvement_passed, improvement_failures = _improvement_passed(best, baseline)
        materialized = _materialize_candidate(full_base, label_key, best) if improvement_passed else None
        for item in rows:
            scan_rows.append(_flatten(label_key, item, baseline))
        summaries[label_key] = {
            "label": LABELS[label_key],
            "baseline_current_formal": baseline,
            "best_candidate": best,
            "delta_vs_baseline": _delta(best, baseline),
            "improvement_gate": {
                "passed": improvement_passed,
                "failed_reasons": improvement_failures,
            },
            "materialized_candidate": materialized,
        }

    scan_csv = REPORT_DIR / f"rank_blend_{suffix}_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "research_only_four_year_rank_blend_all_labels",
        "observation_start": START_DATE,
        "candidate_db": str(EXPERIMENT_DB),
        "labels": summaries,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
            "label_maturity_respected_for_evaluation": True,
        },
    }
    report_json = REPORT_DIR / f"rank_blend_{suffix}_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = REPORT_DIR / f"rank_blend_{suffix}_report.md"
    lines = [
        "# 四年观察期 rank-blend 全标签研究",
        "",
        "- 边界：research-only；未训练、未发布 formal、未生成信号、未回测。",
        f"- 扫描明细：`{scan_csv}`",
        f"- 实验库：`{EXPERIMENT_DB}`",
        "",
        "| 标签 | 是否生成研究候选 | 权重 | Full RankIC Δ | Full Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for label_key, item in summaries.items():
        delta = item["delta_vs_baseline"]
        mat = item["materialized_candidate"]
        lines.append(
            "| {label} | {yes} | `{weights}` | {rank:.6f} | {top5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key,
                yes="是" if mat else "否",
                weights=item["best_candidate"]["weights"],
                rank=delta["full_rank_ic"] or 0.0,
                top5=delta["full_top5"] or 0.0,
                r63=delta["recent63_top5"] or 0.0,
                r20=delta["recent20_top5"] or 0.0,
            )
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report_json": str(report_json), "scan_csv": str(scan_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
