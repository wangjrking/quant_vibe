from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_1d_rank_blend_repair_20260714"
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_1d_rank_blend_repair_20260714.duckdb"
)
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
START_DATE = "20220606"
LABEL = "executable_1d_open_return"
TARGET_TABLE = "stock_predict_data_model_agent_four_year_1d_rank_blend_repair_20260714_research"
TOP_K = [1, 3, 5, 10, 20]

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
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


def _read_scores() -> pd.DataFrame:
    manifests = {key: _load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        for key, item in manifests.items():
            con.execute(f"ATTACH '{Path(item['db_path']).as_posix()}' AS p{key.replace('d', '')} (READ_ONLY)")
        con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
        return con.execute(
            f"""
            SELECT
                p1.trade_date,
                p1.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d,
                labels.{LABEL} AS label_value
            FROM p1.{manifests['1d']['table']} p1
            JOIN p3.{manifests['3d']['table']} p3 USING (trade_date, stock_code)
            JOIN p5.{manifests['5d']['table']} p5 USING (trade_date, stock_code)
            JOIN p10.{manifests['10d']['table']} p10 USING (trade_date, stock_code)
            JOIN labels.{LABEL_TABLE} labels USING (trade_date, stock_code)
            WHERE p1.trade_date >= ?
              AND p1.stock_code NOT LIKE '%.BJ'
              AND labels.{LABEL} IS NOT NULL
              AND p1.pred_prob IS NOT NULL
              AND p3.pred_prob IS NOT NULL
              AND p5.pred_prob IS NOT NULL
              AND p10.pred_prob IS NOT NULL
            ORDER BY p1.trade_date, p1.stock_code
            """,
            [START_DATE],
        ).fetchdf()


def _read_full_scores() -> pd.DataFrame:
    manifests = {key: _load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        for key, item in manifests.items():
            con.execute(f"ATTACH '{Path(item['db_path']).as_posix()}' AS p{key.replace('d', '')} (READ_ONLY)")
        return con.execute(
            f"""
            SELECT
                p1.trade_date,
                p1.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d
            FROM p1.{manifests['1d']['table']} p1
            JOIN p3.{manifests['3d']['table']} p3 USING (trade_date, stock_code)
            JOIN p5.{manifests['5d']['table']} p5 USING (trade_date, stock_code)
            JOIN p10.{manifests['10d']['table']} p10 USING (trade_date, stock_code)
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


def _daily_eval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rows": int(len(ordered)),
            "rank_ic": _to_float(ordered[score_col].corr(ordered["label_value"], method="spearman")),
        }
        for k in TOP_K:
            row[f"top{k}"] = _to_float(ordered.head(k)["label_value"].mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    return daily


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "trade_days": int(len(daily)),
        "rank_ic": _to_float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": _to_float((daily["rank_ic"] > 0).mean()),
    }
    for k in TOP_K:
        out[f"top{k}"] = _to_float(daily[f"top{k}"].mean())
        out[f"top{k}_positive_ratio"] = _to_float((daily[f"top{k}"] > 0).mean())
    return out


def _period_stability(daily: pd.DataFrame, by: str) -> dict[str, Any]:
    grouped = daily.groupby(by, sort=True)[["rank_ic", *[f"top{k}" for k in TOP_K]]].mean().reset_index()
    out: dict[str, Any] = {"periods": int(len(grouped))}
    for col in ["rank_ic", *[f"top{k}" for k in TOP_K]]:
        out[f"{col}_positive_period_ratio"] = _to_float((grouped[col] > 0).mean())
        out[f"min_{col}"] = _to_float(grouped[col].min())
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


def _weight_grid() -> list[tuple[float, float, float, float]]:
    candidates: set[tuple[float, float, float, float]] = set()
    for w1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for w3 in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for w5 in [0.0, 0.1, 0.2, 0.3, 0.4]:
                w10 = round(1.0 - w1 - w3 - w5, 10)
                if w10 < 0.0 or w10 > 0.7:
                    continue
                candidates.add((w1, w3, w5, w10))
    candidates.add((1.0, 0.0, 0.0, 0.0))
    candidates.add((0.0, 0.0, 0.0, 1.0))
    return sorted(candidates)


def _evaluate_weights(base: pd.DataFrame, weights: tuple[float, float, float, float]) -> dict[str, Any]:
    w1, w3, w5, w10 = weights
    score_col = "candidate_score"
    frame = base.copy()
    frame[score_col] = (
        w1 * frame["pred_1d_rank"]
        + w3 * frame["pred_3d_rank"]
        + w5 * frame["pred_5d_rank"]
        + w10 * frame["pred_10d_rank"]
    )
    daily = _daily_eval(frame, score_col)
    full = _metrics(daily)
    recent63 = _metrics(daily.tail(63))
    recent20 = _metrics(daily.tail(20))
    row: dict[str, Any] = {
        "weights": {"w1": w1, "w3": w3, "w5": w5, "w10": w10},
        "eval_min_trade_date": str(daily["trade_date"].min()),
        "eval_max_trade_date": str(daily["trade_date"].max()),
        "eval_trade_days": int(len(daily)),
        "full": full,
        "recent63": recent63,
        "recent20": recent20,
        "monthly_stability": _period_stability(daily, "month"),
        "annual_stability": _period_stability(daily, "year"),
    }
    passed, failures = _gate(row)
    row["four_year_observation_gate"] = {"passed": passed, "failed_reasons": failures}
    row["objective"] = (
        (full.get("top5") or -1.0) * 1.0
        + (recent63.get("top5") or -1.0) * 0.6
        + (recent20.get("top5") or -1.0) * 0.6
        + (full.get("rank_ic") or -1.0) * 0.05
    )
    return row


def _materialize_candidate(full_base: pd.DataFrame, best: dict[str, Any]) -> dict[str, Any]:
    weights = best["weights"]
    out = full_base[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = (
        weights["w1"] * full_base["pred_1d_rank"]
        + weights["w3"] * full_base["pred_3d_rank"]
        + weights["w5"] * full_base["pred_5d_rank"]
        + weights["w10"] * full_base["pred_10d_rank"]
    )
    out["score_source"] = "rank_blend_repair_1d_research"
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_frame", out)
        con.execute(f"CREATE OR REPLACE TABLE {TARGET_TABLE} AS SELECT * FROM candidate_frame")
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
                        FROM {TARGET_TABLE}
                        GROUP BY 1,2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups
            FROM {TARGET_TABLE}
            """
        ).fetchdf().iloc[0].to_dict()
    return {key: _to_float(value) if hasattr(value, "item") else value for key, value in quality.items()}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = _add_rank_columns(_read_scores())
    full_base = _add_rank_columns(_read_full_scores())
    baseline = _evaluate_weights(base, (1.0, 0.0, 0.0, 0.0))
    rows = [_evaluate_weights(base, weights) for weights in _weight_grid()]
    rows = sorted(rows, key=lambda item: (bool(item["four_year_observation_gate"]["passed"]), item["objective"]), reverse=True)
    best = rows[0]
    quality = _materialize_candidate(full_base, best) if best["four_year_observation_gate"]["passed"] else None

    flat_rows: list[dict[str, Any]] = []
    for item in rows:
        flat_rows.append(
            {
                **item["weights"],
                "passed": item["four_year_observation_gate"]["passed"],
                "failed_reasons": ";".join(item["four_year_observation_gate"]["failed_reasons"]),
                "objective": item["objective"],
                "full_rank_ic": item["full"]["rank_ic"],
                "full_top1": item["full"]["top1"],
                "full_top3": item["full"]["top3"],
                "full_top5": item["full"]["top5"],
                "full_top10": item["full"]["top10"],
                "full_top20": item["full"]["top20"],
                "recent63_top5": item["recent63"]["top5"],
                "recent20_top5": item["recent20"]["top5"],
                "monthly_top5_positive_period_ratio": item["monthly_stability"].get("top5_positive_period_ratio"),
                "annual_top5_positive_period_ratio": item["annual_stability"].get("top5_positive_period_ratio"),
            }
        )
    scan_csv = REPORT_DIR / "rank_blend_repair_scan.csv"
    pd.DataFrame(flat_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "research_only_1d_rank_blend_repair",
        "label": LABEL,
        "observation_start": START_DATE,
        "baseline_current_formal": baseline,
        "best_candidate": best,
        "evaluation_asset_scope": {
            "min_trade_date": str(base["trade_date"].min()),
            "max_trade_date": str(base["trade_date"].max()),
            "rows": int(len(base)),
            "label_maturity_respected": True,
        },
        "full_coverage_asset_scope": {
            "min_trade_date": str(full_base["trade_date"].min()),
            "max_trade_date": str(full_base["trade_date"].max()),
            "rows": int(len(full_base)),
            "label_maturity_respected": False,
            "scope_note": "Full-coverage prediction asset includes dates after mature label cutoff; those rows are for scoring coverage only, not effect evaluation.",
        },
        "candidate_duckdb": str(EXPERIMENT_DB) if quality else None,
        "candidate_table": TARGET_TABLE if quality else None,
        "candidate_quality": quality,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "rank_blend_repair_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = REPORT_DIR / "rank_blend_repair_report.md"
    lines = [
        "# 1D 四年观察期 rank-blend 修复研究",
        "",
        "- 边界：research-only；未训练、未发布 formal、未生成信号、未回测。",
        f"- 候选表：`{EXPERIMENT_DB}::{TARGET_TABLE}`" if quality else "- 未生成候选表：没有组合通过四年观察门。",
        "",
        "## 当前 formal 基线",
        "",
        f"- Full RankIC：`{baseline['full']['rank_ic']:.6f}`",
        f"- Full Top5：`{baseline['full']['top5']:.6f}`",
        f"- Recent63 Top5：`{baseline['recent63']['top5']:.6f}`",
        f"- Recent20 Top5：`{baseline['recent20']['top5']:.6f}`",
        f"- 观察门：`{baseline['four_year_observation_gate']['passed']}`；失败原因：`{';'.join(baseline['four_year_observation_gate']['failed_reasons'])}`",
        "",
        "## 最优研究候选",
        "",
        f"- 权重：`{best['weights']}`",
        f"- Full RankIC：`{best['full']['rank_ic']:.6f}`",
        f"- Full Top1/3/5/10/20：`{best['full']['top1']:.6f}` / `{best['full']['top3']:.6f}` / `{best['full']['top5']:.6f}` / `{best['full']['top10']:.6f}` / `{best['full']['top20']:.6f}`",
        f"- Recent63 Top5：`{best['recent63']['top5']:.6f}`",
        f"- Recent20 Top5：`{best['recent20']['top5']:.6f}`",
        f"- 观察门：`{best['four_year_observation_gate']['passed']}`；失败原因：`{';'.join(best['four_year_observation_gate']['failed_reasons'])}`",
        "",
        "## 输出",
        "",
        f"- 扫描明细：`{scan_csv}`",
        f"- JSON 报告：`{report_json}`",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report_json": str(report_json), "scan_csv": str(scan_csv), "candidate_quality": quality}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
