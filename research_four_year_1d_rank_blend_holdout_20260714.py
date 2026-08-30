from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_year_1d_rank_blend_holdout_20260714"
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_1d_rank_blend_holdout_20260714.duckdb"
)
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
START_DATE = "20220606"
TRAIN_END_DATE = "20251231"
HOLDOUT_START_DATE = "20260101"
LABEL = "executable_1d_open_return"
TARGET_TABLE = "stock_predict_data_model_agent_four_year_1d_rank_blend_holdout_20260714_research"
TOP_K = [1, 3, 5, 10, 20]
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
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


def _read_scores(with_label: bool) -> pd.DataFrame:
    manifests = {key: _load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{Path(manifests['1d']['db_path']).as_posix()}' AS p1 (READ_ONLY)")
        con.execute(f"ATTACH '{Path(manifests['10d']['db_path']).as_posix()}' AS p10 (READ_ONLY)")
        if with_label:
            con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
            label_select = f", labels.{LABEL} AS label_value"
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
                p10.pred_prob AS pred_10d
                {label_select}
            FROM p1.{manifests['1d']['table']} p1
            JOIN p10.{manifests['10d']['table']} p10 USING (trade_date, stock_code)
            {label_join}
            WHERE p1.trade_date >= ?
              AND p1.stock_code NOT LIKE '%.BJ'
              AND p1.pred_prob IS NOT NULL
              AND p10.pred_prob IS NOT NULL
            ORDER BY p1.trade_date, p1.stock_code
            """,
            [START_DATE],
        ).fetchdf()


def _add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rank_1d"] = out.groupby("trade_date")["pred_1d"].rank(method="average", pct=True)
    out["rank_10d"] = out.groupby("trade_date")["pred_10d"].rank(method="average", pct=True)
    return out


def _daily_eval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group[group["label_value"].notna()]
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
    if daily.empty:
        raise ValueError("no daily eval rows")
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


def _period_delta(candidate_daily: pd.DataFrame, baseline_daily: pd.DataFrame, by: str) -> pd.DataFrame:
    cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    cand = candidate_daily.groupby(by, sort=True)[cols].mean().reset_index()
    base = baseline_daily.groupby(by, sort=True)[cols].mean().reset_index()
    merged = cand.merge(base, on=by, suffixes=("_candidate", "_baseline"))
    for col in cols:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    return merged


def _evaluate_weight(base: pd.DataFrame, w1: float) -> dict[str, Any]:
    frame = base[base["label_value"].notna()].copy()
    w10 = round(1.0 - w1, 10)
    frame["candidate_score"] = w1 * frame["rank_1d"] + w10 * frame["rank_10d"]
    daily = _daily_eval(frame, "candidate_score")
    train_daily = daily[daily["trade_date"] <= TRAIN_END_DATE]
    holdout_daily = daily[daily["trade_date"] >= HOLDOUT_START_DATE]
    row = {
        "weights": {"w1": round(w1, 2), "w10": round(w10, 2)},
        "eval_min_trade_date": str(daily["trade_date"].min()),
        "eval_max_trade_date": str(daily["trade_date"].max()),
        "eval_trade_days": int(len(daily)),
        "train": _metrics(train_daily),
        "holdout": _metrics(holdout_daily),
        "full": _metrics(daily),
        "recent63": _metrics(daily.tail(63)),
        "recent20": _metrics(daily.tail(20)),
    }
    row["train_objective"] = (
        (row["train"].get("top5") or -1.0)
        + 0.5 * (row["train"].get("top10") or -1.0)
        + 0.05 * (row["train"].get("rank_ic") or -1.0)
    )
    return row


def _quality(table: str) -> dict[str, Any]:
    with duckdb.connect(str(EXPERIMENT_DB), read_only=True) as con:
        latest = con.execute(f"SELECT max(trade_date) FROM {table}").fetchone()[0]
        row = con.execute(
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
            [latest, latest],
        ).fetchdf().iloc[0].to_dict()
    return {key: (_to_float(value) if hasattr(value, "item") else value) for key, value in row.items()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = _add_ranks(_read_scores(with_label=True))
    full_base = _add_ranks(_read_scores(with_label=False))

    baseline_frame = base[base["label_value"].notna()].copy()
    baseline_frame["baseline_score"] = baseline_frame["rank_1d"]
    baseline_daily = _daily_eval(baseline_frame, "baseline_score")
    baseline_train = baseline_daily[baseline_daily["trade_date"] <= TRAIN_END_DATE]
    baseline_holdout = baseline_daily[baseline_daily["trade_date"] >= HOLDOUT_START_DATE]

    rows = [_evaluate_weight(base, i / 20) for i in range(0, 21)]
    rows.sort(key=lambda item: item["train_objective"], reverse=True)
    best = rows[0]
    weights = best["weights"]

    materialized = full_base[["trade_date", "stock_code"]].copy()
    materialized["pred_prob"] = weights["w1"] * full_base["rank_1d"] + weights["w10"] * full_base["rank_10d"]
    materialized["score_source"] = "rank_blend_holdout_1d_10d_research"
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_frame", materialized)
        con.execute(f"CREATE OR REPLACE TABLE {TARGET_TABLE} AS SELECT * FROM candidate_frame")

    quality = _quality(TARGET_TABLE)
    file_sha256 = _sha256(EXPERIMENT_DB)

    candidate_frame = base[base["label_value"].notna()].copy()
    candidate_frame["candidate_score"] = weights["w1"] * candidate_frame["rank_1d"] + weights["w10"] * candidate_frame["rank_10d"]
    candidate_daily = _daily_eval(candidate_frame, "candidate_score")
    candidate_train = candidate_daily[candidate_daily["trade_date"] <= TRAIN_END_DATE]
    candidate_holdout = candidate_daily[candidate_daily["trade_date"] >= HOLDOUT_START_DATE]

    annual_delta = _period_delta(candidate_daily, baseline_daily, "year")
    monthly_delta = _period_delta(candidate_daily, baseline_daily, "month")
    scan_csv = REPORT_DIR / "holdout_weight_scan.csv"
    pd.DataFrame(
        [
            {
                "w1": item["weights"]["w1"],
                "w10": item["weights"]["w10"],
                "train_objective": item["train_objective"],
                "train_rank_ic": item["train"]["rank_ic"],
                "train_top5": item["train"]["top5"],
                "holdout_rank_ic": item["holdout"]["rank_ic"],
                "holdout_top5": item["holdout"]["top5"],
                "full_rank_ic": item["full"]["rank_ic"],
                "full_top5": item["full"]["top5"],
                "recent63_top5": item["recent63"]["top5"],
                "recent20_top5": item["recent20"]["top5"],
            }
            for item in rows
        ]
    ).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    annual_delta.to_csv(REPORT_DIR / "annual_delta.csv", index=False, encoding="utf-8-sig")
    monthly_delta.to_csv(REPORT_DIR / "monthly_delta.csv", index=False, encoding="utf-8-sig")

    def diff_metrics(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        out = {"rank_ic_delta": _to_float((candidate.get("rank_ic") or 0.0) - (baseline.get("rank_ic") or 0.0))}
        for k in TOP_K:
            out[f"top{k}_delta"] = _to_float((candidate.get(f"top{k}") or 0.0) - (baseline.get(f"top{k}") or 0.0))
        return out

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "research_only_1d_rank_blend_holdout_validation",
        "label": LABEL,
        "observation_start": START_DATE,
        "train_selection_window": {"min_trade_date": START_DATE, "max_trade_date": TRAIN_END_DATE},
        "holdout_validation_window": {"min_trade_date": HOLDOUT_START_DATE, "max_trade_date": str(candidate_holdout["trade_date"].max())},
        "selection_rule": "Pick best weight by train-only objective; holdout metrics are not used for weight selection.",
        "baseline_current_formal": {
            "train": _metrics(baseline_train),
            "holdout": _metrics(baseline_holdout),
            "full": _metrics(baseline_daily),
            "recent63": _metrics(baseline_daily.tail(63)),
            "recent20": _metrics(baseline_daily.tail(20)),
        },
        "best_candidate": {
            "weights": weights,
            "train": _metrics(candidate_train),
            "holdout": _metrics(candidate_holdout),
            "full": _metrics(candidate_daily),
            "recent63": _metrics(candidate_daily.tail(63)),
            "recent20": _metrics(candidate_daily.tail(20)),
        },
        "delta_vs_baseline": {
            "train": diff_metrics(_metrics(candidate_train), _metrics(baseline_train)),
            "holdout": diff_metrics(_metrics(candidate_holdout), _metrics(baseline_holdout)),
            "full": diff_metrics(_metrics(candidate_daily), _metrics(baseline_daily)),
            "recent63": diff_metrics(_metrics(candidate_daily.tail(63)), _metrics(baseline_daily.tail(63))),
            "recent20": diff_metrics(_metrics(candidate_daily.tail(20)), _metrics(baseline_daily.tail(20))),
        },
        "period_delta": {
            "annual_delta_csv": str(REPORT_DIR / "annual_delta.csv"),
            "monthly_delta_csv": str(REPORT_DIR / "monthly_delta.csv"),
            "positive_top5_years": int((annual_delta["top5_delta"] > 0).sum()),
            "min_year_top5_delta": _to_float(annual_delta["top5_delta"].min()),
            "min_year_rank_ic_delta": _to_float(annual_delta["rank_ic_delta"].min()),
        },
        "candidate_asset": {
            "db_path": str(EXPERIMENT_DB),
            "table": TARGET_TABLE,
            "quality": quality,
            "file_sha256": file_sha256,
            "score_formula": "w1 * cross_section_percent_rank(active_formal_1d.pred_prob) + w10 * cross_section_percent_rank(active_formal_10d.pred_prob)",
            "source_manifests": {key: str(path) for key, path in MANIFESTS.items()},
            "lineage_note": "The key domain and score formula both depend only on active formal 1D and 10D prediction tables.",
        },
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
    report_json = REPORT_DIR / "holdout_research_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = REPORT_DIR / "holdout_research_report.md"
    lines = [
        "# 1D rank-blend holdout 研究报告",
        "",
        "- 边界：research-only；未训练、未发布 formal、未生成信号、未回测。",
        f"- 选权窗口：`{START_DATE}` 到 `{TRAIN_END_DATE}`。",
        f"- 验证窗口：`{HOLDOUT_START_DATE}` 到 `{payload['holdout_validation_window']['max_trade_date']}`。",
        f"- 候选表：`{EXPERIMENT_DB}::{TARGET_TABLE}`。",
        f"- 文件 SHA256：`{file_sha256}`。",
        "",
        "## 最优权重",
        "",
        f"- 权重：`{weights}`。",
        f"- Full Top5 增量：`{payload['delta_vs_baseline']['full']['top5_delta']:.6f}`。",
        f"- Holdout Top5 增量：`{payload['delta_vs_baseline']['holdout']['top5_delta']:.6f}`。",
        f"- Recent63 Top5 增量：`{payload['delta_vs_baseline']['recent63']['top5_delta']:.6f}`。",
        f"- Recent20 Top5 增量：`{payload['delta_vs_baseline']['recent20']['top5_delta']:.6f}`。",
        "",
        "## 输出",
        "",
        f"- JSON 报告：`{report_json}`",
        f"- 扫描明细：`{scan_csv}`",
        f"- 年度增量：`{REPORT_DIR / 'annual_delta.csv'}`",
        f"- 月度增量：`{REPORT_DIR / 'monthly_delta.csv'}`",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report_json": str(report_json), "weights": weights, "quality": quality}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
