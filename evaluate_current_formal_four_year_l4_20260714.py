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
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "model_agent_current_formal_four_year_eval_20260714"
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


def _mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _to_float(series.mean())


def _load_manifest(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(MAIN))
    from prediction_manifest import load_prediction_source_manifest

    return load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)


def _read_joined(manifest: dict[str, Any], label: str) -> pd.DataFrame:
    db_path = Path(manifest["db_path"])
    table = str(manifest["table"])
    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{db_path.as_posix()}' AS p (READ_ONLY)")
        con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS l (READ_ONLY)")
        return con.execute(
            f"""
            SELECT
                p.trade_date,
                p.stock_code,
                p.pred_prob,
                l.{label} AS label_value
            FROM p.{table} p
            JOIN l.{LABEL_TABLE} l
              ON p.trade_date = l.trade_date
             AND p.stock_code = l.stock_code
            WHERE p.trade_date >= ?
              AND l.{label} IS NOT NULL
              AND p.pred_prob IS NOT NULL
            ORDER BY p.trade_date, p.stock_code
            """,
            [START_DATE],
        ).fetchdf()


def _daily_eval(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values("pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rows": int(len(ordered)),
            "rank_ic": _to_float(ordered["pred_prob"].corr(ordered["label_value"], method="spearman")),
            "pearson_ic": _to_float(ordered["pred_prob"].corr(ordered["label_value"], method="pearson")),
        }
        for k in TOP_K:
            row[f"top{k}"] = _to_float(ordered.head(k)["label_value"].mean())
        rows.append(row)
    if not rows:
        raise ValueError("no valid daily evaluation rows")
    out = pd.DataFrame(rows)
    out["year"] = out["trade_date"].str.slice(0, 4)
    out["month"] = out["trade_date"].str.slice(0, 6)
    return out


def _window_metrics(daily: pd.DataFrame, name: str, subset: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "window": name,
        "trade_days": int(len(subset)),
        "rank_ic": _mean(subset["rank_ic"]),
        "rank_ic_positive_ratio": _to_float((subset["rank_ic"] > 0).mean()) if len(subset) else None,
    }
    for k in TOP_K:
        metrics[f"top{k}"] = _mean(subset[f"top{k}"])
        metrics[f"top{k}_positive_ratio"] = _to_float((subset[f"top{k}"] > 0).mean()) if len(subset) else None
    return metrics


def _aggregate_period(daily: pd.DataFrame, by: str) -> pd.DataFrame:
    metric_cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    rows: list[dict[str, Any]] = []
    for key, group in daily.groupby(by, sort=True):
        row: dict[str, Any] = {by: str(key), "trade_days": int(len(group))}
        for col in metric_cols:
            row[col] = _mean(group[col])
            row[f"{col}_positive_ratio"] = _to_float((group[col] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _stability(period: pd.DataFrame, period_key: str) -> dict[str, Any]:
    if period.empty:
        return {"period_key": period_key, "periods": 0}
    result: dict[str, Any] = {
        "period_key": period_key,
        "periods": int(len(period)),
        "rank_ic_positive_period_ratio": _to_float((period["rank_ic"] > 0).mean()),
        "min_rank_ic": _to_float(period["rank_ic"].min()),
    }
    for k in TOP_K:
        col = f"top{k}"
        result[f"{col}_positive_period_ratio"] = _to_float((period[col] > 0).mean())
        result[f"min_{col}"] = _to_float(period[col].min())
        worst = period.sort_values(col, ascending=True).head(1)
        result[f"worst_{col}_period"] = str(worst.iloc[0][period_key]) if not worst.empty else None
    return result


def _pass_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(summary["eval_min_trade_date"]) > "20220630":
        reasons.append("eval_min_trade_date_after_20220630")
    if int(summary["eval_trade_days"]) < 950:
        reasons.append("eval_trade_days_below_950")
    full = summary["windows"]["full"]
    recent63 = summary["windows"]["recent63"]
    recent20 = summary["windows"]["recent20"]
    if (full.get("rank_ic") or 0.0) <= 0.0:
        reasons.append("full_rank_ic_non_positive")
    if (full.get("top5") or 0.0) <= 0.0:
        reasons.append("full_top5_non_positive")
    if (recent63.get("top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_non_positive")
    if (recent20.get("top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_non_positive")
    monthly = summary["monthly_stability"]
    if (monthly.get("top5_positive_period_ratio") or 0.0) < 0.5:
        reasons.append("monthly_top5_positive_ratio_below_0_5")
    yearly = summary["annual_stability"]
    if (yearly.get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    return not reasons, reasons


def evaluate_one(label_key: str, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    label = str(manifest["manifest"].get("label") or "")
    if not label:
        raise ValueError(f"manifest missing label: {manifest_path}")
    frame = _read_joined(manifest, label)
    daily = _daily_eval(frame)
    annual = _aggregate_period(daily, "year")
    monthly = _aggregate_period(daily, "month")

    windows = {"full": _window_metrics(daily, "full", daily)}
    for n in RECENT_WINDOWS:
        windows[f"recent{n}"] = _window_metrics(daily, f"recent{n}", daily.tail(n))

    summary: dict[str, Any] = {
        "label_key": label_key,
        "label": label,
        "manifest_path": str(manifest_path),
        "source_type": manifest["source_type"],
        "approval_status": manifest["approval_status"],
        "db_path": str(manifest["db_path"]),
        "table": manifest["table"],
        "eval_min_trade_date": str(daily["trade_date"].min()),
        "eval_max_trade_date": str(daily["trade_date"].max()),
        "eval_trade_days": int(daily["trade_date"].nunique()),
        "eval_rows": int(len(frame)),
        "windows": windows,
        "annual_stability": _stability(annual, "year"),
        "monthly_stability": _stability(monthly, "month"),
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_signal": True,
            "no_backtest": True,
            "label_maturity_respected": True,
        },
    }
    gate_passed, gate_failures = _pass_gate(summary)
    summary["four_year_observation_gate"] = {
        "passed": gate_passed,
        "failed_reasons": gate_failures,
        "gate_note": "This gate is a model-side discussion gate only; it does not publish or switch production assets.",
    }

    prefix = f"{label}_current_formal_four_year"
    daily.to_csv(REPORT_DIR / f"{prefix}_daily_eval.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(REPORT_DIR / f"{prefix}_annual_stability.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(REPORT_DIR / f"{prefix}_monthly_stability.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / f"{prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {label_key: evaluate_one(label_key, path) for label_key, path in MANIFESTS.items()}
    rows: list[dict[str, Any]] = []
    for label_key, item in summaries.items():
        row = {
            "label_key": label_key,
            "label": item["label"],
            "eval_min_trade_date": item["eval_min_trade_date"],
            "eval_max_trade_date": item["eval_max_trade_date"],
            "eval_trade_days": item["eval_trade_days"],
            "four_year_observation_passed": item["four_year_observation_gate"]["passed"],
            "failed_reasons": ";".join(item["four_year_observation_gate"]["failed_reasons"]),
            "full_rank_ic": item["windows"]["full"]["rank_ic"],
            "full_top1": item["windows"]["full"]["top1"],
            "full_top3": item["windows"]["full"]["top3"],
            "full_top5": item["windows"]["full"]["top5"],
            "full_top10": item["windows"]["full"]["top10"],
            "full_top20": item["windows"]["full"]["top20"],
            "recent63_rank_ic": item["windows"]["recent63"]["rank_ic"],
            "recent63_top5": item["windows"]["recent63"]["top5"],
            "recent20_rank_ic": item["windows"]["recent20"]["rank_ic"],
            "recent20_top5": item["windows"]["recent20"]["top5"],
            "monthly_top5_positive_period_ratio": item["monthly_stability"].get("top5_positive_period_ratio"),
            "annual_top5_positive_period_ratio": item["annual_stability"].get("top5_positive_period_ratio"),
        }
        rows.append(row)
    summary_csv = REPORT_DIR / "current_formal_four_year_eval_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "current_active_formal_l4_four_year_model_eval",
        "observation_start": START_DATE,
        "label_db": str(LABEL_DB),
        "label_table": LABEL_TABLE,
        "labels": summaries,
        "summary_csv": str(summary_csv),
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_signal": True,
            "no_backtest": True,
            "not_a_production_publish": True,
        },
    }
    summary_json = REPORT_DIR / "current_formal_four_year_eval_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 当前 L4 formal 四年观察期模型评价",
        "",
        f"- 生成时间：`{payload['generated_at']}`",
        f"- 观察起点：`{START_DATE}`",
        "- 边界：只做模型侧评价；未训练、未预测、未生成信号、未回测、未切换生产资产。",
        "",
        "## 摘要",
        "",
        "| 标签 | 可评价区间 | 交易日 | RankIC | Top1 | Top3 | Top5 | Top10 | Top20 | Recent63 Top5 | Recent20 Top5 | 四年观察门 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label_key} | {date_range} | {days} | {rank_ic:.6f} | {top1:.6f} | {top3:.6f} | {top5:.6f} | {top10:.6f} | {top20:.6f} | {r63:.6f} | {r20:.6f} | {gate} |".format(
                label_key=row["label_key"],
                date_range=f"{row['eval_min_trade_date']}-{row['eval_max_trade_date']}",
                days=row["eval_trade_days"],
                rank_ic=row["full_rank_ic"] or 0.0,
                top1=row["full_top1"] or 0.0,
                top3=row["full_top3"] or 0.0,
                top5=row["full_top5"] or 0.0,
                top10=row["full_top10"] or 0.0,
                top20=row["full_top20"] or 0.0,
                r63=row["recent63_top5"] or 0.0,
                r20=row["recent20_top5"] or 0.0,
                gate="通过" if row["four_year_observation_passed"] else f"未通过：{row['failed_reasons']}",
            )
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 汇总 JSON：`{summary_json}`",
            f"- 汇总 CSV：`{summary_csv}`",
            "- 每个标签另有 daily / annual / monthly 明细 CSV。",
        ]
    )
    summary_md = REPORT_DIR / "current_formal_four_year_eval_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "report_dir": str(REPORT_DIR),
                "summary_json": str(summary_json),
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
