"""Read-only mature-label deterioration diagnostics for active formal L4 assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("quant")
MANIFEST_DIR = ROOT / "main/config/prediction_manifests"
LABEL_DB = ROOT / "data_file/production_assets/duckdb/l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
OUT = ROOT / "data_file/reports/model_agent_formal_l4_recent_degradation_20260817"
SPECS = {
    "1D": ("executable_1d_open_return_l4_formal_20260619.json", "executable_1d_open_return"),
    "3D": ("executable_3d_open_return_l4_formal_20260617.json", "executable_3d_open_return"),
    "5D": ("executable_5d_open_return_l4_formal_20260620.json", "executable_5d_open_return"),
    "10D": ("executable_10d_open_return_l4_formal_20260617.json", "executable_10d_open_return"),
}


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metric_day(group: pd.DataFrame) -> dict[str, float | int] | None:
    if len(group) < 30 or group["pred_prob"].nunique() < 2 or group["label"].nunique() < 2:
        return None
    ranked_score = group["pred_prob"].rank(method="average")
    ranked_label = group["label"].rank(method="average")
    top = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort").head(10)
    return {
        "rank_ic": float(ranked_score.corr(ranked_label)),
        "top10_return": float(top["label"].mean()),
        "top10_excess": float(top["label"].mean() - group["label"].median()),
        "score_std": float(group["pred_prob"].std(ddof=0)),
        "distinct_scores": int(group["pred_prob"].nunique()),
        "rows": int(len(group)),
    }


def summarize(days: pd.DataFrame) -> dict[str, float | int | None]:
    if days.empty:
        return {"trade_days": 0, "rank_ic": None, "top10_return": None, "top10_excess": None, "score_std": None, "distinct_scores": None}
    return {
        "trade_days": int(len(days)),
        "rank_ic": float(days["rank_ic"].mean()),
        "top10_return": float(days["top10_return"].mean()),
        "top10_excess": float(days["top10_excess"].mean()),
        "score_std": float(days["score_std"].mean()),
        "distinct_scores": float(days["distinct_scores"].mean()),
    }


def front_rank_daily(group: pd.DataFrame) -> dict[str, float] | None:
    if len(group) < 30 or group["pred_prob"].nunique() < 2:
        return None
    ordered = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
    median = float(group["label"].median())
    result: dict[str, float] = {}
    for count in (1, 3, 5, 10):
        value = float(ordered.head(count)["label"].mean())
        result[f"top{count}_return"] = value
        result[f"top{count}_excess"] = value - median
    return result


def summarize_front(days: pd.DataFrame) -> dict[str, float | int | None]:
    if days.empty:
        return {"trade_days": 0, **{f"top{count}_{kind}": None for count in (1, 3, 5, 10) for kind in ("return", "excess")}}
    return {
        "trade_days": int(len(days)),
        **{f"top{count}_{kind}": float(days[f"top{count}_{kind}"].mean()) for count in (1, 3, 5, 10) for kind in ("return", "excess")},
    }


def read_joined(manifest: dict[str, object], label: str) -> pd.DataFrame:
    db_path = (MANIFEST_DIR / str(manifest["db_path"])).resolve()
    table = str(manifest["table"])
    connection = duckdb.connect()
    try:
        prediction_literal = str(db_path).replace("'", "''")
        label_literal = str(LABEL_DB.resolve()).replace("'", "''")
        connection.execute(f"ATTACH '{prediction_literal}' AS pred (READ_ONLY)")
        connection.execute(f"ATTACH '{label_literal}' AS lab (READ_ONLY)")
        result = connection.execute(
            f"SELECT p.trade_date, p.stock_code, p.pred_prob, l.{q(label)} AS label "
            f"FROM pred.{q(table)} AS p "
            f"INNER JOIN lab.{q(LABEL_TABLE)} AS l USING (trade_date, stock_code) "
            f"WHERE p.trade_date >= '20250101' AND p.stock_code NOT LIKE '%.BJ' "
            f"AND l.{q(label)} IS NOT NULL "
            "ORDER BY p.trade_date, p.stock_code"
        ).fetchdf()
    finally:
        connection.close()
    result["trade_date"] = result["trade_date"].astype(str).str.replace("-", "", regex=False)
    result["stock_code"] = result["stock_code"].astype(str)
    result["pred_prob"] = pd.to_numeric(result["pred_prob"], errors="coerce")
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    if result.empty or result.duplicated(["trade_date", "stock_code"]).any() or result["stock_code"].str.endswith(".BJ").any():
        raise RuntimeError("blocked_invalid_same_key_or_no_bj_input")
    if not np.isfinite(result[["pred_prob", "label"]]).all().all():
        raise RuntimeError("blocked_nonfinite_score_or_label")
    return result


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    OUT = Path(parser.parse_args().output_dir)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite:{OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, object] = {}
    for name, (filename, label) in SPECS.items():
        manifest_path = MANIFEST_DIR / filename
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scores = read_joined(manifest, label)
        daily_rows: list[dict[str, object]] = []
        skipped = 0
        for trade_date, group in scores.groupby("trade_date", sort=True):
            values = metric_day(group)
            if values is None:
                skipped += 1
                continue
            daily_rows.append({"trade_date": trade_date, "month": trade_date[:6], **values})
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            raise RuntimeError(f"blocked_no_valid_metric_days:{name}")
        monthly = {
            month: summarize(frame.drop(columns=["month"]))
            for month, frame in daily.groupby("month", sort=True)
        }
        baseline = summarize(daily.loc[(daily["trade_date"] >= "20250101") & (daily["trade_date"] <= "20251231")])
        recent = summarize(daily.loc[daily["trade_date"] >= "20260101"])
        last20 = summarize(daily.tail(20))
        preceding63 = summarize(daily.iloc[max(0, len(daily) - 83): max(0, len(daily) - 20)])
        front_rows: list[dict[str, object]] = []
        for trade_date, group in scores.groupby("trade_date", sort=True):
            values = front_rank_daily(group)
            if values is not None:
                front_rows.append({"trade_date": trade_date, **values})
        front = pd.DataFrame(front_rows)
        front_baseline = summarize_front(front.loc[(front["trade_date"] >= "20250101") & (front["trade_date"] <= "20251231")])
        front_recent = summarize_front(front.loc[front["trade_date"] >= "20260101"])
        front_last20 = summarize_front(front.tail(20))
        front_prior63 = summarize_front(front.iloc[max(0, len(front) - 83): max(0, len(front) - 20)])
        front_deltas = {
            f"top{count}_excess_vs_2025": front_recent[f"top{count}_excess"] - front_baseline[f"top{count}_excess"]
            for count in (1, 3, 5, 10)
        } | {
            f"last20_top{count}_excess_vs_preceding63": front_last20[f"top{count}_excess"] - front_prior63[f"top{count}_excess"]
            for count in (1, 3, 5, 10)
        }
        decline = {
            "rank_ic_vs_2025": recent["rank_ic"] - baseline["rank_ic"],
            "top10_excess_vs_2025": recent["top10_excess"] - baseline["top10_excess"],
            "last20_rank_ic_vs_preceding63": last20["rank_ic"] - preceding63["rank_ic"],
            "last20_top10_excess_vs_preceding63": last20["top10_excess"] - preceding63["top10_excess"],
        }
        status = (
            "deteriorating"
            if decline["rank_ic_vs_2025"] < 0 and decline["top10_excess_vs_2025"] < 0
            else "mixed_or_stable"
        )
        all_results[name] = {
            "label": label,
            "manifest": str(manifest_path),
            "formal_db": str((MANIFEST_DIR / str(manifest["db_path"])).resolve()),
            "table": manifest["table"],
            "mature_score_date_range": [str(daily["trade_date"].min()), str(daily["trade_date"].max())],
            "mature_metric_trade_days": int(len(daily)),
            "skipped_constant_or_small_cross_section_days": skipped,
            "quality": {"duplicate_key_groups": 0, "bj_rows": 0, "score_or_label_nonfinite": 0},
            "monthly": monthly,
            "baseline_2025": baseline,
            "recent_2026": recent,
            "last20_mature": last20,
            "preceding63_mature": preceding63,
            "front_rank_only": {
                "baseline_2025": front_baseline,
                "recent_2026": front_recent,
                "last20_mature": front_last20,
                "preceding63_mature": front_prior63,
                "deltas": front_deltas,
            },
            "deltas": decline,
            "assessment": status,
        }
        daily.to_csv(OUT / f"{name.lower()}_daily_metrics.csv", index=False)
        front.to_csv(OUT / f"{name.lower()}_front_rank_daily_metrics.csv", index=False)
    report = {
        "report_type": "read_only_formal_l4_recent_mature_label_degradation_diagnosis",
        "scope": "active formal 1D/3D/5D/10D only; mature labels only; no strategy backtest",
        "production_unchanged": True,
        "label_freshness_boundary": "Only non-null label rows are evaluated. August predictions are excluded because they are not mature labels.",
        "results": all_results,
    }
    dump(OUT / "recent_degradation_summary.json", report)
    lines = ["# Formal L4 Recent Mature-Label Diagnosis", "", "Read-only; no strategy backtest or production change.", ""]
    for name, item in all_results.items():
        recent = item["recent_2026"]
        baseline = item["baseline_2025"]
        delta = item["deltas"]
        lines += [
            f"## {name}",
            f"- Mature range: {item['mature_score_date_range'][0]} to {item['mature_score_date_range'][1]}",
            f"- 2025 RankIC / Top10 excess: {baseline['rank_ic']:.6f} / {baseline['top10_excess']:.6f}",
            f"- 2026 RankIC / Top10 excess: {recent['rank_ic']:.6f} / {recent['top10_excess']:.6f}",
            f"- Delta vs 2025: RankIC {delta['rank_ic_vs_2025']:.6f}; Top10 excess {delta['top10_excess_vs_2025']:.6f}",
            f"- Assessment: {item['assessment']}", "",
        ]
    (OUT / "recent_degradation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
