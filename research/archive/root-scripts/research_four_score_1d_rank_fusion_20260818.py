"""Research-only diagnostic: combine formal L4 ranks for the 1D label.

This script deliberately does not write a prediction asset.  Current formal
history is a model-history replay rather than a strict PIT/OOF score family,
so results are useful for ranking diagnosis only and cannot support release or
strategy admission without a separate PIT rebuild.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
LABEL_COLUMN = "executable_1d_open_return"
OUT = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_score_1d_rank_fusion_research_20260818"

MANIFESTS = {
    "1d": "executable_1d_open_return_l4_formal_20260619.json",
    "3d": "executable_3d_open_return_l4_formal_20260617.json",
    "5d": "executable_5d_open_return_l4_formal_20260620.json",
    "10d": "executable_10d_open_return_l4_formal_20260617.json",
}

# Fixed before reading labels. Selection uses 2022H2-2023 only.
FORMULAS = {
    "baseline_1d": {"1d": 1.00, "3d": 0.00, "5d": 0.00, "10d": 0.00},
    "rank_1d3d_70_30": {"1d": 0.70, "3d": 0.30, "5d": 0.00, "10d": 0.00},
    "rank_1d5d_70_30": {"1d": 0.70, "3d": 0.00, "5d": 0.30, "10d": 0.00},
    "rank_1d10d_70_30": {"1d": 0.70, "3d": 0.00, "5d": 0.00, "10d": 0.30},
    "rank_1d5d10d_60_25_15": {"1d": 0.60, "3d": 0.00, "5d": 0.25, "10d": 0.15},
    "rank_all4_55_20_15_10": {"1d": 0.55, "3d": 0.20, "5d": 0.15, "10d": 0.10},
}
SELECTION_START = "20220606"
SELECTION_END = "20231229"
HOLDOUT_START = "20240101"
HOLDOUT_END = "20241231"


def qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def source_meta() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for horizon, filename in MANIFESTS.items():
        manifest_path = MANIFEST_DIR / filename
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        result[horizon] = {
            "manifest": str(manifest_path),
            "db_path": str((manifest_path.parent / payload["db_path"]).resolve()),
            "table": payload["table"],
            "source_type": payload["source_type"],
            "approval_status": payload["approval_status"],
            "pred_prob_sha256": payload.get("pred_prob_sha256"),
        }
    return result


def build_base_view(
    conn: duckdb.DuckDBPyConnection,
    sources: dict[str, dict[str, str]],
    start_date: str = SELECTION_START,
    end_date: str = HOLDOUT_END,
) -> None:
    conn.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
    for horizon, source in sources.items():
        db_literal = source["db_path"].replace("'", "''")
        conn.execute(f"ATTACH '{db_literal}' AS score_{horizon} (READ_ONLY)")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW base AS
        SELECT
            CAST(p1.trade_date AS VARCHAR) AS trade_date,
            p1.stock_code,
            p1.pred_prob AS p1,
            p3.pred_prob AS p3,
            p5.pred_prob AS p5,
            p10.pred_prob AS p10,
            lab.{qi(LABEL_COLUMN)} AS realized
        FROM score_1d.{qi(sources['1d']['table'])} AS p1
        INNER JOIN score_3d.{qi(sources['3d']['table'])} AS p3
          USING (trade_date, stock_code)
        INNER JOIN score_5d.{qi(sources['5d']['table'])} AS p5
          USING (trade_date, stock_code)
        INNER JOIN score_10d.{qi(sources['10d']['table'])} AS p10
          USING (trade_date, stock_code)
        INNER JOIN labels.{qi(LABEL_TABLE)} AS lab
          USING (trade_date, stock_code)
        WHERE p1.stock_code NOT LIKE '%.BJ'
          AND lab.{qi(LABEL_COLUMN)} IS NOT NULL
          AND isfinite(p1.pred_prob)
          AND isfinite(p3.pred_prob)
          AND isfinite(p5.pred_prob)
          AND isfinite(p10.pred_prob)
          AND isfinite(lab.{qi(LABEL_COLUMN)})
          AND CAST(p1.trade_date AS VARCHAR) BETWEEN '{start_date}' AND '{end_date}'
        """
    )


def query_daily_metrics(conn: duckdb.DuckDBPyConnection, formula: dict[str, float]) -> list[dict[str, float | int | str]]:
    score = " + ".join(
        f"{weight:.8f} * r{horizon}" for horizon, weight in formula.items() if weight
    )
    rows = conn.execute(
        f"""
        WITH ranks AS (
          SELECT *,
            percent_rank() OVER (PARTITION BY trade_date ORDER BY p1) AS r1d,
            percent_rank() OVER (PARTITION BY trade_date ORDER BY p3) AS r3d,
            percent_rank() OVER (PARTITION BY trade_date ORDER BY p5) AS r5d,
            percent_rank() OVER (PARTITION BY trade_date ORDER BY p10) AS r10d
          FROM base
        ), scored AS (
          SELECT *, ({score}) AS fusion_score,
            row_number() OVER (PARTITION BY trade_date ORDER BY ({score}) DESC, stock_code) AS rn,
            avg(realized) OVER (PARTITION BY trade_date) AS market_mean
          FROM ranks
        )
        SELECT
          trade_date,
          COUNT(*) AS stocks,
          corr(fusion_score, realized) AS rank_ic,
          avg(realized) FILTER (WHERE rn <= 1) AS top1,
          avg(realized) FILTER (WHERE rn <= 3) AS top3,
          avg(realized) FILTER (WHERE rn <= 5) AS top5,
          avg(realized) FILTER (WHERE rn <= 10) AS top10,
          avg(realized) FILTER (WHERE rn <= 1) - max(market_mean) AS top1_excess,
          avg(realized) FILTER (WHERE rn <= 3) - max(market_mean) AS top3_excess,
          avg(realized) FILTER (WHERE rn <= 5) - max(market_mean) AS top5_excess,
          avg(realized) FILTER (WHERE rn <= 10) - max(market_mean) AS top10_excess
        FROM scored
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchall()
    names = [item[0] for item in conn.description]
    return [dict(zip(names, row)) for row in rows]


def summarize(rows: list[dict[str, float | int | str]], start: str, end: str) -> dict[str, float | int | None]:
    segment = [row for row in rows if start <= str(row["trade_date"]) <= end]
    if not segment:
        return {"days": 0, "rank_ic": None, "top1": None, "top3": None, "top5": None, "top10": None,
                "top1_excess": None, "top3_excess": None, "top5_excess": None, "top10_excess": None}
    metrics = ["rank_ic", "top1", "top3", "top5", "top10", "top1_excess", "top3_excess", "top5_excess", "top10_excess"]
    return {
        "days": len(segment),
        **{metric: sum(float(row[metric]) for row in segment) / len(segment) for metric in metrics},
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite:{OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    sources = source_meta()
    if not all(item["source_type"] == "duckdb_table" for item in sources.values()):
        raise RuntimeError("blocked_non_duckdb_formal_source")
    conn = duckdb.connect()
    try:
        build_base_view(conn, sources)
        base_quality = conn.execute(
            "SELECT COUNT(*) AS row_count, COUNT(DISTINCT trade_date) AS day_count, COUNT(DISTINCT stock_code) AS stock_count, "
            "COUNT(*) - COUNT(DISTINCT (trade_date, stock_code)) AS duplicate_key_count, "
            "SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_row_count, "
            "MIN(trade_date), MAX(trade_date) FROM base"
        ).fetchone()
        all_results: dict[str, dict[str, object]] = {}
        for name, weights in FORMULAS.items():
            daily = query_daily_metrics(conn, weights)
            with (OUT / f"{name}_daily.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(daily[0].keys()))
                writer.writeheader()
                writer.writerows(daily)
            all_results[name] = {
                "weights": weights,
                "selection_2022h2_2023": summarize(daily, SELECTION_START, SELECTION_END),
                "holdout_2024": summarize(daily, HOLDOUT_START, HOLDOUT_END),
            }
        selected = max(
            FORMULAS,
            key=lambda name: (
                float(all_results[name]["selection_2022h2_2023"]["top10_excess"]),
                float(all_results[name]["selection_2022h2_2023"]["top3_excess"]),
                float(all_results[name]["selection_2022h2_2023"]["rank_ic"]),
            ),
        )
        baseline = all_results["baseline_1d"]
        selected_result = all_results[selected]
        summary = {
            "research_only": True,
            "approval_status": "research_only_not_for_l5",
            "purpose": "diagnose whether fixed daily cross-sectional rank fusion improves the 1D model label",
            "strict_pit_oof": False,
            "pit_limitation": "formal historical scores are current-model history replay; this evidence cannot support strategy admission or production release",
            "selection_protocol": {
                "predeclared_formulas": FORMULAS,
                "selection_window": [SELECTION_START, SELECTION_END],
                "selection_metric": "mean_daily_top10_excess_for_executable_1d_open_return",
                "holdout_window": [HOLDOUT_START, HOLDOUT_END],
                "holdout_not_used_for_selection": True,
            },
            "input_quality": {
                "rows": base_quality[0], "trade_days": base_quality[1], "stocks": base_quality[2],
                "duplicate_keys": base_quality[3], "bj_rows": base_quality[4],
                "date_range": [str(base_quality[5]), str(base_quality[6])],
            },
            "sources": sources,
            "results": all_results,
            "selected_by_development_only": selected,
            "selected_holdout_delta_vs_baseline": {
                metric: float(selected_result["holdout_2024"][metric]) - float(baseline["holdout_2024"][metric])
                for metric in ("rank_ic", "top1_excess", "top3_excess", "top5_excess", "top10_excess")
            },
            "next_step": "Only if the holdout is not weaker should a separately authorized strict PIT/OOF four-score build test the frozen formula. No formal or strategy action follows from this diagnostic.",
        }
        (OUT / "four_score_1d_rank_fusion_summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        lines = [
            "# Four-score 1D rank fusion research",
            "",
            "Research-only. No prediction asset, model, manifest, or strategy asset was modified.",
            "",
            f"Selected on 2022H2-2023 only: `{selected}`.",
            f"2024 Top10 excess delta vs 1D baseline: `{summary['selected_holdout_delta_vs_baseline']['top10_excess']:.6f}`.",
            "",
            "Important limitation: formal historical scores are not strict PIT/OOF, so this is a diagnostic, not a release or strategy-admission result.",
        ]
        (OUT / "four_score_1d_rank_fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
