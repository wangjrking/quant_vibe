"""Research-only pre-2026 tuning with a one-time 2026 holdout evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_four_score_1d_rank_fusion_20260818.py"
OUT = ROOT / "quant" / "data_file" / "reports" / "model_agent_four_score_1d_pre2026_tune_20260818"
DEV_START = "20220606"
DEV_END = "20251231"
HOLDOUT_START = "20260101"
HOLDOUT_END = "20260616"  # current mature 1D label boundary


def load_base_module():
    spec = importlib.util.spec_from_file_location("four_score_rank_fusion", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def mean(rows: list[dict], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows)


def development_summary(rows: list[dict]) -> dict[str, object]:
    by_year: dict[str, list[dict]] = {}
    for row in rows:
        by_year.setdefault(str(row["trade_date"])[:4], []).append(row)
    annual = {
        year: {
            "days": len(items),
            "rank_ic": mean(items, "rank_ic"),
            "top1_excess": mean(items, "top1_excess"),
            "top3_excess": mean(items, "top3_excess"),
            "top5_excess": mean(items, "top5_excess"),
            "top10_excess": mean(items, "top10_excess"),
        }
        for year, items in sorted(by_year.items())
    }
    return {
        "days": len(rows),
        "rank_ic": mean(rows, "rank_ic"),
        "top1_excess": mean(rows, "top1_excess"),
        "top3_excess": mean(rows, "top3_excess"),
        "top5_excess": mean(rows, "top5_excess"),
        "top10_excess": mean(rows, "top10_excess"),
        "front_utility": 0.50 * mean(rows, "top1_excess") + 0.30 * mean(rows, "top3_excess") + 0.20 * mean(rows, "top10_excess"),
        "annual": annual,
    }


def evaluate_phase(module, sources: dict, start_date: str, end_date: str, formulas: dict) -> tuple[dict, dict]:
    conn = duckdb.connect()
    try:
        module.build_base_view(conn, sources, start_date, end_date)
        quality = conn.execute(
            "SELECT COUNT(*) AS row_count, COUNT(DISTINCT trade_date) AS day_count, "
            "COUNT(DISTINCT stock_code) AS stock_count, "
            "COUNT(*) - COUNT(DISTINCT (trade_date, stock_code)) AS duplicate_key_count, "
            "SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_row_count "
            "FROM base"
        ).fetchone()
        results = {name: module.query_daily_metrics(conn, weights) for name, weights in formulas.items()}
    finally:
        conn.close()
    return results, {
        "rows": quality[0], "trade_days": quality[1], "stocks": quality[2],
        "duplicate_keys": quality[3], "bj_rows": quality[4], "date_range": [start_date, end_date],
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite:{OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_base_module()
    sources = module.source_meta()
    formulas = module.FORMULAS

    # Stage 1: no 2026 rows are readable while choosing weights.
    development_daily, development_quality = evaluate_phase(module, sources, DEV_START, DEV_END, formulas)
    development = {name: development_summary(rows) for name, rows in development_daily.items()}
    selected = max(
        formulas,
        key=lambda name: (
            development[name]["front_utility"],
            development[name]["top10_excess"],
            development[name]["rank_ic"],
        ),
    )

    # Stage 2 opens 2026 once; only the frozen selection and baseline are evaluated.
    frozen_formulas = {"baseline_1d": formulas["baseline_1d"], selected: formulas[selected]}
    holdout_daily, holdout_quality = evaluate_phase(module, sources, HOLDOUT_START, HOLDOUT_END, frozen_formulas)
    holdout = {name: development_summary(rows) for name, rows in holdout_daily.items()}
    baseline = holdout["baseline_1d"]
    selected_holdout = holdout[selected]
    result = {
        "research_only": True,
        "approval_status": "research_only_not_for_l5",
        "strict_pit_oof": False,
        "pit_limitation": "Current formal historical scores are model-history replay, not strict PIT/OOF. This result cannot support formal release or strategy admission.",
        "protocol": {
            "development_window": [DEV_START, DEV_END],
            "development_uses_2026": False,
            "holdout_window": [HOLDOUT_START, HOLDOUT_END],
            "holdout_candidate_grid_not_evaluated": True,
            "selection_metric": "0.50*Top1 excess + 0.30*Top3 excess + 0.20*Top10 excess",
            "predeclared_formulas": formulas,
        },
        "sources": sources,
        "development_input_quality": development_quality,
        "development_results": development,
        "frozen_selected_formula": selected,
        "frozen_selected_weights": formulas[selected],
        "holdout_input_quality": holdout_quality,
        "holdout_results": holdout,
        "holdout_delta_vs_baseline": {
            metric: float(selected_holdout[metric]) - float(baseline[metric])
            for metric in ("rank_ic", "top1_excess", "top3_excess", "top5_excess", "top10_excess", "front_utility")
        },
        "next_step": "No production action. A separate authorized strict PIT/OOF four-score asset must reproduce the frozen weights before any strategy or formal discussion.",
    }
    (OUT / "pre2026_tune_2026_holdout_summary.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Pre-2026 four-score 1D rank fusion tune",
        "",
        f"Frozen formula selected without reading 2026: `{selected}`.",
        f"Weights: `{formulas[selected]}`.",
        "",
        "2026 is a one-time holdout. The candidate grid was not evaluated in the holdout stage.",
        "",
        "Research-only. Current formal historical scores are not strict PIT/OOF; no release or strategy conclusion follows.",
    ]
    (OUT / "pre2026_tune_2026_holdout_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
