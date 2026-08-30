from __future__ import annotations

import csv
import importlib
from pathlib import Path


grid = importlib.import_module("research_formal_5d10d_l5_candidate_grid_20260621")

ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_tail_direction_refine_20260622"
)


def _assets() -> dict[str, dict]:
    manifest = grid.build_fusion_db()
    return grid._assets(manifest)


def _params() -> list[dict]:
    assets = _assets()
    runs = []
    asset_names = [
        "std_10d",
        "std_5d",
        "rank_10d90_5d10",
        "rank_10d70_5d30",
        "rank_10d50_5d50",
        "rank_min_5d10d",
        "rank_max_5d10d",
    ]
    filters = [
        {"top_k": 5, "holding_days": 5, "max_positions": 5, "target_total_pct": 0.98, "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "max_atr_ratio": None},
        {"top_k": 5, "holding_days": 7, "max_positions": 5, "target_total_pct": 0.98, "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "max_atr_ratio": None},
        {"top_k": 8, "holding_days": 5, "max_positions": 8, "target_total_pct": 0.98, "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "max_atr_ratio": None},
        {"top_k": 5, "holding_days": 5, "max_positions": 5, "target_total_pct": 0.98, "max_total_mv": 120000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3, "max_atr_ratio": None},
    ]
    for asset_name in asset_names:
        for item in filters:
            runs.append(
                {
                    **assets[asset_name],
                    **item,
                    "direction": "bottom",
                    "weight_mode": "equal",
                    "open_score_exit": 0,
                    "max_daily_sells": 1,
                    "score_continue_entry_ratio": 1.0,
                }
            )
    return runs


def _slug(params: dict) -> str:
    return (
        f"{params['asset']}_bottom"
        f"_tk{params['top_k']}_h{params['holding_days']}_mp{params['max_positions']}"
        f"_mv{grid._safe(params['max_total_mv'])}_amt{grid._safe(params['min_amount'])}"
        f"_turn{grid._safe(params['min_turnover_rate'])}"
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sort_by_annual(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("annual") or -999.0),
        float(row.get("sharpe") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _sort_by_sharpe(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    runs = _params()
    for index, params in enumerate(runs, start=1):
        slug = _slug(params)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            rows = grid._load_rows(params)
            grid._write_signal(params, rows, signal_file)
        returncode = grid._run_backtest(params, signal_file, log_file)
        indicator = grid._extract_indicator(log_file) or {}
        row = {
            **{key: value for key, value in params.items() if key not in {"db_path", "manifest_path"}},
            "db_path": str(params["db_path"]),
            "manifest_path": str(params["manifest_path"]),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **grid._signal_stats(signal_file),
            **grid._exposure_stats(log_file),
        }
        results.append(row)
        print(
            f"[{index}/{len(runs)}] {slug} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}",
            flush=True,
        )
    _write_rows(REPORT_DIR / "summary.csv", results)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
    target = [
        row
        for row in results
        if float(row.get("annual") or -999.0) >= 3.0
        and float(row.get("sharpe") or -999.0) >= 4.0
        and float(row.get("avg_invested_pct") or -999.0) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_target_hits.csv", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
