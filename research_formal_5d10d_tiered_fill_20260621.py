from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import research_formal_5d10d_l5_candidate_grid_20260621 as base
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_tiered_fill_20260621"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


PRIMARY_POOLS = [
    {"asset": "rank_10d90_5d10", "max_total_mv": 150000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
    {"asset": "rank_10d80_5d20", "max_total_mv": 150000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
    {"asset": "rank_10d80_5d20", "max_total_mv": 100000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
]

FALLBACK_POOLS = [
    {"asset": "rank_10d80_5d20", "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5},
    {"asset": "rank_10d60_5d40", "max_total_mv": 200000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
    {"asset": "rank_10d50_5d50", "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5},
]

PRIMARY_LIMITS = [2, 3, 5]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _load_asset_map() -> dict[str, dict]:
    manifest = json.loads((SOURCE_DIR / "fusion_5d10d_manifest.json").read_text(encoding="utf-8"))
    return {
        combo["combo_name"]: {
            "db_path": Path(manifest["output_db"]),
            "table": combo["table"],
            "manifest_path": SOURCE_DIR / "fusion_5d10d_manifest.json",
        }
        for combo in manifest["combos"]
    }


def _pool_params(asset_map: dict, pool: dict, top_k: int, holding_days: int, max_positions: int) -> dict:
    return {
        **asset_map[pool["asset"]],
        **pool,
        "direction": "top",
        "top_k": top_k,
        "holding_days": holding_days,
        "max_positions": max_positions,
        "max_atr_ratio": None,
        "weight_mode": "equal",
        "target_total_pct": 0.98,
    }


def _group_top(rows: list[dict], limit: int, tier_offset: float) -> list[dict]:
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        by_date.setdefault(str(row.get("trade_date") or ""), []).append(row)
    selected = []
    for trade_date in sorted(by_date):
        day_rows = sorted(
            by_date[trade_date],
            key=lambda item: (float(item.get("pred_prob") or -999.0), str(item.get("stock_code") or "")),
            reverse=True,
        )[:limit]
        for idx, row in enumerate(day_rows, start=1):
            out = dict(row)
            out["raw_pred_prob"] = row.get("pred_prob")
            out["pred_prob"] = tier_offset + float(row.get("pred_prob") or 0.0) - (idx * 1e-6)
            selected.append(out)
    return selected


def _merge_tiers(primary_rows: list[dict], fallback_rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged = []
    for row in [*primary_rows, *fallback_rows]:
        key = (str(row.get("trade_date") or ""), str(row.get("stock_code") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _write_signal(params: dict, signal_file: Path) -> None:
    primary_rows = base._load_rows(
        _pool_params(params["asset_map"], params["primary"], params["primary_limit"], params["holding_days"], params["max_positions"])
    )
    fallback_rows = base._load_rows(
        _pool_params(params["asset_map"], params["fallback"], 5, params["holding_days"], params["max_positions"])
    )
    rows = _merge_tiers(
        _group_top(primary_rows, int(params["primary_limit"]), 2.0),
        _group_top(fallback_rows, 5, 1.0),
    )
    market_rows = load_market_rows_by_trade_date(MARKET_DB, base.START_DATE, base.END_DATE)
    signals = build_gm_signal_rows(
        rows,
        config=SelectionConfig(
            top_k=5,
            pred_col="pred_prob",
            min_pred_prob=None,
            max_atr_ratio=None,
            min_amount=None,
            min_turnover_rate=None,
            max_total_mv=None,
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_delisting=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=int(params["max_positions"]),
        weight_mode="equal",
        target_total_pct=0.98,
    )
    write_gm_signals_csv(signals, signal_file)


def _slug(params: dict) -> str:
    primary = params["primary"]
    fallback = params["fallback"]
    return (
        f"p{primary['asset']}_mv{_safe(primary['max_total_mv'])}_amt{_safe(primary['min_amount'])}_turn{_safe(primary['min_turnover_rate'])}"
        f"_plim{params['primary_limit']}"
        f"_f{fallback['asset']}_mv{_safe(fallback['max_total_mv'])}_amt{_safe(fallback['min_amount'])}_turn{_safe(fallback['min_turnover_rate'])}"
        f"_mp{params['max_positions']}_h{params['holding_days']}_exit{params['open_score_exit']}"
    )


def _extract_indicator(log_file: Path) -> dict | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _run_backtest(params: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(params["open_score_exit"]),
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.95",
            "GM_MAX_DAILY_SELLS": str(params["max_daily_sells"]),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
    fallback = params["fallback"]
    asset_map = params["asset_map"]
    score_asset = fallback["asset"]
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(params["max_positions"]),
        "--holding-days",
        str(params["holding_days"]),
        "--max-holding-days",
        str(params["holding_days"]),
        "--target-position-pct",
        str(0.98 / float(int(params["max_positions"]))),
        "--score-db",
        str(asset_map[score_asset]["db_path"]),
        "--score-table",
        str(asset_map[score_asset]["table"]),
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _sort_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grid() -> list[dict]:
    asset_map = _load_asset_map()
    runs = []
    for primary in PRIMARY_POOLS:
        for fallback in FALLBACK_POOLS:
            for primary_limit in PRIMARY_LIMITS:
                for max_positions in [8, 10]:
                    runs.append(
                        {
                            "asset_map": asset_map,
                            "primary": primary,
                            "fallback": fallback,
                            "primary_limit": primary_limit,
                            "holding_days": 4,
                            "max_positions": max_positions,
                            "open_score_exit": 1,
                            "max_daily_sells": 1,
                        }
                    )
    return runs


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    grid = _grid()
    for index, params in enumerate(grid, start=1):
        slug = _slug(params)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            _write_signal(params, signal_file)
        returncode = _run_backtest(params, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            "primary_asset": params["primary"]["asset"],
            "primary_max_total_mv": params["primary"]["max_total_mv"],
            "primary_min_amount": params["primary"]["min_amount"],
            "primary_min_turnover_rate": params["primary"]["min_turnover_rate"],
            "primary_limit": params["primary_limit"],
            "fallback_asset": params["fallback"]["asset"],
            "fallback_max_total_mv": params["fallback"]["max_total_mv"],
            "fallback_min_amount": params["fallback"]["min_amount"],
            "fallback_min_turnover_rate": params["fallback"]["min_turnover_rate"],
            "max_positions": params["max_positions"],
            "holding_days": params["holding_days"],
            "open_score_exit": params["open_score_exit"],
            "max_daily_sells": params["max_daily_sells"],
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_key, reverse=True))
        print(
            f"[{index}/{len(grid)}] {slug} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_key, reverse=True)
        if float(row.get("annual") or -999) >= 2.0
        and float(row.get("sharpe") or -999) >= 3.0
        and float(row.get("avg_invested_pct") or -999) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
