from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "timing1d_scores.duckdb"
SCORE_TABLE = "score_timing_55_30_10_05_g1p4_top5_dyn"

CONTRIB_STOCK = REPORT_DIR / "high_return_contribution_by_stock_20260630.csv"
CONTRIB_JSON = REPORT_DIR / "high_return_contribution_summary_20260630.json"

OUT_SIGNAL_DIR = REPORT_DIR / "high_return_fragility_ablation_signals"
OUT_LOG_DIR = REPORT_DIR / "high_return_fragility_ablation_logs"
OUT_CSV = REPORT_DIR / "high_return_fragility_ablation_20260630.csv"
OUT_JSON = REPORT_DIR / "high_return_fragility_ablation_20260630.json"


CASES: dict[str, dict[str, Any]] = {
    "hef_top3_deep_skip_drop_scale": {
        "signal_file": REPORT_DIR / "high_exposure_dayrisk_filter_signals" / "hef_top3_deep_skip_drop_scale.csv",
        "baseline_annual": 7.0825517567412914,
        "baseline_sharpe": 1.0062946793586385,
        "baseline_max_drawdown": 0.6361324608412972,
    },
    "thex_t55_top3_s1000_cap99_h2m3_ddoff": {
        "signal_file": REPORT_DIR / "timing_high_exposure_extension_signals" / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
        "baseline_annual": 6.166572732425358,
        "baseline_sharpe": 0.9137458723007826,
        "baseline_max_drawdown": 0.6336130020015878,
    },
}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
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


def _top_symbols_by_case() -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    with CONTRIB_STOCK.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            case = str(row.get("case") or "")
            symbol = str(row.get("symbol") or "")
            if case not in CASES or not symbol:
                continue
            pnl = float(row.get("pnl") or 0.0)
            grouped.setdefault(case, []).append((pnl, symbol))
    return {case: [symbol for _, symbol in sorted(items, reverse=True)] for case, items in grouped.items()}


def _top_month_by_case() -> dict[str, str]:
    rows = json.loads(CONTRIB_JSON.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for row in rows:
        case = str(row.get("case") or "")
        month = str((row.get("top_month") or {}).get("month") or "")
        if case and month:
            out[case] = month
    return out


def _make_signal(case: str, mode: str, remove_symbols: list[str] | None = None, remove_month: str | None = None) -> tuple[Path, dict[str, Any]]:
    cfg = CASES[case]
    rows = list(csv.DictReader(cfg["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    remove_set = set(remove_symbols or [])
    out: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        symbol = str(row.get("symbol") or "")
        buy_date = str(row.get("buy_date") or "")
        if symbol in remove_set:
            removed += 1
            continue
        if remove_month and buy_date.startswith(remove_month):
            removed += 1
            continue
        out.append(row)
    counts: dict[str, int] = {}
    for row in out:
        counts[str(row.get("signal_date") or "")] = counts.get(str(row.get("signal_date") or ""), 0) + 1
    output = OUT_SIGNAL_DIR / f"{case}__{mode}.csv"
    _write_rows(output, out)
    return output, {
        "input_rows": len(rows),
        "signal_rows": len(out),
        "removed_rows": removed,
        "signal_days": len(counts),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "days_below_3": sum(1 for value in counts.values() if value < 3),
        "remove_symbols": ",".join(remove_symbols or []),
        "remove_month": remove_month or "",
    }


def _run(case: str, mode: str, remove_symbols: list[str] | None = None, remove_month: str | None = None) -> dict[str, Any]:
    signal_file, meta = _make_signal(case, mode, remove_symbols=remove_symbols, remove_month=remove_month)
    log_file = OUT_LOG_DIR / f"{case}__{mode}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": "1",
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_SYNC_POSITIONS": "1",
                "GM_CASH_BUFFER": "0.99",
                "GM_VERBOSE_TRADES": "1",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": "0.98",
                "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.99",
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.995",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
                "GM_RESIZE_HELD_ON_SIGNAL": "0",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
            }
        )
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "3",
            "--holding-days",
            "2",
            "--max-holding-days",
            "3",
            "--target-position-pct",
            "0.99",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            "2022-06-07 09:00:00",
            "--backtest-end",
            "2026-06-29 15:30:00",
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    annual = indicator.get("pnl_ratio_annual") if indicator else None
    sharpe = indicator.get("sharp_ratio") if indicator else None
    max_drawdown = indicator.get("max_drawdown") if indicator else None
    baseline = CASES[case]
    return {
        "case": case,
        "mode": mode,
        **meta,
        "returncode": returncode,
        "annual": annual,
        "annual_drop_vs_baseline": annual - baseline["baseline_annual"] if annual is not None else None,
        "sharpe": sharpe,
        "sharpe_delta_vs_baseline": sharpe - baseline["baseline_sharpe"] if sharpe is not None else None,
        "max_drawdown": max_drawdown,
        "max_drawdown_delta_vs_baseline": max_drawdown - baseline["baseline_max_drawdown"] if max_drawdown is not None else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "baseline_annual": baseline["baseline_annual"],
        "baseline_sharpe": baseline["baseline_sharpe"],
        "baseline_max_drawdown": baseline["baseline_max_drawdown"],
        "target_hit_500_sharpe4_mdd40": bool(
            annual is not None and sharpe is not None and max_drawdown is not None and annual >= 5.0 and sharpe >= 4.0 and max_drawdown <= 0.40
        ),
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "note": "research-only fragility ablation; production unchanged",
    }


def main() -> None:
    top_symbols = _top_symbols_by_case()
    top_months = _top_month_by_case()
    runs: list[tuple[str, str, list[str] | None, str | None]] = []
    for case in CASES:
        symbols = top_symbols.get(case, [])
        if symbols:
            runs.append((case, f"remove_top1_{symbols[0].replace('.', '')}", [symbols[0]], None))
            runs.append((case, "remove_top3_stocks", symbols[:3], None))
        if top_months.get(case):
            runs.append((case, f"remove_top_month_{top_months[case]}", None, top_months[case]))
    results: list[dict[str, Any]] = []
    for case, mode, symbols, month in runs:
        row = _run(case, mode, remove_symbols=symbols, remove_month=month)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

