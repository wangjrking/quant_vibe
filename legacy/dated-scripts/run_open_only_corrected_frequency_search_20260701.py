from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_open_only_corrected_frequency_search_20260701"


BASE_SCRIPT = MAIN / "run_adaptive70w_latest_top3_frequency_grid_20260630.py"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("latest_l4_grid_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


TOP_CASES = [
    {"name": "top1_pos70", "topn": 1, "max_positions": 1, "target": 0.70},
    {"name": "top1_pos60", "topn": 1, "max_positions": 1, "target": 0.60},
    {"name": "top2_pos35", "topn": 2, "max_positions": 2, "target": 0.35},
    {"name": "top3_pos25", "topn": 3, "max_positions": 3, "target": 0.25},
    {"name": "top5_pos15", "topn": 5, "max_positions": 5, "target": 0.15},
]

BUY_RULES = [
    {"name": "cool2d20", "two_day_cap": 0.20, "combo_cap": None, "turnover_floor": None},
    {"name": "cool2d18", "two_day_cap": 0.18, "combo_cap": None, "turnover_floor": None},
    {"name": "nocool", "two_day_cap": None, "combo_cap": None, "turnover_floor": None},
]

SELL_RULES = [
    {
        "name": "h1m2_e098_c099_min1",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit_ratio": 0.98,
        "score_continue_ratio": 0.99,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
    {
        "name": "h2m3_e097_c098_min1",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_ratio": 0.97,
        "score_continue_ratio": 0.98,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
    {
        "name": "h3m5_e096_c097_min1",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit_ratio": 0.96,
        "score_continue_ratio": 0.97,
        "min_score_exit_days": 1,
        "day_drop_ratio": None,
    },
]

FULL_SLICE = ("full", "2022-06-07 09:00:00", "2026-06-29 15:30:00")
CHECK_SLICES = [
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-29 15:30:00"),
    ("recent60", "2026-04-01 09:00:00", "2026-06-29 15:30:00"),
]


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


def _run_case(
    meta: dict[str, Any],
    top_case: dict[str, Any],
    sell_rule: dict[str, Any],
    tag: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    log_file = REPORT_DIR / "logs" / f"{meta['case_key']}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": "0",
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
                "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(sell_rule["min_score_exit_days"]),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(sell_rule["score_continue_ratio"]),
                "GM_SCORE_EXIT_ENTRY_RATIO": str(sell_rule["score_exit_ratio"]),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "none",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": f"{700000.0 * float(top_case['target']):.2f}",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
            }
        )
        cmd = [
            str(base.JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(base.STRATEGY_DIR),
            "--signal-file",
            str(meta["signal_file"]),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(top_case["max_positions"]),
            "--holding-days",
            str(sell_rule["holding_days"]),
            "--max-holding-days",
            str(sell_rule["max_holding_days"]),
            "--target-position-pct",
            str(top_case["target"]),
            "--score-db",
            str(base.SCORE_DB),
            "--score-table",
            str(meta["score_table"]),
            "--market-db",
            str(base.MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            end,
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
            "--score-exit-entry-ratio",
            str(sell_rule["score_exit_ratio"]),
            "--score-continue-entry-ratio",
            str(sell_rule["score_continue_ratio"]),
            "--min-holding-days-before-score-exit",
            str(sell_rule["min_score_exit_days"]),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "slice": tag,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def _summary_row(
    meta: dict[str, Any],
    top_case: dict[str, Any],
    buy_rule: dict[str, Any],
    sell_rule: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out = {
        "case_key": meta["case_key"],
        "top_case": top_case["name"],
        "buy_rule": buy_rule["name"],
        "sell_rule": sell_rule["name"],
        "target_pct_each": top_case["target"],
        "signal_rows": meta["signal_rows"],
        "signal_file": str(meta["signal_file"]),
        "score_table": meta["score_table"],
    }
    by_slice = {row["slice"]: row for row in rows}
    for tag in [FULL_SLICE[0], *[item[0] for item in CHECK_SLICES]]:
        item = by_slice.get(tag, {})
        out[f"{tag}_annual"] = item.get("annual")
        out[f"{tag}_pnl_ratio"] = item.get("pnl_ratio")
        out[f"{tag}_sharpe"] = item.get("sharpe")
        out[f"{tag}_max_drawdown"] = item.get("max_drawdown")
        out[f"{tag}_win_ratio"] = item.get("win_ratio")
        out[f"{tag}_open_count"] = item.get("open_count")
        out[f"{tag}_close_count"] = item.get("close_count")
        out[f"{tag}_log_file"] = item.get("log_file")
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    for top_case in TOP_CASES:
        for buy_rule in BUY_RULES:
            for sell_rule in SELL_RULES:
                meta = base._ensure_score_and_signal(top_case, buy_rule, sell_rule)
                full = _run_case(meta, top_case, sell_rule, *FULL_SLICE)
                row = {
                    "case_key": meta["case_key"],
                    "top_case": top_case["name"],
                    "buy_rule": buy_rule["name"],
                    "sell_rule": sell_rule["name"],
                    "target_pct_each": top_case["target"],
                    "signal_rows": meta["signal_rows"],
                    "signal_file": str(meta["signal_file"]),
                    "score_table": meta["score_table"],
                    "full_annual": full["annual"],
                    "full_pnl_ratio": full["pnl_ratio"],
                    "full_sharpe": full["sharpe"],
                    "full_max_drawdown": full["max_drawdown"],
                    "full_win_ratio": full["win_ratio"],
                    "full_open_count": full["open_count"],
                    "full_close_count": full["close_count"],
                    "full_log_file": full["log_file"],
                }
                screened.append(row)
                screened.sort(
                    key=lambda item: (
                        float(item["full_annual"]) if item["full_annual"] is not None else -999.0,
                        float(item["full_sharpe"]) if item["full_sharpe"] is not None else -999.0,
                    ),
                    reverse=True,
                )
                _write_rows(REPORT_DIR / "screen_full.csv", screened)
                print(json.dumps({"case_key": meta["case_key"], "full_annual": full["annual"], "full_sharpe": full["sharpe"]}, ensure_ascii=False), flush=True)

    top_candidates = screened[:10]
    for item in top_candidates:
        top_case = next(case for case in TOP_CASES if case["name"] == item["top_case"])
        buy_rule = next(rule for rule in BUY_RULES if rule["name"] == item["buy_rule"])
        sell_rule = next(rule for rule in SELL_RULES if rule["name"] == item["sell_rule"])
        meta = {
            "case_key": item["case_key"],
            "signal_rows": item["signal_rows"],
            "signal_file": Path(item["signal_file"]),
            "score_table": item["score_table"],
        }
        rows = [_run_case(meta, top_case, sell_rule, *FULL_SLICE)]
        rows.extend(_run_case(meta, top_case, sell_rule, tag, start, end) for tag, start, end in CHECK_SLICES)
        summary_rows.append(_summary_row(meta, top_case, buy_rule, sell_rule, rows))
        summary_rows.sort(
            key=lambda item: (
                float(item["full_annual"]) if item["full_annual"] is not None else -999.0,
                float(item["full_sharpe"]) if item["full_sharpe"] is not None else -999.0,
            ),
            reverse=True,
        )
        _write_rows(REPORT_DIR / "top_candidates_summary.csv", summary_rows)

    (REPORT_DIR / "top_candidates_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"report_dir": str(REPORT_DIR), "screened_cases": len(screened), "top_candidates": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
