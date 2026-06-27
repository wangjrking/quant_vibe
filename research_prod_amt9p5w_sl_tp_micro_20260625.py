from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path

import research_prod_filter_amount_interpolate_20260625 as filt


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_amt9p5w_sl_tp_micro_20260625"

CASES = [
    {
        "case_name": "amt9p5_best_sl055_tp065",
        "signal_file": DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv",
        "target_pct": 0.8975,
        "dd_soft": 0.08,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "stop_loss": 0.055,
        "take_profit": 0.065,
    },
    {
        "case_name": "amt9p5_best_sl055_tp07",
        "signal_file": DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv",
        "target_pct": 0.8975,
        "dd_soft": 0.08,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "stop_loss": 0.055,
        "take_profit": 0.07,
    },
    {
        "case_name": "amt9p5_best_sl055_tp075",
        "signal_file": DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv",
        "target_pct": 0.8975,
        "dd_soft": 0.08,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "stop_loss": 0.055,
        "take_profit": 0.075,
    },
    {
        "case_name": "amt9p5_best_sl06_tp065",
        "signal_file": DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv",
        "target_pct": 0.8975,
        "dd_soft": 0.08,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "stop_loss": 0.06,
        "take_profit": 0.065,
    },
    {
        "case_name": "amt9p5_best_sl06_tp07",
        "signal_file": DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv",
        "target_pct": 0.8975,
        "dd_soft": 0.08,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "stop_loss": 0.06,
        "take_profit": 0.07,
    },
    {
        "case_name": "amt9p5_best_sl06_tp075",
        "signal_file": DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv",
        "target_pct": 0.8975,
        "dd_soft": 0.08,
        "dd_hard": 0.115,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
        "stop_loss": 0.06,
        "take_profit": 0.075,
    },
]

SLICES = [
    ("full", "2024-06-05 09:00:00", "2026-06-23 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-23 15:30:00"),
    ("recent60", "2026-03-24 09:00:00", "2026-06-23 15:30:00"),
    ("late20250701", "2025-07-01 09:00:00", "2026-06-23 15:30:00"),
    ("late20251009", "2025-10-09 09:00:00", "2026-06-23 15:30:00"),
    ("late20260105", "2026-01-05 09:00:00", "2026-06-23 15:30:00"),
]


def _write_rows(path: Path, rows: list[dict]) -> None:
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


def _run(case: dict, slice_name: str, start: str, end: str) -> dict:
    log_file = REPORT_DIR / "logs" / f"{case['case_name']}_{slice_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(filt.BASE_ENV)
        env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(case["dd_soft"])
        env["GM_EQUITY_DD_HARD_TRIGGER"] = str(case["dd_hard"])
        env["GM_EQUITY_DD_SOFT_SCALE"] = str(case["dd_soft_scale"])
        env["GM_EQUITY_DD_HARD_SCALE"] = str(case["dd_hard_scale"])
        command = [
            str(filt.JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(filt.STRATEGY_DIR),
            "--signal-file", str(case["signal_file"]),
            "--log-file", str(log_file),
            "--max-positions", "1",
            "--holding-days", "2",
            "--max-holding-days", "3",
            "--target-position-pct", str(case["target_pct"]),
            "--score-db", str(filt.SCORE_DB),
            "--score-table", filt.SCORE_TABLE,
            "--market-db", str(filt.MARKET_DB),
            "--backtest-start", start,
            "--backtest-end", end,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0.0015",
            "--stop-loss-pct", str(case["stop_loss"]),
            "--take-profit-pct", str(case["take_profit"]),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        indicator = _extract_indicator(log_file)
    return {
        "case_name": case["case_name"],
        "target_pct": case["target_pct"],
        "dd_soft": case["dd_soft"],
        "dd_hard": case["dd_hard"],
        "dd_soft_scale": case["dd_soft_scale"],
        "dd_hard_scale": case["dd_hard_scale"],
        "stop_loss": case["stop_loss"],
        "take_profit": case["take_profit"],
        "slice": slice_name,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail = []
    summary = []
    for case in CASES:
        case_rows = []
        for slice_name, start, end in SLICES:
            row = _run(case, slice_name, start, end)
            detail.append(row)
            case_rows.append(row)
        by_slice = {row["slice"]: row for row in case_rows}
        summary.append(
            {
                "case_name": case["case_name"],
                "stop_loss": case["stop_loss"],
                "take_profit": case["take_profit"],
                "target_pct": case["target_pct"],
                "dd_soft": case["dd_soft"],
                "dd_hard": case["dd_hard"],
                "dd_soft_scale": case["dd_soft_scale"],
                "dd_hard_scale": case["dd_hard_scale"],
                "full_annual": by_slice["full"]["annual"],
                "full_sharpe": by_slice["full"]["sharpe"],
                "full_max_drawdown": by_slice["full"]["max_drawdown"],
                "recent120_annual": by_slice["recent120"]["annual"],
                "recent60_annual": by_slice["recent60"]["annual"],
                "late20250701_annual": by_slice["late20250701"]["annual"],
                "late20251009_annual": by_slice["late20251009"]["annual"],
                "late20260105_annual": by_slice["late20260105"]["annual"],
            }
        )
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    _write_rows(REPORT_DIR / "detail.csv", detail)
    _write_rows(REPORT_DIR / "summary.csv", summary)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
