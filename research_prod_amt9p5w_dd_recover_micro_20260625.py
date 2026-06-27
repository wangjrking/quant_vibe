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
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_amt9p5w_dd_recover_micro_20260625"

RECOVERS = [0.02, 0.025, 0.03, 0.035, 0.04]
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


def _run(dd_recover: float, slice_name: str, start: str, end: str) -> dict:
    case_name = f"ddrecover_{str(dd_recover).replace('.', 'p')}"
    log_file = REPORT_DIR / "logs" / f"{case_name}_{slice_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(filt.BASE_ENV)
        env["GM_EQUITY_DD_SOFT_TRIGGER"] = "0.08"
        env["GM_EQUITY_DD_HARD_TRIGGER"] = "0.115"
        env["GM_EQUITY_DD_RECOVER_TRIGGER"] = str(dd_recover)
        env["GM_EQUITY_DD_SOFT_SCALE"] = "0.75"
        env["GM_EQUITY_DD_HARD_SCALE"] = "0.55"
        command = [
            str(filt.JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(filt.STRATEGY_DIR),
            "--signal-file", str(DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv"),
            "--log-file", str(log_file),
            "--max-positions", "1",
            "--holding-days", "2",
            "--max-holding-days", "3",
            "--target-position-pct", "0.8975",
            "--score-db", str(filt.SCORE_DB),
            "--score-table", filt.SCORE_TABLE,
            "--market-db", str(filt.MARKET_DB),
            "--backtest-start", start,
            "--backtest-end", end,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0.0015",
            "--stop-loss-pct", "0.06",
            "--take-profit-pct", "0.07",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        indicator = _extract_indicator(log_file)
    return {
        "case_name": case_name,
        "dd_recover": dd_recover,
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
    for dd_recover in RECOVERS:
        case_rows = []
        for slice_name, start, end in SLICES:
            row = _run(dd_recover, slice_name, start, end)
            detail.append(row)
            case_rows.append(row)
        by_slice = {row["slice"]: row for row in case_rows}
        summary.append(
            {
                "dd_recover": dd_recover,
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
