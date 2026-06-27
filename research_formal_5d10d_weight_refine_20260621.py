from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weight_refine_20260621"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_tiered_score_refine_20260621"
    / "tiered_scores.db"
)
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


BASES = [
    {
        "base_name": "p90_f50",
        "signal_file": ROOT
        / "quant"
        / "main"
        / "strategy_library"
        / "exploration"
        / "l5_formal_5d10d_candidate_20260621"
        / "signals"
        / "candidate_signals.csv",
        "score_table": "tier_p90_f50_plim2_pmv150000p0_fmv200000p0",
    },
    {
        "base_name": "p90_f60",
        "signal_file": DATA_DIR
        / "reports"
        / "strategy_agent_model_application_20260621"
        / "formal_5d10d_tiered_score_refine_20260621"
        / "signals"
        / "p90_f60_plim2_mp10_h5_exit1_ser0p95_mhse2_sells1.csv",
        "score_table": "tier_p90_f60_plim2_pmv150000p0_fmv200000p0",
    },
]


WEIGHT_VARIANTS = [
    {"name": "equal098", "mode": "fixed", "primary_pct": 0.098, "fallback_pct": 0.098, "max_target_pct": 0.098},
    {"name": "p13_f0767", "mode": "fixed", "primary_pct": 0.13, "fallback_pct": 0.0766666667, "max_target_pct": 0.13},
    {"name": "p15_f0633", "mode": "fixed", "primary_pct": 0.15, "fallback_pct": 0.0633333333, "max_target_pct": 0.15},
    {"name": "p17_f0500", "mode": "fixed", "primary_pct": 0.17, "fallback_pct": 0.05, "max_target_pct": 0.17},
    {"name": "p14_f0800_t52", "mode": "fixed", "primary_pct": 0.14, "fallback_pct": 0.08, "max_target_pct": 0.14},
    {"name": "p16_f0667_t52", "mode": "fixed", "primary_pct": 0.16, "fallback_pct": 0.0666666667, "max_target_pct": 0.16},
    {"name": "score049", "mode": "score", "daily_total": 0.49, "max_target_pct": 0.16},
    {"name": "score052", "mode": "score", "daily_total": 0.52, "max_target_pct": 0.17},
]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _write_weighted_signal(source: Path, dest: Path, variant: dict) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
        fieldnames = list(rows[0].keys())
    if variant["mode"] == "fixed":
        for row in rows:
            score = float(row.get("pred_prob") or 0.0)
            row["target_pct"] = f"{(variant['primary_pct'] if score >= 2.0 else variant['fallback_pct']):.10f}"
    elif variant["mode"] == "score":
        by_date: dict[str, list[dict]] = {}
        for row in rows:
            by_date.setdefault(str(row.get("buy_date") or ""), []).append(row)
        for day_rows in by_date.values():
            scores = [max(float(row.get("pred_prob") or 0.0), 0.0) for row in day_rows]
            total_score = sum(scores) or 1.0
            for row, score in zip(day_rows, scores):
                row["target_pct"] = f"{(variant['daily_total'] * score / total_score):.10f}"
    else:
        raise ValueError(f"unsupported mode: {variant['mode']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _run_backtest(base: dict, variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.95",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_STOP_LOSS_PCT": "0.08",
            "GM_TAKE_PROFIT_PCT": "none",
            "GM_LIGHT_STOP_LOSS_PCT": "none",
            "GM_SCORE_EXIT_RANK": "none",
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
        "10",
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--target-position-pct",
        str(variant["max_target_pct"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        str(base["score_table"]),
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


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    grid = [(base, variant) for base in BASES for variant in WEIGHT_VARIANTS]
    for index, (base, variant) in enumerate(grid, start=1):
        slug = f"{base['base_name']}_{variant['name']}"
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            _write_weighted_signal(Path(base["signal_file"]), signal_file, variant)
        returncode = _run_backtest(base, variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            "base_name": base["base_name"],
            **variant,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "score_db": str(SCORE_DB),
            "score_table": base["score_table"],
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
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
