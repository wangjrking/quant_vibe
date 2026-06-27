from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
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
    / "formal_10d_sell_risk_overlay_20260622"
)
BASE_REPORT = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_effective_target_refine_20260622"
)
SCORE_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
    / "weak_day_pool_scores.db"
)
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-18 15:30:00"


BASES = [
    {
        "base": "t0p23_h6_mh6_sync1",
        "signal_file": BASE_REPORT / "signals" / "t0p23_h6_mh6_sync1.csv",
        "target_pct": 0.23,
        "holding_days": 6,
        "max_holding_days": 6,
        "max_positions": 5,
        "base_env": {"GM_SYNC_POSITIONS": "1"},
    },
    {
        "base": "t0p21_h5_mh6_sync1",
        "signal_file": BASE_REPORT / "signals" / "t0p21_h5_mh6_sync1.csv",
        "target_pct": 0.21,
        "holding_days": 5,
        "max_holding_days": 6,
        "max_positions": 5,
        "base_env": {"GM_SYNC_POSITIONS": "1"},
    },
]


def _variant(base: dict, name: str, extra_env: dict[str, str], *, max_daily_sells: int = 1) -> dict:
    return {
        **base,
        "name": f"{base['base']}__{name}",
        "max_daily_sells": max_daily_sells,
        "extra_env": {**base.get("base_env", {}), **extra_env},
    }


VARIANTS: list[dict] = []
for base in BASES:
    VARIANTS.append(_variant(base, "repro", {}))
    for score_exit in ["0.95", "0.98", "1.00", "1.03"]:
        for min_hold in ["1", "2", "3"]:
            VARIANTS.append(
                _variant(
                    base,
                    f"score_exit{score_exit.replace('.', 'p')}_minh{min_hold}",
                    {
                        "GM_SCORE_EXIT_ENTRY_RATIO": score_exit,
                        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": min_hold,
                    },
                )
            )
    for stop_loss in ["0.04", "0.06", "0.08"]:
        for take_profit in ["none", "0.10", "0.16"]:
            VARIANTS.append(
                _variant(
                    base,
                    f"intraday_sl{stop_loss.replace('.', 'p')}_tp{take_profit.replace('.', 'p')}",
                    {
                        "GM_INTRADAY_RISK_MODE": "1",
                        "GM_INTRADAY_REPLACE_BUY": "0",
                        "GM_STOP_LOSS_PCT": stop_loss,
                        "GM_TAKE_PROFIT_PCT": take_profit,
                    },
                )
            )
    for stop_loss in ["0.04", "0.06"]:
        VARIANTS.append(
            _variant(
                base,
                f"intraday_replace_sl{stop_loss.replace('.', 'p')}",
                {
                    "GM_INTRADAY_RISK_MODE": "1",
                    "GM_INTRADAY_REPLACE_BUY": "1",
                    "GM_INTRADAY_REPLACE_MAX_BUYS": "1",
                    "GM_STOP_LOSS_PCT": stop_loss,
                    "GM_TAKE_PROFIT_PCT": "none",
                },
            )
        )
    for soft_trigger, hard_trigger, soft_scale, hard_scale in [
        ("0.06", "0.12", "0.80", "0.60"),
        ("0.08", "0.16", "0.85", "0.65"),
        ("0.10", "0.20", "0.90", "0.70"),
    ]:
        VARIANTS.append(
            _variant(
                base,
                f"eqdd_s{soft_trigger.replace('.', 'p')}_h{hard_trigger.replace('.', 'p')}",
                {
                    "GM_EQUITY_DD_RISK_MODE": "1",
                    "GM_EQUITY_DD_SOFT_TRIGGER": soft_trigger,
                    "GM_EQUITY_DD_HARD_TRIGGER": hard_trigger,
                    "GM_EQUITY_DD_SOFT_SCALE": soft_scale,
                    "GM_EQUITY_DD_HARD_SCALE": hard_scale,
                },
            )
        )
    for threshold in ["-0.010", "-0.015", "-0.020", "-0.030", "-0.040"]:
        for buy_scale in ["0.0", "0.35", "0.60"]:
            tag = threshold.replace("-", "m").replace(".", "p")
            scale_tag = buy_scale.replace(".", "p")
            VARIANTS.append(
                _variant(
                    base,
                    f"indexrisk_cc{tag}_scale{scale_tag}",
                    {
                        "GM_INDEX_RISK_EXIT_MODE": "1",
                        "GM_INDEX_RISK_CC_THRESHOLD": threshold,
                        "GM_INDEX_RISK_INTRADAY_THRESHOLD": threshold,
                        "GM_INDEX_RISK_BUY_SCALE": buy_scale,
                    },
                )
            )
    for stop_loss, take_profit in [("0.08", "0.16"), ("0.06", "0.16"), ("0.04", "0.10")]:
        for threshold in ["-0.015", "-0.020", "-0.030"]:
            for buy_scale in ["0.35", "0.60"]:
                sl_tag = stop_loss.replace(".", "p")
                tp_tag = take_profit.replace(".", "p")
                th_tag = threshold.replace("-", "m").replace(".", "p")
                scale_tag = buy_scale.replace(".", "p")
                VARIANTS.append(
                    _variant(
                        base,
                        f"intraday_sl{sl_tag}_tp{tp_tag}_indexrisk_cc{th_tag}_scale{scale_tag}",
                        {
                            "GM_INTRADAY_RISK_MODE": "1",
                            "GM_INTRADAY_REPLACE_BUY": "0",
                            "GM_STOP_LOSS_PCT": stop_loss,
                            "GM_TAKE_PROFIT_PCT": take_profit,
                            "GM_INDEX_RISK_EXIT_MODE": "1",
                            "GM_INDEX_RISK_CC_THRESHOLD": threshold,
                            "GM_INDEX_RISK_INTRADAY_THRESHOLD": threshold,
                            "GM_INDEX_RISK_BUY_SCALE": buy_scale,
                        },
                    )
                )
    for up_ratio, avg_pct in [("0.06", "-3.0"), ("0.08", "-3.0"), ("0.10", "-2.5"), ("0.12", "-2.0"), ("0.18", "-1.2")]:
        for buy_scale in ["0.0", "0.35", "0.60"]:
            up_tag = up_ratio.replace(".", "p")
            avg_tag = avg_pct.replace("-", "m").replace(".", "p")
            scale_tag = buy_scale.replace(".", "p")
            VARIANTS.append(
                _variant(
                    base,
                    f"breadth_up{up_tag}_avg{avg_tag}_scale{scale_tag}",
                    {
                        "GM_BREADTH_RISK_EXIT_MODE": "1",
                        "GM_BREADTH_RISK_UP_RATIO_THRESHOLD": up_ratio,
                        "GM_BREADTH_RISK_AVG_PCT_THRESHOLD": avg_pct,
                        "GM_BREADTH_RISK_BUY_SCALE": buy_scale,
                    },
                )
            )
    for up_ratio, avg_pct, stop_loss in [("0.08", "-3.0", "0.08"), ("0.10", "-2.5", "0.08"), ("0.12", "-2.0", "0.06")]:
        for buy_scale in ["0.35", "0.60"]:
            up_tag = up_ratio.replace(".", "p")
            avg_tag = avg_pct.replace("-", "m").replace(".", "p")
            sl_tag = stop_loss.replace(".", "p")
            scale_tag = buy_scale.replace(".", "p")
            VARIANTS.append(
                _variant(
                    base,
                    f"breadth_intraday_up{up_tag}_avg{avg_tag}_sl{sl_tag}_scale{scale_tag}",
                    {
                        "GM_BREADTH_RISK_EXIT_MODE": "1",
                        "GM_BREADTH_RISK_UP_RATIO_THRESHOLD": up_ratio,
                        "GM_BREADTH_RISK_AVG_PCT_THRESHOLD": avg_pct,
                        "GM_BREADTH_RISK_BUY_SCALE": buy_scale,
                        "GM_INTRADAY_RISK_MODE": "1",
                        "GM_INTRADAY_REPLACE_BUY": "0",
                        "GM_STOP_LOSS_PCT": stop_loss,
                        "GM_TAKE_PROFIT_PCT": "none",
                    },
                )
            )


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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


def _signal_stats(path: Path) -> dict:
    rows = _load_rows(path)
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _run_backtest(variant: dict, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
            "GM_MAX_DAILY_SELLS": str(int(variant["max_daily_sells"])),
            "GM_STOP_LOSS_PCT": "0.08",
            "GM_TAKE_PROFIT_PCT": "none",
            "GM_LIGHT_STOP_LOSS_PCT": "none",
            "GM_SCORE_EXIT_RANK": "none",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.99",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.90",
            "GM_EQUITY_DD_HARD_SCALE": "0.70",
        }
    )
    env.update({str(key): str(value) for key, value in variant.get("extra_env", {}).items()})
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(variant["signal_file"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(int(variant["max_positions"])),
        "--holding-days",
        str(int(variant["holding_days"])),
        "--max-holding-days",
        str(int(variant["max_holding_days"])),
        "--target-position-pct",
        str(float(variant["target_pct"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
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


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    name_filter = os.environ.get("NAME_FILTER", "").strip()
    selected = [variant for variant in VARIANTS if not name_filter or name_filter in variant["name"]]
    max_variants = int(os.environ.get("MAX_VARIANTS", str(len(selected))))
    selected = selected[:max_variants]
    for index, variant in enumerate(selected, start=1):
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        returncode = _run_backtest(variant, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "base": variant["base"],
            "target_pct": variant["target_pct"],
            "holding_days": variant["holding_days"],
            "max_holding_days": variant["max_holding_days"],
            "max_daily_sells": variant["max_daily_sells"],
            "extra_env_json": json.dumps(variant["extra_env"], ensure_ascii=False, sort_keys=True),
            "returncode": returncode,
            "signal_file": str(variant["signal_file"]),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(_signal_stats(Path(variant["signal_file"])))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(selected)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(
        REPORT_DIR / "summary_by_objective.csv",
        sorted(
            results,
            key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80),
            reverse=True,
        ),
    )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
