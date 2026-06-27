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
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap02_concentrated_20260622"
SOURCE_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_10d_gap_strong_weak_fill_20260622" / "signals" / "gap02_s0p32_w0p18_h6_mh6.csv"
SCORE_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
}


def _variant(name: str, **kwargs) -> dict:
    cfg = {
        "name": name,
        "top_k": 1,
        "max_positions": 1,
        "target": 0.80,
        "holding_days": 3,
        "max_holding_days": 5,
        "strong_only": False,
        "prefer_strong": True,
        "max_gap": None,
        "env": dict(BASE_ENV),
    }
    cfg.update(kwargs)
    return cfg


VARIANTS = [
    _variant("top1_t80_h3", top_k=1, max_positions=1, target=0.80, holding_days=3, max_holding_days=5),
    _variant("top1_t95_h3", top_k=1, max_positions=1, target=0.95, holding_days=3, max_holding_days=5),
    _variant("top1_t80_h5", top_k=1, max_positions=1, target=0.80, holding_days=5, max_holding_days=6),
    _variant("top1_strong_t80_h3", top_k=1, max_positions=1, target=0.80, holding_days=3, max_holding_days=5, strong_only=True),
    _variant("top2_t45_h3", top_k=2, max_positions=2, target=0.45, holding_days=3, max_holding_days=5),
    _variant("top2_t50_h3", top_k=2, max_positions=2, target=0.50, holding_days=3, max_holding_days=5),
    _variant("top2_t45_h5", top_k=2, max_positions=2, target=0.45, holding_days=5, max_holding_days=6),
    _variant("top3_t32_h3", top_k=3, max_positions=3, target=0.32, holding_days=3, max_holding_days=5),
    _variant("top3_t32_h5", top_k=3, max_positions=3, target=0.32, holding_days=5, max_holding_days=6),
    _variant("top2_gaplt006_t50_h3", top_k=2, max_positions=2, target=0.50, holding_days=3, max_holding_days=5, max_gap=0.06),
]


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


def _passes(row: dict, cfg: dict) -> bool:
    if cfg.get("strong_only") and str(row.get("pool_role") or "") != "strong":
        return False
    if cfg.get("max_gap") is not None and _to_float(row.get("pred_gap"), 999.0) > float(cfg["max_gap"]):
        return False
    return True


def _sort_key(row: dict, cfg: dict):
    role_rank = 0 if str(row.get("pool_role") or "") == "strong" else 1
    rank = int(float(row.get("rank") or 999))
    pred = _to_float(row.get("pred_prob"), -999.0)
    gap = _to_float(row.get("pred_gap"), 999.0)
    if cfg.get("prefer_strong"):
        return (role_rank, rank, gap, -pred)
    return (rank, gap, -pred)


def _write_signal(cfg: dict, signal_file: Path) -> None:
    rows = _load_rows(SOURCE_SIGNAL)
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        if _passes(row, cfg):
            by_date.setdefault(str(row.get("buy_date")), []).append(dict(row))
    out: list[dict] = []
    fields = list(rows[0].keys())
    for date, day_rows in sorted(by_date.items()):
        selected = sorted(day_rows, key=lambda row: _sort_key(row, cfg))[: int(cfg["top_k"])]
        for row in selected:
            row["target_pct"] = f"{float(cfg['target']):.5f}"
            row["holding_days"] = str(int(cfg["holding_days"]))
            out.append(row)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in out])


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
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    cmd = [
        str(JUEJIN_PYTHON), str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir", str(STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", str(int(cfg["max_positions"])),
        "--holding-days", str(int(cfg["holding_days"])),
        "--max-holding-days", str(int(cfg["max_holding_days"])),
        "--target-position-pct", str(float(cfg["target"])),
        "--score-db", str(SCORE_DB),
        "--score-table", SCORE_TABLE,
        "--market-db", str(MARKET_DB),
        "--backtest-start", "2024-06-05 09:00:00",
        "--backtest-end", "2026-06-18 15:30:00",
        "--backtest-adjust", "none",
        "--backtest-initial-cash", "600000",
        "--backtest-slippage-ratio", "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


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
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, cfg in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        _write_signal(cfg, signal_file)
        returncode = _run_backtest(cfg, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}")
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
