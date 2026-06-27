from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_feature_pool_gm_20260622"
)
SOURCE_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "core10d_original_signal_score_cut_20260621"
    / "signals"
    / "drop_gt0p5_all.csv"
)
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
    / "fusion_5d10d.db"
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


def _variant(
    name: str,
    rule: dict,
    target_pct: float,
    holding_days: int,
    max_holding_days: int,
    max_positions: int = 5,
    weak_target_pct: float | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "rule": rule,
        "target_pct": target_pct,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "max_positions": max_positions,
        "weak_target_pct": weak_target_pct,
        "extra_env": extra_env or {"GM_SYNC_POSITIONS": "1"},
    }


RULES = {
    "mv150_500": {"total_mv": {"min": 150000.0, "max": 500000.0}},
    "gap05": {"pred_gap": {"max": 0.05}},
    "mv100_300": {"total_mv": {"min": 100000.0, "max": 300000.0}},
    "amt20_mv200_gap05": {
        "amount": {"min": 20000.0},
        "total_mv": {"max": 200000.0},
        "pred_gap": {"max": 0.05},
    },
    "turn03_3": {"turnover_rate": {"min": 0.3, "max": 3.0}},
}

VARIANTS: list[dict] = []
for rule_name, rule in RULES.items():
    for target in [0.23, 0.25, 0.28]:
        for hold, max_hold in [(5, 6), (6, 6)]:
            VARIANTS.append(
                _variant(
                    f"{rule_name}_t{str(target).replace('.', 'p')}_h{hold}_mh{max_hold}",
                    rule,
                    target,
                    hold,
                    max_hold,
                )
            )

for weak_target in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]:
    for strong_target in [0.25, 0.28]:
        for hold, max_hold in [(5, 6), (6, 6)]:
            VARIANTS.append(
                _variant(
                    f"gap05_dynamic_s{str(strong_target).replace('.', 'p')}_w{str(weak_target).replace('.', 'p')}_h{hold}_mh{max_hold}",
                    RULES["gap05"],
                    strong_target,
                    hold,
                    max_hold,
                    max_positions=5,
                    weak_target_pct=weak_target,
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


def _stock_code(row: dict) -> str:
    stock = str(row.get("stock_code") or "")
    if stock:
        return stock
    symbol = str(row.get("symbol") or "")
    if symbol.startswith("SZSE."):
        return symbol[5:] + ".SZ"
    if symbol.startswith("SHSE."):
        return symbol[5:] + ".SH"
    return symbol


def _load_features(rows: list[dict]) -> dict[tuple[str, str], dict]:
    keys = {(str(row.get("signal_date") or ""), _stock_code(row)) for row in rows}
    dates = sorted({date for date, _stock in keys if date})
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    features = {}
    try:
        for start in range(0, len(dates), 100):
            chunk = dates[start : start + 100]
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_5d, pred_10d, rank_5d, rank_10d,
                       amount, turnover_rate, total_mv, atr_qfq, close
                FROM fusion_rank_base
                WHERE trade_date IN ({placeholders})
                """,
                chunk,
            ):
                key = (str(row["trade_date"]), str(row["stock_code"]))
                if key not in keys:
                    continue
                pred_5d = _to_float(row["pred_5d"])
                pred_10d = _to_float(row["pred_10d"])
                close = _to_float(row["close"])
                atr = _to_float(row["atr_qfq"])
                features[key] = {
                    "pred_5d": pred_5d,
                    "pred_10d": pred_10d,
                    "rank_5d": _to_float(row["rank_5d"]),
                    "rank_10d": _to_float(row["rank_10d"]),
                    "amount": _to_float(row["amount"]),
                    "turnover_rate": _to_float(row["turnover_rate"]),
                    "total_mv": _to_float(row["total_mv"]),
                    "atr_ratio": atr / close if atr is not None and close and close > 0 else None,
                    "pred_gap": abs(pred_10d - pred_5d) if pred_10d is not None and pred_5d is not None else None,
                }
    finally:
        conn.close()
    return features


def _passes(row: dict, rule: dict) -> bool:
    for field, condition in rule.items():
        value = _to_float(row.get(field))
        if value is None:
            return False
        if "min" in condition and value < float(condition["min"]):
            return False
        if "max" in condition and value > float(condition["max"]):
            return False
    return True


def _write_signal(variant: dict, path: Path) -> None:
    rows = _load_rows(SOURCE_SIGNAL)
    features = _load_features(rows)
    fieldnames = list(rows[0].keys())
    for field in ["pred_5d", "pred_10d", "rank_5d", "rank_10d", "amount", "turnover_rate", "total_mv", "atr_ratio", "pred_gap"]:
        if field not in fieldnames:
            fieldnames.append(field)
    if "pool_role" not in fieldnames:
        fieldnames.append("pool_role")
    selected = []
    for row in rows:
        key = (str(row.get("signal_date") or ""), _stock_code(row))
        out = dict(row)
        out.update(features.get(key, {}))
        if _passes(out, variant["rule"]):
            out["target_pct"] = f"{float(variant['target_pct']):.5f}"
            out["pool_role"] = "strong"
        elif variant.get("weak_target_pct") is not None:
            out["target_pct"] = f"{float(variant['weak_target_pct']):.5f}"
            out["pool_role"] = "weak"
        else:
            continue
        out["holding_days"] = str(int(variant["holding_days"]))
        selected.append(out)
    selected.sort(
        key=lambda item: (
            str(item.get("signal_date")),
            0 if item.get("pool_role") == "strong" else 1,
            int(float(item.get("rank") or 999999)),
            str(item.get("stock_code")),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in selected])


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
    counts = {}
    for row in rows:
        date = str(row.get("signal_date") or "")
        counts[date] = counts.get(date, 0) + 1
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_count_per_signal_day": len(rows) / len(counts) if counts else None,
        "days_ge3": sum(1 for value in counts.values() if value >= 3),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _run_backtest(variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
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
        str(signal_file),
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    name_filter = os.environ.get("NAME_FILTER", "").strip()
    selected = [variant for variant in VARIANTS if not name_filter or name_filter in variant["name"]]
    max_variants = int(os.environ.get("MAX_VARIANTS", str(len(selected))))
    selected = selected[:max_variants]
    for index, variant in enumerate(selected, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        _write_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "rule_json": json.dumps(variant["rule"], ensure_ascii=False, sort_keys=True),
            "target_pct": variant["target_pct"],
            "holding_days": variant["holding_days"],
            "max_holding_days": variant["max_holding_days"],
            "max_positions": variant["max_positions"],
            "weak_target_pct": variant.get("weak_target_pct"),
            "extra_env_json": json.dumps(variant["extra_env"], ensure_ascii=False, sort_keys=True),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(_signal_stats(signal_file))
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(selected)}] {variant['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80), reverse=True))
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
