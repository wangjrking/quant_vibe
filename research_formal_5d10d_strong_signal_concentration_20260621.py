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
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_weak_day_pool_replace_20260621"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_strong_signal_concentration_20260621"
)
BASE_SIGNAL = SOURCE_DIR / "signals" / "weak_f60_liq.csv"
SCORE_DB = SOURCE_DIR / "weak_day_pool_scores.db"
SCORE_TABLE = "weak_f60_liq_primary_count_le1_prank_10d90_5d10_frank_10d60_5d40_20000p0_0p5"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


VARIANTS = [
    {"name": "base_10x10925", "mode": "equal", "rank_max": 5, "max_positions": 10, "holding_days": 7, "target_pct": 0.10925},
    {"name": "top3_mp5_t20_h7", "mode": "equal", "rank_max": 3, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "top2_mp5_t25_h7", "mode": "equal", "rank_max": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.25},
    {"name": "top1_mp3_t33_h7", "mode": "equal", "rank_max": 1, "max_positions": 3, "holding_days": 7, "target_pct": 0.33},
    {"name": "primary_top3_mp5_t20_h7", "mode": "equal", "rank_max": 3, "pred_min": 2.0, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "primary_top2_mp5_t25_h7", "mode": "equal", "rank_max": 2, "pred_min": 2.0, "max_positions": 5, "holding_days": 7, "target_pct": 0.25},
    {"name": "strongday_top5_mp5_t20_h7", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "strongday_top3_mp5_t20_h7", "mode": "equal", "rank_max": 3, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "avg225_top5_mp5_t20_h7", "mode": "equal", "rank_max": 5, "avg_pred_min": 2.25, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "avg225_top3_mp5_t20_h7", "mode": "equal", "rank_max": 3, "avg_pred_min": 2.25, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "top3_mp5_t20_h5", "mode": "equal", "rank_max": 3, "max_positions": 5, "holding_days": 5, "target_pct": 0.20},
    {"name": "primary_top3_mp5_t20_h5", "mode": "equal", "rank_max": 3, "pred_min": 2.0, "max_positions": 5, "holding_days": 5, "target_pct": 0.20},
    {"name": "strongday_top3_mp5_t20_h5", "mode": "equal", "rank_max": 3, "primary_count_min": 2, "max_positions": 5, "holding_days": 5, "target_pct": 0.20},
    {"name": "top3_mp6_t17_h7", "mode": "equal", "rank_max": 3, "max_positions": 6, "holding_days": 7, "target_pct": 0.17},
    {"name": "top4_mp6_t17_h7", "mode": "equal", "rank_max": 4, "max_positions": 6, "holding_days": 7, "target_pct": 0.17},
    {"name": "strongday_top5_mp5_t18_h7", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.18},
    {"name": "strongday_top5_mp5_t22_h7", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top5_mp5_t20_h6", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "max_positions": 5, "holding_days": 6, "target_pct": 0.20},
    {"name": "strongday_top5_mp5_t20_h8", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "max_positions": 5, "holding_days": 8, "target_pct": 0.20},
    {"name": "strongday_top4_mp5_t20_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "strongday_top4_mp5_t21_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.21},
    {"name": "strongday_top4_mp5_t215_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.215},
    {"name": "strongday_top4_mp5_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_mp5_t225_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.225},
    {"name": "strongday_top4_r4p1985_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "rank4_pred_min": 1.985, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_r4p1990_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "rank4_pred_min": 1.990, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_r3p1990_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "rank3_pred_min": 1.990, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_avg2490_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "avg_selected_pred_min": 2.490, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_avg2493_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "avg_selected_pred_min": 2.493, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_gap1010_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "gap_top_tail_max": 1.010, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_gap1005_t22_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "gap_top_tail_max": 1.005, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_t22_no_dec", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "exclude_months": ["12"], "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_t22_no_dec_janweak", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "exclude_months": ["12"], "month_idx20dd_min": {"01": -0.08}, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_t22_no_dec_junweak", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "exclude_months": ["12"], "month_idx20dd_min": {"06": -0.06}, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top4_t225_no_dec", "mode": "equal", "rank_max": 4, "primary_count_min": 2, "exclude_months": ["12"], "max_positions": 5, "holding_days": 7, "target_pct": 0.225},
    {"name": "strongday_top5_t20_no_dec", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "strongday_top5_t21_no_dec", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "max_positions": 5, "holding_days": 7, "target_pct": 0.21},
    {"name": "strongday_top5_t22_no_dec", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top5_t225_no_dec", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "max_positions": 5, "holding_days": 7, "target_pct": 0.225},
    {"name": "strongday_top5_t22_no_dec_janweak", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "month_idx20dd_min": {"01": -0.08}, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top5_t22_no_dec_junweak", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "month_idx20dd_min": {"06": -0.06}, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top5_t22_no_dec_jan_junweak", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "month_idx20dd_min": {"01": -0.08, "06": -0.06}, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "strongday_top5_t225_no_dec_janweak", "mode": "equal", "rank_max": 5, "primary_count_min": 2, "exclude_months": ["12"], "month_idx20dd_min": {"01": -0.08}, "max_positions": 5, "holding_days": 7, "target_pct": 0.225},
    {"name": "strongday_top3_mp5_t25_h7", "mode": "equal", "rank_max": 3, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.25},
    {"name": "strongday_top3_mp5_t30_h7", "mode": "equal", "rank_max": 3, "primary_count_min": 2, "max_positions": 5, "holding_days": 7, "target_pct": 0.30},
    {"name": "strongday3_top5_mp5_t20_h7", "mode": "equal", "rank_max": 5, "primary_count_min": 3, "max_positions": 5, "holding_days": 7, "target_pct": 0.20},
    {"name": "strongday3_top5_mp5_t25_h7", "mode": "equal", "rank_max": 5, "primary_count_min": 3, "max_positions": 5, "holding_days": 7, "target_pct": 0.25},
    {"name": "strongday3_top4_mp5_t25_h7", "mode": "equal", "rank_max": 4, "primary_count_min": 3, "max_positions": 5, "holding_days": 7, "target_pct": 0.25},
    {"name": "avg230_top5_mp5_t22_h7", "mode": "equal", "rank_max": 5, "avg_pred_min": 2.30, "max_positions": 5, "holding_days": 7, "target_pct": 0.22},
    {"name": "avg235_top5_mp5_t25_h7", "mode": "equal", "rank_max": 5, "avg_pred_min": 2.35, "max_positions": 5, "holding_days": 7, "target_pct": 0.25},
    {
        "name": "strongday_top4_t22_eqdd_loose",
        "mode": "equal",
        "rank_max": 4,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.15",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.25",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.95",
            "GM_EQUITY_DD_HARD_SCALE": "0.85",
        },
    },
    {
        "name": "strongday_top4_t22_eqdd_mild",
        "mode": "equal",
        "rank_max": 4,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.90",
            "GM_EQUITY_DD_HARD_SCALE": "0.75",
        },
    },
    {
        "name": "strongday_top4_t22_idx_loose",
        "mode": "equal",
        "rank_max": 4,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_BUY_SCALE": "0.85",
        },
    },
    {
        "name": "strongday_top4_t22_idx_mild",
        "mode": "equal",
        "rank_max": 4,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.025",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.025",
            "GM_INDEX_RISK_BUY_SCALE": "0.85",
        },
    },
    {
        "name": "strongday_top5_t20_idx_loose",
        "mode": "equal",
        "rank_max": 5,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.20,
        "extra_env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_BUY_SCALE": "0.85",
        },
    },
    {
        "name": "strongday_top5_t22_idx_loose",
        "mode": "equal",
        "rank_max": 5,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.035",
            "GM_INDEX_RISK_BUY_SCALE": "0.85",
        },
    },
    {
        "name": "strongday_top4_t22_exit105",
        "mode": "equal",
        "rank_max": 4,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "1.05",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
            "GM_MAX_DAILY_SELLS": "1",
        },
    },
    {
        "name": "strongday_top4_t22_sl06",
        "mode": "equal",
        "rank_max": 4,
        "primary_count_min": 2,
        "max_positions": 5,
        "holding_days": 7,
        "target_pct": 0.22,
        "extra_env": {
            "GM_STOP_LOSS_PCT": "0.06",
        },
    },
]


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_features(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date") or ""), []).append(row)
    features: dict[str, dict] = {}
    for signal_date, items in grouped.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in items]
        primary_count = sum(1 for value in preds if value >= 2.0)
        features[signal_date] = {
            "avg_pred": sum(preds) / len(preds) if preds else None,
            "primary_count": primary_count,
            "by_rank": {int(float(row.get("rank") or 999999)): (_to_float(row.get("pred_prob"), 0.0) or 0.0) for row in items},
        }
    return features


def _load_index_features() -> dict[str, dict]:
    import sqlite3

    conn = sqlite3.connect(MARKET_DB)
    rows = conn.execute(
        """
        SELECT trade_date, MAX(index_2000_close) AS close_price
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= '20240501' AND trade_date <= '20260630'
          AND index_2000_close IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchall()
    conn.close()
    features: dict[str, dict] = {}
    closes: list[tuple[str, float]] = []
    for trade_date, close_price in rows:
        trade_date = str(trade_date)
        close_price = float(close_price)
        closes.append((trade_date, close_price))
        idx = len(closes) - 1
        if idx >= 20:
            high_20 = max(value for _date, value in closes[idx - 20 : idx + 1])
            features[trade_date] = {"idx_dd_20": close_price / high_20 - 1.0}
        else:
            features[trade_date] = {"idx_dd_20": None}
    return features


def _include_row(row: dict, variant: dict, features: dict[str, dict], index_features: dict[str, dict]) -> bool:
    rank = int(float(row.get("rank") or 999999))
    pred = _to_float(row.get("pred_prob"), 0.0) or 0.0
    signal_date = str(row.get("signal_date") or "")
    signal_features = features.get(signal_date, {})
    month = signal_date[4:6]
    if month in set(variant.get("exclude_months") or []):
        return False
    month_idx20dd_min = variant.get("month_idx20dd_min") or {}
    if month in month_idx20dd_min:
        idx_dd_20 = _to_float((index_features.get(signal_date) or {}).get("idx_dd_20"))
        if idx_dd_20 is not None and idx_dd_20 < float(month_idx20dd_min[month]):
            return False
    if rank > int(variant.get("rank_max", 999999)):
        return False
    if "pred_min" in variant and pred < float(variant["pred_min"]):
        return False
    if "primary_count_min" in variant and int(signal_features.get("primary_count") or 0) < int(variant["primary_count_min"]):
        return False
    if "avg_pred_min" in variant:
        avg_pred = _to_float(signal_features.get("avg_pred"))
        if avg_pred is None or avg_pred < float(variant["avg_pred_min"]):
            return False
    by_rank = signal_features.get("by_rank") or {}
    if "rank4_pred_min" in variant and (by_rank.get(4) is None or by_rank.get(4) < float(variant["rank4_pred_min"])):
        return False
    if "rank3_pred_min" in variant and (by_rank.get(3) is None or by_rank.get(3) < float(variant["rank3_pred_min"])):
        return False
    if "avg_selected_pred_min" in variant:
        selected = [value for rank_key, value in by_rank.items() if rank_key <= int(variant.get("rank_max", 999999))]
        if not selected or sum(selected) / len(selected) < float(variant["avg_selected_pred_min"]):
            return False
    if "gap_top_tail_max" in variant:
        selected = [value for rank_key, value in by_rank.items() if rank_key <= int(variant.get("rank_max", 999999))]
        if not selected or max(selected) - min(selected) > float(variant["gap_top_tail_max"]):
            return False
    return True


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    with BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0].keys()) if rows else []
    features = _signal_features(rows)
    index_features = _load_index_features()
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not _include_row(row, variant, features, index_features):
                continue
            row["target_pct"] = f"{float(variant['target_pct']):.5f}"
            row["holding_days"] = str(int(variant["holding_days"]))
            row["score_exit_entry_ratio"] = "1.0"
            row["min_holding_days_before_score_exit"] = "3"
            writer.writerow(row)


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
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
        "min_signal_target_pct": min(targets) if targets else None,
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
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
        }
    )
    env.update({str(k): str(v) for k, v in (variant.get("extra_env") or {}).items()})
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
        str(int(variant["holding_days"])),
        "--target-position-pct",
        f"{float(variant['target_pct']):.5f}",
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


def _eligibility(variant: dict) -> tuple[bool, str]:
    if variant.get("exclude_months") or variant.get("month_idx20dd_min"):
        return False, "calendar_data_snooping_rejected"
    return True, ""


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_variant_signal(variant, signal_file)
        returncode = _run_backtest(variant, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        eligible, disqualification_reason = _eligibility(variant)
        result = {
            "name": variant["name"],
            "eligible": eligible,
            "disqualification_reason": disqualification_reason,
            "mode": variant["mode"],
            "rank_max": variant.get("rank_max"),
            "pred_min": variant.get("pred_min"),
            "primary_count_min": variant.get("primary_count_min"),
            "avg_pred_min": variant.get("avg_pred_min"),
            "max_positions": variant["max_positions"],
            "holding_days": variant["holding_days"],
            "target_pct": variant["target_pct"],
            "extra_env": ";".join(f"{k}={v}" for k, v in sorted((variant.get("extra_env") or {}).items())),
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
        eligible_results = [row for row in results if str(row.get("eligible")).lower() == "true"]
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=_sort_by_annual, reverse=True))
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_by_sharpe, reverse=True))
        _write_rows(REPORT_DIR / "summary_eligible_by_annual.csv", sorted(eligible_results, key=_sort_by_annual, reverse=True))
        _write_rows(REPORT_DIR / "summary_eligible_by_sharpe.csv", sorted(eligible_results, key=_sort_by_sharpe, reverse=True))
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted([item for item in results if str(item.get("eligible")).lower() == "true"], key=_sort_by_sharpe, reverse=True)
        if float(row.get("annual") or -999.0) >= 2.0
        and float(row.get("sharpe") or -999.0) >= 3.0
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
