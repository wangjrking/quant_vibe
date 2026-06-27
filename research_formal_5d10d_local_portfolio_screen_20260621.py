from __future__ import annotations

import csv
import itertools
import math
import sqlite3
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_local_portfolio_screen_20260621"
BASE_SIGNAL = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_weak_day_pool_replace_20260621" / "signals" / "weak_f60_liq.csv"
FUSION_DB = DATA_DIR / "reports" / "strategy_agent_model_application_20260621" / "formal_5d10d_l5_candidate_grid_20260621" / "fusion_5d10d.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"


def _to_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return default if math.isnan(out) or math.isinf(out) else out


def _load_signals() -> list[dict]:
    with BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_fusion() -> dict[tuple[str, str], dict]:
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT trade_date, stock_code, pred_5d, pred_10d, rank_5d, rank_10d,
               amount, turnover_rate, total_mv, atr_qfq, close
        FROM fusion_rank_base
        """
    ).fetchall()
    conn.close()
    out = {}
    for row in rows:
        item = dict(row)
        close = _to_float(item.get("close"))
        atr = _to_float(item.get("atr_qfq"))
        item["atr_ratio"] = atr / close if atr is not None and close else None
        out[(str(row["trade_date"]), str(row["stock_code"]))] = item
    return out


def _load_open_prices(stock_codes: set[str]) -> tuple[list[str], dict[tuple[str, str], float]]:
    conn = sqlite3.connect(MARKET_DB)
    dates = [
        str(row[0])
        for row in conn.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date").fetchall()
    ]
    rows = []
    codes = sorted(stock_codes)
    for offset in range(0, len(codes), 400):
        chunk = codes[offset : offset + 400]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            conn.execute(
                f"SELECT trade_date, stock_code, open FROM STOCK_DAILY_DATA WHERE stock_code IN ({placeholders})",
                chunk,
            ).fetchall()
        )
    conn.close()
    prices = {}
    for trade_date, stock_code, open_price in rows:
        value = _to_float(open_price)
        if value is not None and value > 0:
            prices[(str(stock_code), str(trade_date))] = value
    return dates, prices


def _signal_day_features(rows: list[dict]) -> dict[str, dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["signal_date"]), []).append(row)
    out = {}
    for signal_date, items in grouped.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in items]
        out[signal_date] = {
            "avg_pred": sum(preds) / len(preds) if preds else 0.0,
            "primary_count": sum(1 for value in preds if value >= 2.0),
        }
    return out


def _passes(row: dict, feature: dict, day_feature: dict, rule: dict) -> bool:
    rank = int(float(row.get("rank") or 999999))
    if rank > rule["rank_max"]:
        return False
    if int(day_feature.get("primary_count") or 0) < rule["primary_count_min"]:
        return False
    if float(day_feature.get("avg_pred") or 0.0) < rule["post_avg_min"]:
        return False
    checks = [
        ("max_total_mv", "total_mv", lambda actual, threshold: actual <= threshold),
        ("min_total_mv", "total_mv", lambda actual, threshold: actual >= threshold),
        ("min_turnover", "turnover_rate", lambda actual, threshold: actual >= threshold),
        ("max_turnover", "turnover_rate", lambda actual, threshold: actual <= threshold),
        ("min_rank_10d", "rank_10d", lambda actual, threshold: actual >= threshold),
        ("min_rank_5d", "rank_5d", lambda actual, threshold: actual >= threshold),
        ("min_pred_5d", "pred_5d", lambda actual, threshold: actual >= threshold),
        ("max_pred_5d", "pred_5d", lambda actual, threshold: actual <= threshold),
        ("max_atr_ratio", "atr_ratio", lambda actual, threshold: actual <= threshold),
    ]
    for rule_key, feature_key, predicate in checks:
        threshold = rule.get(rule_key)
        if threshold is None:
            continue
        actual = _to_float(feature.get(feature_key))
        if actual is None or not predicate(actual, float(threshold)):
            return False
    return True


def _annualized(returns: list[float]) -> tuple[float, float, float]:
    if not returns:
        return 0.0, 0.0, 0.0
    nav = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        nav *= 1.0 + ret
        peak = max(peak, nav)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - nav / peak)
    years = len(returns) / 252.0
    annual = nav ** (1.0 / years) - 1.0 if years > 0 and nav > 0 else -1.0
    mean = sum(returns) / len(returns)
    variance = sum((ret - mean) ** 2 for ret in returns) / max(len(returns) - 1, 1)
    sharpe = mean / math.sqrt(variance) * math.sqrt(252.0) if variance > 0 else 0.0
    return annual, sharpe, max_dd


def _simulate(rows_by_buy_date: dict[str, list[dict]], dates: list[str], prices: dict[tuple[str, str], float], rule: dict) -> dict:
    date_index = {date: idx for idx, date in enumerate(dates)}
    positions: dict[str, dict] = {}
    daily_returns = []
    exposures = []
    start_idx = min(date_index.get(date, 10**9) for date in rows_by_buy_date)
    end_idx = max(date_index.get(date, -1) for date in rows_by_buy_date)
    for idx in range(start_idx, min(end_idx + int(rule["holding_days"]) + 2, len(dates) - 1)):
        today = dates[idx]
        next_day = dates[idx + 1]
        for code in list(positions):
            pos = positions[code]
            if idx - pos["entry_idx"] >= int(rule["holding_days"]):
                del positions[code]
        rows = rows_by_buy_date.get(today, [])
        open_slots = max(int(rule["max_positions"]) - len(positions), 0)
        for row in rows:
            if open_slots <= 0:
                break
            code = row["stock_code"]
            if code in positions:
                continue
            if (code, today) not in prices:
                continue
            rank = int(float(row["rank"]))
            target = float(rule["rank_targets"].get(rank, rule["target_pct"]))
            positions[code] = {"entry_idx": idx, "target": target}
            open_slots -= 1
        day_ret = 0.0
        exposure = 0.0
        for code, pos in positions.items():
            p0 = prices.get((code, today))
            p1 = prices.get((code, next_day))
            weight = float(pos["target"])
            exposure += weight
            if p0 and p1:
                day_ret += min(weight, 1.0) * (p1 / p0 - 1.0)
        exposure = min(exposure, 1.0)
        daily_returns.append(day_ret)
        exposures.append(exposure)
    annual, sharpe, max_dd = _annualized(daily_returns)
    return {
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_max_drawdown": max_dd,
        "local_avg_exposure": sum(exposures) / len(exposures) if exposures else 0.0,
        "local_ge80": sum(1 for value in exposures if value >= 0.8) / len(exposures) if exposures else 0.0,
        "local_days": len(daily_returns),
    }


def _rule_name(rule: dict) -> str:
    parts = [
        f"r{rule['rank_max']}",
        f"h{rule['holding_days']}",
        f"mp{rule['max_positions']}",
        f"avg{str(rule['post_avg_min']).replace('.', 'p')}",
    ]
    for key in ["max_total_mv", "min_rank_10d", "min_rank_5d", "min_pred_5d", "max_pred_5d", "min_turnover", "max_turnover", "max_atr_ratio"]:
        if rule.get(key) is not None:
            parts.append(f"{key}{str(rule[key]).replace('.', 'p')}")
    return "_".join(parts)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_signals()
    fusion = _load_fusion()
    dates, prices = _load_open_prices({str(row["stock_code"]) for row in rows})
    day_features = _signal_day_features(rows)
    rank_targets_by_max = {
        3: {1: 0.40, 2: 0.34, 3: 0.24},
        4: {1: 0.30, 2: 0.26, 3: 0.23, 4: 0.20},
        5: {1: 0.25, 2: 0.23, 3: 0.21, 4: 0.18, 5: 0.15},
    }
    rules = []
    for rank_max, holding_days, post_avg_min, max_total_mv, min_rank_10d, min_rank_5d, pred5_band, turnover_band in itertools.product(
        [3, 4, 5],
        [5, 7, 10, 15],
        [1.90, 2.00, 2.05, 2.10],
        [None, 110000, 130000, 150000, 180000],
        [None, 0.99, 0.995, 0.9966],
        [None, 0.99, 0.9955],
        [None, "mid"],
        [None, "high", "low"],
    ):
        active_filters = sum(value is not None for value in [max_total_mv, min_rank_10d, min_rank_5d, pred5_band, turnover_band])
        if active_filters > 2:
            continue
        rule = {
            "rank_max": rank_max,
            "max_positions": rank_max,
            "primary_count_min": 1,
            "holding_days": holding_days,
            "post_avg_min": post_avg_min,
            "target_pct": rank_targets_by_max[rank_max][rank_max],
            "rank_targets": rank_targets_by_max[rank_max],
            "max_total_mv": max_total_mv,
            "min_rank_10d": min_rank_10d,
            "min_rank_5d": min_rank_5d,
            "min_pred_5d": 0.0057 if pred5_band == "mid" else None,
            "max_pred_5d": 0.0345 if pred5_band == "mid" else None,
            "min_turnover": 4.7 if turnover_band == "high" else None,
            "max_turnover": 1.7 if turnover_band == "low" else None,
            "max_atr_ratio": None,
        }
        rules.append(rule)
    results = []
    for rule in rules:
        filtered_by_buy_date: dict[str, list[dict]] = {}
        for row in rows:
            feature = fusion.get((str(row["signal_date"]), str(row["stock_code"]))) or {}
            if not feature:
                continue
            if not _passes(row, feature, day_features.get(str(row["signal_date"]), {}), rule):
                continue
            filtered_by_buy_date.setdefault(str(row["buy_date"]), []).append(row)
        for items in filtered_by_buy_date.values():
            items.sort(key=lambda item: int(float(item.get("rank") or 999999)))
        signal_count = sum(len(items) for items in filtered_by_buy_date.values())
        if signal_count < 300:
            continue
        metrics = _simulate(filtered_by_buy_date, dates, prices, rule)
        results.append(
            {
                "name": _rule_name(rule),
                **rule,
                "rank_targets": rule["rank_targets"],
                "signal_count": signal_count,
                "buy_days": len(filtered_by_buy_date),
                **metrics,
            }
        )
    results.sort(key=lambda row: (row["local_annual"], row["local_sharpe"], row["local_avg_exposure"]), reverse=True)
    out = REPORT_DIR / "local_screen_summary.csv"
    fieldnames = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    for row in results[:20]:
        print(row["name"], row["local_annual"], row["local_sharpe"], row["local_avg_exposure"], row["signal_count"])
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
