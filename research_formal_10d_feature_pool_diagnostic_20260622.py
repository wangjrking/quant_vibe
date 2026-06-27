from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_feature_pool_diagnostic_20260622"
)
SIGNAL_FILE = (
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
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"


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


def _load_market() -> tuple[list[str], dict[tuple[str, str], float]]:
    conn = sqlite3.connect(MARKET_DB)
    rows = conn.execute(
        """
        SELECT trade_date, stock_code, open
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= '20240601' AND trade_date <= '20260630'
        """
    ).fetchall()
    conn.close()
    dates = sorted({str(row[0]) for row in rows})
    open_map = {(str(row[0]), str(row[1])): _to_float(row[2]) for row in rows}
    return dates, open_map


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
                       amount, turnover_rate, total_mv, atr_qfq, close, limit_times
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
                    "limit_times": row["limit_times"],
                }
    finally:
        conn.close()
    return features


def _enrich(holding_days: int = 5) -> list[dict]:
    rows = _load_rows(SIGNAL_FILE)
    dates, open_map = _load_market()
    date_index = {date: index for index, date in enumerate(dates)}
    features = _load_features(rows)
    enriched = []
    for row in rows:
        signal_date = str(row.get("signal_date") or "")
        buy_date = str(row.get("buy_date") or "")
        stock = _stock_code(row)
        buy_index = date_index.get(buy_date)
        if buy_index is None or buy_index + holding_days >= len(dates):
            continue
        exit_date = dates[buy_index + holding_days]
        buy_open = open_map.get((buy_date, stock))
        exit_open = open_map.get((exit_date, stock))
        if buy_open in (None, 0) or exit_open in (None, 0):
            continue
        out = dict(row)
        out["stock_code"] = stock
        out["exit_date"] = exit_date
        out["open_ret"] = exit_open / buy_open - 1.0
        out.update(features.get((signal_date, stock), {}))
        enriched.append(out)
    return enriched


def _passes(row: dict, rule: dict) -> bool:
    for field, condition in rule.items():
        value = _to_float(row.get(field))
        if value is None:
            return False
        if "min" in condition and value < condition["min"]:
            return False
        if "max" in condition and value > condition["max"]:
            return False
    return True


def _score_rule(name: str, rows: list[dict], rule: dict) -> dict:
    selected = [row for row in rows if _passes(row, rule)]
    returns = [_to_float(row.get("open_ret")) for row in selected]
    returns = [value for value in returns if value is not None]
    days = {row.get("signal_date") for row in selected}
    if not returns:
        return {"name": name, "count": 0, "days": 0, "rule": rule}
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
    std = math.sqrt(variance)
    win_rate = sum(1 for value in returns if value > 0) / len(returns)
    proxy = mean / std * math.sqrt(252 / 5) if std > 0 else None
    counts_by_day: dict[str, int] = {}
    for row in selected:
        date = str(row.get("signal_date"))
        counts_by_day[date] = counts_by_day.get(date, 0) + 1
    return {
        "name": name,
        "rule": rule,
        "count": len(returns),
        "days": len(days),
        "avg_count_per_day": len(returns) / len(days) if days else None,
        "days_ge3": sum(1 for value in counts_by_day.values() if value >= 3),
        "mean_open_ret_5d": mean,
        "std_open_ret_5d": std,
        "win_rate": win_rate,
        "sharpe_proxy": proxy,
        "min_ret": min(returns),
        "max_ret": max(returns),
    }


def main() -> int:
    rows = _enrich(holding_days=5)
    rules: list[tuple[str, dict]] = [("all", {})]
    for amount_min in [10000, 20000, 50000, 80000, 100000]:
        rules.append((f"amount_ge{amount_min}", {"amount": {"min": amount_min}}))
        for mv_max in [100000, 150000, 200000, 300000, 500000]:
            rules.append(
                (
                    f"amount_ge{amount_min}_mv_le{mv_max}",
                    {"amount": {"min": amount_min}, "total_mv": {"max": mv_max}},
                )
            )
    for mv_min, mv_max in [(0, 100000), (0, 150000), (50000, 200000), (100000, 300000), (150000, 500000)]:
        rules.append((f"mv_{mv_min}_{mv_max}", {"total_mv": {"min": mv_min, "max": mv_max}}))
    for turn_min, turn_max in [(0.3, 3), (0.5, 5), (1, 5), (1, 8), (2, 8), (3, 12)]:
        rules.append((f"turn_{turn_min}_{turn_max}", {"turnover_rate": {"min": turn_min, "max": turn_max}}))
    for gap_max in [0.015, 0.02, 0.03, 0.05, 0.08, 0.12]:
        rules.append((f"gap_le{gap_max}", {"pred_gap": {"max": gap_max}}))
        rules.append(
            (
                f"gap_le{gap_max}_amount_ge50000",
                {"pred_gap": {"max": gap_max}, "amount": {"min": 50000}},
            )
        )
    for pred5_min in [-0.02, 0.0, 0.02, 0.05, 0.08]:
        rules.append((f"pred5_ge{pred5_min}", {"pred_5d": {"min": pred5_min}}))
        rules.append(
            (
                f"pred5_ge{pred5_min}_amount_ge50000",
                {"pred_5d": {"min": pred5_min}, "amount": {"min": 50000}},
            )
        )
    for atr_max in [0.04, 0.06, 0.08, 0.10, 0.12]:
        rules.append((f"atr_le{atr_max}", {"atr_ratio": {"max": atr_max}}))
    combo_rules = []
    for amount_min in [20000, 50000, 80000]:
        for mv_max in [150000, 200000, 300000]:
            for gap_max in [0.02, 0.03, 0.05]:
                combo_rules.append(
                    (
                        f"amt{amount_min}_mv{mv_max}_gap{gap_max}",
                        {"amount": {"min": amount_min}, "total_mv": {"max": mv_max}, "pred_gap": {"max": gap_max}},
                    )
                )
    rules.extend(combo_rules)
    scored = [_score_rule(name, rows, rule) for name, rule in rules]
    scored_all = sorted(
        scored,
        key=lambda row: (
            _to_float(row.get("sharpe_proxy"), -999),
            _to_float(row.get("mean_open_ret_5d"), -999),
            int(row.get("count") or 0),
        ),
        reverse=True,
    )
    scored_high_coverage = [
        row for row in scored if int(row.get("count") or 0) >= 1000 and int(row.get("days") or 0) >= 300
    ]
    scored_high_coverage.sort(
        key=lambda row: (
            _to_float(row.get("sharpe_proxy"), -999),
            _to_float(row.get("mean_open_ret_5d"), -999),
            int(row.get("days_ge3") or 0),
        ),
        reverse=True,
    )
    _write_rows(REPORT_DIR / "ten_signal_feature_rules_all.csv", scored_all)
    _write_rows(REPORT_DIR / "ten_signal_feature_rules_high_coverage.csv", scored_high_coverage)
    _write_rows(REPORT_DIR / "ten_signal_enriched_sample.csv", rows[:5000])
    print("enriched", len(rows))
    for row in scored_high_coverage[:20]:
        print(
            row["name"],
            "count",
            row["count"],
            "days",
            row["days"],
            "avg_day",
            row["avg_count_per_day"],
            "mean",
            row["mean_open_ret_5d"],
            "proxy",
            row["sharpe_proxy"],
            "win",
            row["win_rate"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
