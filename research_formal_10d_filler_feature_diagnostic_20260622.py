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
    / "formal_10d_filler_feature_diagnostic_20260622"
)
CORE_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_gap_v20260622"
    / "signals"
    / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
)
TEN_SIGNAL = (
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
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _load_features(keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    out = {}
    dates = sorted({date for date, _stock in keys})
    try:
        for start in range(0, len(dates), 100):
            chunk = dates[start : start + 100]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_5d, pred_10d, rank_5d, rank_10d,
                       amount, turnover_rate, total_mv, atr_qfq, close, limit_times
                FROM fusion_rank_base
                WHERE trade_date IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                key = (str(row["trade_date"]), str(row["stock_code"]))
                if key not in keys:
                    continue
                pred_5d = _to_float(row["pred_5d"])
                pred_10d = _to_float(row["pred_10d"])
                close = _to_float(row["close"])
                atr = _to_float(row["atr_qfq"])
                out[key] = {
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
    return out


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


def _enrich_returns(rows: list[dict], holding_days: int) -> list[dict]:
    dates, open_map = _load_market()
    date_index = {date: index for index, date in enumerate(dates)}
    keys = {(str(row.get("signal_date")), _stock_code(row)) for row in rows}
    features = _load_features(keys)
    enriched = []
    for row in rows:
        signal_date = str(row.get("signal_date"))
        buy_date = str(row.get("buy_date"))
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


def _rule_pass(row: dict, rule: dict) -> bool:
    for key, condition in rule.items():
        value = _to_float(row.get(key))
        if value is None:
            return False
        if "min" in condition and value < condition["min"]:
            return False
        if "max" in condition and value > condition["max"]:
            return False
    return True


def _score_rule(name: str, rows: list[dict], rule: dict) -> dict:
    selected = [row for row in rows if _rule_pass(row, rule)]
    returns = [_to_float(row.get("open_ret")) for row in selected]
    returns = [value for value in returns if value is not None]
    if not returns:
        return {"name": name, "count": 0}
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
    std = math.sqrt(variance)
    win_rate = sum(1 for value in returns if value > 0) / len(returns)
    sharpe_proxy = mean / std * math.sqrt(252 / 5) if std > 0 else None
    return {
        "name": name,
        "rule": rule,
        "count": len(returns),
        "days": len({row.get("signal_date") for row in selected}),
        "mean_open_ret_5d": mean,
        "std_open_ret_5d": std,
        "win_rate": win_rate,
        "sharpe_proxy": sharpe_proxy,
        "min_ret": min(returns),
        "max_ret": max(returns),
    }


def main() -> int:
    core_rows = _load_rows(CORE_SIGNAL)
    ten_rows = _load_rows(TEN_SIGNAL)
    core_keys = {(str(row.get("signal_date")), _stock_code(row)) for row in core_rows}
    core_day_counts: dict[str, int] = {}
    for row in core_rows:
        date = str(row.get("signal_date"))
        core_day_counts[date] = core_day_counts.get(date, 0) + 1
    filler_rows = [
        row
        for row in ten_rows
        if (str(row.get("signal_date")), _stock_code(row)) not in core_keys
    ]
    enriched = _enrich_returns(filler_rows, holding_days=5)
    for row in enriched:
        row["core_count"] = core_day_counts.get(str(row.get("signal_date")), 0)
    rules = []
    for core_max in [0, 1, 2, 3, 4]:
        for amount_min in [0, 10000, 20000, 50000]:
            for mv_max in [50000, 100000, 150000, 200000, 400000]:
                rules.append(
                    (
                        f"core_le{core_max}_amt{amount_min}_mv{mv_max}",
                        {
                            "core_count": {"max": core_max},
                            "amount": {"min": amount_min},
                            "total_mv": {"max": mv_max},
                        },
                    )
                )
        for turn_min, turn_max in [(0.3, 3), (0.5, 5), (1, 5), (1, 8), (3, 8)]:
            rules.append(
                (
                    f"core_le{core_max}_turn{turn_min}_{turn_max}",
                    {
                        "core_count": {"max": core_max},
                        "turnover_rate": {"min": turn_min, "max": turn_max},
                    },
                )
            )
        for gap_max in [0.02, 0.03, 0.05, 0.08]:
            rules.append(
                (
                    f"core_le{core_max}_gap_le{gap_max}",
                    {"core_count": {"max": core_max}, "pred_gap": {"max": gap_max}},
                )
            )
        for atr_max in [0.04, 0.06, 0.08, 0.10]:
            rules.append(
                (
                    f"core_le{core_max}_atr_le{atr_max}",
                    {"core_count": {"max": core_max}, "atr_ratio": {"max": atr_max}},
                )
            )
    scored = [_score_rule(name, enriched, rule) for name, rule in rules]
    scored = [row for row in scored if int(row.get("count") or 0) >= 80]
    scored.sort(
        key=lambda row: (
            _to_float(row.get("sharpe_proxy"), -999),
            _to_float(row.get("mean_open_ret_5d"), -999),
            int(row.get("days") or 0),
        ),
        reverse=True,
    )
    _write_rows(REPORT_DIR / "filler_rule_diagnostic.csv", scored)
    _write_rows(REPORT_DIR / "filler_enriched_sample.csv", enriched[:5000])
    print("enriched", len(enriched))
    for row in scored[:20]:
        print(
            row["name"],
            "count",
            row["count"],
            "days",
            row["days"],
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
