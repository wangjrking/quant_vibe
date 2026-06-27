from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_forward_quality_20260622"
)
SIGNAL_FILE = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_best_v20260621"
    / "signals"
    / "backtest_signals_best_noncal_avgpred205_rw2121201919.csv"
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


def _load_signals() -> list[dict]:
    with SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_fusion_features(keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    if not keys:
        return {}
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    features = {}
    try:
        dates = sorted({date for date, _stock in keys})
        for start in range(0, len(dates), 100):
            chunk = dates[start : start + 100]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT trade_date, stock_code, pred_5d, pred_10d, rank_5d, rank_10d,
                       amount, turnover_rate, total_mv, atr_qfq, close
                FROM fusion_rank_base
                WHERE trade_date IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                key = (str(row["trade_date"]), str(row["stock_code"]))
                if key not in keys:
                    continue
                close = _to_float(row["close"])
                atr = _to_float(row["atr_qfq"])
                pred_5d = _to_float(row["pred_5d"])
                pred_10d = _to_float(row["pred_10d"])
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


def _load_market() -> tuple[list[str], dict[tuple[str, str], float]]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, open
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20240601' AND trade_date <= '20260630'
              AND open IS NOT NULL AND open > 0
            """
        ).fetchall()
    finally:
        conn.close()
    dates = sorted({str(row[0]) for row in rows})
    opens = {(str(row[0]), str(row[1])): float(row[2]) for row in rows}
    return dates, opens


def _forward_return(
    trade_dates: list[str],
    opens: dict[tuple[str, str], float],
    buy_date: str,
    stock_code: str,
    horizon: int,
) -> float | None:
    if buy_date not in trade_dates:
        return None
    start_index = trade_dates.index(buy_date)
    end_index = start_index + int(horizon)
    if end_index >= len(trade_dates):
        return None
    buy_open = opens.get((buy_date, stock_code))
    sell_open = opens.get((trade_dates[end_index], stock_code))
    if buy_open is None or sell_open is None or buy_open <= 0:
        return None
    return sell_open / buy_open - 1.0


def _bin_numeric(value: float | None, bins: list[tuple[str, float | None, float | None]]) -> str:
    if value is None:
        return "missing"
    for label, lower, upper in bins:
        if lower is not None and value < lower:
            continue
        if upper is not None and value >= upper:
            continue
        return label
    return "other"


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _win_rate(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(1 for value in values if value > 0) / len(values) if values else None


def _aggregate(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key) or "missing"), []).append(row)
    out = []
    for group, items in sorted(groups.items()):
        returns_5 = [_to_float(row.get("ret_5d")) for row in items]
        returns_7 = [_to_float(row.get("ret_7d")) for row in items]
        returns_10 = [_to_float(row.get("ret_10d")) for row in items]
        out.append(
            {
                "group_key": group_key,
                "group": group,
                "count": len(items),
                "ret_5d_mean": _mean(returns_5),
                "ret_7d_mean": _mean(returns_7),
                "ret_10d_mean": _mean(returns_10),
                "ret_7d_win_rate": _win_rate(returns_7),
                "avg_pred_prob": _mean([_to_float(row.get("pred_prob")) for row in items]),
                "avg_pred_5d": _mean([_to_float(row.get("pred_5d")) for row in items]),
                "avg_pred_10d": _mean([_to_float(row.get("pred_10d")) for row in items]),
                "avg_atr_ratio": _mean([_to_float(row.get("atr_ratio")) for row in items]),
                "avg_total_mv": _mean([_to_float(row.get("total_mv")) for row in items]),
                "avg_turnover_rate": _mean([_to_float(row.get("turnover_rate")) for row in items]),
            }
        )
    return out


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    signals = _load_signals()
    keys = {(row["signal_date"], row["stock_code"]) for row in signals}
    features = _load_fusion_features(keys)
    trade_dates, opens = _load_market()
    enriched = []
    for row in signals:
        key = (row["signal_date"], row["stock_code"])
        item = dict(row)
        item.update(features.get(key, {}))
        for horizon in [1, 3, 5, 7, 10]:
            item[f"ret_{horizon}d"] = _forward_return(trade_dates, opens, row["buy_date"], row["stock_code"], horizon)
        item["rank_bin"] = str(int(float(row.get("rank") or 0)))
        item["pred_prob_bin"] = _bin_numeric(
            _to_float(row.get("pred_prob")),
            [("lt1p98", None, 1.98), ("1p98_2p0", 1.98, 2.0), ("2p0_2p5", 2.0, 2.5), ("ge2p5", 2.5, None)],
        )
        item["atr_bin"] = _bin_numeric(
            _to_float(item.get("atr_ratio")),
            [("lt4pct", None, 0.04), ("4_6pct", 0.04, 0.06), ("6_8pct", 0.06, 0.08), ("ge8pct", 0.08, None)],
        )
        item["mv_bin"] = _bin_numeric(
            _to_float(item.get("total_mv")),
            [("lt50b", None, 500000.0), ("50_100b", 500000.0, 1000000.0), ("100_200b", 1000000.0, 2000000.0), ("ge200b", 2000000.0, None)],
        )
        item["amount_bin"] = _bin_numeric(
            _to_float(item.get("amount")),
            [("lt20k", None, 20000.0), ("20_50k", 20000.0, 50000.0), ("50_100k", 50000.0, 100000.0), ("ge100k", 100000.0, None)],
        )
        item["turnover_bin"] = _bin_numeric(
            _to_float(item.get("turnover_rate")),
            [("lt1", None, 1.0), ("1_3", 1.0, 3.0), ("3_6", 3.0, 6.0), ("ge6", 6.0, None)],
        )
        item["gap_bin"] = _bin_numeric(
            _to_float(item.get("pred_gap")),
            [("lt0p02", None, 0.02), ("0p02_0p05", 0.02, 0.05), ("0p05_0p10", 0.05, 0.10), ("ge0p10", 0.10, None)],
        )
        item["rank10_bin"] = _bin_numeric(
            _to_float(item.get("rank_10d")),
            [("lt0p2", None, 0.2), ("0p2_0p5", 0.2, 0.5), ("0p5_0p8", 0.5, 0.8), ("ge0p8", 0.8, None)],
        )
        item["rank5_bin"] = _bin_numeric(
            _to_float(item.get("rank_5d")),
            [("lt0p2", None, 0.2), ("0p2_0p5", 0.2, 0.5), ("0p5_0p8", 0.5, 0.8), ("ge0p8", 0.8, None)],
        )
        enriched.append(item)
    _write_rows(REPORT_DIR / "signal_forward_rows.csv", enriched)
    aggregates = []
    for group_key in [
        "rank_bin",
        "pred_prob_bin",
        "atr_bin",
        "mv_bin",
        "amount_bin",
        "turnover_bin",
        "gap_bin",
        "rank10_bin",
        "rank5_bin",
    ]:
        aggregates.extend(_aggregate(enriched, group_key))
    _write_rows(REPORT_DIR / "signal_forward_group_summary.csv", aggregates)
    summary = {
        "signal_count": len(enriched),
        "source_signal_file": str(SIGNAL_FILE),
        "note": "本地前向收益只用于筛选候选，正式绩效仍以掘金回测为准。",
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
