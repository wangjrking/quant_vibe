from __future__ import annotations

import csv
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_signal_state_rule_scan_20260622"
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
CORE_LOG = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_current_l5_execution_refine_20260622"
    / "logs"
    / "current_repro.log"
)
TEN_SIGNAL = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "core10d_original_signal_score_cut_20260621"
    / "signals"
    / "drop_gt0p5_all.csv"
)
TEN_LOG = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_effective_target_refine_20260622"
    / "logs"
    / "t0p23_h6_mh6_sync1.log"
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


def _parse_nav_returns(log_path: Path) -> dict[str, dict]:
    pattern = re.compile(
        r"EXPOSURE\s+(\d{8})\s+post_buy\s+invested_pct=([0-9.]+)\s+active_positions=(\d+)\s+market_value=([0-9.]+)\s+nav=([0-9.]+)"
    )
    rows = []
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="ignore")):
        rows.append(
            {
                "date": match.group(1),
                "invested_pct": float(match.group(2)),
                "active_positions": int(match.group(3)),
                "market_value": float(match.group(4)),
                "nav": float(match.group(5)),
            }
        )
    rows.sort(key=lambda row: row["date"])
    out = {}
    prev_nav = None
    for row in rows:
        nav = row["nav"]
        row["nav_return"] = None if prev_nav in (None, 0) else nav / prev_nav - 1.0
        prev_nav = nav
        out[row["date"]] = row
    return out


def _signal_date_by_buy_date(rows: list[dict]) -> dict[str, str]:
    mapping = {}
    for row in rows:
        buy_date = str(row.get("buy_date") or "")
        signal_date = str(row.get("signal_date") or "")
        if buy_date and signal_date:
            mapping[buy_date] = max(mapping.get(buy_date, ""), signal_date)
    return mapping


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _signal_features(rows: list[dict], prefix: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("signal_date") or "")].append(row)
    features = {}
    for date, day_rows in grouped.items():
        targets = [_to_float(row.get("target_pct")) for row in day_rows]
        preds = [_to_float(row.get("pred_prob")) for row in day_rows]
        pred_5d = [_to_float(row.get("pred_5d")) for row in day_rows]
        pred_10d = [_to_float(row.get("pred_10d")) for row in day_rows]
        gaps = [_to_float(row.get("pred_gap")) for row in day_rows]
        amount = [_to_float(row.get("amount")) for row in day_rows]
        turnover = [_to_float(row.get("turnover_rate")) for row in day_rows]
        total_mv = [_to_float(row.get("total_mv")) for row in day_rows]
        features[date] = {
            f"{prefix}_count": len(day_rows),
            f"{prefix}_target_sum": sum(value for value in targets if value is not None),
            f"{prefix}_pred_avg": _mean(preds),
            f"{prefix}_pred_max": max([value for value in preds if value is not None], default=None),
            f"{prefix}_pred_min": min([value for value in preds if value is not None], default=None),
            f"{prefix}_pred_std": _std(preds),
            f"{prefix}_pred5_avg": _mean(pred_5d),
            f"{prefix}_pred10_avg": _mean(pred_10d),
            f"{prefix}_gap_avg": _mean(gaps),
            f"{prefix}_gap_max": max([value for value in gaps if value is not None], default=None),
            f"{prefix}_gap_min": min([value for value in gaps if value is not None], default=None),
            f"{prefix}_amount_avg": _mean(amount),
            f"{prefix}_turnover_avg": _mean(turnover),
            f"{prefix}_mv_avg": _mean(total_mv),
        }
    return features


def _market_features(dates: set[str]) -> dict[str, dict]:
    if not dates:
        return {}
    conn = sqlite3.connect(MARKET_DB)
    rows = []
    ordered_dates = sorted(dates)
    for offset in range(0, len(ordered_dates), 80):
        chunk = ordered_dates[offset : offset + 80]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            conn.execute(
                f"""
                SELECT
                    trade_date,
                    AVG(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio,
                    AVG(pct_chg) AS avg_pct_chg,
                    AVG(turnover_rate) AS avg_turnover,
                    AVG(amount) AS avg_amount,
                    AVG(CASE WHEN limit_times IS NOT NULL AND limit_times != '' AND CAST(limit_times AS REAL) > 0 THEN 1.0 ELSE 0.0 END) AS limit_ratio,
                    MAX(index_2000_close) AS index_2000_close,
                    MAX(index_2000_open) AS index_2000_open
                FROM STOCK_DAILY_DATA
                WHERE trade_date IN ({placeholders})
                GROUP BY trade_date
                """,
                chunk,
            ).fetchall()
        )
    prev_rows = conn.execute(
        """
        SELECT trade_date, MAX(index_2000_close) AS index_2000_close
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= '20240501' AND trade_date <= '20260618'
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchall()
    conn.close()
    prev_close_by_date = {}
    last_close = None
    for date, close_price in prev_rows:
        prev_close_by_date[str(date)] = last_close
        close_price = _to_float(close_price)
        if close_price is not None:
            last_close = close_price
    features = {}
    for row in rows:
        date, up_ratio, avg_pct_chg, avg_turnover, avg_amount, limit_ratio, idx_close, idx_open = row
        idx_close = _to_float(idx_close)
        idx_open = _to_float(idx_open)
        prev_close = prev_close_by_date.get(str(date))
        features[str(date)] = {
            "mkt_up_ratio": _to_float(up_ratio),
            "mkt_avg_pct_chg": _to_float(avg_pct_chg),
            "mkt_avg_turnover": _to_float(avg_turnover),
            "mkt_avg_amount": _to_float(avg_amount),
            "mkt_limit_ratio": _to_float(limit_ratio),
            "mkt_index_intraday": idx_close / idx_open - 1.0 if idx_close is not None and idx_open not in (None, 0) else None,
            "mkt_index_cc": idx_close / prev_close - 1.0 if idx_close is not None and prev_close not in (None, 0) else None,
        }
    return features


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


def _metric(returns: list[float]) -> dict:
    returns = [value for value in returns if value is not None]
    if not returns:
        return {"n": 0, "mean": None, "std": None, "ann_ret_proxy": None, "sharpe_proxy": None}
    mean = sum(returns) / len(returns)
    std = _std(returns) or 0.0
    return {
        "n": len(returns),
        "mean": mean,
        "std": std,
        "ann_ret_proxy": mean * 242,
        "sharpe_proxy": mean / std * math.sqrt(242) if std > 0 else None,
    }


def _scan_rules(panel: list[dict], return_key: str, prefix: str) -> list[dict]:
    feature_names = [
        key
        for key in panel[0].keys()
        if key not in {"date", "signal_date", return_key}
        and not key.endswith("_date")
        and isinstance(panel[0].get(key), (int, float, type(None)))
    ]
    rows = []
    base_returns = [row.get(return_key) for row in panel if row.get(return_key) is not None]
    base_metric = _metric(base_returns)
    for feature in feature_names:
        values = sorted({row.get(feature) for row in panel if row.get(feature) is not None})
        if len(values) < 20:
            continue
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = [values[int((len(values) - 1) * q)] for q in qs]
        for threshold in thresholds:
            for op in ["ge", "le"]:
                selected = [
                    row
                    for row in panel
                    if row.get(feature) is not None
                    and row.get(return_key) is not None
                    and ((row[feature] >= threshold) if op == "ge" else (row[feature] <= threshold))
                ]
                if len(selected) < 80:
                    continue
                m = _metric([row.get(return_key) for row in selected])
                coverage = len(selected) / len([row for row in panel if row.get(return_key) is not None])
                rows.append(
                    {
                        "source": prefix,
                        "feature": feature,
                        "op": op,
                        "threshold": threshold,
                        "coverage": coverage,
                        "n": m["n"],
                        "mean": m["mean"],
                        "ann_ret_proxy": m["ann_ret_proxy"],
                        "sharpe_proxy": m["sharpe_proxy"],
                        "base_ann_ret_proxy": base_metric["ann_ret_proxy"],
                        "base_sharpe_proxy": base_metric["sharpe_proxy"],
                    }
                )
    rows.sort(key=lambda row: (row["sharpe_proxy"] or -999, row["ann_ret_proxy"] or -999), reverse=True)
    return rows


def _build_panel(signal_path: Path, log_path: Path, sig_prefix: str) -> list[dict]:
    rows = _load_rows(signal_path)
    nav = _parse_nav_returns(log_path)
    signal_map = _signal_date_by_buy_date(rows)
    sig_features = _signal_features(rows, sig_prefix)
    market = _market_features(set(signal_map.values()))
    panel = []
    for buy_date, nav_row in nav.items():
        signal_date = signal_map.get(buy_date)
        row = {"date": buy_date, "signal_date": signal_date, "nav_return": nav_row.get("nav_return")}
        if signal_date:
            row.update(sig_features.get(signal_date, {}))
            row.update(market.get(signal_date, {}))
        row.update({f"nav_{key}": value for key, value in nav_row.items() if key not in {"date", "nav_return"}})
        panel.append(row)
    return panel


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    core_panel = _build_panel(CORE_SIGNAL, CORE_LOG, "core")
    ten_panel = _build_panel(TEN_SIGNAL, TEN_LOG, "ten")
    _write_rows(REPORT_DIR / "core_panel.csv", core_panel)
    _write_rows(REPORT_DIR / "ten_panel.csv", ten_panel)
    core_rules = _scan_rules(core_panel, "nav_return", "core_l5")
    ten_rules = _scan_rules(ten_panel, "nav_return", "ten10d")
    _write_rows(REPORT_DIR / "core_rule_scan.csv", core_rules)
    _write_rows(REPORT_DIR / "ten_rule_scan.csv", ten_rules)
    print("core top")
    for row in core_rules[:10]:
        print(row)
    print("ten top")
    for row in ten_rules[:10]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
