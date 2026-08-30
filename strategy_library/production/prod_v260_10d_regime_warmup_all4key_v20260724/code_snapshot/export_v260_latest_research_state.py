# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260


ROOT = Path(__file__).resolve().parents[2]
L2 = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
PROTOCOL = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v258_position_warmup_v260_20260723/preregistered_protocol.json"
)
L4_MANIFEST = (
    ROOT
    / "quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json"
)


def append_execution_date(arrays: dict[str, np.ndarray], buy_date: str) -> None:
    old_dates = arrays["dates"].astype(str)
    old_count = len(old_dates)
    arrays["dates"] = np.append(old_dates, buy_date).astype("U8")
    for key, value in list(arrays.items()):
        if key in {"dates", "stocks"} or not isinstance(value, np.ndarray):
            continue
        if value.ndim < 1 or value.shape[0] != old_count:
            continue
        shape = (1, *value.shape[1:])
        if np.issubdtype(value.dtype, np.bool_):
            extra = np.zeros(shape, dtype=value.dtype)
        elif np.issubdtype(value.dtype, np.integer):
            extra = np.full(shape, -1, dtype=value.dtype)
        else:
            extra = np.full(shape, np.nan, dtype=value.dtype)
        arrays[key] = np.concatenate([value, extra], axis=0)


def truncate_to_manifest(arrays: dict[str, np.ndarray], signal_date: str) -> None:
    dates = arrays["dates"].astype(str)
    keep = int(np.searchsorted(dates, signal_date, side="right"))
    if keep <= 0 or str(dates[keep - 1]) != signal_date:
        raise RuntimeError(f"manifest signal date absent from active arrays: {signal_date}")
    for key, value in list(arrays.items()):
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == len(dates)
        ):
            arrays[key] = value[:keep]


def active_holdings(actions: pd.DataFrame) -> pd.DataFrame:
    holdings: dict[str, dict] = {}
    for row in actions.itertuples(index=False):
        stock = str(row.stock_code)
        if str(row.action).upper() == "BUY":
            holdings[stock] = {
                "stock_code": stock,
                "entry_signal_date": str(row.signal_date),
                "entry_buy_date": str(row.buy_date),
                "entry_target_pct": float(row.target_pct),
                "entry_open_raw": float(row.execution_open_raw),
            }
        else:
            holdings.pop(stock, None)
    return pd.DataFrame(holdings.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="V260最新研究信号与理论持仓")
    parser.add_argument("--buy-date", required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(L4_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("approval_status") != "approved_for_l5"
        or manifest.get("source_type") != "duckdb_table"
    ):
        raise RuntimeError("active 10D manifest is not approved DuckDB formal")
    arrays = v260.v258.v252.fresh_arrays()
    buy_date = str(args.buy_date)
    manifest_max_date = str(manifest["max_trade_date"])
    table_max_date = str(arrays["dates"][-1])
    with duckdb.connect(L2.as_posix(), read_only=True) as con:
        signal_date = str(
            con.execute(
                "SELECT MAX(trade_date) FROM STOCK_DAILY_DATA WHERE trade_date < ?",
                [buy_date],
            ).fetchone()[0]
        )
    if signal_date > manifest_max_date:
        raise RuntimeError(
            f"previous trade date exceeds approved manifest: {signal_date} > {manifest_max_date}"
        )
    truncate_to_manifest(arrays, signal_date)
    with duckdb.connect(L2.as_posix(), read_only=True) as con:
        next_date = con.execute(
            "SELECT MIN(trade_date) FROM STOCK_DAILY_DATA WHERE trade_date > ?",
            [signal_date],
        ).fetchone()[0]
        if str(next_date) != buy_date:
            raise RuntimeError(
                f"buy date mismatch: expected next L2 date {next_date}, got {buy_date}"
            )
        latest_rows = con.execute(
            "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date = ?",
            [buy_date],
        ).fetchone()[0]
        if int(latest_rows) <= 0:
            raise RuntimeError(f"L2 buy-day market unavailable: {buy_date}")

    original_count = len(arrays["dates"])
    if not np.isfinite(arrays["buy_open"][-1]).any():
        raise RuntimeError("latest signal row has no next-day raw open")
    append_execution_date(arrays, buy_date)
    if len(arrays["dates"]) != original_count + 1:
        raise RuntimeError("execution-date append failed")

    definition = v260.definition_for(protocol, 50)
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    daily, actions = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        signal_date,
        record_actions=True,
    )

    out = (
        ROOT
        / f"quant/data_file/reports/strategy_agent_v260_latest_signal_{buy_date}"
    )
    out.mkdir(parents=True, exist_ok=True)
    actions_path = out / "research_actions_through_buy_date.csv"
    holdings_path = out / "theoretical_holdings_after_open.csv"
    today_path = out / "today_open_actions.csv"
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")

    today = actions[actions["buy_date"].astype(str) == buy_date].copy()
    holdings = active_holdings(actions)
    stocks = arrays["stocks"].astype(str)
    stock_index = {stock: idx for idx, stock in enumerate(stocks)}
    t = len(arrays["dates"]) - 2
    valid_scores = score[t][np.isfinite(score[t])]

    with duckdb.connect(L2.as_posix(), read_only=True) as con:
        names = con.execute(
            """
            SELECT stock_code, any_value(name) AS name, any_value(market) AS market
            FROM STOCK_DAILY_DATA
            WHERE trade_date = ?
            GROUP BY stock_code
            """,
            [buy_date],
        ).fetchdf()
    names = names.drop_duplicates("stock_code").set_index("stock_code")

    def enrich(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["name"] = frame["stock_code"].map(
            lambda stock: str(names.loc[stock, "name"]) if stock in names.index else ""
        )
        frame["market"] = frame["stock_code"].map(
            lambda stock: str(names.loc[stock, "market"]) if stock in names.index else ""
        )
        frame["strategy_score"] = frame["stock_code"].map(
            lambda stock: float(score[t, stock_index[stock]])
            if stock in stock_index and np.isfinite(score[t, stock_index[stock]])
            else np.nan
        )
        frame["score_rank"] = frame["strategy_score"].map(
            lambda value: int(np.sum(valid_scores > value) + 1)
            if np.isfinite(value)
            else None
        )
        frame["score_denominator"] = len(valid_scores)
        return frame

    today = enrich(today)
    holdings = enrich(holdings)
    today.to_csv(today_path, index=False, encoding="utf-8-sig")
    holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")

    result = {
        "status": "research_only_not_production_signal",
        "strategy_id": "v260_748bdb6520590687",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "l4_latest_trade_date": signal_date,
        "l4_manifest_max_trade_date": manifest_max_date,
        "l4_table_observed_max_trade_date": table_max_date,
        "manifest_table_date_drift_detected": table_max_date != manifest_max_date,
        "l2_buy_day_rows": int(latest_rows),
        "today_actions": today.to_dict("records"),
        "theoretical_holdings_after_open": holdings.to_dict("records"),
        "production_registry_current": "",
        "actual_broker_holdings_included": False,
        "paths": {
            "actions": str(actions_path),
            "today": str(today_path),
            "holdings": str(holdings_path),
        },
    }
    (out / "research_state.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
