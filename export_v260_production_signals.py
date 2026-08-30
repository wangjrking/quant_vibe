from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_score_deterioration_v174_20260722 as v174
import research_active_l4_v258_position_warmup_v260_20260723 as v260


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant/main"
DATA = ROOT / "quant/data_file"
STRATEGY_ID = "prod_v260_10d_regime_warmup_v20260724"
STRATEGY_DIR = MAIN / "strategy_library/production" / STRATEGY_ID
REGISTRY = MAIN / "strategy_library/registry.json"
L2 = DATA / "production_assets/duckdb/l2_stock_daily_data.duckdb"
FORMAL_MANIFEST = (
    MAIN
    / "config/prediction_manifests/"
    "executable_10d_open_return_l4_formal_20260617.json"
)
PROTOCOL = STRATEGY_DIR / "inputs/preregistered_protocol.json"
FROZEN_ACTIONS = STRATEGY_DIR / "signals/full_history_actions.csv"
CURRENT_ACTIONS = STRATEGY_DIR / "signals/full_history_actions_current.csv"
VERIFIER = STRATEGY_DIR / "code_snapshot/verify_candidate.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def clean_sql(alias: str) -> str:
    return f"""
      coalesce(cast({alias}.ST_TYPE as varchar), '') in ('', '0', '0.0', 'None', 'NONE')
      and upper(coalesce(cast({alias}.ST_TYPE_name as varchar), '')) not like '%ST%'
      and upper(coalesce(cast({alias}.name as varchar), '')) not like 'ST%'
      and upper(coalesce(cast({alias}.name as varchar), '')) not like '*ST%'
      and coalesce(cast({alias}.name as varchar), '') not like '%退%'
    """


def validate_route() -> dict:
    registry = load_json(REGISTRY)
    production = registry.get("production", {})
    if production.get("current") != STRATEGY_ID:
        raise RuntimeError("production.current does not point to V260")
    if production.get("state") != "production_active":
        raise RuntimeError(
            f"production strategy is not active: {production.get('state')}"
        )
    strategies = production.get("strategies", [])
    if len(strategies) != 1 or strategies[0].get("strategy_id") != STRATEGY_ID:
        raise RuntimeError("production strategy registry is not uniquely V260")

    manifest = load_json(FORMAL_MANIFEST)
    if (
        manifest.get("approval_status") != "approved_for_l5"
        or manifest.get("source_type") != "duckdb_table"
    ):
        raise RuntimeError("10D manifest is not approved DuckDB formal")
    serialized = json.dumps(manifest, ensure_ascii=False).lower()
    if "sqlite" in serialized or "odb.db" in serialized or "research_only" in serialized:
        raise RuntimeError("legacy or research-only production input detected")
    if str(manifest.get("max_trade_date")) != "20260723":
        raise RuntimeError("unexpected 10D formal signal date")
    subprocess.run([sys.executable, str(VERIFIER)], cwd=str(ROOT), check=True)
    return manifest


def resolve_formal_source(manifest: dict) -> tuple[Path, str]:
    db_path = (FORMAL_MANIFEST.parent / str(manifest["db_path"])).resolve()
    if db_path.suffix.lower() != ".duckdb" or not db_path.is_file():
        raise RuntimeError("formal 10D DuckDB source is unavailable")
    return db_path, str(manifest["table"])


def build_arrays(manifest: dict) -> dict[str, np.ndarray]:
    prediction_db, prediction_table = resolve_formal_source(manifest)
    signal_date = str(manifest["max_trade_date"])
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{L2.as_posix()}' AS m (READ_ONLY)")
        con.execute(f"ATTACH '{prediction_db.as_posix()}' AS p10 (READ_ONLY)")
        dates = [
            str(row[0])
            for row in con.execute(
                """
                SELECT DISTINCT trade_date
                FROM m.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND ?
                ORDER BY trade_date
                """,
                [signal_date],
            ).fetchall()
        ]
        stocks = [
            str(row[0])
            for row in con.execute(
                f"""
                SELECT DISTINCT stock_code
                FROM p10."{prediction_table}"
                WHERE stock_code NOT LIKE '%.BJ'
                ORDER BY stock_code
                """
            ).fetchall()
        ]
        if not dates or dates[-1] != signal_date:
            raise RuntimeError("L2 does not contain the formal signal date")

        shape = (len(dates), len(stocks))
        arrays: dict[str, np.ndarray] = {
            "rank_1d": np.zeros(shape, dtype=np.float32),
            "rank_3d": np.zeros(shape, dtype=np.float32),
            "rank_5d": np.zeros(shape, dtype=np.float32),
            "rank_10d": np.full(shape, np.nan, dtype=np.float32),
            "amount": np.full(shape, np.nan, dtype=np.float32),
            "total_mv": np.full(shape, np.nan, dtype=np.float32),
            "atr_qfq": np.full(shape, np.nan, dtype=np.float32),
            "close_qfq": np.full(shape, np.nan, dtype=np.float32),
            "turnover_rate": np.full(shape, np.nan, dtype=np.float32),
            "listed_days": np.full(shape, -1, dtype=np.int16),
            "signal_clean": np.zeros(shape, dtype=np.bool_),
            "buy_clean": np.zeros(shape, dtype=np.bool_),
            "buy_open": np.full(shape, np.nan, dtype=np.float32),
            "buy_pre_close": np.full(shape, np.nan, dtype=np.float32),
        }
        date_map = pd.DataFrame(
            {"trade_date": dates, "d_idx": np.arange(len(dates), dtype=np.int32)}
        )
        stock_map = pd.DataFrame(
            {"stock_code": stocks, "s_idx": np.arange(len(stocks), dtype=np.int32)}
        )
        con.register("date_map", date_map)
        con.register("stock_map", stock_map)
        query = f"""
        WITH calendar AS (
          SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
          FROM (SELECT DISTINCT trade_date FROM m.STOCK_DAILY_DATA)
        ), scores AS (
          SELECT trade_date, stock_code,
                 percent_rank() OVER (
                   PARTITION BY trade_date ORDER BY pred_prob, stock_code
                 ) AS rank_10d
          FROM p10."{prediction_table}"
          WHERE trade_date BETWEEN '20220606' AND ?
            AND stock_code NOT LIKE '%.BJ'
        )
        SELECT dm.d_idx, sm.s_idx, s.rank_10d,
               sig.amount, sig.total_mv, sig.atr_qfq, sig.close_qfq,
               sig.turnover_rate,
               greatest(
                 0,
                 date_diff(
                   'day',
                   try_strptime(sig.list_date, '%Y%m%d'),
                   try_strptime(sig.trade_date, '%Y%m%d')
                 )
               ) AS listed_days,
               ({clean_sql("sig")}) AS signal_clean,
               buy.trade_date IS NOT NULL AND ({clean_sql("buy")}) AS buy_clean,
               buy.open AS buy_open,
               buy.pre_close AS buy_pre_close
        FROM scores s
        JOIN date_map dm USING (trade_date)
        JOIN stock_map sm USING (stock_code)
        JOIN m.STOCK_DAILY_DATA sig
          ON sig.trade_date=s.trade_date AND sig.stock_code=s.stock_code
        LEFT JOIN calendar c ON c.trade_date=s.trade_date
        LEFT JOIN m.STOCK_DAILY_DATA buy
          ON buy.trade_date=c.buy_date AND buy.stock_code=s.stock_code
        ORDER BY dm.d_idx, sm.s_idx
        """
        reader = con.execute(query, [signal_date]).fetch_record_batch(
            rows_per_batch=250000
        )
        for batch in reader:
            frame = batch.to_pandas()
            d = frame["d_idx"].to_numpy(dtype=np.intp, copy=False)
            s = frame["s_idx"].to_numpy(dtype=np.intp, copy=False)
            for column in (
                "rank_10d",
                "amount",
                "total_mv",
                "atr_qfq",
                "close_qfq",
                "turnover_rate",
            ):
                arrays[column][d, s] = frame[column].to_numpy(
                    dtype=np.float32, copy=False
                )
            arrays["listed_days"][d, s] = frame["listed_days"].fillna(-1).to_numpy(
                dtype=np.int16
            )
            arrays["signal_clean"][d, s] = frame["signal_clean"].fillna(False).to_numpy(
                dtype=np.bool_
            )
            arrays["buy_clean"][d, s] = frame["buy_clean"].fillna(False).to_numpy(
                dtype=np.bool_
            )
            arrays["buy_open"][d, s] = frame["buy_open"].to_numpy(
                dtype=np.float32, copy=False
            )
            arrays["buy_pre_close"][d, s] = frame["buy_pre_close"].to_numpy(
                dtype=np.float32, copy=False
            )
    finally:
        con.close()
    arrays["dates"] = np.asarray(dates, dtype="U8")
    arrays["stocks"] = np.asarray(stocks, dtype="U9")
    return arrays


def active_holdings(actions: pd.DataFrame, dates: np.ndarray) -> dict[str, int]:
    date_index = {str(value): idx for idx, value in enumerate(dates.astype(str))}
    holdings: dict[str, int] = {}
    for row in actions.itertuples(index=False):
        stock = str(row.stock_code)
        if str(row.action).upper() == "BUY":
            holdings[stock] = date_index[str(row.signal_date)]
        else:
            holdings.pop(stock, None)
    return holdings


def compare_frozen_prefix(actions: pd.DataFrame) -> None:
    frozen = pd.read_csv(FROZEN_ACTIONS, dtype={"signal_date": str, "buy_date": str})
    if list(frozen.columns) != list(actions.columns):
        raise RuntimeError("frozen action schema mismatch")
    if len(frozen) != len(actions):
        raise RuntimeError(
            f"frozen action row mismatch: expected {len(frozen)}, got {len(actions)}"
        )
    for column in ("signal_date", "buy_date", "action", "stock_code"):
        if not frozen[column].astype(str).equals(actions[column].astype(str)):
            raise RuntimeError(f"frozen action key mismatch: {column}")
    for column in ("target_pct", "execution_open_raw"):
        if not np.allclose(
            frozen[column].to_numpy(dtype=float),
            actions[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        ):
            raise RuntimeError(f"frozen action value mismatch: {column}")


def next_weekday(date_text: str) -> str:
    value = datetime.strptime(date_text, "%Y%m%d").date() + timedelta(days=1)
    while value.weekday() >= 5:
        value += timedelta(days=1)
    return value.strftime("%Y%m%d")


def names_for(signal_date: str) -> pd.DataFrame:
    with duckdb.connect(str(L2), read_only=True) as con:
        frame = con.execute(
            """
            SELECT stock_code, any_value(name) AS name, any_value(market) AS market
            FROM STOCK_DAILY_DATA
            WHERE trade_date=?
            GROUP BY stock_code
            """,
            [signal_date],
        ).fetchdf()
    return frame.drop_duplicates("stock_code").set_index("stock_code")


def pending_intents(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    actions: pd.DataFrame,
    protocol: dict,
    definition: dict,
    signal_date: str,
    buy_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    stock_index = {stock: idx for idx, stock in enumerate(stocks)}
    t = int(np.where(dates == signal_date)[0][0])
    holdings = active_holdings(actions, dates)

    selection = v174.selection_mask(
        arrays, float(definition["max_rank_deterioration"])
    )[t]
    clean = arrays["signal_clean"][t] & np.isfinite(score[t]) & selection
    clean &= arrays["listed_days"][t] >= int(protocol["fixed_universe"]["listed_days_min"])
    clean &= np.isfinite(arrays["amount"][t]) & (
        arrays["amount"][t] >= int(protocol["fixed_universe"]["amount_min"])
    )
    clean &= np.isfinite(arrays["total_mv"][t]) & (
        arrays["total_mv"][t] >= int(protocol["fixed_universe"]["mv_min"])
    )
    clean &= np.isfinite(arrays["turnover_rate"][t])
    clean &= arrays["turnover_rate"][t] >= 0
    clean &= arrays["turnover_rate"][t] <= float(
        protocol["fixed_universe"]["turnover_max"]
    )
    ranked = [int(idx) for idx in order[t] if clean[int(idx)]]
    best_unheld = max(
        (
            float(score[t, idx])
            for idx in ranked
            if stocks[idx] not in holdings
        ),
        default=-np.inf,
    )
    min_hold = v260.v258.v252.v162.min_hold_schedule(
        arrays, definition["min_hold_policy"]
    )
    sell_stocks: list[str] = []
    sell_reasons: dict[str, str] = {}
    for stock, entry_t in sorted(holdings.items()):
        idx = stock_index[stock]
        age = t - entry_t
        current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
        score_exit = (
            age >= int(min_hold[t])
            and current < float(definition["sell_score_below"])
            and best_unheld - current >= float(definition["replacement_advantage"])
        )
        if age >= int(definition["max_hold_days"]) or score_exit:
            sell_stocks.append(stock)
            sell_reasons[stock] = (
                "max_hold"
                if age >= int(definition["max_hold_days"])
                else "score_replacement"
            )

    holdings_after_sell = set(holdings) - set(sell_stocks)
    max_positions = int(v260.position_schedule(arrays, definition, None)[t])
    slots = min(1, max(max_positions - len(holdings_after_sell), 0))
    buy_indices = [
        idx for idx in ranked if stocks[idx] not in holdings_after_sell
    ][:slots]
    target = v260.v258.v252.v162.v153.schedules(arrays, definition)[1]
    multiplier = v260.v258.v252.warmup_multiplier(
        arrays, score, definition, protocol, None
    )
    names = names_for(signal_date)
    valid_scores = score[t][np.isfinite(score[t])]

    def enrich(stock: str, idx: int) -> dict:
        value = float(score[t, idx]) if np.isfinite(score[t, idx]) else None
        return {
            "stock_code": stock,
            "name": str(names.loc[stock, "name"]) if stock in names.index else "",
            "market": str(names.loc[stock, "market"]) if stock in names.index else "",
            "strategy_score": value,
            "score_rank": (
                int(np.sum(valid_scores > value) + 1) if value is not None else None
            ),
            "score_denominator": int(len(valid_scores)),
        }

    rows: list[dict] = []
    for stock in sell_stocks:
        idx = stock_index[stock]
        rows.append(
            {
                "strategy_id": STRATEGY_ID,
                "signal_date": signal_date,
                "buy_date": buy_date,
                "action": "SELL",
                **enrich(stock, idx),
                "target_pct": 0.0,
                "reason": sell_reasons[stock],
                "execution_open_raw": None,
                "status": "pending_buy_day_hard_gate",
            }
        )
    for idx in buy_indices:
        stock = stocks[idx]
        rows.append(
            {
                "strategy_id": STRATEGY_ID,
                "signal_date": signal_date,
                "buy_date": buy_date,
                "action": "BUY",
                **enrich(stock, idx),
                "target_pct": float(target[t] * multiplier[t, idx]),
                "reason": "top_signal_candidate_after_planned_sells",
                "execution_open_raw": None,
                "status": "pending_buy_day_hard_gate",
            }
        )

    candidates = []
    for rank_position, idx in enumerate(ranked[:20], start=1):
        stock = stocks[idx]
        candidates.append(
            {
                "rank_position": rank_position,
                **enrich(stock, idx),
                "already_held": stock in holdings,
                "planned_sell": stock in sell_stocks,
                "amount_thousand_cny": float(arrays["amount"][t, idx]),
                "turnover_rate": float(arrays["turnover_rate"][t, idx]),
                "status": "pending_buy_day_hard_gate",
            }
        )
    audit = {
        "signal_day_universe_count": int(np.sum(clean)),
        "holding_count_before": len(holdings),
        "planned_sell_count": len(sell_stocks),
        "planned_buy_count": len(buy_indices),
        "max_positions": max_positions,
        "holding_count_after_planned_actions": (
            len(holdings_after_sell) + len(buy_indices)
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(candidates), audit


def write_pending_assets(
    manifest: dict, signal_dir: Path, buy_date: str
) -> dict:
    protocol = load_json(PROTOCOL)
    arrays = build_arrays(manifest)
    signal_date = str(manifest["max_trade_date"])
    dates = arrays["dates"].astype(str)
    signal_idx = int(np.where(dates == signal_date)[0][0])
    previous_signal_date = str(dates[signal_idx - 1])
    definition = v260.definition_for(protocol, 50)
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)

    _, frozen_prefix = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        "20260721",
        record_actions=True,
    )
    compare_frozen_prefix(frozen_prefix)

    _, actions = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        previous_signal_date,
        record_actions=True,
    )
    actions.to_csv(CURRENT_ACTIONS, index=False, encoding="utf-8-sig")
    intents, candidates, audit = pending_intents(
        arrays,
        score,
        order,
        actions,
        protocol,
        definition,
        signal_date,
        buy_date,
    )
    if intents.empty:
        raise RuntimeError("V260 produced an empty pending intent set")
    duplicate_keys = int(
        intents.duplicated(["signal_date", "buy_date", "action", "stock_code"]).sum()
    )
    bj_rows = int(intents["stock_code"].astype(str).str.endswith(".BJ").sum())
    if duplicate_keys or bj_rows:
        raise RuntimeError("pending signal duplicate or BJ gate failed")

    signal_dir.mkdir(parents=True, exist_ok=True)
    latest_path = signal_dir / f"{STRATEGY_ID}_latest.csv"
    archive_path = (
        STRATEGY_DIR
        / "signals"
        / f"latest_signal_{signal_date}_for_{buy_date}.csv"
    )
    candidate_path = (
        STRATEGY_DIR
        / "signals"
        / f"latest_candidates_{signal_date}_for_{buy_date}.csv"
    )
    intents.to_csv(latest_path, index=False, encoding="utf-8-sig")
    intents.to_csv(archive_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    status = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "status": "pending_buy_day_hard_gate",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_date_source": "next_weekday_estimate_l2_buy_day_market_not_available",
        "source_manifest": str(FORMAL_MANIFEST),
        "source_prediction_table": str(manifest["table"]),
        "source_file": str(archive_path),
        "output_file": str(latest_path),
        "candidate_file": str(candidate_path),
        "current_action_history": str(CURRENT_ACTIONS),
        "row_count": int(len(intents)),
        "stock_count": int(intents["stock_code"].nunique()),
        "buy_count": int((intents["action"] == "BUY").sum()),
        "sell_count": int((intents["action"] == "SELL").sum()),
        "duplicate_key_count": duplicate_keys,
        "bj_rows": bj_rows,
        "buy_day_market_available": False,
        "buy_day_hard_gate_complete": False,
        "l7_execution_allowed": False,
        "formal_signal_generated": True,
        "formal_signal_semantics": "l5_strategy_intent_pending_l7_buy_day_gate",
        "frozen_prefix_action_sha256": sha256(FROZEN_ACTIONS),
        "legacy_or_research_data_input_used": False,
        "audit": audit,
        "note": "该资产是正式 L5 策略意图，不是可执行交易指令；20260724 真实开盘行情硬门控和 L5/L6 审计均未完成。",
    }
    status_path = STRATEGY_DIR / "signals/latest_signal_status.json"
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (signal_dir / f"{STRATEGY_ID}_latest_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    strategy_manifest_path = STRATEGY_DIR / "strategy_manifest.json"
    strategy_manifest = load_json(strategy_manifest_path)
    strategy_manifest["latest_signal"] = {
        "status": status["status"],
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_day_market_available": False,
        "buy_day_hard_gate_complete": False,
        "l7_execution_allowed": False,
    }
    strategy_manifest["current_signal"] = {
        "latest_file": str(latest_path),
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "buy_day_hard_gate_complete": False,
        "status": status["status"],
        "l7_execution_allowed": False,
    }
    strategy_manifest_path.write_text(
        json.dumps(strategy_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 正式 L5 信号导出入口")
    parser.add_argument(
        "--signal-dir", default=str(DATA / "production_signals")
    )
    parser.add_argument("--buy-date")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = validate_route()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "strategy_id": STRATEGY_ID,
                    "manifest_max_trade_date": manifest["max_trade_date"],
                    "formal_signal_generated": False,
                },
                ensure_ascii=False,
            )
        )
        return
    buy_date = str(args.buy_date or next_weekday(str(manifest["max_trade_date"])))
    with duckdb.connect(str(L2), read_only=True) as con:
        buy_day_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date=?",
                [buy_date],
            ).fetchone()[0]
        )
    if buy_day_rows:
        raise RuntimeError(
            "buy-day market is already available; pending-only L5 generation refuses "
            "to bypass the separate execution hard-gate stage"
        )
    status = write_pending_assets(manifest, Path(args.signal_dir).resolve(), buy_date)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
