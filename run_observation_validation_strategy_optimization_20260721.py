from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
CONFIG = MAIN / "config" / "strategy_research" / "observation_validation_strategy_optimization_20260721.json"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_observation_validation_optimization_20260721"
POOL_DB = OUT / "current_l4_pool.duckdb"
SIGNALS = OUT / "signals"
SCORES = OUT / "score_assets"
LOGS = OUT / "juejin_logs"
LOCK = OUT / "preregistration_lock.json"
SCREEN = OUT / "entry_screen_results.csv"
ENTRY_FINALISTS = OUT / "entry_finalists.json"
JUEJIN_TUNE = OUT / "juejin_tuning_results.csv"
JUEJIN_FINALISTS = OUT / "juejin_finalists_frozen.json"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_score_quantized_20260720"
    / "research_code_snapshot"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"manifest is not approved_for_l5: {path}")
    if payload.get("source_type") != "duckdb_table":
        raise RuntimeError(f"manifest is not duckdb_table: {path}")
    db = (path.parent / payload["db_path"]).resolve()
    return {"manifest": path, "db": db, "table": payload["table"], "payload": payload}


def sources(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], Path]:
    manifests = {
        label: resolve_manifest(ROOT / relpath)
        for label, relpath in config["input_contract"]["manifests"].items()
    }
    market = (ROOT / config["input_contract"]["market_db"]).resolve()
    return manifests, market


def table_fingerprint(db: Path, table: str) -> dict[str, Any]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute(
            f'''SELECT count(*), count(distinct trade_date), min(trade_date), max(trade_date),
                       bit_xor(hash(trade_date, stock_code, pred_prob))
                FROM "{table}"'''
        ).fetchone()
    finally:
        con.close()
    stat = db.stat()
    return {
        "db": str(db),
        "table": table,
        "file_size": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "rows": int(row[0]),
        "trade_days": int(row[1]),
        "min_date": str(row[2]),
        "max_date": str(row[3]),
        "content_xor": str(row[4]),
    }


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG)
    manifests, market = sources(config)
    manifest_rows = {}
    for label, source in manifests.items():
        manifest_rows[label] = {
            "manifest": str(source["manifest"]),
            "manifest_sha256": sha256(source["manifest"]),
            **table_fingerprint(source["db"], source["table"]),
        }
    con = duckdb.connect(str(market), read_only=True)
    try:
        market_row = con.execute(
            "SELECT count(*), count(distinct trade_date), min(trade_date), max(trade_date), "
            "bit_xor(hash(trade_date, stock_code, open, pre_close, amount, total_mv, ST_TYPE, ST_TYPE_name)) "
            "FROM STOCK_DAILY_DATA"
        ).fetchone()
    finally:
        con.close()
    lock = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "frozen_before_result_access",
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "runner_sha256": sha256(RUNNER),
        "strategy_snapshot": str(STRATEGY),
        "strategy_main_sha256": sha256(STRATEGY / "main.py"),
        "l4": manifest_rows,
        "l2": {
            "db": str(market),
            "file_size": market.stat().st_size,
            "file_mtime_ns": market.stat().st_mtime_ns,
            "rows": int(market_row[0]),
            "trade_days": int(market_row[1]),
            "min_date": str(market_row[2]),
            "max_date": str(market_row[3]),
            "content_xor": str(market_row[4]),
        },
        "holdout_status": "not_accessed",
    }
    write_json(LOCK, lock)


def verify_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], Path]:
    if not LOCK.exists():
        raise RuntimeError("preregistration lock does not exist; run --stage freeze first")
    lock = read_json(LOCK)
    if sha256(CONFIG) != lock["config_sha256"]:
        raise RuntimeError("config changed after preregistration freeze")
    if sha256(Path(__file__).resolve()) != lock["script_sha256"]:
        raise RuntimeError("script changed after preregistration freeze")
    config = read_json(CONFIG)
    manifests, market = sources(config)
    for label, source in manifests.items():
        expected = lock["l4"][label]
        current = table_fingerprint(source["db"], source["table"])
        for key in ("rows", "trade_days", "min_date", "max_date", "content_xor"):
            if str(current[key]) != str(expected[key]):
                raise RuntimeError(f"L4 input changed after freeze: {label}.{key}")
    con = duckdb.connect(str(market), read_only=True)
    try:
        market_row = con.execute(
            "SELECT count(*), count(distinct trade_date), min(trade_date), max(trade_date), "
            "bit_xor(hash(trade_date, stock_code, open, pre_close, amount, total_mv, ST_TYPE, ST_TYPE_name)) "
            "FROM STOCK_DAILY_DATA"
        ).fetchone()
    finally:
        con.close()
    current_market = {
        "rows": int(market_row[0]),
        "trade_days": int(market_row[1]),
        "min_date": str(market_row[2]),
        "max_date": str(market_row[3]),
        "content_xor": str(market_row[4]),
    }
    for key, value in current_market.items():
        if str(value) != str(lock["l2"][key]):
            raise RuntimeError(f"L2 input changed after freeze: {key}")
    return config, lock, manifests, market


def clean_expr(alias: str) -> str:
    return (
        f"coalesce(cast({alias}.ST_TYPE as varchar), '') in ('', '0', '0.0', 'None', 'NONE') "
        f"and upper(coalesce(cast({alias}.ST_TYPE_name as varchar), '')) not like '%ST%' "
        f"and upper(coalesce(cast({alias}.name as varchar), '')) not like 'ST%' "
        f"and upper(coalesce(cast({alias}.name as varchar), '')) not like '*ST%' "
        f"and coalesce(cast({alias}.name as varchar), '') not like '%退%'"
    )


def build_pool(manifests: dict[str, dict[str, Any]], market: Path) -> None:
    POOL_DB.unlink(missing_ok=True)
    con = duckdb.connect(str(POOL_DB))
    try:
        con.execute("PRAGMA threads=4")
        con.execute(f"ATTACH '{market.as_posix()}' AS m (READ_ONLY)")
        for label, source in manifests.items():
            con.execute(f"ATTACH '{source['db'].as_posix()}' AS p{label} (READ_ONLY)")
        table = {label: source["table"] for label, source in manifests.items()}
        con.execute(
            f'''
            CREATE TABLE pool AS
            WITH calendar AS (
              SELECT trade_date AS signal_date,
                     lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                     lead(trade_date, 2) OVER (ORDER BY trade_date) AS next_date,
                     lead(trade_date, 6) OVER (ORDER BY trade_date) AS eval_date_5d
              FROM (SELECT DISTINCT trade_date FROM m.STOCK_DAILY_DATA)
            ), breadth AS (
              SELECT trade_date,
                     avg(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) market_breadth_up,
                     avg(pct_chg) market_mean_pct
              FROM m.STOCK_DAILY_DATA
              WHERE stock_code NOT LIKE '%.BJ' AND pct_chg IS NOT NULL
              GROUP BY trade_date
            ), joined AS (
              SELECT p10.trade_date, p10.stock_code,
                     p1.pred_prob raw_1d, p3.pred_prob raw_3d,
                     p5.pred_prob raw_5d, p10.pred_prob raw_10d
              FROM p10d."{table['10d']}" p10
              JOIN p5d."{table['5d']}" p5 USING(trade_date, stock_code)
              JOIN p3d."{table['3d']}" p3 USING(trade_date, stock_code)
              JOIN p1d."{table['1d']}" p1 USING(trade_date, stock_code)
              WHERE p10.stock_code NOT LIKE '%.BJ'
            ), ranked AS (
              SELECT *,
                     percent_rank() OVER(PARTITION BY trade_date ORDER BY raw_1d) rank_1d,
                     percent_rank() OVER(PARTITION BY trade_date ORDER BY raw_3d) rank_3d,
                     percent_rank() OVER(PARTITION BY trade_date ORDER BY raw_5d) rank_5d,
                     percent_rank() OVER(PARTITION BY trade_date ORDER BY raw_10d) rank_10d
              FROM joined
            )
            SELECT r.trade_date signal_date, c.buy_date, c.next_date, r.stock_code,
                   s.name, r.raw_1d, r.raw_3d, r.raw_5d, r.raw_10d,
                   r.rank_1d, r.rank_3d, r.rank_5d, r.rank_10d,
                   s.amount, s.turnover_rate, s.total_mv, s.atr_qfq,
                   s.pct_chg signal_pct_chg_raw,
                   br.market_breadth_up, br.market_mean_pct,
                   b.open buy_open_raw, b.pre_close buy_pre_close_raw,
                   100.0 * (b.open / nullif(b.pre_close, 0) - 1.0) buy_open_gap_raw_pct,
                   n.open next_open_raw,
                   n.open / nullif(b.open, 0) - 1.0 next_open_return_raw,
                   e5.open / nullif(b.open, 0) - 1.0 open_return_5d_raw
            FROM ranked r
            JOIN calendar c ON c.signal_date=r.trade_date AND c.buy_date IS NOT NULL
            JOIN breadth br ON br.trade_date=r.trade_date
            JOIN m.STOCK_DAILY_DATA s ON s.trade_date=r.trade_date AND s.stock_code=r.stock_code
            JOIN m.STOCK_DAILY_DATA b ON b.trade_date=c.buy_date AND b.stock_code=r.stock_code
            LEFT JOIN m.STOCK_DAILY_DATA n ON n.trade_date=c.next_date AND n.stock_code=r.stock_code
            LEFT JOIN m.STOCK_DAILY_DATA e5 ON e5.trade_date=c.eval_date_5d AND e5.stock_code=r.stock_code
            WHERE {clean_expr('s')} AND {clean_expr('b')}
              AND s.amount > 0 AND s.total_mv IS NOT NULL AND s.atr_qfq IS NOT NULL
              AND b.open > 0 AND b.pre_close > 0
              AND b.open < b.pre_close * CASE
                    WHEN r.stock_code LIKE '300%' OR r.stock_code LIKE '301%' OR r.stock_code LIKE '688%'
                    THEN 1.195 ELSE 1.095 END
            '''
        )
    finally:
        con.close()


def entry_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["entry_grid"]
    rows = []
    values = itertools.product(
        grid["blends"].items(), grid["top_n"], grid["blend_rank_min"],
        grid["cross_horizon_gate"], grid["amount_min"], grid["total_mv_min"],
        grid["signal_pct_band"], grid["buy_open_gate"], grid["market_regime"],
    )
    for (blend, weights), topn, rank_min, support, amount, mv, pct, open_gate, regime in values:
        name = (
            f"{blend}_n{topn}_r{str(rank_min).replace('.', 'p')}_{support['name']}_"
            f"a{int(amount)}_mv{int(mv)}_{pct['name']}_{open_gate['name']}_{regime['name']}"
        )
        rows.append({
            "name": name, "blend": blend, "weights": weights, "topn": topn,
            "blend_rank_min": rank_min, "support": support, "amount_min": amount,
            "mv_min": mv, "pct": pct, "open_gate": open_gate, "regime": regime,
        })
    return rows


def load_blend_frame(
    con: duckdb.DuckDBPyConnection,
    weights: dict[str, float],
    min_rank: float,
    include_holdout: bool,
) -> pd.DataFrame:
    expr = " + ".join(f"{weight} * rank_{label}" for label, weight in weights.items())
    holdout_clause = "" if include_holdout else "AND signal_date <= '20251231'"
    source_columns = (
        "* EXCLUDE(next_open_raw, next_open_return_raw, open_return_5d_raw)"
        if include_holdout else "*"
    )
    return con.execute(
        f'''
        WITH scored AS (
          SELECT {source_columns}, ({expr}) blend_score
          FROM pool
          WHERE amount >= 90000 AND total_mv >= 200000 {holdout_clause}
        ), ranked_score AS (
          SELECT *, percent_rank() OVER(PARTITION BY signal_date ORDER BY blend_score) blend_rank
          FROM scored
        )
        SELECT * FROM ranked_score WHERE blend_rank >= {min_rank}
        ORDER BY signal_date, blend_score DESC, stock_code
        '''
    ).fetchdf()


def filter_case_frame(base: pd.DataFrame, case: dict[str, Any]) -> pd.DataFrame:
    support = case["support"]
    pct = case["pct"]
    gate = case["open_gate"]
    regime = case["regime"]
    mask = (
        (base["blend_rank"] >= float(case["blend_rank_min"]))
        & (base["rank_10d"] >= float(support["rank_10d_min"]))
        & (base["rank_5d"] >= float(support["rank_5d_min"]))
        & (base["amount"] >= float(case["amount_min"]))
        & (base["total_mv"] >= float(case["mv_min"]))
        & base["signal_pct_chg_raw"].between(float(pct["min"]), float(pct["max"]))
        & base["buy_open_gap_raw_pct"].between(float(gate["reject_below"]), float(gate["reject_above"]))
        & (base["market_breadth_up"] >= float(regime["breadth_up_min"]))
        & (base["market_mean_pct"] >= float(regime["market_mean_pct_min"]))
    )
    out = base.loc[mask].copy()
    out["open_scale"] = np.where(
        out["buy_open_gap_raw_pct"] > float(gate["scale_above"]), float(gate["high_scale"]), 1.0
    )
    out = out.sort_values(["signal_date", "blend_score", "stock_code"], ascending=[True, False, True])
    out["pick_rank"] = out.groupby("signal_date").cumcount() + 1
    return out[out["pick_rank"] <= int(case["topn"])].copy()


def period_metrics(
    frame: pd.DataFrame,
    calendar: pd.Series,
    start: str,
    end: str,
    cost: float,
    topn: int,
    return_column: str,
) -> dict[str, Any]:
    part = frame[(frame["buy_date"] >= start) & (frame["buy_date"] <= end)].copy()
    part = part.dropna(subset=[return_column])
    if part.empty:
        return {"annual": -1.0, "sharpe": -99.0, "mdd": 1.0, "buy_days": 0, "rows": 0}
    part["weight"] = 0.9 / topn * part["open_scale"]
    part["net"] = part["weight"] * (part[return_column] - cost)
    daily = part.groupby("buy_date")["net"].sum().sort_index()
    all_dates = calendar[(calendar >= start) & (calendar <= end)].sort_values().drop_duplicates()
    daily = daily.reindex(all_dates.astype(str), fill_value=0.0)
    equity = (1.0 + daily).cumprod()
    if len(daily) == 0 or equity.iloc[-1] <= 0:
        annual = -1.0
    else:
        annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0)
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float(abs((equity / equity.cummax() - 1.0).min()))
    return {
        "annual": annual, "sharpe": sharpe, "mdd": mdd,
        "buy_days": int(part["buy_date"].nunique()), "rows": int(len(part)),
        "avg_names": float(len(part) / max(part["buy_date"].nunique(), 1)),
    }


def rank_rows(rows: list[dict[str, Any]], metric_prefixes: list[str]) -> list[dict[str, Any]]:
    rows = [
        row for row in rows
        if all(float(row.get(f"{prefix}_buy_days", 0)) >= 100 for prefix in metric_prefixes)
        and all(float(row.get(f"{prefix}_avg_names", 0)) >= 0.8 for prefix in metric_prefixes)
    ]
    def key(row: dict[str, Any]) -> tuple:
        sharpes = [float(row[f"{prefix}_sharpe"]) for prefix in metric_prefixes]
        annuals = [float(row[f"{prefix}_annual"]) for prefix in metric_prefixes]
        mdds = [float(row[f"{prefix}_mdd"]) for prefix in metric_prefixes]
        return (min(sharpes), float(np.median(sharpes)), min(annuals), -max(mdds), row["name"])
    return sorted(rows, key=key, reverse=True)


def signal_columns(frame: pd.DataFrame, case: dict[str, Any], cap: float = 0.9) -> pd.DataFrame:
    out = frame.copy()
    out["symbol"] = np.where(
        out["stock_code"].str.endswith(".SH"), "SHSE." + out["stock_code"].str[:6],
        np.where(out["stock_code"].str.endswith(".SZ"), "SZSE." + out["stock_code"].str[:6], out["stock_code"]),
    )
    out["rank"] = out["pick_rank"].astype(int)
    out["pred_prob"] = out["blend_score"].astype(float)
    out["entry_score"] = out["blend_score"].astype(float)
    out["target_pct"] = cap / int(case["topn"]) * out["open_scale"].astype(float)
    out["holding_days"] = 1
    out["max_holding_days"] = 3
    out["score_exit_entry_ratio"] = 0.0
    out["min_holding_days_before_score_exit"] = 1
    out["score_continue_entry_ratio"] = 99.0
    out["strategy_variant"] = case["name"]
    out["filter_name"] = "preregistered_current_l4_only"
    out["entry_weight_name"] = case["blend"]
    out["dynamic_hold_name"] = "independent_replace"
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "raw_1d", "raw_3d", "raw_5d", "raw_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw",
        "market_breadth_up", "market_mean_pct", "target_pct",
        "holding_days", "max_holding_days", "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "strategy_variant", "filter_name", "entry_weight_name",
        "dynamic_hold_name", "buy_open_gap_raw_pct",
    ]
    return out[cols]


def build_score_asset(con: duckdb.DuckDBPyConnection, blend: str, weights: dict[str, float]) -> Path:
    path = SCORES / f"{blend}.duckdb"
    if path.exists():
        return path
    expr = " + ".join(f"{weight} * rank_{label}" for label, weight in weights.items())
    con.execute(f"ATTACH '{path.as_posix()}' AS scoreout")
    con.execute(
        f"CREATE TABLE scoreout.blended_rank_score AS "
        f"SELECT signal_date trade_date, stock_code, cast({expr} AS DOUBLE) pred_prob FROM pool"
    )
    con.execute("DETACH scoreout")
    return path


def screen_entries() -> None:
    config, lock, manifests, market = verify_lock()
    for path in (OUT, SIGNALS, SCORES, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    build_pool(manifests, market)
    con = duckdb.connect(str(POOL_DB))
    try:
        pool_audit = con.execute(
            "SELECT count(*), count(distinct signal_date), min(signal_date), max(signal_date), "
            "count(*) filter(where stock_code like '%.BJ') FROM pool"
        ).fetchone()
        rows = []
        cost = float(config["local_screen"]["round_trip_cost_proxy"])
        all_cases = entry_cases(config)
        calendar = con.execute("SELECT DISTINCT buy_date FROM pool ORDER BY buy_date").fetchdf()["buy_date"].astype(str)
        min_rank = min(float(item) for item in config["entry_grid"]["blend_rank_min"])
        blend_cache = {
            blend: load_blend_frame(con, weights, min_rank, include_holdout=False)
            for blend, weights in config["entry_grid"]["blends"].items()
        }
        for case in all_cases:
            frame = filter_case_frame(blend_cache[case["blend"]], case)
            row = {"name": case["name"], "case": json.dumps(case, ensure_ascii=False, sort_keys=True)}
            for prefix, window in (
                ("s1", config["walk_forward"]["entry_screen_stage_1"]),
                ("s2", config["walk_forward"]["entry_screen_stage_2"]),
                ("s3", config["walk_forward"]["entry_screen_stage_3"]),
            ):
                metrics = period_metrics(
                    frame, calendar, window[0], window[1], cost, int(case["topn"]),
                    str(config["local_screen"]["evaluation_return_column"]),
                )
                row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            rows.append(row)
        stage1 = rank_rows(rows, ["s1"])
        top12_names = {row["name"] for row in stage1[: config["walk_forward"]["entry_promote_counts"][0]]}
        stage2 = rank_rows([row for row in rows if row["name"] in top12_names], ["s1", "s2"])
        top6_names = {row["name"] for row in stage2[: config["walk_forward"]["entry_promote_counts"][1]]}
        stage3 = rank_rows([row for row in rows if row["name"] in top6_names], ["s1", "s2", "s3"])
        finalists = stage3[: config["walk_forward"]["entry_promote_counts"][2]]
        pd.DataFrame(rows).to_csv(SCREEN, index=False, encoding="utf-8-sig")
        final_payload = []
        case_map = {case["name"]: case for case in all_cases}
        full_blend_cache: dict[str, pd.DataFrame] = {}
        for ranked, row in enumerate(finalists, 1):
            case = case_map[row["name"]]
            if case["blend"] not in full_blend_cache:
                full_blend_cache[case["blend"]] = load_blend_frame(
                    con, case["weights"], min_rank, include_holdout=True
                )
            frame = filter_case_frame(full_blend_cache[case["blend"]], case)
            signal = signal_columns(frame, case)
            signal_path = SIGNALS / f"entry_{ranked}_{case['name']}.csv"
            signal.to_csv(signal_path, index=False, encoding="utf-8-sig")
            score_path = build_score_asset(con, case["blend"], case["weights"])
            final_payload.append({
                "entry_rank": ranked, "case": case, "screen_metrics": row,
                "signal": str(signal_path), "signal_sha256": sha256(signal_path),
                "score_db": str(score_path), "score_sha256": sha256(score_path),
                "score_table": "blended_rank_score",
            })
    finally:
        con.close()
    payload = {
        "status": "entry_finalists_frozen_before_holdout",
        "preregistration_lock_sha256": sha256(LOCK),
        "pool_audit": {
            "rows": int(pool_audit[0]), "trade_days": int(pool_audit[1]),
            "min_date": str(pool_audit[2]), "max_date": str(pool_audit[3]), "bj_rows": int(pool_audit[4]),
        },
        "holdout_return_accessed": False,
        "finalists": final_payload,
    }
    write_json(ENTRY_FINALISTS, payload)


def parse_indicator(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        raw = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(raw)
        except Exception:
            result = {}
            for key in ("pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"):
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", raw)
                if match:
                    result[key] = float(match.group(1))
            return result
    return {}


def run_juejin(signal: Path, score_db: Path, name: str, start: str, end: str, slippage: float, exit_case: dict[str, Any], market: Path, max_positions: int, target_position_ceiling: float) -> dict[str, Any]:
    log = LOGS / f"{name}.log"
    env = os.environ.copy()
    env.update({
        "GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0",
        "GM_OPEN_DAILY_SCORE_EXIT": "0", "GM_INDEPENDENT_REPLACE_MODE": "1",
        "GM_INDEPENDENT_REPLACE_MIN_HOLD": str(exit_case["min_hold"]),
        "GM_INDEPENDENT_REPLACE_MAX_HOLD": str(exit_case["max_hold"]),
        "GM_INDEPENDENT_REPLACE_RATIO": "1.0",
        "GM_INDEPENDENT_REPLACE_EDGE": str(exit_case["replace_edge"]),
        "GM_INDEPENDENT_REPLACE_SCORE_DECIMALS": "10", "GM_RESIZE_HELD_ON_SIGNAL": "0",
        "GM_INDEPENDENT_EARLY_OPEN_GAP_PCT": "", "GM_EQUITY_DD_RISK_MODE": "0",
    })
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(signal),
        "--log-file", str(log), "--max-positions", str(max_positions),
        "--target-position-pct", str(target_position_ceiling),
        "--holding-days", str(exit_case["min_hold"]),
        "--max-holding-days", str(exit_case["max_hold"]), "--score-db", str(score_db),
        "--score-table", "blended_rank_score", "--market-db", str(market), "--backtest-adjust", "none",
        "--backtest-initial-cash", "600000", "--backtest-slippage-ratio", str(slippage),
        "--backtest-start", start, "--backtest-end", end,
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log)
    return {
        "returncode": process.returncode, "annual": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "mdd": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "log": str(log),
    }


def materialize_tuning_signal(source: Path, name: str, cap: float, exit_case: dict[str, Any]) -> Path:
    frame = pd.read_csv(source, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    topn = int(pd.to_numeric(frame["rank"], errors="coerce").max())
    base = pd.to_numeric(frame["target_pct"], errors="coerce")
    current_cap = frame.groupby("buy_date")["target_pct"].transform("sum").replace(0.0, np.nan)
    frame["target_pct"] = base * np.minimum(1.0, cap / current_cap).fillna(1.0)
    frame["holding_days"] = exit_case["min_hold"]
    frame["max_holding_days"] = exit_case["max_hold"]
    path = SIGNALS / f"tune_{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def tune_juejin() -> None:
    config, _lock, _manifests, market = verify_lock()
    entry = read_json(ENTRY_FINALISTS)
    cases = []
    for finalist, exit_case, cap in itertools.product(
        entry["finalists"], config["exit_grid"], config["position_caps"]
    ):
        name = f"e{finalist['entry_rank']}_{exit_case['name']}_c{int(cap * 100)}"
        signal = materialize_tuning_signal(Path(finalist["signal"]), name, float(cap), exit_case)
        cases.append({
            "name": name, "entry": finalist, "exit": exit_case, "cap": cap,
            "signal": signal, "score_db": Path(finalist["score_db"]),
        })
    rows = []
    for index, case in enumerate(cases, 1):
        result = run_juejin(
            case["signal"], case["score_db"], case["name"] + "_s1",
            "2022-06-07 09:00:00", "2023-12-29 15:30:00",
            config["execution_cost"]["juejin_primary_one_side_slippage"], case["exit"], market,
            int(config["max_positions"]), float(config["strategy_target_position_ceiling"]),
        )
        row = {"name": case["name"], "entry_rank": case["entry"]["entry_rank"],
               "exit": json.dumps(case["exit"], sort_keys=True), "cap": case["cap"],
               "signal": str(case["signal"]), "score_db": str(case["score_db"]),
               **{f"s1_{key}": value for key, value in result.items()}}
        rows.append(row)
        pd.DataFrame(rows).to_csv(JUEJIN_TUNE, index=False, encoding="utf-8-sig")
        print(json.dumps({"progress": f"{index}/{len(cases)}", "name": case["name"], **result}, ensure_ascii=False), flush=True)
    ranked1 = rank_juejin(rows, ["s1"])
    top12 = ranked1[: config["walk_forward"]["juejin_promote_counts"][0]]
    for row in top12:
        exit_case = json.loads(row["exit"])
        result = run_juejin(
            Path(row["signal"]), Path(row["score_db"]), row["name"] + "_s2",
            "2024-01-02 09:00:00", "2024-12-31 15:30:00",
            config["execution_cost"]["juejin_primary_one_side_slippage"], exit_case, market,
            int(config["max_positions"]), float(config["strategy_target_position_ceiling"]),
        )
        row.update({f"s2_{key}": value for key, value in result.items()})
    top6 = rank_juejin(top12, ["s1", "s2"])[: config["walk_forward"]["juejin_promote_counts"][1]]
    for row in top6:
        exit_case = json.loads(row["exit"])
        result = run_juejin(
            Path(row["signal"]), Path(row["score_db"]), row["name"] + "_s3",
            "2025-01-02 09:00:00", "2025-12-31 15:30:00",
            config["execution_cost"]["juejin_primary_one_side_slippage"], exit_case, market,
            int(config["max_positions"]), float(config["strategy_target_position_ceiling"]),
        )
        row.update({f"s3_{key}": value for key, value in result.items()})
    finalists = rank_juejin(top6, ["s1", "s2", "s3"])[: config["walk_forward"]["juejin_promote_counts"][2]]
    pd.DataFrame(rows).to_csv(JUEJIN_TUNE, index=False, encoding="utf-8-sig")
    payload = {
        "status": "finalists_frozen_before_2026_holdout",
        "holdout_accessed": False,
        "selection_rule": config["objective"]["selection_order"],
        "finalists": finalists,
    }
    write_json(JUEJIN_FINALISTS, payload)


def rank_juejin(rows: list[dict[str, Any]], prefixes: list[str]) -> list[dict[str, Any]]:
    valid = [row for row in rows if all(row.get(f"{prefix}_sharpe") is not None for prefix in prefixes)]
    def key(row: dict[str, Any]) -> tuple:
        sharpes = [float(row[f"{prefix}_sharpe"]) for prefix in prefixes]
        annuals = [float(row[f"{prefix}_annual"]) for prefix in prefixes]
        mdds = [float(row[f"{prefix}_mdd"]) for prefix in prefixes]
        return (min(sharpes), float(np.median(sharpes)), min(annuals), -max(mdds), row["name"])
    return sorted(valid, key=key, reverse=True)


def run_holdout() -> None:
    config, _lock, _manifests, market = verify_lock()
    if not bool(config["walk_forward"].get("final_holdout_currently_available")):
        raise RuntimeError("future unseen admission window is not available after 20260717")
    frozen = read_json(JUEJIN_FINALISTS)
    results = []
    for row in frozen["finalists"]:
        exit_case = json.loads(row["exit"])
        for window, start, end, slip in (
            ("holdout_2026", "2026-01-05 09:00:00", "2026-07-17 15:30:00", 0.003),
            ("full_repeat_a", "2022-06-07 09:00:00", "2026-07-17 15:30:00", 0.003),
            ("full_repeat_b", "2022-06-07 09:00:00", "2026-07-17 15:30:00", 0.003),
            ("full_stress", "2022-06-07 09:00:00", "2026-07-17 15:30:00", 0.004),
        ):
            result = run_juejin(
                Path(row["signal"]), Path(row["score_db"]), row["name"] + "_" + window,
                start, end, slip, exit_case, market,
                int(config["max_positions"]), float(config["strategy_target_position_ceiling"]),
            )
            results.append({"name": row["name"], "window": window, "slippage": slip, **result})
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "holdout_and_full_results.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "holdout_and_full_results.json", {
        "status": "research_only_pending_full_admission_audit",
        "holdout_accessed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "screen", "juejin-tune", "holdout"), required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "screen":
        screen_entries()
    elif args.stage == "juejin-tune":
        tune_juejin()
    else:
        run_holdout()


if __name__ == "__main__":
    main()
