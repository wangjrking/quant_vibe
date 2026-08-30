from __future__ import annotations

import ast
import csv
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "variants_0939_weight_rank_ls08"
RUN_DIR = REPORT_DIR / "juejin_runs_0939_weight_rank_ls08"
SUMMARY_CSV = REPORT_DIR / "juejin_0939_weight_rank_ls08_summary.csv"
RAW_CSV = REPORT_DIR / "juejin_0939_weight_rank_ls08_raw.csv"
MANIFEST_CSV = REPORT_DIR / "variants_0939_weight_rank_ls08_manifest.csv"

MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}

CASES = [
    ("w15_35_00_50", (0.15, 0.35, 0.00, 0.50)),
    ("w25_25_00_50", (0.25, 0.25, 0.00, 0.50)),
    ("w35_15_00_50", (0.35, 0.15, 0.00, 0.50)),
    ("w10_15_25_50", (0.10, 0.15, 0.25, 0.50)),
    ("w10_10_20_60", (0.10, 0.10, 0.20, 0.60)),
    ("w00_20_20_60", (0.00, 0.20, 0.20, 0.60)),
    ("w00_00_30_70", (0.00, 0.00, 0.30, 0.70)),
    ("w00_00_00_100", (0.00, 0.00, 0.00, 1.00)),
]


def _load_manifest(path: Path) -> dict:
    import json

    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"manifest not approved_for_l5: {path}")
    if obj.get("source_type") != "duckdb_table":
        raise RuntimeError(f"manifest is not duckdb_table: {path}")
    db_path = (path.parent / obj["db_path"]).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    return {"db_path": db_path, "table": obj["table"]}


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _indicator_from_log(text: str) -> dict:
    match = re.search(r"GM_BACKTEST_INDICATOR:\s*(\{.*?\})(?:\r?\n|$)", text, re.S)
    if not match:
        return {}
    raw = match.group(1)
    out: dict[str, object] = {}
    account_match = re.search(r"'account_id':\s*'([^']+)'", raw)
    if account_match:
        out["account_id"] = account_match.group(1)
    for key in [
        "pnl_ratio",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "risk_ratio",
        "open_count",
        "close_count",
        "win_count",
        "lose_count",
        "win_ratio",
        "calmar_ratio",
    ]:
        value_match = re.search(rf"'{key}':\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", raw)
        if value_match:
            value = float(value_match.group(1))
            out[key] = int(value) if key.endswith("_count") else value
    return out


def _build_base(con: duckdb.DuckDBPyConnection, sources: dict[str, dict]) -> None:
    for label, src in sources.items():
        con.execute(f"ATTACH '{src['db_path'].as_posix()}' AS l4_{label} (READ_ONLY)")
    con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
    tables = {k: v["table"] for k, v in sources.items()}
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE base AS
        WITH cal AS (
            SELECT
                trade_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
            FROM (
                SELECT DISTINCT trade_date
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260702'
            )
        ),
        preds AS (
            SELECT
                p10.trade_date,
                p10.stock_code,
                p1.pred_prob AS pred_1d,
                p3.pred_prob AS pred_3d,
                p5.pred_prob AS pred_5d,
                p10.pred_prob AS pred_10d
            FROM l4_10d."{tables['10d']}" p10
            JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
            JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
            JOIN l4_1d."{tables['1d']}" p1 USING (trade_date, stock_code)
            WHERE p10.trade_date BETWEEN '20220606' AND '20260701'
              AND p10.stock_code NOT LIKE '%.BJ'
        ),
        ranked AS (
            SELECT
                *,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
            FROM preds
        ),
        md AS (
            SELECT
                trade_date,
                stock_code,
                name,
                open,
                close,
                pre_close,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                pct_chg,
                ST_TYPE,
                ST_TYPE_name
            FROM marketdb.STOCK_DAILY_DATA
            WHERE trade_date BETWEEN '20220606' AND '20260702'
        )
        SELECT
            r.*,
            cal.buy_date,
            sig.name,
            sig.amount AS signal_amount,
            sig.turnover_rate AS signal_turnover_rate,
            sig.total_mv AS signal_total_mv,
            sig.atr_qfq AS signal_atr_qfq,
            sig.pct_chg AS signal_pct_chg,
            buy.open AS buy_open_raw,
            buy.pre_close AS buy_pre_close_raw,
            buy.open / NULLIF(sig.close, 0) - 1 AS buy_open_gap_raw,
            buy.amount AS buy_amount,
            buy.turnover_rate AS buy_turnover_rate,
            buy.total_mv AS buy_total_mv,
            buy.atr_qfq AS buy_atr_qfq,
            percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.amount) AS amount_rank,
            percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.total_mv) AS mv_rank,
            percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.atr_qfq / NULLIF(sig.close, 0)) AS atr_rank
        FROM ranked r
        JOIN cal ON cal.trade_date = r.trade_date
        JOIN md sig ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
        JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
        WHERE cal.buy_date IS NOT NULL
          AND coalesce(sig.ST_TYPE, '') IN ('', '0')
          AND coalesce(sig.ST_TYPE_name, '') NOT LIKE '%ST%'
          AND coalesce(sig.name, '') NOT LIKE 'ST%'
          AND coalesce(sig.name, '') NOT LIKE '*ST%'
          AND coalesce(sig.name, '') NOT LIKE '%退%'
          AND coalesce(buy.ST_TYPE, '') IN ('', '0')
          AND coalesce(buy.ST_TYPE_name, '') NOT LIKE '%ST%'
          AND sig.close IS NOT NULL
          AND buy.open IS NOT NULL
          AND buy.pre_close IS NOT NULL
        """
    )


def _export_case(con: duckdb.DuckDBPyConnection, name: str, weights: tuple[float, float, float, float]) -> dict:
    w1, w3, w5, w10 = weights
    score_expr = f"({w1} * r1 + {w3} * r3 + {w5} * r5 + {w10} * r10)"
    rerank_expr = f"{score_expr} + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank"
    signal_file = OUT_DIR / f"0939_{name}_rank95_amt50000_top1_t50_h2_ls08.csv"
    score_db = OUT_DIR / f"score_0939_{name}.duckdb"
    case_name = signal_file.stem
    score_df = con.execute(
        f"""
        SELECT trade_date, stock_code, {score_expr}::DOUBLE AS pred_prob
        FROM base
        """
    ).fetchdf()
    with duckdb.connect(str(score_db)) as out_con:
        out_con.register("score_df", score_df)
        out_con.execute("CREATE OR REPLACE TABLE score AS SELECT * FROM score_df")

    sig = con.execute(
        f"""
        WITH scored AS (
            SELECT
                *,
                {score_expr}::DOUBLE AS entry_score,
                {rerank_expr}::DOUBLE AS rerank_score
            FROM base
        ),
        rank_scored AS (
            SELECT
                *,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY entry_score) AS entry_score_pct_rank
            FROM scored
        ),
        filtered AS (
            SELECT *
            FROM rank_scored
            WHERE entry_score_pct_rank >= 0.95
              AND signal_amount >= 50000
              AND buy_amount >= 50000
              AND buy_open_gap_raw >= -0.03
              AND buy_open_gap_raw <= 0.015
              AND buy_open_raw / NULLIF(buy_pre_close_raw, 0) - 1 < CASE
                  WHEN stock_code LIKE '300%' OR stock_code LIKE '301%' OR stock_code LIKE '688%' THEN 0.195
                  ELSE 0.095
              END
        ),
        picked AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY buy_date ORDER BY buy_open_gap_raw ASC, rerank_score DESC, stock_code) AS pick_rank
            FROM filtered
        )
        SELECT
            trade_date AS signal_date,
            buy_date,
            stock_code,
            name,
            pick_rank AS rank,
            entry_score AS pred_prob,
            entry_score,
            pred_1d,
            pred_3d,
            pred_5d,
            pred_10d,
            r1 AS rank_1d,
            r3 AS rank_3d,
            r5 AS rank_5d,
            r10 AS rank_10d,
            entry_score_pct_rank AS score_pct_rank,
            signal_pct_chg,
            buy_open_gap_raw * 100.0 AS buy_open_gap_raw_pct,
            signal_amount,
            buy_amount,
            signal_turnover_rate,
            buy_turnover_rate,
            signal_total_mv,
            buy_total_mv,
            signal_atr_qfq,
            buy_atr_qfq
        FROM picked
        WHERE pick_rank <= 1
        ORDER BY signal_date, pick_rank
        """
    ).fetchdf()
    sig.insert(2, "symbol", sig["stock_code"].map(_symbol))
    sig["target_pct"] = 0.50
    sig["holding_days"] = 2
    sig["max_holding_days"] = 2
    sig["score_exit_entry_ratio"] = 9.99
    sig["min_holding_days_before_score_exit"] = 2
    sig["score_continue_entry_ratio"] = 9.99
    sig["buy_day_market_available"] = True
    sig["buy_day_hard_gate_complete"] = True
    sig["buy_day_st_rejected"] = False
    sig["buy_day_open_limit_up_rejected"] = False
    sig["latest_market_date"] = "20260702"
    sig.to_csv(signal_file, index=False, encoding="utf-8-sig")
    return {
        "name": case_name,
        "weights": weights,
        "signal_file": str(signal_file),
        "score_db": str(score_db),
        "score_table": "score",
        "signal_rows": int(len(sig)),
        "signal_buy_days": int(sig["buy_date"].nunique()),
    }


def _run_juejin(row: dict) -> dict:
    log_file = RUN_DIR / f"{row['name']}.log"
    cmd = [
        str(PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        "2",
        "--target-position-pct",
        "0.50",
        "--score-db",
        row["score_db"],
        "--score-table",
        "score",
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        "9.99",
        "--min-holding-days-before-score-exit",
        "2",
        "--score-continue-entry-ratio",
        "9.99",
        "--max-holding-days",
        "2",
        "--light-stop-loss-pct",
        "0.08",
        "--min-holding-days-before-light-stop",
        "1",
        "--backtest-adjust",
        "none",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=180)
    text = completed.stdout + "\n" + completed.stderr
    if not log_file.exists() and text:
        log_file.write_text(text, encoding="utf-8", errors="ignore")
    indicator = _indicator_from_log(log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else text)
    out = {
        **row,
        "log_file": str(log_file),
        "returncode": completed.returncode,
        "has_indicator": bool(indicator),
    }
    for key in [
        "pnl_ratio",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "open_count",
        "close_count",
        "win_ratio",
        "calmar_ratio",
        "risk_ratio",
    ]:
        out[key] = indicator.get(key)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sources = {k: _load_manifest(v) for k, v in MANIFESTS.items()}
    con = duckdb.connect()
    try:
        _build_base(con, sources)
        export_rows = [_export_case(con, name, weights) for name, weights in CASES]
    finally:
        con.close()
    pd.DataFrame(export_rows).to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")

    results = []
    for row in export_rows:
        print(f"RUN {row['name']}", flush=True)
        results.append(_run_juejin(row))
    pd.DataFrame(results).to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(results).sort_values(
        ["pnl_ratio_annual", "sharp_ratio", "max_drawdown"],
        ascending=[False, False, True],
    )
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(summary[[
        "name",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "open_count",
        "close_count",
        "win_ratio",
        "signal_rows",
        "signal_buy_days",
    ]].to_string(index=False))
    print(SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
