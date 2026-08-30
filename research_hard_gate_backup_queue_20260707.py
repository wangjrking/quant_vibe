from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from prediction_manifest import load_prediction_source_manifest  # noqa: E402
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402
from update_fw_soft_deepdrop_l5_signal import (  # noqa: E402
    COLUMNS,
    MANIFESTS,
    apply_feature_weight,
    clean_status_expr,
    symbol,
)


OUT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_hard_gate_backup_queue_20260707"
SIGNAL_DIR = OUT_DIR / "signals"
SUMMARY_PATH = OUT_DIR / "backup_queue_generation_summary.json"
REPORT_PATH = OUT_DIR / "backup_queue_generation_report.md"
PRODUCTION_HISTORY = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_fw_soft_deepdrop_weight_v20260706"
    / "signals"
    / "full_history_fw_soft_deepdrop_weight.csv"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def attach_sources(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for label, manifest_path in MANIFESTS.items():
        source = load_prediction_source_manifest(manifest_path, require_approved=True, allow_legacy=False)
        if source["source_type"] != "duckdb_table":
            raise RuntimeError(f"L4 manifest is not DuckDB-only: {source['manifest_path']}")
        sources[label] = source
        con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)
    con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS market (READ_ONLY)")
    sources["_market"] = {"db_path": str(market_db), "table": "STOCK_DAILY_DATA"}
    return sources


def build_candidate_pool(
    con: duckdb.DuckDBPyConnection,
    sources: dict[str, dict[str, Any]],
    restrict_to_production_signal_days: bool,
) -> dict[str, Any]:
    tables = {label: sources[label]["table"] for label in ("1d", "3d", "5d", "10d")}
    signal_day_clause = ""
    if restrict_to_production_signal_days:
        history = pd.read_csv(PRODUCTION_HISTORY, dtype={"signal_date": str})
        signal_days = sorted(history["signal_date"].dropna().astype(str).unique().tolist())
        con.execute("CREATE OR REPLACE TEMP TABLE production_signal_days(signal_date varchar)")
        con.executemany("INSERT INTO production_signal_days VALUES (?)", [(day,) for day in signal_days])
        signal_day_clause = "AND r.trade_date IN (SELECT signal_date FROM production_signal_days)"
    con.execute(
        f'''
        CREATE OR REPLACE TEMP TABLE trade_calendar AS
        SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
        FROM (SELECT DISTINCT trade_date FROM market.STOCK_DAILY_DATA)
        '''
    )
    con.execute(
        f'''
        CREATE OR REPLACE TEMP TABLE raw_pool AS
        WITH preds AS (
            SELECT
                p10.trade_date,
                p10.stock_code,
                p1.pred_prob AS raw_pred_1d,
                p3.pred_prob AS raw_pred_3d,
                p5.pred_prob AS raw_pred_5d,
                p10.pred_prob AS raw_pred_10d
            FROM l4_10d."{tables["10d"]}" p10
            JOIN l4_5d."{tables["5d"]}" p5 USING (trade_date, stock_code)
            JOIN l4_3d."{tables["3d"]}" p3 USING (trade_date, stock_code)
            JOIN l4_1d."{tables["1d"]}" p1 USING (trade_date, stock_code)
            WHERE p10.stock_code NOT LIKE '%.BJ'
        ),
        ranked AS (
            SELECT
                *,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_1d) AS rank_1d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_3d) AS rank_3d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_5d) AS rank_5d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_10d) AS rank_10d
            FROM preds
        )
        SELECT
            r.trade_date AS signal_date,
            cal.buy_date,
            r.stock_code,
            sig.name,
            r.rank_1d,
            r.rank_3d,
            r.rank_5d,
            r.rank_10d,
            (0.25 * r.rank_1d + 0.25 * r.rank_3d + 0.50 * r.rank_10d) AS base_entry_score,
            sig.amount,
            sig.turnover_rate,
            sig.total_mv,
            sig.atr_qfq,
            sig.pct_chg,
            buy.open AS buy_open_raw,
            buy.pre_close AS buy_pre_close_raw,
            CASE
                WHEN buy.open IS NOT NULL AND buy.pre_close IS NOT NULL AND buy.pre_close > 0
                THEN ((buy.open / buy.pre_close) - 1.0) * 100.0
                ELSE NULL
            END AS buy_open_gap_raw_pct,
            CASE
                WHEN {clean_status_expr("buy")}
                 AND buy.open IS NOT NULL
                 AND buy.pre_close IS NOT NULL
                 AND buy.pre_close > 0
                 AND ((buy.open / buy.pre_close) - 1.0) * 100.0 <= 1.5
                 AND buy.open / NULLIF(buy.pre_close, 0) - 1.0 < CASE
                    WHEN r.stock_code LIKE '300%' OR r.stock_code LIKE '301%' OR r.stock_code LIKE '688%' THEN 0.195
                    ELSE 0.095
                 END
                THEN 1 ELSE 0
            END AS buy_day_hard_gate_pass
        FROM ranked r
        JOIN trade_calendar cal ON cal.trade_date = r.trade_date
        JOIN market.STOCK_DAILY_DATA sig
          ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
        JOIN market.STOCK_DAILY_DATA buy
          ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
        WHERE cal.buy_date IS NOT NULL
          {signal_day_clause}
          AND {clean_status_expr("sig")}
          AND sig.amount >= 90000
          AND sig.total_mv >= 200000
          AND sig.pct_chg <= -1.75
          AND r.rank_10d >= 0.70
          AND r.rank_1d >= 0.00
        '''
    )
    row = con.execute(
        '''
        SELECT min(signal_date), max(signal_date), count(*), count(distinct signal_date),
               sum(CASE WHEN buy_day_hard_gate_pass = 1 THEN 1 ELSE 0 END)
        FROM raw_pool
        '''
    ).fetchone()
    return {
        "min_signal_date": row[0],
        "max_signal_date": row[1],
        "candidate_rows": int(row[2] or 0),
        "signal_days": int(row[3] or 0),
        "hard_gate_pass_rows": int(row[4] or 0),
    }


def to_rows(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        entry_score = float(row["base_entry_score"])
        rows.append(
            {
                "signal_date": str(row["signal_date"]),
                "buy_date": str(row["buy_date"]),
                "symbol": symbol(str(row["stock_code"])),
                "stock_code": str(row["stock_code"]),
                "name": row["name"],
                "rank": int(idx + 1),
                "pred_prob": entry_score,
                "entry_score": entry_score,
                "pred_1d": float(row["rank_1d"]),
                "pred_3d": float(row["rank_3d"]),
                "pred_5d": float(row["rank_5d"]),
                "pred_10d": float(row["rank_10d"]),
                "amount": float(row["amount"]) if pd.notna(row["amount"]) else None,
                "turnover_rate": float(row["turnover_rate"]) if pd.notna(row["turnover_rate"]) else None,
                "total_mv": float(row["total_mv"]) if pd.notna(row["total_mv"]) else None,
                "atr_qfq": float(row["atr_qfq"]) if pd.notna(row["atr_qfq"]) else None,
                "signal_pct_chg_raw": float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
                "target_pct": 0.435,
                "holding_days": 1,
                "max_holding_days": 1,
                "score_exit_entry_ratio": 0.96,
                "min_holding_days_before_score_exit": 1,
                "score_continue_entry_ratio": 9.99,
                "signal_stop_loss_pct": 0.05,
                "signal_take_profit_pct": 0.08,
                "strategy_variant": variant,
                "source_strategy_variant": "research_hard_gate_backup_queue_20260707",
                "filter_name": "sigpct_le_m175_rank10d70_then_buyday_gate",
                "entry_weight_name": "w25_25_00_50_backup_queue",
                "dynamic_hold_name": "h1m1_reconstructed",
                "buy_day_market_available": True,
                "buy_day_hard_gate_complete": True,
                "buy_day_st_rejected": False,
                "buy_day_open_limit_up_rejected": False,
                "latest_market_date": None,
                "buy_open_gap_pct": None,
                "hybrid_source": "base_backup_queue",
                "buy_open_gap_raw_pct": float(row["buy_open_gap_raw_pct"])
                if pd.notna(row["buy_open_gap_raw_pct"])
                else None,
                "feature_weight_scale": 1.0,
                "daily_target_sum_after_cap": None,
            }
        )
    out = pd.DataFrame(rows, columns=COLUMNS)
    return apply_feature_weight(out)


def build_variant(con: duckdb.DuckDBPyConnection, name: str, queue_size: int) -> tuple[Path, dict[str, Any]]:
    selected = con.execute(
        f'''
        WITH queued AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY signal_date
                    ORDER BY base_entry_score DESC, stock_code
                ) AS signal_queue_rank
            FROM raw_pool
        ),
        hard_pass AS (
            SELECT *
            FROM queued
            WHERE signal_queue_rank <= {queue_size}
              AND buy_day_hard_gate_pass = 1
        ),
        picked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY signal_date
                    ORDER BY signal_queue_rank, base_entry_score DESC, stock_code
                ) AS final_rank
            FROM hard_pass
        )
        SELECT *
        FROM picked
        WHERE final_rank <= 3
        ORDER BY signal_date, final_rank, stock_code
        '''
    ).fetchdf()

    frames: list[pd.DataFrame] = []
    for (_signal_date, _buy_date), group in selected.groupby(["signal_date", "buy_date"], sort=True):
        frames.append(to_rows(group.sort_values(["final_rank", "stock_code"]), name))
    signals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    out_file = SIGNAL_DIR / f"{name}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_file, index=False, encoding="utf-8-sig")

    per_day = signals.groupby("signal_date").size() if not signals.empty else pd.Series(dtype=int)
    audit = {
        "variant": name,
        "queue_size": queue_size,
        "signal_file": str(out_file),
        "rows": int(len(signals)),
        "signal_days": int(signals["signal_date"].nunique()) if not signals.empty else 0,
        "buy_days": int(signals["buy_date"].nunique()) if not signals.empty else 0,
        "days_below_3": int((per_day < 3).sum()) if not signals.empty else 0,
        "duplicate_buy_stock_keys": int(signals.duplicated(["buy_date", "stock_code"]).sum()) if not signals.empty else 0,
        "bj_rows": int(signals["stock_code"].astype(str).str.endswith(".BJ").sum()) if not signals.empty else 0,
        "target_sum_min": float(signals.groupby("buy_date")["target_pct"].sum().min()) if not signals.empty else 0.0,
        "target_sum_max": float(signals.groupby("buy_date")["target_pct"].sum().max()) if not signals.empty else 0.0,
        "target_sum_mean": float(signals.groupby("buy_date")["target_pct"].sum().mean()) if not signals.empty else 0.0,
    }
    return out_file, audit


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        sources = attach_sources(con)
        pool_audit = build_candidate_pool(con, sources, restrict_to_production_signal_days=True)
        variants = []
        for name, queue_size in [
            ("no_backup_top3_then_gate", 3),
            ("backup_top5_then_gate", 5),
            ("backup_top8_then_gate", 8),
            ("backup_top12_then_gate", 12),
        ]:
            _file, audit = build_variant(con, name, queue_size)
            variants.append(audit)
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "strategy_boundary": "research_only_not_production",
        "input_contract": "active formal L4 DuckDB manifests, allow_legacy=false",
        "market_contract": "DuckDB L2 STOCK_DAILY_DATA raw open/pre_close for execution gates",
        "pool_audit": pool_audit,
        "variants": variants,
    }
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(
        "# 买入日硬门槛补位队列研究生成报告\n\n"
        "本报告为 research-only，不修改生产策略参数，不生成 L7 交易交付。\n\n"
        f"- 候选池日期：`{pool_audit['min_signal_date']}` 到 `{pool_audit['max_signal_date']}`\n"
        f"- 候选池行数：`{pool_audit['candidate_rows']}`\n"
        f"- 买入日硬门槛通过行数：`{pool_audit['hard_gate_pass_rows']}`\n"
        f"- 输出目录：`{SIGNAL_DIR}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
