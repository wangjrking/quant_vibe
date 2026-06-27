from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_1d3d_fusion_grid_20260622"
)
FUSION_DB = REPORT_DIR / "fusion_1d3d.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"

COMBOS = {
    "rank_1d50_3d50": "(rank_1d * 0.50) + (rank_3d * 0.50)",
    "rank_1d30_3d70": "(rank_1d * 0.30) + (rank_3d * 0.70)",
    "rank_1d70_3d30": "(rank_1d * 0.70) + (rank_3d * 0.30)",
    "rank_min_1d3d": "MIN(rank_1d, rank_3d)",
    "rank_max_1d3d": "MAX(rank_1d, rank_3d)",
    "rank_1d20_3d40_5d20_10d20": "(rank_1d * 0.20) + (rank_3d * 0.40) + (rank_5d * 0.20) + (rank_10d * 0.20)",
    "rank_1d20_3d30_5d20_10d30": "(rank_1d * 0.20) + (rank_3d * 0.30) + (rank_5d * 0.20) + (rank_10d * 0.30)",
    "rank_1d10_3d30_5d20_10d40": "(rank_1d * 0.10) + (rank_3d * 0.30) + (rank_5d * 0.20) + (rank_10d * 0.40)",
    "rank_5d80_10d20": "(rank_5d * 0.80) + (rank_10d * 0.20)",
    "rank_5d70_10d20_3d10": "(rank_5d * 0.70) + (rank_10d * 0.20) + (rank_3d * 0.10)",
    "rank_5d70_10d20_1d10": "(rank_5d * 0.70) + (rank_10d * 0.20) + (rank_1d * 0.10)",
    "rank_5d70_10d15_3d10_1d05": "(rank_5d * 0.70) + (rank_10d * 0.15) + (rank_3d * 0.10) + (rank_1d * 0.05)",
    "rank_10d60_5d30_3d10": "(rank_10d * 0.60) + (rank_5d * 0.30) + (rank_3d * 0.10)",
}


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


def _rank_pct_sql(column: str) -> str:
    return f"""(
        CAST(RANK() OVER (PARTITION BY trade_date ORDER BY {column} ASC) AS REAL)
        + (CAST(COUNT(*) OVER (PARTITION BY trade_date, {column}) AS REAL) - 1.0) / 2.0
    ) / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL)"""


def _create_combo_table(conn: sqlite3.Connection, combo_name: str, formula: str) -> dict:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote_ident(combo_name)} AS
        SELECT
            trade_date,
            stock_code,
            {formula} AS pred_prob,
            pred_1d,
            pred_3d,
            pred_5d,
            pred_10d,
            rank_1d,
            rank_3d,
            rank_5d,
            rank_10d,
            name,
            pre_close,
            open,
            close,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            limit_times
        FROM fusion_rank_base;
        CREATE INDEX IF NOT EXISTS idx_{combo_name}_trade_pred ON {_quote_ident(combo_name)}(trade_date, pred_prob DESC);
        """
    )
    stats = conn.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT trade_date), COUNT(DISTINCT stock_code), MIN(trade_date), MAX(trade_date) FROM {_quote_ident(combo_name)}"
    ).fetchone()
    return {
        "combo_name": combo_name,
        "formula": formula,
        "table": combo_name,
        "row_count": int(stats[0] or 0),
        "trade_days": int(stats[1] or 0),
        "stock_count": int(stats[2] or 0),
        "min_trade_date": stats[3],
        "max_trade_date": stats[4],
    }


def _load_formal_manifest(label: str) -> dict:
    candidates = []
    for manifest_path in MANIFEST_DIR.glob("*.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("label") != label:
            continue
        if manifest.get("approval_status") != "approved_for_l5":
            continue
        if manifest.get("source_type", "sqlite_table") != "sqlite_table":
            continue
        if not manifest.get("table") or not manifest.get("db_path"):
            continue
        candidates.append((str(manifest.get("max_trade_date") or ""), str(manifest.get("generated_at") or ""), manifest_path, manifest))
    if not candidates:
        raise RuntimeError(f"missing approved formal manifest for {label}")
    _, _, manifest_path, manifest = sorted(candidates)[-1]
    return {
        "label": label,
        "manifest_path": manifest_path.resolve(),
        "db_path": (manifest_path.parent / str(manifest["db_path"])).resolve(),
        "table": str(manifest["table"]),
        "max_trade_date": str(manifest.get("max_trade_date") or ""),
        "row_count": manifest.get("row_count"),
        "trade_days": manifest.get("trade_days"),
    }


def build_fusion_db() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {
        "1d": _load_formal_manifest("executable_1d_open_return"),
        "3d": _load_formal_manifest("executable_3d_open_return"),
        "5d": _load_formal_manifest("executable_5d_open_return"),
        "10d": _load_formal_manifest("executable_10d_open_return"),
    }
    resolved_dbs = {Path(item["db_path"]).resolve() for item in manifests.values()}
    if resolved_dbs != {MODEL_DB.resolve()}:
        raise RuntimeError(f"formal assets resolve to unexpected DBs: {sorted(str(path) for path in resolved_dbs)}")
    manifest_path = REPORT_DIR / "fusion_1d3d_manifest.json"
    if FUSION_DB.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_combo_names = {str(row.get("combo_name")) for row in manifest.get("combos", [])}
        missing_combos = [(name, formula) for name, formula in COMBOS.items() if name not in existing_combo_names]
        if not missing_combos:
            return manifest
        conn = sqlite3.connect(FUSION_DB)
        try:
            has_base = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fusion_rank_base'"
            ).fetchone()
            if not has_base:
                raise RuntimeError(f"{FUSION_DB} exists but missing fusion_rank_base")
            combo_rows = list(manifest.get("combos", []))
            for combo_name, formula in missing_combos:
                combo_rows.append(_create_combo_table(conn, combo_name, formula))
            conn.commit()
        finally:
            conn.close()
        manifest["combos"] = combo_rows
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
    if FUSION_DB.exists():
        FUSION_DB.unlink()

    conn = sqlite3.connect(FUSION_DB)
    try:
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(MODEL_DB))} AS model")
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(MARKET_DB))} AS market")
        t1 = _quote_ident(str(manifests["1d"]["table"]))
        t3 = _quote_ident(str(manifests["3d"]["table"]))
        t5 = _quote_ident(str(manifests["5d"]["table"]))
        t10 = _quote_ident(str(manifests["10d"]["table"]))
        conn.executescript(
            f"""
            CREATE TABLE fusion_rank_base AS
            WITH joined AS (
                SELECT
                    d1.trade_date AS trade_date,
                    d1.stock_code AS stock_code,
                    d1.pred_prob AS pred_1d,
                    d3.pred_prob AS pred_3d,
                    d5.pred_prob AS pred_5d,
                    d10.pred_prob AS pred_10d,
                    m.name,
                    m.pre_close,
                    m.open,
                    m.close,
                    m.amount,
                    m.turnover_rate,
                    m.total_mv,
                    m.atr_qfq,
                    m.limit_times
                FROM model.{t1} d1
                INNER JOIN model.{t3} d3
                  ON d1.trade_date = d3.trade_date AND d1.stock_code = d3.stock_code
                INNER JOIN model.{t5} d5
                  ON d1.trade_date = d5.trade_date AND d1.stock_code = d5.stock_code
                INNER JOIN model.{t10} d10
                  ON d1.trade_date = d10.trade_date AND d1.stock_code = d10.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON d1.trade_date = m.trade_date AND d1.stock_code = m.stock_code
                WHERE d1.trade_date >= '{START_DATE}'
                  AND d1.trade_date <= '{END_DATE}'
                  AND d1.pred_prob IS NOT NULL
                  AND d3.pred_prob IS NOT NULL
                  AND d5.pred_prob IS NOT NULL
                  AND d10.pred_prob IS NOT NULL
            ),
            ranked AS (
                SELECT
                    joined.*,
                    {_rank_pct_sql("pred_1d")} AS rank_1d,
                    {_rank_pct_sql("pred_3d")} AS rank_3d,
                    {_rank_pct_sql("pred_5d")} AS rank_5d,
                    {_rank_pct_sql("pred_10d")} AS rank_10d
                FROM joined
            )
            SELECT * FROM ranked;

            CREATE INDEX idx_fusion_rank_base_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_rank_base_trade_rank3d ON fusion_rank_base(trade_date, rank_3d);
            """
        )
        combo_rows = []
        for combo_name, formula in COMBOS.items():
            combo_rows.append(_create_combo_table(conn, combo_name, formula))
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "generated_from": {
            "model_db": str(MODEL_DB),
            "market_db": str(MARKET_DB),
            "formal_assets": {
                key: {field: str(value) if isinstance(value, Path) else value for field, value in item.items()}
                for key, item in manifests.items()
            },
            "source_mode": "latest_approved_formal_l4_1d3d_with_5d10d_comparison",
        },
        "output_db": str(FUSION_DB),
        "combos": combo_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _asset_rows(manifest: dict) -> dict[str, dict]:
    formal = manifest["generated_from"]["formal_assets"]
    assets = {
        "std_1d": {
            "asset": "std_1d",
            "db_path": Path(formal["1d"]["db_path"]),
            "table": formal["1d"]["table"],
            "manifest_path": Path(formal["1d"]["manifest_path"]),
        },
        "std_3d": {
            "asset": "std_3d",
            "db_path": Path(formal["3d"]["db_path"]),
            "table": formal["3d"]["table"],
            "manifest_path": Path(formal["3d"]["manifest_path"]),
        },
    }
    for combo in manifest["combos"]:
        assets[combo["combo_name"]] = {
            "asset": combo["combo_name"],
            "db_path": Path(manifest["output_db"]),
            "table": combo["table"],
            "manifest_path": REPORT_DIR / "fusion_1d3d_manifest.json",
        }
    return assets


def _grid(manifest: dict) -> list[dict]:
    assets = _asset_rows(manifest)
    focus = [
        "std_1d",
        "std_3d",
        "rank_1d50_3d50",
        "rank_1d30_3d70",
        "rank_1d70_3d30",
        "rank_min_1d3d",
        "rank_max_1d3d",
        "rank_1d20_3d40_5d20_10d20",
        "rank_1d20_3d30_5d20_10d30",
        "rank_1d10_3d30_5d20_10d40",
        "rank_5d80_10d20",
        "rank_5d70_10d20_3d10",
        "rank_5d70_10d20_1d10",
        "rank_5d70_10d15_3d10_1d05",
        "rank_10d60_5d30_3d10",
    ]
    rows = []
    base_variants = [
        {
            "suffix": "tk1_h2_mp2_p80_a8000_t0p3_mv200000_exit0",
            "top_k": 1,
            "holding_days": 2,
            "max_positions": 2,
            "target_total_pct": 0.80,
            "min_amount": 8000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "score_exit": False,
        },
        {
            "suffix": "tk2_h2_mp3_p80_a8000_t0p3_mv200000_exit0",
            "top_k": 2,
            "holding_days": 2,
            "max_positions": 3,
            "target_total_pct": 0.80,
            "min_amount": 8000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "score_exit": False,
        },
        {
            "suffix": "tk3_h3_mp5_p90_a8000_t0p3_mv200000_exit0",
            "top_k": 3,
            "holding_days": 3,
            "max_positions": 5,
            "target_total_pct": 0.90,
            "min_amount": 8000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "score_exit": False,
        },
        {
            "suffix": "tk3_h3_mp5_p90_a12000_t0p5_mv150000_exit0",
            "top_k": 3,
            "holding_days": 3,
            "max_positions": 5,
            "target_total_pct": 0.90,
            "min_amount": 12000.0,
            "min_turnover_rate": 0.5,
            "max_total_mv": 150000.0,
            "score_exit": False,
        },
        {
            "suffix": "tk3_h3_mp5_p90_a8000_t0p3_mv200000_exit1",
            "top_k": 3,
            "holding_days": 3,
            "max_positions": 5,
            "target_total_pct": 0.90,
            "min_amount": 8000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "score_exit": True,
            "score_exit_entry_ratio": 0.95,
            "min_holding_days_before_score_exit": 1,
            "max_daily_sells": 1,
        },
        {
            "suffix": "tk5_h5_mp5_p98_a10000_t0p3_mv200000_exit0",
            "top_k": 5,
            "holding_days": 5,
            "max_positions": 5,
            "target_total_pct": 0.98,
            "min_amount": 10000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "score_exit": False,
        },
        {
            "suffix": "tk5_h5_mp8_p120_a10000_t0p3_mv200000_exit0",
            "top_k": 5,
            "holding_days": 5,
            "max_positions": 8,
            "target_total_pct": 1.20,
            "min_amount": 10000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "score_exit": False,
        },
        {
            "suffix": "tk5_h5_mp8_p120_a10000_t0p3_mv200000_r1d50_r3d50_exit0",
            "top_k": 5,
            "holding_days": 5,
            "max_positions": 8,
            "target_total_pct": 1.20,
            "min_amount": 10000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "rank_1d_min": 0.50,
            "rank_3d_min": 0.50,
            "score_exit": False,
        },
        {
            "suffix": "tk5_h5_mp8_p120_a10000_t0p3_mv200000_r1d60_r3d60_exit0",
            "top_k": 5,
            "holding_days": 5,
            "max_positions": 8,
            "target_total_pct": 1.20,
            "min_amount": 10000.0,
            "min_turnover_rate": 0.3,
            "max_total_mv": 200000.0,
            "rank_1d_min": 0.60,
            "rank_3d_min": 0.60,
            "score_exit": False,
        },
    ]
    for asset_name in focus:
        for variant in base_variants:
            asset = assets[asset_name]
            name = f"{asset_name}_{variant['suffix']}"
            rows.append({**asset, **variant, "name": name})
    return rows


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()}


def _read_rows(db_path: Path, table: str, params: dict) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = _table_columns(conn, table)
        filters = ["trade_date >= ?", "trade_date <= ?", "pred_prob IS NOT NULL"]
        values: list[object] = [START_DATE, END_DATE]
        if "amount" in columns and params.get("min_amount") is not None:
            filters.append("amount IS NOT NULL AND amount >= ?")
            values.append(float(params["min_amount"]))
        if "turnover_rate" in columns and params.get("min_turnover_rate") is not None:
            filters.append("turnover_rate IS NOT NULL AND turnover_rate >= ?")
            values.append(float(params["min_turnover_rate"]))
        if "total_mv" in columns and params.get("max_total_mv") is not None:
            filters.append("total_mv IS NOT NULL AND total_mv <= ?")
            values.append(float(params["max_total_mv"]))
        if "rank_1d" in columns and params.get("rank_1d_min") is not None:
            filters.append("rank_1d IS NOT NULL AND rank_1d >= ?")
            values.append(float(params["rank_1d_min"]))
        if "rank_3d" in columns and params.get("rank_3d_min") is not None:
            filters.append("rank_3d IS NOT NULL AND rank_3d >= ?")
            values.append(float(params["rank_3d_min"]))
        if "rank_5d" in columns and params.get("rank_5d_min") is not None:
            filters.append("rank_5d IS NOT NULL AND rank_5d >= ?")
            values.append(float(params["rank_5d_min"]))
        if "rank_10d" in columns and params.get("rank_10d_min") is not None:
            filters.append("rank_10d IS NOT NULL AND rank_10d >= ?")
            values.append(float(params["rank_10d_min"]))
        if "stock_code" in columns:
            filters.append("stock_code NOT LIKE '%.BJ'")
        if "name" in columns:
            filters.append("(name IS NULL OR (name NOT LIKE 'ST%' AND name NOT LIKE '*ST%' AND name NOT LIKE '%退%'))")
        if "limit_times" in columns:
            filters.append("(limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) <= 0.0)")
        where_sql = " AND ".join(filters)
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) AS rn
                FROM {_quote_ident(table)}
                WHERE {where_sql}
            )
            SELECT *
            FROM ranked
            WHERE rn <= 500
            """,
            values,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _write_signal(params: dict, signal_file: Path) -> None:
    rows = _read_rows(Path(params["db_path"]), str(params["table"]), params)
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    config = SelectionConfig(
        top_k=int(params["top_k"]),
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        min_amount=float(params["min_amount"]) if params.get("min_amount") is not None else None,
        min_turnover_rate=float(params["min_turnover_rate"]) if params.get("min_turnover_rate") is not None else None,
        max_total_mv=float(params["max_total_mv"]) if params.get("max_total_mv") is not None else None,
        max_per_industry=999,
        exclude_bj=True,
        exclude_st=True,
        exclude_delisting=True,
        exclude_current_limit=True,
    )
    signals = build_gm_signal_rows(
        rows,
        config=config,
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=int(params["max_positions"]),
        weight_mode="equal",
        target_total_pct=float(params["target_total_pct"]),
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        for field in (
            "pred_1d",
            "pred_3d",
            "pred_5d",
            "pred_10d",
            "rank_1d",
            "rank_3d",
            "rank_5d",
            "rank_10d",
            "amount",
            "turnover_rate",
            "total_mv",
            "limit_times",
        ):
            if field in source:
                signal[field] = source.get(field)
    write_gm_signals_csv(signals, signal_file)


def _extract_indicator(log_file: Path) -> dict | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _run(params: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1" if params.get("score_exit") else "0",
            "GM_MAX_DAILY_SELLS": str(int(params.get("max_daily_sells") or 1)),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "999",
        }
    )
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(params["max_positions"]),
        "--holding-days",
        str(params["holding_days"]),
        "--max-holding-days",
        str(max(int(params["holding_days"]) + 2, 4)),
        "--target-position-pct",
        str(float(params["target_total_pct"]) / float(params["max_positions"])),
        "--score-db",
        str(params["db_path"]),
        "--score-table",
        str(params["table"]),
        "--market-db",
        str(MARKET_DB),
        "--score-continue-entry-ratio",
        "999",
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    if params.get("score_exit"):
        command.extend(
            [
                "--score-exit-entry-ratio",
                str(params.get("score_exit_entry_ratio", 0.95)),
                "--min-holding-days-before-score-exit",
                str(params.get("min_holding_days_before_score_exit", 1)),
            ]
        )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    manifest = build_fusion_db()
    results = []
    for params in _grid(manifest):
        signal_file = REPORT_DIR / "signals" / f"{params['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{params['name']}.log"
        if not signal_file.exists():
            _write_signal(params, signal_file)
        returncode = _run(params, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **params,
            "db_path": str(params["db_path"]),
            "manifest_path": str(params["manifest_path"]),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)

    if not results:
        raise SystemExit("no results generated")
    fieldnames = list(dict.fromkeys(key for row in results for key in row.keys()))
    outputs = {
        "summary.csv": results,
        "summary_all_completed.csv": results,
        "summary_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_all_by_sharpe.csv": sorted(results, key=lambda row: float(row.get("sharpe") or -999), reverse=True),
        "summary_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_all_by_annual.csv": sorted(results, key=lambda row: float(row.get("annual") or -999), reverse=True),
        "summary_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
        "summary_all_avg80_by_sharpe.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
        "summary_target_hits.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 3.0
            and float(row.get("sharpe") or -999) >= 4.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
        "summary_all_target_hits.csv": [
            row
            for row in sorted(results, key=lambda item: float(item.get("sharpe") or -999), reverse=True)
            if float(row.get("annual") or -999) >= 3.0
            and float(row.get("sharpe") or -999) >= 4.0
            and float(row.get("avg_invested_pct") or -999) >= 0.80
        ],
    }
    for filename, data in outputs.items():
        with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
