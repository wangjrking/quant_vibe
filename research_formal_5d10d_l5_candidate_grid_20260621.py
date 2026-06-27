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
    / "formal_5d10d_l5_candidate_grid_20260621"
)
FUSION_DB = REPORT_DIR / "fusion_5d10d.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


COMBOS = {
    "rank_10d90_5d10": "(rank_10d * 0.90) + (rank_5d * 0.10)",
    "rank_10d80_5d20": "(rank_10d * 0.80) + (rank_5d * 0.20)",
    "rank_10d70_5d30": "(rank_10d * 0.70) + (rank_5d * 0.30)",
    "rank_10d60_5d40": "(rank_10d * 0.60) + (rank_5d * 0.40)",
    "rank_10d50_5d50": "(rank_10d * 0.50) + (rank_5d * 0.50)",
    "rank_min_5d10d": "MIN(rank_5d, rank_10d)",
    "rank_max_5d10d": "MAX(rank_5d, rank_10d)",
}


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rank_pct_sql(column: str) -> str:
    return f"""(
        CAST(RANK() OVER (PARTITION BY trade_date ORDER BY {column} ASC) AS REAL)
        + (CAST(COUNT(*) OVER (PARTITION BY trade_date, {column}) AS REAL) - 1.0) / 2.0
    ) / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL)"""


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
    m5 = _load_formal_manifest("executable_5d_open_return")
    m10 = _load_formal_manifest("executable_10d_open_return")
    if Path(m5["db_path"]).resolve() != Path(m10["db_path"]).resolve():
        raise RuntimeError("5D and 10D formal assets must resolve to the same DB for this research script")
    if FUSION_DB.exists():
        manifest_path = REPORT_DIR / "fusion_5d10d_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        FUSION_DB.unlink()
    conn = sqlite3.connect(FUSION_DB)
    try:
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(m5['db_path']))} AS model")
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(MARKET_DB))} AS market")
        t5 = _quote_ident(str(m5["table"]))
        t10 = _quote_ident(str(m10["table"]))
        conn.executescript(
            f"""
            CREATE TABLE fusion_rank_base AS
            WITH joined AS (
                SELECT
                    d10.trade_date AS trade_date,
                    d10.stock_code AS stock_code,
                    d10.pred_prob AS pred_10d,
                    d5.pred_prob AS pred_5d,
                    m.name,
                    m.pre_close,
                    m.open,
                    m.close,
                    m.amount,
                    m.turnover_rate,
                    m.total_mv,
                    m.atr_qfq,
                    m.limit_times
                FROM model.{t10} d10
                INNER JOIN model.{t5} d5
                  ON d10.trade_date = d5.trade_date AND d10.stock_code = d5.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON d10.trade_date = m.trade_date AND d10.stock_code = m.stock_code
                WHERE d10.trade_date >= '{START_DATE}'
                  AND d10.trade_date <= '{END_DATE}'
                  AND d10.pred_prob IS NOT NULL
                  AND d5.pred_prob IS NOT NULL
            ),
            ranked AS (
                SELECT
                    joined.*,
                    {_rank_pct_sql("pred_5d")} AS rank_5d,
                    {_rank_pct_sql("pred_10d")} AS rank_10d
                FROM joined
            )
            SELECT * FROM ranked;

            CREATE INDEX idx_fusion_rank_base_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_rank_base_trade_rank10d ON fusion_rank_base(trade_date, rank_10d);
            """
        )
        combo_rows = []
        for combo_name, formula in COMBOS.items():
            conn.executescript(
                f"""
                CREATE TABLE {_quote_ident(combo_name)} AS
                SELECT
                    trade_date,
                    stock_code,
                    {formula} AS pred_prob,
                    pred_5d,
                    pred_10d,
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
                CREATE INDEX idx_{combo_name}_trade_pred ON {_quote_ident(combo_name)}(trade_date, pred_prob DESC);
                """
            )
            stats = conn.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT trade_date), COUNT(DISTINCT stock_code), MIN(trade_date), MAX(trade_date) FROM {_quote_ident(combo_name)}"
            ).fetchone()
            combo_rows.append(
                {
                    "combo_name": combo_name,
                    "formula": formula,
                    "table": combo_name,
                    "row_count": int(stats[0] or 0),
                    "trade_days": int(stats[1] or 0),
                    "stock_count": int(stats[2] or 0),
                    "min_trade_date": stats[3],
                    "max_trade_date": stats[4],
                }
            )
        conn.commit()
    finally:
        conn.close()
    manifest = {
        "generated_from": {
            "model_db": str(m5["db_path"]),
            "market_db": str(MARKET_DB),
            "formal_5d": {key: str(value) if isinstance(value, Path) else value for key, value in m5.items()},
            "formal_10d": {key: str(value) if isinstance(value, Path) else value for key, value in m10.items()},
            "source_mode": "latest_formal_l4_5d10d_only",
        },
        "output_db": str(FUSION_DB),
        "combos": combo_rows,
    }
    (REPORT_DIR / "fusion_5d10d_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _assets(manifest: dict) -> dict[str, dict]:
    generated = manifest["generated_from"]
    rows = {
        "std_5d": {
            "asset": "std_5d",
            "db_path": Path(generated["formal_5d"]["db_path"]),
            "table": generated["formal_5d"]["table"],
            "manifest_path": Path(generated["formal_5d"]["manifest_path"]),
        },
        "std_10d": {
            "asset": "std_10d",
            "db_path": Path(generated["formal_10d"]["db_path"]),
            "table": generated["formal_10d"]["table"],
            "manifest_path": Path(generated["formal_10d"]["manifest_path"]),
        },
    }
    for combo in manifest["combos"]:
        rows[combo["combo_name"]] = {
            "asset": combo["combo_name"],
            "db_path": Path(manifest["output_db"]),
            "table": combo["table"],
            "manifest_path": REPORT_DIR / "fusion_5d10d_manifest.json",
        }
    return rows


def _params_grid(manifest: dict) -> list[dict]:
    assets = _assets(manifest)
    focus_assets = [
        "std_10d",
        "std_5d",
        "rank_10d90_5d10",
        "rank_10d80_5d20",
        "rank_10d70_5d30",
        "rank_10d60_5d40",
        "rank_10d50_5d50",
        "rank_min_5d10d",
        "rank_max_5d10d",
    ]
    filters = [
        {"max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 5, "max_positions": 8, "max_atr_ratio": None},
        {"max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 5, "max_positions": 10, "max_atr_ratio": None},
        {"max_total_mv": 300000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 5, "max_positions": 8, "max_atr_ratio": None},
        {"max_total_mv": 500000.0, "min_amount": 50000.0, "min_turnover_rate": 1.0, "top_k": 5, "holding_days": 5, "max_positions": 8, "max_atr_ratio": None},
        {"max_total_mv": 500000.0, "min_amount": 20000.0, "min_turnover_rate": 2.0, "top_k": 8, "holding_days": 6, "max_positions": 10, "max_atr_ratio": None},
    ]
    runs = []
    for asset_name in focus_assets:
        for direction in ("top",):
            for filter_params in filters:
                runs.append(
                    {
                        **assets[asset_name],
                        **filter_params,
                        "direction": direction,
                        "target_total_pct": 0.98,
                        "weight_mode": "equal",
                        "open_score_exit": 0,
                        "max_daily_sells": 1,
                        "score_continue_entry_ratio": 1.0,
                    }
                )
    return runs


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _slug(params: dict) -> str:
    return (
        f"{params['asset']}_{params['direction']}"
        f"_tk{params['top_k']}_h{params['holding_days']}_mp{params['max_positions']}"
        f"_mv{_safe(params['max_total_mv'])}_amt{_safe(params['min_amount'])}"
        f"_turn{_safe(params['min_turnover_rate'])}_atr{_safe(params['max_atr_ratio'])}"
    )


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def _load_rows(params: dict) -> list[dict]:
    where = ["p.trade_date >= ?", "p.trade_date <= ?", "p.pred_prob IS NOT NULL"]
    values: list[object] = [START_DATE, END_DATE]
    if params.get("max_total_mv") is not None:
        where.append("m.total_mv IS NOT NULL AND m.total_mv <= ?")
        values.append(float(params["max_total_mv"]))
    if params.get("min_amount") is not None:
        where.append("m.amount IS NOT NULL AND m.amount >= ?")
        values.append(float(params["min_amount"]))
    if params.get("min_turnover_rate") is not None:
        where.append("m.turnover_rate IS NOT NULL AND m.turnover_rate >= ?")
        values.append(float(params["min_turnover_rate"]))
    if params.get("max_atr_ratio") is not None:
        where.append("m.atr_qfq IS NOT NULL AND m.close IS NOT NULL AND m.close > 0 AND m.atr_qfq / m.close <= ?")
        values.append(float(params["max_atr_ratio"]))
    direction_sql = "ASC" if params["direction"] == "bottom" else "DESC"
    score_expr = "-p.pred_prob" if params["direction"] == "bottom" else "p.pred_prob"
    buffer_k = max(int(params["top_k"]), min(300, int(params["top_k"]) * 20 + 20))
    conn = _connect_readonly(Path(params["db_path"]))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(MARKET_DB))} AS market")
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    p.trade_date,
                    p.stock_code,
                    {score_expr} AS pred_prob,
                    p.pred_prob AS raw_pred_prob,
                    m.name,
                    m.pre_close,
                    m.open,
                    m.close,
                    m.amount,
                    m.turnover_rate,
                    m.total_mv,
                    m.atr_qfq,
                    m.limit_times,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.trade_date
                        ORDER BY p.pred_prob {direction_sql}, p.stock_code
                    ) AS __rn
                FROM "{params['table']}" p
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON p.trade_date = m.trade_date AND p.stock_code = m.stock_code
                WHERE {" AND ".join(where)}
                  AND p.stock_code NOT LIKE '%.BJ'
                  AND COALESCE(m.name, '') NOT LIKE 'ST%'
                  AND COALESCE(m.name, '') NOT LIKE '*ST%'
                  AND COALESCE(m.name, '') NOT LIKE '%退市%'
                  AND COALESCE(m.name, '') NOT LIKE '退%'
                  AND (m.limit_times IS NULL OR m.limit_times = '' OR m.limit_times = 'None' OR CAST(m.limit_times AS REAL) = 0)
            )
            WHERE __rn <= ?
            ORDER BY trade_date, pred_prob DESC, stock_code
            """,
            [*values, buffer_k],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _write_signal(params: dict, rows: list[dict], signal_file: Path) -> None:
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    signals = build_gm_signal_rows(
        rows,
        config=SelectionConfig(
            top_k=int(params["top_k"]),
            pred_col="pred_prob",
            min_pred_prob=None,
            max_atr_ratio=params.get("max_atr_ratio"),
            min_amount=params.get("min_amount"),
            min_turnover_rate=params.get("min_turnover_rate"),
            max_total_mv=params.get("max_total_mv"),
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_delisting=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=int(params["max_positions"]),
        weight_mode=str(params["weight_mode"]),
        target_total_pct=float(params["target_total_pct"]),
    )
    if not signals:
        raise RuntimeError(f"no signals generated for {_slug(params)}")
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
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _run_backtest(params: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(params["open_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(params["max_daily_sells"]),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(params["score_continue_entry_ratio"]),
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
        str(params["holding_days"]),
        "--target-position-pct",
        str(float(params["target_total_pct"]) / max(1, int(params["max_positions"]))),
        "--score-db",
        str(params["db_path"]),
        "--score-table",
        str(params["table"]),
        "--market-db",
        str(MARKET_DB),
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
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _sort_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    manifest = build_fusion_db()
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    grid = _params_grid(manifest)
    for index, params in enumerate(grid, start=1):
        slug = _slug(params)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            _write_signal(params, _load_rows(params), signal_file)
        returncode = _run_backtest(params, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **{key: value for key, value in params.items() if key not in {"db_path", "manifest_path"}},
            "db_path": str(params["db_path"]),
            "manifest_path": str(params["manifest_path"]),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_key, reverse=True))
        print(
            f"[{index}/{len(grid)}] {slug} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_key, reverse=True)
        if float(row.get("annual") or -999) >= 2.0
        and float(row.get("sharpe") or -999) >= 3.0
        and float(row.get("avg_invested_pct") or -999) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
