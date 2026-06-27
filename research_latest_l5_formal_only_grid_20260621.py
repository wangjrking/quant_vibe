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
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "latest_l5_formal_only_grid_20260621"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


LABELS = {
    "3d": "executable_3d_open_return",
    "5d": "executable_5d_open_return",
    "10d": "executable_10d_open_return",
}

SCORE_SQL = {
    "std_5d": "rank_5d",
    "std_10d": "rank_10d",
    "blend_10d60_5d30_3d10": "(0.60 * rank_10d) + (0.30 * rank_5d) + (0.10 * rank_3d)",
    "blend_10d80_5d20": "(0.80 * rank_10d) + (0.20 * rank_5d)",
    "min_consensus": "MIN(rank_3d, rank_5d, rank_10d)",
    "max_any": "MAX(rank_3d, rank_5d, rank_10d)",
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
        candidates.append(
            (
                str(manifest.get("max_trade_date") or ""),
                str(manifest.get("generated_at") or ""),
                manifest_path,
                manifest,
            )
        )
    if not candidates:
        raise RuntimeError(f"missing approved_for_l5 manifest for {label}")
    _, _, manifest_path, manifest = sorted(candidates)[-1]
    return {
        "label": label,
        "table": str(manifest["table"]),
        "db_path": (manifest_path.parent / str(manifest["db_path"])).resolve(),
        "manifest_path": manifest_path.resolve(),
        "max_trade_date": str(manifest.get("max_trade_date") or ""),
        "candidate_id": str(manifest.get("candidate_id") or ""),
    }


def _formal_assets() -> dict[str, dict]:
    assets = {key: _load_formal_manifest(label) for key, label in LABELS.items()}
    db_paths = {str(row["db_path"]) for row in assets.values()}
    if len(db_paths) != 1:
        raise RuntimeError(f"formal manifests point to multiple db files: {sorted(db_paths)}")
    stale = {key: row["max_trade_date"] for key, row in assets.items() if row["max_trade_date"] < END_DATE}
    if stale:
        raise RuntimeError(f"formal manifests do not cover {END_DATE}: {stale}")
    return assets


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _params_grid() -> list[dict]:
    assets = _formal_assets()
    runs = []
    filters = [
        {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 5, "max_positions": 10},
        {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 7, "max_positions": 10},
        {"min_total_mv": None, "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 5, "max_positions": 10},
        {"min_total_mv": 120000.0, "max_total_mv": 250000.0, "min_amount": 20000.0, "min_turnover_rate": 1.0, "top_k": 5, "holding_days": 6, "max_positions": 10},
        {"min_total_mv": 120000.0, "max_total_mv": 200000.0, "min_amount": 50000.0, "min_turnover_rate": 1.0, "top_k": 8, "holding_days": 6, "max_positions": 10},
    ]
    score_modes = [
        ("std_5d", "top"),
        ("std_10d", "top"),
        ("blend_10d60_5d30_3d10", "top"),
        ("blend_10d80_5d20", "top"),
        ("min_consensus", "top"),
        ("min_consensus", "bottom"),
        ("max_any", "top"),
    ]
    for score_mode, direction in score_modes:
        for filter_params in filters:
            runs.append(
                {
                    **filter_params,
                    "score_mode": score_mode,
                    "direction": direction,
                    "target_total_pct": 0.98,
                    "weight_mode": "equal",
                    "score_continue_entry_ratio": 1.0,
                    "open_score_exit": 0,
                    "max_daily_sells": 1,
                    "model_db": str(assets["5d"]["db_path"]),
                    "table_3d": assets["3d"]["table"],
                    "table_5d": assets["5d"]["table"],
                    "table_10d": assets["10d"]["table"],
                    "manifest_3d": str(assets["3d"]["manifest_path"]),
                    "manifest_5d": str(assets["5d"]["manifest_path"]),
                    "manifest_10d": str(assets["10d"]["manifest_path"]),
                    "candidate_3d": assets["3d"]["candidate_id"],
                    "candidate_5d": assets["5d"]["candidate_id"],
                    "candidate_10d": assets["10d"]["candidate_id"],
                }
            )
    return runs


def _slug(params: dict) -> str:
    return (
        f"{params['score_mode']}_{params['direction']}"
        f"_tk{params['top_k']}_h{params['holding_days']}_mp{params['max_positions']}"
        f"_mv{_safe(params['min_total_mv'])}_{_safe(params['max_total_mv'])}"
        f"_amt{_safe(params['min_amount'])}_turn{_safe(params['min_turnover_rate'])}"
    )


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def _score_expression(params: dict) -> str:
    expression = SCORE_SQL[params["score_mode"]]
    return f"(1.0 - ({expression}))" if params["direction"] == "bottom" else expression


def _load_rows(params: dict) -> list[dict]:
    where = [
        "d10.trade_date >= ?",
        "d10.trade_date <= ?",
        "d3.pred_prob IS NOT NULL",
        "d5.pred_prob IS NOT NULL",
        "d10.pred_prob IS NOT NULL",
        "d10.stock_code NOT LIKE '%.BJ'",
        "COALESCE(m.name, '') NOT LIKE 'ST%'",
        "COALESCE(m.name, '') NOT LIKE '*ST%'",
        "COALESCE(m.name, '') NOT LIKE '%退市%'",
        "COALESCE(m.name, '') NOT LIKE '退%'",
        "(m.limit_times IS NULL OR m.limit_times = '' OR m.limit_times = 'None' OR CAST(m.limit_times AS REAL) = 0)",
    ]
    values: list[object] = [START_DATE, END_DATE]
    if params.get("min_total_mv") is not None:
        where.append("m.total_mv IS NOT NULL AND m.total_mv >= ?")
        values.append(float(params["min_total_mv"]))
    if params.get("max_total_mv") is not None:
        where.append("m.total_mv IS NOT NULL AND m.total_mv <= ?")
        values.append(float(params["max_total_mv"]))
    if params.get("min_amount") is not None:
        where.append("m.amount IS NOT NULL AND m.amount >= ?")
        values.append(float(params["min_amount"]))
    if params.get("min_turnover_rate") is not None:
        where.append("m.turnover_rate IS NOT NULL AND m.turnover_rate >= ?")
        values.append(float(params["min_turnover_rate"]))

    buffer_k = max(int(params["top_k"]), min(300, int(params["top_k"]) * 20 + 20))
    score_expr = _score_expression(params)
    conn = _connect_readonly(Path(params["model_db"]))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS market", (str(MARKET_DB),))
        rows = conn.execute(
            f"""
            WITH joined AS (
                SELECT
                    d10.trade_date,
                    d10.stock_code,
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
                FROM "{params['table_10d']}" d10
                INNER JOIN "{params['table_5d']}" d5
                  ON d10.trade_date = d5.trade_date AND d10.stock_code = d5.stock_code
                INNER JOIN "{params['table_3d']}" d3
                  ON d10.trade_date = d3.trade_date AND d10.stock_code = d3.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON d10.trade_date = m.trade_date AND d10.stock_code = m.stock_code
                WHERE {" AND ".join(where)}
            ),
            ranked AS (
                SELECT
                    joined.*,
                    CAST(RANK() OVER (PARTITION BY trade_date ORDER BY pred_3d ASC) AS REAL)
                      / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL) AS rank_3d,
                    CAST(RANK() OVER (PARTITION BY trade_date ORDER BY pred_5d ASC) AS REAL)
                      / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL) AS rank_5d,
                    CAST(RANK() OVER (PARTITION BY trade_date ORDER BY pred_10d ASC) AS REAL)
                      / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL) AS rank_10d
                FROM joined
            ),
            scored AS (
                SELECT
                    ranked.*,
                    {score_expr} AS pred_prob,
                    ROW_NUMBER() OVER (
                        PARTITION BY trade_date
                        ORDER BY {score_expr} DESC, stock_code
                    ) AS __rn
                FROM ranked
            )
            SELECT *
            FROM scored
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
            max_atr_ratio=None,
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
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
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
        str(params["model_db"]),
        "--score-table",
        str(params["table_10d"]),
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
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    grid = _params_grid()
    results: list[dict] = []
    for index, params in enumerate(grid, start=1):
        slug = _slug(params)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        error = None
        returncode = None
        indicator = {}
        try:
            if not signal_file.exists():
                _write_signal(params, _load_rows(params), signal_file)
            returncode = _run_backtest(params, signal_file, log_file)
            indicator = _extract_indicator(log_file) or {}
            signal_stats = _signal_stats(signal_file)
        except Exception as exc:
            error = str(exc)
            signal_stats = {"signal_count": 0, "buy_days": 0}
        result = {
            **params,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "error": error,
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **signal_stats,
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_key, reverse=True))
        print(f"[{index}/{len(grid)}] {slug} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}", flush=True)
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
