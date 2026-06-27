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

from gm_signal_module import to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_core_lowvol_filler_v2_20260622"
)
L5_STRATEGY_DIR = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_formal_5d10d_gap_v20260622"
)
CORE_SIGNAL_FILE = (
    L5_STRATEGY_DIR
    / "signals"
    / "backtest_signals_gap_t0p03_g1p08_b0p88.csv"
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


def _load_manifest(label: str) -> dict:
    candidates = []
    for path in MANIFEST_DIR.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if item.get("label") != label:
            continue
        if item.get("approval_status") != "approved_for_l5":
            continue
        if item.get("source_type") != "sqlite_table":
            continue
        candidates.append((str(item.get("max_trade_date") or ""), str(item.get("generated_at") or ""), path, item))
    if not candidates:
        raise RuntimeError(f"missing approved formal manifest: {label}")
    _, _, path, item = sorted(candidates)[-1]
    return {
        "path": path.resolve(),
        "db_path": (path.parent / str(item["db_path"])).resolve(),
        "table": str(item["table"]),
        "max_trade_date": str(item.get("max_trade_date") or ""),
        "candidate_id": str(item.get("candidate_id") or ""),
    }


def _formal_assets() -> dict[str, dict]:
    assets = {key: _load_manifest(label) for key, label in LABELS.items()}
    db_paths = {str(row["db_path"]) for row in assets.values()}
    if len(db_paths) != 1:
        raise RuntimeError(f"formal manifests point to multiple model db files: {sorted(db_paths)}")
    stale = {key: row["max_trade_date"] for key, row in assets.items() if row["max_trade_date"] < END_DATE}
    if stale:
        raise RuntimeError(f"formal manifests do not cover {END_DATE}: {stale}")
    return assets


def _l5_score_asset() -> dict:
    rules = json.loads((L5_STRATEGY_DIR / "trading_rules.json").read_text(encoding="utf-8"))
    score_asset = rules["score_rule"]["score_asset"]
    return {"db_path": str(Path(score_asset["db_path"]).resolve()), "table": str(score_asset["table"])}


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_core_signals(scale: float) -> list[dict]:
    rows = []
    with CORE_SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            out = dict(row)
            target = _to_float(out.get("target_pct"), 0.0) or 0.0
            out["target_pct"] = f"{target * scale:.5f}"
            out["sleeve"] = "core"
            rows.append(out)
    return rows


def _load_filler_rows(params: dict, assets: dict[str, dict]) -> list[dict]:
    score_sql = {
        "consensus": "(0.50 * rank_10d) + (0.25 * rank_5d) + (0.25 * rank_3d)",
        "min_consensus": "MIN(rank_3d, rank_5d, rank_10d)",
        "ten_lowvol": "(0.75 * rank_10d) + (0.15 * rank_5d) + (0.10 * rank_3d) - (atr_ratio * 0.35)",
        "three_confirm": "(0.60 * rank_10d) + (0.25 * rank_3d) + (0.15 * rank_5d)",
    }[params["score_mode"]]
    where = [
        "pred_3d IS NOT NULL",
        "pred_5d IS NOT NULL",
        "pred_10d IS NOT NULL",
        "stock_code NOT LIKE '%.BJ'",
        "COALESCE(name, '') NOT LIKE 'ST%'",
        "COALESCE(name, '') NOT LIKE '*ST%'",
        "COALESCE(name, '') NOT LIKE '%退市%'",
        "COALESCE(name, '') NOT LIKE '退%'",
        "(limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)",
        "open IS NOT NULL AND open > 0",
        "close IS NOT NULL AND close > 0",
        "atr_qfq IS NOT NULL AND atr_qfq > 0",
        "(atr_qfq / close) <= ?",
        "amount IS NOT NULL AND amount >= ?",
        "turnover_rate IS NOT NULL AND turnover_rate >= ?",
        "total_mv IS NOT NULL AND total_mv >= ?",
        "total_mv <= ?",
    ]
    values: list[object] = [
        float(params["max_atr_ratio"]),
        float(params["min_amount"]),
        float(params["min_turnover_rate"]),
        float(params["min_total_mv"]),
        float(params["max_total_mv"]),
    ]
    if params.get("min_rank_10d") is not None:
        where.append("rank_10d >= ?")
        values.append(float(params["min_rank_10d"]))
    if params.get("min_rank_3d") is not None:
        where.append("rank_3d >= ?")
        values.append(float(params["min_rank_3d"]))
    if params.get("max_pred_gap") is not None:
        where.append("ABS(pred_10d - pred_5d) <= ?")
        values.append(float(params["max_pred_gap"]))

    model_db = assets["10d"]["db_path"]
    conn = sqlite3.connect(model_db)
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
                FROM "{assets['10d']['table']}" d10
                INNER JOIN "{assets['5d']['table']}" d5
                  ON d10.trade_date = d5.trade_date AND d10.stock_code = d5.stock_code
                INNER JOIN "{assets['3d']['table']}" d3
                  ON d10.trade_date = d3.trade_date AND d10.stock_code = d3.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON d10.trade_date = m.trade_date AND d10.stock_code = m.stock_code
                WHERE d10.trade_date >= ? AND d10.trade_date <= ?
            ),
            ranked AS (
                SELECT
                    joined.*,
                    CAST(RANK() OVER (PARTITION BY trade_date ORDER BY pred_3d ASC) AS REAL)
                      / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL) AS rank_3d,
                    CAST(RANK() OVER (PARTITION BY trade_date ORDER BY pred_5d ASC) AS REAL)
                      / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL) AS rank_5d,
                    CAST(RANK() OVER (PARTITION BY trade_date ORDER BY pred_10d ASC) AS REAL)
                      / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL) AS rank_10d,
                    (atr_qfq / close) AS atr_ratio
                FROM joined
            ),
            filtered AS (
                SELECT
                    *,
                    ABS(pred_10d - pred_5d) AS pred_gap,
                    {score_sql} AS filler_score
                FROM ranked
                WHERE {" AND ".join(where)}
            ),
            picked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY trade_date
                        ORDER BY filler_score DESC, rank_10d DESC, atr_ratio ASC, stock_code
                    ) AS rn
                FROM filtered
            )
            SELECT *
            FROM picked
            WHERE rn <= ?
            ORDER BY trade_date, rn
            """,
            [START_DATE, END_DATE, *values, int(params["filler_top_k"])],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _next_trade_date_map() -> dict[str, str]:
    conn = sqlite3.connect(MARKET_DB)
    try:
        dates = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT DISTINCT trade_date
                FROM STOCK_DAILY_DATA
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date
                """,
                (START_DATE, END_DATE),
            ).fetchall()
        ]
    finally:
        conn.close()
    return {date: dates[idx + 1] for idx, date in enumerate(dates[:-1])}


def _is_next_open_limit_up(stock_code: str, buy_date: str) -> bool:
    conn = sqlite3.connect(MARKET_DB)
    try:
        row = conn.execute(
            """
            SELECT pre_close, open, name, limit_times
            FROM STOCK_DAILY_DATA
            WHERE trade_date = ? AND stock_code = ?
            """,
            (buy_date, stock_code),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return False
    pre_close, open_price, name, limit_times = row
    limit_value = _to_float(limit_times)
    if limit_value is not None and limit_value > 0:
        return True
    pre_close = _to_float(pre_close)
    open_price = _to_float(open_price)
    if pre_close is None or open_price is None or pre_close <= 0:
        return False
    pct = 0.05 if str(name or "").startswith(("ST", "*ST")) else 0.20 if str(stock_code).startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _build_hybrid_signal(params: dict, assets: dict[str, dict], signal_file: Path) -> dict:
    core = _read_core_signals(float(params["core_scale"]))
    filler_source = _load_filler_rows(params, assets)
    next_by_signal = _next_trade_date_map()
    core_keys = {(str(row.get("signal_date")), str(row.get("stock_code"))) for row in core}
    filler = []
    for row in filler_source:
        signal_date = str(row["trade_date"])
        stock_code = str(row["stock_code"])
        if (signal_date, stock_code) in core_keys:
            continue
        buy_date = next_by_signal.get(signal_date)
        if not buy_date:
            continue
        if _is_next_open_limit_up(stock_code, buy_date):
            continue
        rank = 100 + int(row["rn"])
        filler.append(
            {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "symbol": to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": row.get("name"),
                "rank": rank,
                "pred_prob": row.get("filler_score"),
                "holding_days": int(params["filler_holding_days"]),
                "target_pct": f"{float(params['filler_target_pct']):.5f}",
                "score_exit_entry_ratio": "0.98",
                "min_holding_days_before_score_exit": "3",
                "sleeve": "filler",
                "pred_3d": row.get("pred_3d"),
                "pred_5d": row.get("pred_5d"),
                "pred_10d": row.get("pred_10d"),
                "rank_3d": row.get("rank_3d"),
                "rank_5d": row.get("rank_5d"),
                "rank_10d": row.get("rank_10d"),
                "atr_ratio": row.get("atr_ratio"),
                "amount": row.get("amount"),
                "turnover_rate": row.get("turnover_rate"),
                "total_mv": row.get("total_mv"),
                "pred_gap": row.get("pred_gap"),
            }
        )
    rows = [*core, *filler]
    rows.sort(key=lambda item: (str(item.get("buy_date") or ""), int(float(item.get("rank") or 999999))))
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "signal_count": len(rows),
        "core_count": len(core),
        "filler_count": len(filler),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


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


def _run_backtest(params: dict, signal_file: Path, log_file: Path, assets: dict[str, dict], score_asset: dict) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
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
        "7",
        "--max-holding-days",
        "10",
        "--target-position-pct",
        str(params["target_position_cap"]),
        "--score-db",
        str(score_asset["db_path"]),
        "--score-table",
        str(score_asset["table"]),
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
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _grid() -> list[dict]:
    grid = []
    for score_mode in ("consensus", "min_consensus", "ten_lowvol", "three_confirm"):
        for filler_top_k in (1, 2, 3):
            for filler_target_pct in (0.06, 0.08, 0.10):
                grid.append(
                    {
                        "name": (
                            f"{score_mode}_fk{filler_top_k}_ft{_safe(filler_target_pct)}"
                            "_atr0p065_mv80_300_amt20_turn0p4_r10p75_r3p55"
                        ),
                        "score_mode": score_mode,
                        "core_scale": 1.0,
                        "filler_top_k": filler_top_k,
                        "filler_target_pct": filler_target_pct,
                        "filler_holding_days": 5,
                        "max_positions": 8,
                        "target_position_cap": 0.28,
                        "max_daily_sells": 1,
                        "score_continue_entry_ratio": 1.02,
                        "max_atr_ratio": 0.065,
                        "min_amount": 20000.0,
                        "min_turnover_rate": 0.4,
                        "min_total_mv": 80000.0,
                        "max_total_mv": 300000.0,
                        "min_rank_10d": 0.75,
                        "min_rank_3d": 0.55,
                        "max_pred_gap": None,
                    }
                )
    for score_mode in ("consensus", "three_confirm"):
        for filler_top_k in (2, 3):
            for filler_target_pct in (0.08, 0.10):
                grid.append(
                    {
                        "name": (
                            f"{score_mode}_fk{filler_top_k}_ft{_safe(filler_target_pct)}"
                            "_atr0p08_mv50_500_amt50_turn0p8_gap0p08"
                        ),
                        "score_mode": score_mode,
                        "core_scale": 1.0,
                        "filler_top_k": filler_top_k,
                        "filler_target_pct": filler_target_pct,
                        "filler_holding_days": 5,
                        "max_positions": 8,
                        "target_position_cap": 0.28,
                        "max_daily_sells": 1,
                        "score_continue_entry_ratio": 1.02,
                        "max_atr_ratio": 0.08,
                        "min_amount": 50000.0,
                        "min_turnover_rate": 0.8,
                        "min_total_mv": 50000.0,
                        "max_total_mv": 500000.0,
                        "min_rank_10d": 0.70,
                        "min_rank_3d": 0.50,
                        "max_pred_gap": 0.08,
                    }
                )
    return grid


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sort_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def main() -> int:
    assets = _formal_assets()
    score_asset = _l5_score_asset()
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    grid = _grid()
    max_runs = int(os.environ.get("MAX_RUNS", "0") or "0")
    if max_runs > 0:
        grid = grid[:max_runs]
    for index, params in enumerate(grid, start=1):
        signal_file = REPORT_DIR / "signals" / f"{params['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{params['name']}.log"
        error = None
        returncode = None
        indicator = {}
        signal_stats = {"signal_count": None, "core_count": None, "filler_count": None, "buy_days": None}
        try:
            signal_stats = _build_hybrid_signal(params, assets, signal_file)
            returncode = _run_backtest(params, signal_file, log_file, assets, score_asset)
            indicator = _extract_indicator(log_file) or {}
        except Exception as exc:
            error = str(exc)
        result = {
            **params,
            "returncode": returncode,
            "error": error,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
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
        _write_rows(
            REPORT_DIR / "summary_target_hits.csv",
            [
                row
                for row in sorted(results, key=_sort_key, reverse=True)
                if float(row.get("annual") or -999) >= 3.0
                and float(row.get("sharpe") or -999) >= 4.0
                and float(row.get("avg_invested_pct") or -999) >= 0.80
            ],
        )
        print(
            f"[{index}/{len(grid)}] {params['name']} "
            f"annual={result.get('annual')} sharpe={result.get('sharpe')} "
            f"avg={result.get('avg_invested_pct')} error={error}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
