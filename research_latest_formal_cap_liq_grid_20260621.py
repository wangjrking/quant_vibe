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
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "latest_formal_cap_liq_grid_20260621"
)
FUSION_MANIFEST = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "latest_standard_chain_research_new5d_20260621"
    / "fusion"
    / "fusion_manifest.json"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


def _load_formal_asset(label: str, asset: str) -> dict:
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
        raise RuntimeError(f"missing approved formal asset for {label}")
    _, _, manifest_path, manifest = sorted(candidates)[-1]
    return {
        "asset": asset,
        "db_path": (manifest_path.parent / str(manifest["db_path"])).resolve(),
        "table": str(manifest["table"]),
        "manifest_path": manifest_path.resolve(),
        "max_trade_date": str(manifest.get("max_trade_date") or ""),
    }


def _load_fusion_assets() -> list[dict]:
    manifest = json.loads(FUSION_MANIFEST.read_text(encoding="utf-8"))
    output_db = Path(str(manifest["output_db"])).resolve()
    by_table = {row["table"]: row for row in manifest.get("combos", [])}
    assets = []
    for asset, table in {
        "fusion_mincons": "combo_rank_min_consensus",
        "fusion_10d60_5d30_3d10": "combo_rank_10d60_5d30_3d10",
        "fusion_10d80_5d20": "combo_rank_10d80_5d20",
        "fusion_max_any": "combo_rank_max_any",
    }.items():
        if table not in by_table:
            raise RuntimeError(f"fusion table missing: {table}")
        assets.append(
            {
                "asset": asset,
                "db_path": output_db,
                "table": table,
                "manifest_path": FUSION_MANIFEST.resolve(),
                "max_trade_date": str(by_table[table].get("max_trade_date") or ""),
            }
        )
    return assets


def _assets_by_name() -> dict[str, dict]:
    assets = [
        _load_formal_asset("executable_5d_open_return", "std_5d"),
        _load_formal_asset("executable_10d_open_return", "std_10d"),
        *_load_fusion_assets(),
    ]
    return {row["asset"]: row for row in assets}


def _params_grid() -> list[dict]:
    assets = _assets_by_name()
    runs = []
    base_filters = [
        {"max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 5, "max_atr_ratio": None},
        {"max_total_mv": 500000.0, "min_amount": 50000.0, "min_turnover_rate": 1.0, "top_k": 5, "holding_days": 5, "max_atr_ratio": None},
        {"max_total_mv": 1000000.0, "min_amount": 100000.0, "min_turnover_rate": 1.0, "top_k": 5, "holding_days": 6, "max_atr_ratio": 0.10},
        {"max_total_mv": 500000.0, "min_amount": 20000.0, "min_turnover_rate": 2.0, "top_k": 8, "holding_days": 6, "max_atr_ratio": None},
    ]
    asset_directions = [
        ("std_5d", "top"),
        ("std_5d", "bottom"),
        ("std_10d", "top"),
        ("std_10d", "bottom"),
        ("fusion_mincons", "bottom"),
        ("fusion_10d60_5d30_3d10", "top"),
        ("fusion_10d60_5d30_3d10", "bottom"),
        ("fusion_10d80_5d20", "top"),
        ("fusion_max_any", "top"),
    ]
    for asset_name, direction in asset_directions:
        for filter_params in base_filters:
            asset = assets[asset_name]
            top_k = int(filter_params["top_k"])
            holding_days = int(filter_params["holding_days"])
            max_positions = min(top_k * holding_days, 10)
            runs.append(
                {
                    **asset,
                    **filter_params,
                    "direction": direction,
                    "max_positions": max_positions,
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
    direction_sql = "ASC" if params["direction"] == "bottom" else "DESC"
    score_expr = "-p.pred_prob" if params["direction"] == "bottom" else "p.pred_prob"
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

    buffer_k = max(int(params["top_k"]), min(300, int(params["top_k"]) * 20 + 20))
    sql = f"""
        ATTACH DATABASE ? AS market;
    """
    conn = _connect_readonly(Path(params["db_path"]))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(sql, (str(MARKET_DB),))
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
    sharpe = float(row.get("sharpe") or -999.0)
    annual = float(row.get("annual") or -999.0)
    avg = float(row.get("avg_invested_pct") or -999.0)
    return sharpe, annual, avg


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
    row_cache: dict[str, list[dict]] = {}
    results: list[dict] = []
    for index, params in enumerate(_params_grid(), start=1):
        slug = _slug(params)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            rows = _load_rows(params)
            row_cache[slug] = rows
            _write_signal(params, rows, signal_file)
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
        print(f"[{index}/{len(_params_grid())}] {slug} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}", flush=True)
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
