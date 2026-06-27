"""Research-only Juejin tuning for standard-chain published prediction assets.

This script does not touch production strategy archives. It generates research
signal files from approved/standard L4 prediction assets and evaluates them
with the configured Juejin backtest entry.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import itertools
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from model_experiment_grid import write_rows
from selection_module import SelectionConfig


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MANIFEST_DIR = ROOT / "quant" / "main" / "config" / "prediction_manifests"
REPORTS_DIR = DATA_DIR / "reports"
FUSION_DB = (
    REPORTS_DIR
    / "strategy_agent_model_application_20260618"
    / "published_asset_fusions"
    / "fusion_combos.db"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260618"
    / "standard_chain_full_investment_round1"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260612"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-24 15:30:00"


FUSION_COMBO_TABLES = {
    "fusion_mincons": "combo_rank_min_consensus",
    "fusion_10d60_5d30_3d10": "combo_rank_10d60_5d30_3d10",
    "fusion_10d80_5d20": "combo_rank_10d80_5d20",
    "fusion_max_any": "combo_rank_max_any",
}

STANDARD_CHAIN_LABEL_TO_ASSET = {
    "executable_10d_open_return": "std_10d",
    "executable_5d_open_return": "std_5d",
}
STANDARD_CHAIN_ALLOWED_APPROVALS = {"approved_for_l5", "approved_for_l4_only"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=32, help="Maximum number of new/evaluated parameter rows.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--start-date", default=START_DATE, help="Signal selection start trade_date (YYYYMMDD).")
    parser.add_argument("--end-date", default=END_DATE, help="Signal selection end trade_date (YYYYMMDD).")
    parser.add_argument(
        "--min-standard-trade-date",
        default=None,
        help="Require standard-chain 5D/10D assets to cover at least this trade_date (YYYYMMDD).",
    )
    return parser.parse_args(argv)


def _manifest_sort_key(manifest: dict) -> tuple[str, str, str]:
    return (
        str(manifest.get("max_trade_date") or ""),
        str(manifest.get("generated_at") or ""),
        str(manifest.get("table") or ""),
    )


def _fusion_manifest_sort_key(manifest_path: Path, manifest: dict) -> tuple[str, str, str]:
    combo_dates = sorted(str(row.get("max_trade_date") or "") for row in manifest.get("combos", []))
    return (
        combo_dates[-1] if combo_dates else "",
        str(manifest.get("output_db") or ""),
        str(manifest_path),
    )


def resolve_fusion_assets(
    reports_dir: Path = REPORTS_DIR,
    min_trade_date: str | None = None,
) -> list[dict]:
    best: tuple[tuple[str, str, str], dict] | None = None
    for manifest_path in sorted(Path(reports_dir).glob("strategy_agent_model_application_*/**/fusion_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        combos = {str(row.get("table") or ""): row for row in manifest.get("combos", [])}
        required_tables = set(FUSION_COMBO_TABLES.values())
        if not required_tables.issubset(combos):
            continue
        output_db = manifest.get("output_db")
        if not output_db:
            continue
        candidate = {
            "manifest_path": manifest_path.resolve(),
            "db_path": Path(str(output_db)).resolve(),
            "combos": combos,
        }
        sort_key = _fusion_manifest_sort_key(manifest_path, manifest)
        if best is None or sort_key > best[0]:
            best = (sort_key, candidate)

    if best is None:
        raise RuntimeError("missing fusion manifest with required combo tables")

    selected = best[1]
    assets = [
        {
            "asset": asset_name,
            "db_path": selected["db_path"],
            "table": table_name,
            "max_trade_date": str(selected["combos"][table_name].get("max_trade_date") or ""),
            "manifest_path": selected["manifest_path"],
        }
        for asset_name, table_name in FUSION_COMBO_TABLES.items()
    ]
    if min_trade_date:
        stale = [row for row in assets if str(row.get("max_trade_date") or "") < str(min_trade_date)]
        if stale:
            stale_desc = ", ".join(
                f"{row['asset']}={row.get('max_trade_date') or 'missing'}" for row in stale
            )
            raise RuntimeError(f"stale fusion manifest for min_trade_date={min_trade_date}: {stale_desc}")
    return assets


def resolve_standard_chain_assets(
    manifest_dir: Path = MANIFEST_DIR,
    min_trade_date: str | None = None,
) -> list[dict]:
    best_by_label: dict[str, tuple[tuple[str, str, str], dict]] = {}
    for manifest_path in sorted(Path(manifest_dir).glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        label = str(manifest.get("label") or "")
        approval_status = str(manifest.get("approval_status") or "")
        table = str(manifest.get("table") or "")
        source_type = str(manifest.get("source_type") or "sqlite_table")
        raw_db_path = manifest.get("db_path")
        if label not in STANDARD_CHAIN_LABEL_TO_ASSET:
            continue
        if approval_status not in STANDARD_CHAIN_ALLOWED_APPROVALS:
            continue
        if source_type != "sqlite_table":
            continue
        if not raw_db_path or not table:
            continue

        resolved_asset = {
            "asset": STANDARD_CHAIN_LABEL_TO_ASSET[label],
            "db_path": (manifest_path.parent / str(raw_db_path)).resolve(),
            "table": table,
            "approval_status": approval_status,
            "max_trade_date": str(manifest.get("max_trade_date") or ""),
            "manifest_path": manifest_path.resolve(),
        }
        sort_key = _manifest_sort_key(manifest)
        current = best_by_label.get(label)
        if current is None or sort_key > current[0]:
            best_by_label[label] = (sort_key, resolved_asset)

    missing_labels = [label for label in STANDARD_CHAIN_LABEL_TO_ASSET if label not in best_by_label]
    if missing_labels:
        raise RuntimeError(f"missing standard-chain manifest for {', '.join(sorted(missing_labels))}")
    assets = [best_by_label[label][1] for label in sorted(STANDARD_CHAIN_LABEL_TO_ASSET)]
    if min_trade_date:
        stale = [row for row in assets if str(row.get("max_trade_date") or "") < str(min_trade_date)]
        if stale:
            stale_desc = ", ".join(
                f"{row['asset']}={row.get('max_trade_date') or 'missing'}" for row in stale
            )
            raise RuntimeError(
                f"stale standard-chain manifest for min_trade_date={min_trade_date}: {stale_desc}"
            )
    return assets


def resolve_assets(
    manifest_dir: Path = MANIFEST_DIR,
    reports_dir: Path = REPORTS_DIR,
    min_trade_date: str | None = None,
) -> list[dict]:
    return [
        *resolve_fusion_assets(reports_dir=reports_dir, min_trade_date=min_trade_date),
        *resolve_standard_chain_assets(manifest_dir=manifest_dir, min_trade_date=min_trade_date),
    ]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


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
            try:
                return eval(payload, {"__builtins__": {}}, {"datetime": datetime})
            except Exception:
                return None
    return None


def _signal_stats(signal_file: Path) -> dict:
    if not signal_file.exists():
        return {"signal_count": 0, "buy_days": 0}
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
    }


def _existing_result(params: dict, signal_file: Path, log_file: Path) -> dict | None:
    indicator = _extract_indicator(log_file)
    if not indicator:
        return None
    return _result_row(params, signal_file, log_file, indicator, 0)


def _result_row(params: dict, signal_file: Path, log_file: Path, indicator: dict | None, returncode: int) -> dict:
    row = {
        **params,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "returncode": returncode,
        **_signal_stats(signal_file),
    }
    if indicator:
        row.update(
            {
                "pnl_ratio": indicator.get("pnl_ratio"),
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
                "calmar_ratio": indicator.get("calmar_ratio"),
                "risk_ratio": indicator.get("risk_ratio"),
            }
        )
    else:
        row["error"] = "missing_indicator"
    return row


def _run_juejin(signal_file: Path, log_file: Path, params: dict) -> tuple[int, dict | None]:
    env = os.environ.copy()
    env.update(
        {
            "GM_SIGNAL_FILE": str(signal_file),
            "GM_MAX_POSITIONS": str(params["max_positions"]),
            "GM_HOLDING_DAYS": str(params["holding_days"]),
            "GM_TARGET_POSITION_PCT": str(
                float(params["target_total_pct"])
                / float(max(1, min(int(params["max_positions"]), int(params["holding_days"]))))
            ),
            "GM_MAX_HOLDING_DAYS": str(params["holding_days"]),
            "GM_BACKTEST_START": BACKTEST_START,
            "GM_BACKTEST_END": BACKTEST_END,
            "GM_BACKTEST_ADJUST": "none",
            "GM_BACKTEST_INITIAL_CASH": "600000",
            "GM_BACKTEST_SLIPPAGE_RATIO": "0.0015",
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(params.get("score_exit_entry_ratio", 0.95)),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(
                params.get("min_holding_days_before_score_exit", 2)
            ),
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(params.get("score_continue_entry_ratio", 1.0)),
            "GM_SCORE_DB": str(params["db_path"]),
            "GM_SCORE_TABLE": str(params["table"]),
            "GM_MARKET_DB": str(MARKET_DB),
            "GM_VERBOSE_TRADES": "0",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
        }
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(JUEJIN_PYTHON), "main.py"],
            cwd=str(STRATEGY_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    indicator = _extract_indicator(log_file)
    if proc.returncode != 0:
        raise RuntimeError(log_file.read_text(encoding="utf-8", errors="ignore")[-2000:] or "Juejin run failed")
    return proc.returncode, indicator


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path)
    try:
        return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=30)
    except Exception:
        return sqlite3.connect(path, timeout=30)


def _table_columns(db_path: Path, table: str) -> set[str]:
    conn = _connect_readonly(db_path)
    try:
        return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    finally:
        conn.close()


def _load_rows(cache: dict, params: dict, start_date: str = START_DATE, end_date: str = END_DATE) -> list[dict]:
    key = (
        str(params["db_path"]),
        params["table"],
        params["direction"],
        params["top_k"],
        params["max_total_mv"],
        params["min_amount"],
        params["min_turnover_rate"],
        params.get("max_atr_ratio"),
        str(start_date),
        str(end_date),
    )
    if key in cache:
        return cache[key]
    cols = _table_columns(params["db_path"], params["table"])
    where = ["trade_date >= ?", "trade_date <= ?", "pred_prob IS NOT NULL"]
    values: list[object] = [str(start_date), str(end_date)]
    if "stock_code" in cols:
        where.append("stock_code NOT LIKE '%.BJ'")
    if "name" in cols:
        where.append("COALESCE(name, '') NOT LIKE 'ST%'")
        where.append("COALESCE(name, '') NOT LIKE '*ST%'")
        where.append("COALESCE(name, '') NOT LIKE '%退市%'")
        where.append("COALESCE(name, '') NOT LIKE '退%'")
    if params["max_total_mv"] is not None and "total_mv" in cols:
        where.append("total_mv IS NOT NULL AND total_mv <= ?")
        values.append(float(params["max_total_mv"]))
    if params["min_amount"] is not None and "amount" in cols:
        where.append("amount IS NOT NULL AND amount >= ?")
        values.append(float(params["min_amount"]))
    if params["min_turnover_rate"] is not None and "turnover_rate" in cols:
        where.append("turnover_rate IS NOT NULL AND turnover_rate >= ?")
        values.append(float(params["min_turnover_rate"]))
    if params.get("max_atr_ratio") is not None and {"atr_qfq", "close"}.issubset(cols):
        where.append("atr_qfq IS NOT NULL AND close IS NOT NULL AND close > 0 AND atr_qfq / close <= ?")
        values.append(float(params["max_atr_ratio"]))
    if "limit_times" in cols:
        where.append("(limit_times IS NULL OR limit_times = '' OR limit_times = 'None')")

    direction_sql = "ASC" if params["direction"] == "bottom" else "DESC"
    buffer_k = max(int(params["top_k"]), min(300, int(params["top_k"]) * 20 + 20))
    sql = f"""
        SELECT *
        FROM (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY trade_date
                       ORDER BY pred_prob {direction_sql}, stock_code
                   ) AS __rn
            FROM "{params['table']}" p
            WHERE {" AND ".join(where)}
        )
        WHERE __rn <= ?
        ORDER BY trade_date, pred_prob {direction_sql}
    """
    conn = _connect_readonly(params["db_path"])
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute(sql, [*values, buffer_k])]
    finally:
        conn.close()
    if params["direction"] == "bottom":
        rows = [{**row, "pred_prob": -float(row.get("pred_prob") or 0.0)} for row in rows]
    cache[key] = rows
    return rows


def _build_signals(params: dict, rows: list[dict], market_rows: dict[str, dict[str, dict]]) -> list[dict]:
    return build_gm_signal_rows(
        rows,
        SelectionConfig(
            top_k=int(params["top_k"]),
            pred_col="pred_prob",
            min_pred_prob=None,
            min_pred_quantile=None,
            max_atr_ratio=params.get("max_atr_ratio"),
            min_amount=params["min_amount"],
            min_turnover_rate=params["min_turnover_rate"],
            max_total_mv=params["max_total_mv"],
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=int(params["holding_days"]),
        max_positions=int(params["max_positions"]),
        weight_mode=params["weight_mode"],
        target_total_pct=(
            None if params.get("target_total_pct") is None else float(params["target_total_pct"])
        ),
    )


def _params(
    manifest_dir: Path = MANIFEST_DIR,
    reports_dir: Path = REPORTS_DIR,
    min_trade_date: str | None = None,
) -> list[dict]:
    base = []
    for asset, direction, top_k, holding_days, max_total_mv in itertools.product(
        resolve_assets(
            manifest_dir=manifest_dir,
            reports_dir=reports_dir,
            min_trade_date=min_trade_date,
        ),
        ["top", "bottom"],
        [8, 10],
        [5, 6],
        [18000000.0, 20000000.0],
    ):
        base.append(
            {
                **asset,
                "direction": direction,
                "top_k": top_k,
                "max_positions": top_k,
                "holding_days": holding_days,
                "max_total_mv": max_total_mv,
                "min_amount": None,
                "min_turnover_rate": None,
                "max_atr_ratio": 0.10,
                "weight_mode": "equal",
                "target_total_pct": 0.98,
            }
        )
    focus_names = {
        "fusion_mincons",
        "fusion_10d60_5d30_3d10",
        "fusion_10d80_5d20",
        "std_10d",
    }
    focused = [row for row in base if row["asset"] in focus_names]
    tail = [row for row in base if row["asset"] not in focus_names]
    return focused + tail


def _slug(params: dict) -> str:
    return (
        f"{params['asset']}_{params['direction']}"
        f"_top{params['top_k']}_h{params['holding_days']}"
        f"_mv{_safe(params['max_total_mv'])}"
        f"_amt{_safe(params['min_amount'])}"
        f"_turn{_safe(params['min_turnover_rate'])}"
        f"_atr{_safe(params.get('max_atr_ratio'))}_{params['weight_mode']}"
    )


def _sort_key(row: dict) -> tuple[float, float, float]:
    sharpe = float(row.get("sharp_ratio") or -999.0)
    annual = float(row.get("pnl_ratio_annual") or -999.0)
    buy_days = float(row.get("buy_days") or 0.0)
    return sharpe, annual, buy_days


def main(argv=None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    start_date = str(args.start_date)
    end_date = str(args.end_date)
    signal_dir = report_dir / "signals"
    log_dir = report_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    market_rows = None
    row_cache: dict = {}
    results: list[dict] = []

    params_grid = _params(
        manifest_dir=MANIFEST_DIR,
        reports_dir=REPORTS_DIR,
        min_trade_date=(None if args.min_standard_trade_date in (None, "") else str(args.min_standard_trade_date)),
    )

    for index, params in enumerate(params_grid[: args.limit], start=1):
        slug = _slug(params)
        signal_file = signal_dir / f"{slug}.csv"
        log_file = log_dir / f"{slug}.log"
        existing = _existing_result(params, signal_file, log_file)
        if existing:
            result = existing
        else:
            if not signal_file.exists():
                if market_rows is None:
                    market_rows = load_market_rows_by_trade_date(MARKET_DB, start_date, end_date)
                rows = _load_rows(row_cache, params, start_date=start_date, end_date=end_date)
                signals = _build_signals(params, rows, market_rows)
                write_gm_signals_csv(signals, signal_file)
            returncode, indicator = _run_juejin(signal_file, log_file, params)
            result = _result_row(params, signal_file, log_file, indicator, returncode)
        results.append(result)
        write_rows(results, report_dir / "results.csv")
        write_rows(sorted(results, key=_sort_key, reverse=True), report_dir / "results_sorted.csv")
        print(
            f"[{index}/{min(args.limit, len(params_grid))}] {slug} "
            f"annual={result.get('pnl_ratio_annual')} sharpe={result.get('sharp_ratio')} "
            f"buy_days={result.get('buy_days')}",
            flush=True,
        )

    qualified = [
        row
        for row in results
        if float(row.get("pnl_ratio_annual") or 0.0) >= 2.0
        and float(row.get("sharp_ratio") or 0.0) >= 3.0
        and float(row.get("buy_days") or 0.0) >= 394.0
    ]
    write_rows(sorted(qualified, key=_sort_key, reverse=True), report_dir / "qualified_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
