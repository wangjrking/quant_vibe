from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import export_formal_5d10d_l5_signals as exporter
from gm_signal_module import write_gm_signals_csv


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_latest_manifest_pool_grid_20260622"
)
FUSION_DB = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_latest_manifest_fusion_20260622"
    / "fusion_5d10d_latest.db"
)
SCORE_DB = REPORT_DIR / "latest_pool_scores.db"
PRODUCTION_STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_formal_5d10d_gap_v20260622"
JUEJIN_STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "3",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.90",
    "GM_EQUITY_DD_HARD_SCALE": "0.70",
}

BEST_10D_ENV = dict(BASE_ENV)
BEST_10D_ENV.update(
    {
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.09",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.17",
        "GM_EQUITY_DD_SOFT_SCALE": "0.82",
        "GM_EQUITY_DD_HARD_SCALE": "0.58",
    }
)


def _variant(
    name: str,
    *,
    primary_w10: float,
    fallback_w10: float,
    rank_targets: dict[int, float],
    base_target_pct: float = 0.10925,
    primary_limit: int = 2,
    fallback_limit: int = 5,
    primary_count_min: int = 2,
    rank_max: int = 5,
    post_avg: float = 2.05,
    max_positions: int = 5,
    holding_days: int = 7,
    max_holding_days: int = 10,
    env: dict[str, str] | None = None,
    max_single: float = 0.42,
    score_exit_ratio: float = 1.0,
    min_hold_score_exit: int = 3,
) -> dict:
    run_env = dict(env or BASE_ENV)
    run_env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(score_exit_ratio)
    run_env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_hold_score_exit)
    return {
        "name": name,
        "primary_w10": primary_w10,
        "primary_w5": 1.0 - primary_w10,
        "fallback_w10": fallback_w10,
        "fallback_w5": 1.0 - fallback_w10,
        "base_target_pct": base_target_pct,
        "primary_limit": primary_limit,
        "fallback_limit": fallback_limit,
        "primary_count_min": primary_count_min,
        "rank_max": rank_max,
        "post_avg": post_avg,
        "rank_targets": rank_targets,
        "max_positions": max_positions,
        "holding_days": holding_days,
        "max_holding_days": max_holding_days,
        "env": run_env,
        "max_single": max_single,
        "score_exit_ratio": score_exit_ratio,
        "min_hold_score_exit": min_hold_score_exit,
    }


VARIANTS = [
    _variant(
        "prod_rules_latest",
        primary_w10=0.90,
        fallback_w10=0.60,
        rank_targets={1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19},
    ),
    _variant(
        "prod_no_avg_latest",
        primary_w10=0.90,
        fallback_w10=0.60,
        rank_targets={1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19},
        post_avg=0.0,
        primary_count_min=1,
    ),
    _variant(
        "rank4_conc_h6_bestenv",
        primary_w10=0.90,
        fallback_w10=0.60,
        rank_targets={1: 0.28, 2: 0.25, 3: 0.22, 4: 0.19},
        rank_max=4,
        holding_days=6,
        max_holding_days=6,
        env=BEST_10D_ENV,
        max_single=0.32,
        min_hold_score_exit=4,
    ),
    _variant(
        "rank5_conc_h6_bestenv",
        primary_w10=0.90,
        fallback_w10=0.60,
        rank_targets={1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.14},
        holding_days=6,
        max_holding_days=6,
        env=BEST_10D_ENV,
        max_single=0.30,
        min_hold_score_exit=4,
    ),
    _variant(
        "rank5_highgross_h6_bestenv",
        primary_w10=0.90,
        fallback_w10=0.60,
        rank_targets={1: 0.42, 2: 0.42, 3: 0.42, 4: 0.42, 5: 0.42},
        holding_days=6,
        max_holding_days=6,
        env=BEST_10D_ENV,
        max_single=0.42,
        min_hold_score_exit=4,
    ),
    _variant(
        "rank5_10d80_5d20_h6",
        primary_w10=0.80,
        fallback_w10=0.80,
        rank_targets={1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.14},
        holding_days=6,
        max_holding_days=6,
        env=BEST_10D_ENV,
        max_single=0.30,
        min_hold_score_exit=4,
    ),
    _variant(
        "rank5_10d70_5d30_h6",
        primary_w10=0.70,
        fallback_w10=0.70,
        rank_targets={1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.14},
        holding_days=6,
        max_holding_days=6,
        env=BEST_10D_ENV,
        max_single=0.30,
        min_hold_score_exit=4,
    ),
    _variant(
        "rank5_more_liq_h6",
        primary_w10=0.90,
        fallback_w10=0.60,
        rank_targets={1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.14},
        holding_days=6,
        max_holding_days=6,
        env=BEST_10D_ENV,
        max_single=0.30,
        min_hold_score_exit=4,
        primary_count_min=1,
        post_avg=0.0,
    ),
]


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _pool_formula(w10: float, w5: float) -> str:
    return f"(rank_10d * {w10:.12g}) + (rank_5d * {w5:.12g})"


def _score_table_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return f"pool_score_{clean}"


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _build_score_table(cfg: dict) -> str:
    table = _score_table_name(cfg["name"])
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(FUSION_DB),))
        primary_formula = _pool_formula(float(cfg["primary_w10"]), float(cfg["primary_w5"]))
        fallback_formula = _pool_formula(float(cfg["fallback_w10"]), float(cfg["fallback_w5"]))
        tradable = """
            stock_code NOT LIKE '%.BJ'
            AND COALESCE(name, '') NOT LIKE 'ST%'
            AND COALESCE(name, '') NOT LIKE '*ST%'
            AND COALESCE(name, '') NOT LIKE '%退市%'
            AND COALESCE(name, '') NOT LIKE '退%'
            AND COALESCE(name, '') NOT LIKE '%退'
            AND (limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)
            AND rank_5d IS NOT NULL
            AND rank_10d IS NOT NULL
            AND close IS NOT NULL
        """
        primary = f"""
            {tradable}
            AND total_mv IS NOT NULL AND total_mv <= 150000.0
            AND amount IS NOT NULL AND amount >= 10000.0
            AND turnover_rate IS NOT NULL AND turnover_rate >= 0.3
        """
        fallback = f"""
            {tradable}
            AND total_mv IS NOT NULL AND total_mv <= 200000.0
            AND amount IS NOT NULL AND amount >= 20000.0
            AND turnover_rate IS NOT NULL AND turnover_rate >= 0.5
        """
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {_quote_ident(table)};
            CREATE TABLE {_quote_ident(table)} AS
            SELECT
                trade_date,
                stock_code,
                CASE
                    WHEN {primary} THEN 2.0 + ({primary_formula})
                    WHEN {fallback} THEN 1.0 + ({fallback_formula})
                    ELSE ({fallback_formula})
                END AS pred_prob
            FROM fusion.fusion_rank_base;
            CREATE INDEX idx_{table}_trade_stock ON {_quote_ident(table)}(trade_date, stock_code);
            CREATE INDEX idx_{table}_trade_pred ON {_quote_ident(table)}(trade_date, pred_prob DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return table


def _strategy_dir_for(cfg: dict, score_table: str) -> Path:
    out = REPORT_DIR / "strategy_dirs" / str(cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRODUCTION_STRATEGY_DIR / "strategy_manifest.json", out / "strategy_manifest.json")
    rules = json.loads((PRODUCTION_STRATEGY_DIR / "trading_rules.json").read_text(encoding="utf-8"))
    rules["score_rule"]["score_asset"]["db_path"] = str(SCORE_DB)
    rules["score_rule"]["score_asset"]["table"] = score_table
    rules["score_rule"]["fusion_asset"]["db_path"] = str(FUSION_DB)
    pool_rule = rules["score_rule"]["pool_generation_rule"]
    pool_rule["base_target_pct"] = float(cfg["base_target_pct"])
    pool_rule["primary_pool"]["rank_weight_10d"] = float(cfg["primary_w10"])
    pool_rule["primary_pool"]["rank_weight_5d"] = float(cfg["primary_w5"])
    pool_rule["primary_pool"]["limit"] = int(cfg["primary_limit"])
    pool_rule["normal_fallback_pool"]["rank_weight_10d"] = float(cfg["fallback_w10"])
    pool_rule["normal_fallback_pool"]["rank_weight_5d"] = float(cfg["fallback_w5"])
    pool_rule["normal_fallback_pool"]["limit"] = int(cfg["fallback_limit"])
    pool_rule["weak_fallback_pool"]["rank_weight_10d"] = float(cfg["fallback_w10"])
    pool_rule["weak_fallback_pool"]["rank_weight_5d"] = float(cfg["fallback_w5"])
    pool_rule["weak_fallback_pool"]["limit"] = int(cfg["fallback_limit"])
    rules["selection_rule"]["primary_count_min"] = int(cfg["primary_count_min"])
    rules["selection_rule"]["rank_max"] = int(cfg["rank_max"])
    rules["selection_rule"]["post_filter_avg_pred_min"] = float(cfg["post_avg"])
    rules["position_rule"]["max_positions"] = int(cfg["max_positions"])
    rules["position_rule"]["max_single_position_pct"] = float(cfg["max_single"])
    rules["position_rule"]["rank_target_pct"] = {str(key): value for key, value in cfg["rank_targets"].items()}
    rules["holding_rule"]["holding_days"] = int(cfg["holding_days"])
    rules["holding_rule"]["max_holding_days"] = int(cfg["max_holding_days"])
    rules["holding_rule"]["score_exit_entry_ratio"] = float(cfg["score_exit_ratio"])
    rules["holding_rule"]["min_holding_days_before_score_exit"] = int(cfg["min_hold_score_exit"])
    (out / "trading_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_signal(cfg: dict, strategy_dir: Path, signal_file: Path) -> dict:
    rows = exporter._build_signals(strategy_dir, "20240604", "20260618", str(MARKET_DB), None)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    write_gm_signals_csv(rows, signal_file)
    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
    }


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


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path, score_table: str) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    cmd = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(JUEJIN_STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(cfg["max_positions"]),
        "--holding-days",
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["max_holding_days"]),
        "--target-position-pct",
        str(cfg["max_single"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        score_table,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-18 15:30:00",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def _objective(row: dict) -> float:
    return min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80)


def main() -> int:
    if not FUSION_DB.exists():
        raise SystemExit(f"fusion db not found: {FUSION_DB}")
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results: list[dict] = _read_existing_results(REPORT_DIR / "summary.csv")
    completed = {str(row.get("name")) for row in results if row.get("name")}
    for index, cfg in enumerate(VARIANTS, start=1):
        if cfg["name"] in completed and (REPORT_DIR / "logs" / f"{cfg['name']}.log").exists():
            print(f"[{index}/{len(VARIANTS)}] {cfg['name']} skipped_existing", flush=True)
            continue
        score_table = _build_score_table(cfg)
        strategy_dir = _strategy_dir_for(cfg, score_table)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_stats = _write_signal(cfg, strategy_dir, signal_file)
        returncode = _run_backtest(cfg, signal_file, log_file, score_table)
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "score_table": score_table,
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
        }
        row.update(signal_stats)
        row.update(_exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}", flush=True)
    _write_rows(REPORT_DIR / "summary_by_objective.csv", sorted(results, key=_objective, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0
            and _metric(row, "sharpe") >= 4.0
            and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
