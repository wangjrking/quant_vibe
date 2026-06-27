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
CLEAN_REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "latest_formal_5d10d_stclean_refill"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "latest_formal_5d10d_allow_st_probe"
)
FUSION_DB = CLEAN_REPORT_DIR / "fusion_5d10d_latest_20260622.db"
SCORE_DB = REPORT_DIR / "allow_st_pool_scores.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
TEMPLATE_STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_formal_5d10d_gap_v20260622"
JUEJIN_STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
SIGNAL_END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-23 15:30:00"

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "3",
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
    "GM_EQUITY_DD_RISK_MODE": "0",
}


def _variant(
    name: str,
    *,
    w10: float,
    rank_targets: dict[int, float],
    limit: int,
    rank_max: int,
    holding_days: int,
    max_positions: int,
    max_single: float,
    max_total_mv: float = 999999999.0,
    min_amount: float = 8000.0,
    min_turnover: float = 0.0,
    max_turnover: float | None = None,
    open_score_exit: bool = True,
) -> dict:
    env = dict(BASE_ENV)
    env["GM_OPEN_DAILY_SCORE_EXIT"] = "1" if open_score_exit else "0"
    env["GM_MAX_DAILY_SELLS"] = str(max_positions)
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = "1" if open_score_exit else str(holding_days)
    return {
        "name": name,
        "w10": w10,
        "w5": 1.0 - w10,
        "limit": limit,
        "rank_max": rank_max,
        "rank_targets": rank_targets,
        "holding_days": holding_days,
        "max_holding_days": holding_days,
        "max_positions": max_positions,
        "max_single": max_single,
        "max_total_mv": max_total_mv,
        "min_amount": min_amount,
        "min_turnover": min_turnover,
        "max_turnover": max_turnover,
        "env": env,
    }


VARIANTS = [
    _variant(
        "allowst_top3_10d80_full_h3",
        w10=0.80,
        rank_targets={1: 0.34, 2: 0.32, 3: 0.29},
        limit=3,
        rank_max=3,
        holding_days=3,
        max_positions=3,
        max_single=0.36,
    ),
    _variant(
        "allowst_top5_10d80_full_h3",
        w10=0.80,
        rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14},
        limit=5,
        rank_max=5,
        holding_days=3,
        max_positions=5,
        max_single=0.26,
    ),
    _variant(
        "allowst_top5_10d80_full_h10_noexit",
        w10=0.80,
        rank_targets={1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14},
        limit=5,
        rank_max=5,
        holding_days=10,
        max_positions=5,
        max_single=0.26,
        open_score_exit=False,
    ),
    _variant(
        "allowst_top10_10d80_full_h10_noexit",
        w10=0.80,
        rank_targets={1: 0.11, 2: 0.105, 3: 0.10, 4: 0.095, 5: 0.09, 6: 0.09, 7: 0.085, 8: 0.085, 9: 0.08, 10: 0.08},
        limit=10,
        rank_max=10,
        holding_days=10,
        max_positions=10,
        max_single=0.12,
        open_score_exit=False,
    ),
    _variant(
        "allowst_top3_10d80_lowturn_h3",
        w10=0.80,
        rank_targets={1: 0.34, 2: 0.32, 3: 0.29},
        limit=3,
        rank_max=3,
        holding_days=3,
        max_positions=3,
        max_single=0.36,
        max_turnover=0.6,
    ),
]


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _score_table_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return f"allow_st_{clean}"


def _tradable_sql(prefix: str = "") -> str:
    p = prefix
    return f"""
        {p}stock_code NOT LIKE '%.BJ'
        AND COALESCE({p}name, '') NOT LIKE '%退市%'
        AND COALESCE({p}name, '') NOT LIKE '退%'
        AND COALESCE({p}name, '') NOT LIKE '%退'
        AND ({p}limit_times IS NULL OR {p}limit_times = '' OR {p}limit_times = 'None' OR CAST({p}limit_times AS REAL) = 0)
        AND {p}rank_5d IS NOT NULL
        AND {p}rank_10d IS NOT NULL
        AND {p}close IS NOT NULL
    """


def _pool_formula(w10: float, w5: float) -> str:
    return f"(rank_10d * {w10:.12g}) + (rank_5d * {w5:.12g})"


def build_score_table(cfg: dict) -> str:
    table = _score_table_name(cfg["name"])
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(FUSION_DB),))
        formula = _pool_formula(float(cfg["w10"]), float(cfg["w5"]))
        condition = f"""
            {_tradable_sql()}
            AND total_mv IS NOT NULL AND total_mv <= {float(cfg["max_total_mv"]):.12g}
            AND amount IS NOT NULL AND amount >= {float(cfg["min_amount"]):.12g}
            AND turnover_rate IS NOT NULL AND turnover_rate >= {float(cfg["min_turnover"]):.12g}
        """
        if cfg.get("max_turnover") is not None:
            condition += f"\n            AND turnover_rate <= {float(cfg['max_turnover']):.12g}"
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {_quote_ident(table)};
            CREATE TABLE {_quote_ident(table)} AS
            SELECT
                trade_date,
                stock_code,
                CASE WHEN {condition} THEN 1.0 + ({formula}) ELSE ({formula}) END AS pred_prob
            FROM fusion.fusion_rank_base;
            CREATE INDEX idx_{table}_trade_stock ON {_quote_ident(table)}(trade_date, stock_code);
            CREATE INDEX idx_{table}_trade_pred ON {_quote_ident(table)}(trade_date, pred_prob DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return table


def strategy_dir_for(cfg: dict, score_table: str) -> Path:
    out = REPORT_DIR / "strategy_dirs" / str(cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_STRATEGY_DIR / "strategy_manifest.json", out / "strategy_manifest.json")
    rules = json.loads((TEMPLATE_STRATEGY_DIR / "trading_rules.json").read_text(encoding="utf-8"))
    rules["score_rule"]["score_asset"]["db_path"] = str(SCORE_DB)
    rules["score_rule"]["score_asset"]["table"] = score_table
    rules["score_rule"]["fusion_asset"]["db_path"] = str(FUSION_DB)
    pool_rule = rules["score_rule"]["pool_generation_rule"]
    pool_rule["base_target_pct"] = 0.10925
    pool_rule["primary_pool"].update(
        {
            "rank_weight_10d": float(cfg["w10"]),
            "rank_weight_5d": float(cfg["w5"]),
            "limit": int(cfg["limit"]),
            "max_total_mv": float(cfg["max_total_mv"]),
            "min_amount": float(cfg["min_amount"]),
            "min_turnover_rate": float(cfg["min_turnover"]),
        }
    )
    for key in ("normal_fallback_pool", "weak_fallback_pool"):
        pool_rule[key].update(
            {
                "rank_weight_10d": float(cfg["w10"]),
                "rank_weight_5d": float(cfg["w5"]),
                "limit": int(cfg["limit"]) + 1,
                "max_total_mv": float(cfg["max_total_mv"]),
                "min_amount": float(cfg["min_amount"]),
                "min_turnover_rate": float(cfg["min_turnover"]),
            }
        )
    rules["selection_rule"]["primary_count_min"] = 1
    rules["selection_rule"]["rank_max"] = int(cfg["rank_max"])
    rules["selection_rule"]["post_filter_avg_pred_min"] = 0.0
    rules["selection_rule"]["min_pred_10d"] = 0.0
    rules["selection_rule"]["exclude_st"] = False
    rules["position_rule"]["max_positions"] = int(cfg["max_positions"])
    rules["position_rule"]["max_single_position_pct"] = float(cfg["max_single"])
    rules["position_rule"]["rank_target_pct"] = {str(key): value for key, value in cfg["rank_targets"].items()}
    rules["holding_rule"]["holding_days"] = int(cfg["holding_days"])
    rules["holding_rule"]["max_holding_days"] = int(cfg["max_holding_days"])
    rules["holding_rule"]["min_holding_days_before_score_exit"] = 1
    (out / "trading_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_signal(strategy_dir: Path, signal_file: Path) -> dict:
    rows = exporter._build_signals(strategy_dir, START_DATE, SIGNAL_END_DATE, str(MARKET_DB), None)
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


def extract_indicator(log_file: Path) -> dict | None:
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


def exposure_stats(log_file: Path) -> dict:
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


def st_stats(signal_file: Path) -> dict:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        signal_hits = 0
        buy_hits = 0
        both_hits = 0
        for row in rows:
            stock_code = row.get("stock_code")
            signal_date = row.get("signal_date")
            buy_date = row.get("buy_date")
            if not stock_code or not signal_date or not buy_date:
                continue
            q = {
                item["trade_date"]: item
                for item in conn.execute(
                    """
                    SELECT trade_date, ST_TYPE, ST_TYPE_name
                    FROM STOCK_DAILY_DATA
                    WHERE stock_code=? AND trade_date IN (?, ?)
                    """,
                    (stock_code, signal_date, buy_date),
                ).fetchall()
            }

            def hit(item) -> bool:
                if not item:
                    return False
                st_type = str(item["ST_TYPE"] or "").strip().upper()
                st_name = str(item["ST_TYPE_name"] or "")
                return (st_type and st_type not in {"0", "0.0", "NONE", "NAN", "FALSE"}) or ("风险" in st_name)

            s = hit(q.get(signal_date))
            b = hit(q.get(buy_date))
            signal_hits += int(s)
            buy_hits += int(b)
            both_hits += int(s and b)
        return {
            "signal_st_or_risk_warning": signal_hits,
            "buy_st_or_risk_warning": buy_hits,
            "both_signal_buy_st_or_risk_warning": both_hits,
        }
    finally:
        conn.close()


def run_backtest(cfg: dict, signal_file: Path, log_file: Path, score_table: str) -> int:
    if log_file.exists() and extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    command = [
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


def write_rows(path: Path, rows: list[dict]) -> None:
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


def metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def main() -> int:
    if not FUSION_DB.exists():
        raise SystemExit(f"fusion db not found: {FUSION_DB}")
    if not JUEJIN_STRATEGY_DIR.joinpath("main.py").exists():
        raise SystemExit(f"juejin strategy main.py not found: {JUEJIN_STRATEGY_DIR}")
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, cfg in enumerate(VARIANTS, start=1):
        score_table = build_score_table(cfg)
        strategy_dir = strategy_dir_for(cfg, score_table)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        signal_info = write_signal(strategy_dir, signal_file)
        returncode = run_backtest(cfg, signal_file, log_file, score_table)
        indicator = extract_indicator(log_file) or {}
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
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "config_json": json.dumps({key: value for key, value in cfg.items() if key != "env"}, ensure_ascii=False, sort_keys=True),
            "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
        }
        row.update(signal_info)
        row.update(st_stats(signal_file))
        row.update(exposure_stats(log_file))
        results.append(row)
        write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} maxdd={row.get('max_drawdown')} "
            f"buy_st={row.get('buy_st_or_risk_warning')}"
        )
    write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: metric(row, "annual"), reverse=True))
    write_rows(REPORT_DIR / "summary_by_drawdown.csv", sorted(results, key=lambda row: metric(row, "max_drawdown"), reverse=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
