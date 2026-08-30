from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
SOURCE = MAIN / "optimize_current_formal_l4_persistent_edge_20260720.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_persistent_edge_juejin_20260720"
FEATURE_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_persistent_edge_20260720"
    / "persistent_edge_features.duckdb"
)
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_current_formal_l4_robust_rules_20260719"
    / "research_code_snapshot"
)
SIGNAL_DIR = REPORT_DIR / "signals"
SCORE_DIR = REPORT_DIR / "score_assets"
LOG_DIR = REPORT_DIR / "logs"


BLENDS = {
    "w10_100": (0.0, 0.0, 1.0),
    "w5_20_10_80": (0.0, 0.2, 0.8),
    "w3_20_5_20_10_60": (0.2, 0.2, 0.6),
    "w3_20_5_40_10_40": (0.2, 0.4, 0.4),
}

SCORES = {
    "plain": "blend",
    "agree": "blend - 0.20 * spread_3510",
    "persistent": "blend + 0.10 * rank10_lag1 + 0.05 * rank10_lag3",
}

FILTERS = {
    "broad": "signal_pct_chg_raw <= 3.0",
    "no_chase": "signal_pct_chg_raw <= 1.0",
    "multihorizon": "least(rank_3d, rank_5d, rank_10d) >= 0.70 AND signal_pct_chg_raw <= 2.0",
}

CASES = [
    {"blend": "w10_100", "score": "plain", "filter": "multihorizon", "threshold": threshold, "topn": 1, "hold": hold}
    for threshold in [0.90, 0.95]
    for hold in [12, 15]
] + [
    {"blend": "w3_20_5_40_10_40", "score": "persistent", "filter": "no_chase", "threshold": threshold, "topn": 5, "hold": 15}
    for threshold in [0.80, 0.85]
] + [
    {"blend": "w3_20_5_40_10_40", "score": "agree", "filter": "no_chase", "threshold": 0.80, "topn": topn, "hold": 15}
    for topn in [3, 5]
] + [
    {"blend": "w3_20_5_20_10_60", "score": "agree", "filter": "broad", "threshold": 0.95, "topn": 3, "hold": 15}
]


def symbol(stock_code: str) -> str:
    return ("SHSE." if stock_code.endswith(".SH") else "SZSE.") + stock_code[:6]


def parse_indicator(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for line in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            result = {}
            for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "win_ratio"]:
                match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                if match:
                    result[key] = float(match.group(1))
            for key in ["open_count", "close_count"]:
                match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                if match:
                    result[key] = int(match.group(1))
            return result
    return {}


def expressions(case: dict) -> tuple[str, str, str]:
    w3, w5, w10 = BLENDS[case["blend"]]
    blend = f"({w3} * rank_3d + {w5} * rank_5d + {w10} * rank_10d)"
    score = SCORES[case["score"]].replace("blend", blend)
    return blend, score, FILTERS[case["filter"]]


def build_signal(con: duckdb.DuckDBPyConnection, case: dict) -> tuple[str, Path, int, float]:
    blend, score, filt = expressions(case)
    threshold = float(case["threshold"])
    topn = int(case["topn"])
    hold = int(case["hold"])
    frame = con.execute(
        f"""
        WITH eligible AS (
            SELECT *, {blend} AS blend, {score} AS select_score
            FROM persistent_edge_features
            WHERE {blend} >= {threshold} AND {filt}
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY trade_date ORDER BY select_score DESC, stock_code
            ) AS pick_rank
            FROM eligible
        )
        SELECT * FROM ranked WHERE pick_rank <= {topn} ORDER BY trade_date, pick_rank
        """
    ).fetchdf()
    max_positions = topn * hold
    target_pct = 1.0 / max_positions
    name = (
        f"{case['blend']}_{case['score']}_{case['filter']}_m{int(threshold*100)}"
        f"_top{topn}_h{hold}_full100"
    )
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["symbol"] = frame["stock_code"].map(symbol)
    frame["rank"] = frame["pick_rank"].astype(int)
    frame["pred_prob"] = frame["blend"]
    frame["entry_score"] = frame["blend"]
    frame["target_pct"] = target_pct
    frame["holding_days"] = hold
    frame["max_holding_days"] = hold
    frame["score_exit_entry_ratio"] = 9.99
    frame["min_holding_days_before_score_exit"] = 99
    frame["score_continue_entry_ratio"] = 9.99
    frame["strategy_variant"] = name
    frame["filter_name"] = "current_formal_l4_persistent_edge"
    frame["entry_weight_name"] = "rolling_cohort_full100"
    frame["dynamic_hold_name"] = f"fixed_h{hold}"
    frame["buy_day_market_available"] = True
    frame["buy_day_hard_gate_complete"] = True
    frame["buy_day_st_rejected"] = False
    frame["buy_day_open_limit_up_rejected"] = False
    frame["latest_market_date"] = str(frame["buy_date"].max())
    frame["buy_open_gap_pct"] = frame["buy_open_gap_raw_pct"]
    columns = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "signal_pct_chg_raw", "target_pct",
        "holding_days", "max_holding_days", "score_exit_entry_ratio", "min_holding_days_before_score_exit",
        "score_continue_entry_ratio", "strategy_variant", "filter_name", "entry_weight_name", "dynamic_hold_name",
        "buy_day_market_available", "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct", "buy_open_gap_raw_pct",
    ]
    path = SIGNAL_DIR / f"{name}.csv"
    frame[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return name, path, max_positions, target_pct


def build_score(con: duckdb.DuckDBPyConnection, blend_name: str) -> Path:
    w3, w5, w10 = BLENDS[blend_name]
    blend = f"({w3} * rank_3d + {w5} * rank_5d + {w10} * rank_10d)"
    path = SCORE_DIR / f"{blend_name}.duckdb"
    path.unlink(missing_ok=True)
    out = duckdb.connect(str(path))
    try:
        frame = con.execute(
            f"SELECT trade_date, stock_code, {blend} AS pred_prob FROM persistent_edge_features"
        ).fetchdf()
        out.register("score_frame", frame)
        out.execute("CREATE TABLE blended_rank_score AS SELECT * FROM score_frame")
    finally:
        out.close()
    return path


def run_case(case: dict, name: str, signal_file: Path, max_positions: int, target_pct: float, score_db: Path) -> dict:
    log_file = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env.update({"GM_INTRADAY_RISK_MODE": "0", "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "0", "GM_OPEN_DAILY_SCORE_EXIT": "0"})
    command = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY_DIR), "--signal-file", str(signal_file),
        "--log-file", str(log_file), "--max-positions", str(max_positions), "--holding-days", str(case["hold"]),
        "--max-holding-days", str(case["hold"]), "--target-position-pct", str(target_pct),
        "--score-db", str(score_db), "--score-table", "blended_rank_score", "--market-db", str(MARKET_DB),
        "--backtest-adjust", "none", "--backtest-slippage-ratio", "0.003",
        "--backtest-start", "2022-06-07 09:00:00", "--backtest-end", "2026-07-17 15:30:00",
    ]
    process = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    indicator = parse_indicator(log_file)
    return {
        "name": name, **case, "max_positions": max_positions, "target_pct": target_pct,
        "returncode": int(process.returncode), "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio"), "max_drawdown": indicator.get("max_drawdown"),
        "win_ratio": indicator.get("win_ratio"), "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"), "signal_file": str(signal_file), "score_db": str(score_db),
        "log_file": str(log_file),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(exist_ok=True)
    SCORE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB), read_only=True)
    rows = []
    try:
        score_assets = {blend: build_score(con, blend) for blend in {case["blend"] for case in CASES}}
        for case in CASES:
            name, signal_file, max_positions, target_pct = build_signal(con, case)
            result = run_case(case, name, signal_file, max_positions, target_pct, score_assets[case["blend"]])
            rows.append(result)
            pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").to_csv(
                REPORT_DIR / "juejin_results.csv", index=False, encoding="utf-8-sig"
            )
            print(json.dumps({key: result.get(key) for key in ["name", "returncode", "annual_return", "sharpe", "max_drawdown"]}, ensure_ascii=False), flush=True)
    finally:
        con.close()
    (REPORT_DIR / "result.json").write_text(
        json.dumps({"status": "research_only", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
