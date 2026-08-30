from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
SIGNAL_DIR = REPORT_DIR / "signals"
OUT_DIR = REPORT_DIR / "latest_l4_executable_position_variants"
LOG_DIR = OUT_DIR / "juejin_runs"
RUNNER = ROOT / "quant" / "main" / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "code_snapshot_sell_available_safe_20260702"
)
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


BASE_SIGNALS = [
    "w35_15_00_50_pctm1p75_gapm0p08to0p0_r1070_r160",
    "w35_15_00_50_pctm1p75_gapm0p08to0p0_r1080_r160",
    "w15_35_00_50_pctm1p75_gapm0p08to0p0_r1070_r160",
]

POSITION_CASES = [
    {"suffix": "eq32", "mode": "rank_map", "rank_map": {1: 0.32, 2: 0.32, 3: 0.32}},
    {"suffix": "eq30", "mode": "rank_map", "rank_map": {1: 0.30, 2: 0.30, 3: 0.30}},
    {"suffix": "rank40_32_24", "mode": "rank_map", "rank_map": {1: 0.40, 2: 0.32, 3: 0.24}},
    {"suffix": "rank45_30_20", "mode": "rank_map", "rank_map": {1: 0.45, 2: 0.30, 3: 0.20}},
    {"suffix": "rank50_28_18", "mode": "rank_map", "rank_map": {1: 0.50, 2: 0.28, 3: 0.18}},
    {"suffix": "top2_48_48", "mode": "topn_rank_map", "topn": 2, "rank_map": {1: 0.48, 2: 0.48}},
    {"suffix": "top2_55_40", "mode": "topn_rank_map", "topn": 2, "rank_map": {1: 0.55, 2: 0.40}},
    {"suffix": "top1_70", "mode": "topn_rank_map", "topn": 1, "rank_map": {1: 0.70}},
]


def _extract(stdout: str) -> dict:
    payload = json.loads(stdout)
    indicator = payload.get("indicator")
    if isinstance(indicator, str):
        try:
            return ast.literal_eval(indicator)
        except Exception:
            return {}
    return indicator or {}


def _make_signal(base_name: str, case: dict) -> Path:
    src = SIGNAL_DIR / f"{base_name}.csv"
    df = pd.read_csv(src, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    if case["mode"] == "topn_rank_map":
        df = df[df["rank"].astype(int) <= int(case["topn"])].copy()
    rank_map = {int(k): float(v) for k, v in case["rank_map"].items()}
    df["target_pct"] = df["rank"].astype(int).map(rank_map).fillna(0.01).map(lambda v: f"{v:.5f}")
    df["strategy_variant"] = f"{base_name}_{case['suffix']}"
    df["filter_name"] = f"latest_l4_executable_{case['suffix']}"
    out_dir = OUT_DIR / "signals" / f"{base_name}_{case['suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "signals.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for base_name in BASE_SIGNALS:
        for case in POSITION_CASES:
            signal_file = _make_signal(base_name, case)
            name = f"{base_name}_{case['suffix']}"
            log_file = LOG_DIR / f"{name}.log"
            max_positions = int(case.get("topn", 3))
            cmd = [
                sys.executable,
                str(RUNNER),
                "--strategy-dir",
                str(STRATEGY_DIR),
                "--signal-file",
                str(signal_file),
                "--log-file",
                str(log_file),
                "--max-positions",
                str(max_positions),
                "--holding-days",
                "1",
                "--max-holding-days",
                "1",
                "--score-exit-entry-ratio",
                "9.99",
                "--min-holding-days-before-score-exit",
                "1",
                "--score-continue-entry-ratio",
                "9.99",
                "--light-stop-loss-pct",
                "0.05",
                "--min-holding-days-before-light-stop",
                "1",
                "--take-profit-pct",
                "0.08",
                "--score-db",
                str(SCORE_DB),
                "--score-table",
                "score",
                "--market-db",
                str(MARKET_DB),
                "--backtest-adjust",
                "none",
                "--backtest-slippage-ratio",
                "0.0015",
            ]
            print("RUN", name, flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (LOG_DIR / f"{name}.runner.json").write_text(proc.stdout, encoding="utf-8")
            indicator = _extract(proc.stdout) if proc.returncode == 0 else {}
            sig = pd.read_csv(signal_file, dtype={"signal_date": str})
            target_by_day = sig.assign(tp=sig["target_pct"].astype(float)).groupby("signal_date")["tp"].sum()
            rows.append(
                {
                    "name": name,
                    "base": base_name,
                    "position_suffix": case["suffix"],
                    "returncode": proc.returncode,
                    "rows": int(len(sig)),
                    "signal_days": int(sig["signal_date"].nunique()),
                    "avg_target_sum": float(target_by_day.mean()) if len(target_by_day) else 0.0,
                    "max_target_sum": float(target_by_day.max()) if len(target_by_day) else 0.0,
                    "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                    "sharp_ratio": indicator.get("sharp_ratio"),
                    "max_drawdown": indicator.get("max_drawdown"),
                    "open_count": indicator.get("open_count"),
                    "close_count": indicator.get("close_count"),
                    "win_ratio": indicator.get("win_ratio"),
                    "signal_file": str(signal_file),
                    "log_file": str(log_file),
                }
            )
    out = OUT_DIR / "latest_l4_executable_position_summary.csv"
    df = pd.DataFrame(rows).sort_values(["pnl_ratio_annual", "sharp_ratio"], ascending=False)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out)
    print(df.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
