from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
SOURCE_SIGNAL = (
    ROOT
    / "quant/main/strategy_library/production/prod_repro500_p435_s96_p10d70_v20260702/signals/full_history_repro500_p435_s96_p10d70.csv"
)
ACTIVE_WIDE = REPORT_DIR / "active_l4_wide_cache.duckdb"
OUT_DIR = REPORT_DIR / "repro500_feature_frontier_fine_20260704"
SIGNAL_DIR = OUT_DIR / "signals"
LOG_DIR = OUT_DIR / "juejin_runs"
SUMMARY_FILE = OUT_DIR / "feature_frontier_fine_summary.csv"

STRATEGY_DIR = (
    ROOT
    / "quant/data_file/reports/strategy_agent_latest_l4_weight_candidates_20260702/postrank_open_filter_candidates/code_snapshot_sell_available_safe_20260702"
)
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for rule in ["atr_le2_or_amt50w", "atr_le2", "amt_le50w"]:
        for weak in [0.385, 0.395, 0.405, 0.415, 0.425, 0.435]:
            for normal in [0.480, 0.490, 0.500, 0.510]:
                if weak >= normal:
                    continue
                cases.append(
                    {
                        "name": f"{rule}_w{weak:.3f}_n{normal:.3f}".replace(".", "p"),
                        "rule": rule,
                        "weak": weak,
                        "normal": normal,
                    }
                )
    return cases


def load_source() -> pd.DataFrame:
    src = pd.read_csv(SOURCE_SIGNAL, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(ACTIVE_WIDE), read_only=True)
    wide = con.execute(
        """
        select
            cast(signal_date as varchar) as signal_date,
            stock_code,
            buy_open_gap_raw_pct
        from active_l4_wide
        """
    ).fetchdf()
    con.close()
    wide["stock_code"] = wide["stock_code"].astype(str)
    out = src.merge(wide, on=["signal_date", "stock_code"], how="left")
    if out["buy_open_gap_raw_pct"].isna().any():
        raise RuntimeError("missing buy_open_gap_raw_pct")
    return out


def rule_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    atr = df["atr_qfq"].astype(float)
    amt = df["amount"].astype(float)
    if rule == "atr_le2":
        return atr <= 2.0
    if rule == "amt_le50w":
        return amt <= 500000
    if rule == "atr_le2_or_amt50w":
        return (atr <= 2.0) | (amt <= 500000)
    raise ValueError(rule)


def write_signal(src: pd.DataFrame, case: dict) -> tuple[Path, int, float]:
    df = src.copy()
    mask = rule_mask(df, case["rule"])
    target = pd.Series(float(case["normal"]), index=df.index)
    target.loc[mask] = float(case["weak"])
    target = target.clip(lower=0.20, upper=0.52)
    df["target_pct"] = target.map(lambda x: f"{float(x):.5f}")
    df["holding_days"] = "1"
    df["max_holding_days"] = "1"
    df["score_exit_entry_ratio"] = "0.96000"
    df["min_holding_days_before_score_exit"] = "1"
    df["score_continue_entry_ratio"] = "9.99000"
    df["signal_stop_loss_pct"] = "0.05000"
    df["signal_take_profit_pct"] = "0.08000"
    df["strategy_variant"] = case["name"]
    df["filter_name"] = case["name"]
    case_dir = SIGNAL_DIR / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "signals.csv"
    keep = [c for c in df.columns if c not in {"buy_open_gap_raw_pct"}]
    df[keep].to_csv(out, index=False, encoding="utf-8")
    return out, int(mask.sum()), float(target.mean())


def parse_indicator(stdout: str) -> dict:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    raw = indicator if isinstance(indicator, str) else stdout
    if isinstance(indicator, dict):
        return indicator
    parsed: dict[str, object] = {}
    for key in [
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "pnl_ratio",
        "open_count",
        "close_count",
        "win_ratio",
        "calmar_ratio",
    ]:
        match = re.search(rf"'{key}': ([0-9eE+\-.]+)", raw)
        if match:
            value = float(match.group(1))
            parsed[key] = int(value) if key.endswith("_count") else value
    if parsed:
        return parsed
    text = re.sub(r"datetime\.datetime\(.*?\)", "'datetime'", raw)
    try:
        parsed_any = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed_any if isinstance(parsed_any, dict) else {}


def run_case(signal_file: Path, case: dict) -> tuple[int, dict, Path]:
    log_file = LOG_DIR / f"{case['name']}.log"
    runner_json = LOG_DIR / f"{case['name']}.runner.json"
    if runner_json.exists() and log_file.exists():
        stdout = runner_json.read_text(encoding="utf-8", errors="ignore")
        return 0, parse_indicator(stdout), log_file
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
        "3",
        "--holding-days",
        "1",
        "--max-holding-days",
        "1",
        "--score-exit-entry-ratio",
        "0.96",
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
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runner_json.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, parse_indicator(proc.stdout), log_file


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    src = load_source()
    rows = []
    for case in build_cases():
        signal_file, weak_rows, avg_target = write_signal(src, case)
        print("RUN", case["name"], flush=True)
        returncode, indicator, log_file = run_case(signal_file, case)
        rows.append(
            {
                **case,
                "weak_rows": weak_rows,
                "avg_target": avg_target,
                "returncode": returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                "sharp_ratio": indicator.get("sharp_ratio"),
                "max_drawdown": indicator.get("max_drawdown"),
                "open_count": indicator.get("open_count"),
                "close_count": indicator.get("close_count"),
                "win_ratio": indicator.get("win_ratio"),
            }
        )
    rows = sorted(
        rows,
        key=lambda x: (x.get("sharp_ratio") or -999, x.get("pnl_ratio_annual") or -999),
        reverse=True,
    )
    with SUMMARY_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:20], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
