from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top1_three_tier_open_quality"
LOG_DIR = REPORT_DIR / "logs" / "top1_three_tier_open_quality"
BASE_SIGNAL = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_fw_soft_deepdrop_weight_v20260706"
    / "signals"
    / "full_history_fw_soft_deepdrop_weight.csv"
)
STRATEGY_DIR = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_fw_soft_deepdrop_weight_v20260706"
    / "code_snapshot"
)
RUNNER = MAIN / "run_juejin_signal_backtest.py"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_top1() -> pd.DataFrame:
    df = pd.read_csv(BASE_SIGNAL, encoding="utf-8-sig")
    for col in [
        "rank",
        "target_pct",
        "pred_prob",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "buy_open_gap_pct",
        "buy_open_gap_raw_pct",
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])
    gap = df.get("buy_open_gap_raw_pct")
    if gap is None:
        df["exec_open_gap_pct"] = df["buy_open_gap_pct"]
    else:
        df["exec_open_gap_pct"] = gap.fillna(df["buy_open_gap_pct"])
    df = df.sort_values(["buy_date", "rank", "pred_prob"], ascending=[True, True, False])
    top1 = df.groupby("buy_date", as_index=False).head(1).copy()
    return top1


def quality_flags(df: pd.DataFrame, profile: str) -> tuple[pd.Series, pd.Series]:
    gap = df["exec_open_gap_pct"]
    turn = df["turnover_rate"]
    atr = df["atr_qfq"]
    amount = df["amount"]
    pct = df["signal_pct_chg_raw"]
    pred10 = df["pred_10d"]
    pred1 = df["pred_1d"]

    high = (
        (gap >= -5.0)
        & (gap <= 0.5)
        & (turn >= 4.0)
        & (atr <= 8.0)
        & (pct <= -1.75)
        & (pred10 >= 0.70)
    )

    if profile == "mid_gap_amt":
        mid = (gap <= 0.5) & (gap >= -5.0) & (amount >= 200000) & (pct <= -1.75) & (pred10 >= 0.70)
    elif profile == "mid_turn_amt":
        mid = (gap <= 0.75) & (gap >= -5.0) & (turn >= 2.5) & (amount >= 150000) & (pct <= -1.75) & (pred10 >= 0.70)
    elif profile == "mid_pred1":
        mid = (gap <= 0.75) & (gap >= -5.0) & (amount >= 150000) & (pct <= -1.75) & (pred10 >= 0.70) & (pred1 >= 0.75)
    elif profile == "mid_atr12":
        mid = (gap <= 1.0) & (gap >= -6.0) & (amount >= 120000) & (atr <= 12.0) & (pct <= -1.75) & (pred10 >= 0.70)
    elif profile == "mid_deep_only":
        mid = (gap <= 1.0) & (gap >= -7.5) & (pct <= -2.5) & (pred10 >= 0.70)
    else:
        raise ValueError(profile)
    mid = mid & ~high
    return high, mid


def write_signal(base: pd.DataFrame, case: dict) -> dict:
    df = base.copy()
    high, mid = quality_flags(df, case["profile"])
    df["target_pct"] = case["low"]
    df.loc[mid, "target_pct"] = case["mid"]
    df.loc[high, "target_pct"] = case["high"]
    df = df[df["target_pct"] > 0].copy()
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["daily_target_sum_after_cap"] = df["target_pct"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "profile": case["profile"],
        "high": case["high"],
        "mid": case["mid"],
        "low": case["low"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "high_rows": int(high.sum()),
        "mid_rows": int(mid.sum()),
        "low_rows": int((df["target_pct"] == case["low"]).sum()),
        "mean_target": float(df["target_pct"].mean()) if len(df) else 0.0,
    }


def extract_indicator(log_text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                return parse_indicator_text(payload)
    return None


def parse_indicator_text(payload: str) -> dict:
    out: dict[str, float | int] = {}
    for key in [
        "pnl_ratio",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "risk_ratio",
        "win_ratio",
        "calmar_ratio",
    ]:
        m = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
        if m:
            out[key] = float(m.group(1))
    for key in ["open_count", "close_count", "win_count", "lose_count"]:
        m = re.search(rf"'{key}':\s*([0-9]+)", payload)
        if m:
            out[key] = int(m.group(1))
    return out


def run_juejin(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    result = dict(row)
    result["returncode"] = proc.returncode
    result["log_file"] = str(log)
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            ind = payload.get("indicator") or {}
            if isinstance(ind, str):
                ind = parse_indicator_text(ind)
        except Exception:
            ind = None
    else:
        text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
        ind = extract_indicator(text)
    if isinstance(ind, dict):
        for key in [
            "pnl_ratio",
            "pnl_ratio_annual",
            "sharp_ratio",
            "max_drawdown",
            "risk_ratio",
            "open_count",
            "close_count",
            "win_count",
            "lose_count",
            "win_ratio",
            "calmar_ratio",
        ]:
            result[key] = ind.get(key)
    else:
        result["indicator_error"] = "missing"
        result["stdout"] = proc.stdout[-1000:]
        result["stderr"] = proc.stderr[-1000:]
    return result


def main() -> None:
    base = load_top1()
    cases = []
    profiles = ["mid_gap_amt", "mid_turn_amt", "mid_pred1", "mid_atr12", "mid_deep_only"]
    weight_sets = [
        (0.91, 0.35, 0.05),
        (0.91, 0.30, 0.10),
        (0.91, 0.25, 0.15),
        (0.85, 0.35, 0.10),
        (0.85, 0.30, 0.15),
        (0.80, 0.35, 0.15),
        (0.80, 0.30, 0.20),
    ]
    for profile in profiles:
        for high, mid, low in weight_sets:
            cases.append(
                {
                    "case": f"tier_{profile}_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}",
                    "profile": profile,
                    "high": high,
                    "mid": mid,
                    "low": low,
                }
            )
    manifest = [write_signal(base, case) for case in cases]
    results = [run_juejin(row) for row in manifest]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "top1_three_tier_open_quality_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top1_three_tier_open_quality_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
