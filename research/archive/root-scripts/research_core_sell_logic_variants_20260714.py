from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_sell_logic_variants"
LOG_DIR = REPORT_DIR / "logs" / "core_sell_logic_variants"
OUT_CSV = REPORT_DIR / "core_sell_logic_variants_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_sell_logic_variants_juejin_results_20260714.json"
SUMMARY_JSON = REPORT_DIR / "core_sell_logic_variants_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "x115": REPORT_DIR / "signals" / "market_regime_scaling" / "mkt_panic_deep_x115.csv",
    "x125": REPORT_DIR / "signals" / "market_regime_scaling" / "mkt_panic_deep_x125.csv",
}

CASES = [
    {"suffix": "base_explicit", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "h1m1", "hold": 1, "max_hold": 1, "exit": 0.98, "cont": 9.99, "stop": 0.05, "take": 0.08},
    {"suffix": "h1m2_cont100", "hold": 1, "max_hold": 2, "exit": 0.98, "cont": 1.00, "stop": 0.05, "take": 0.08},
    {"suffix": "h1m3_cont100", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.00, "stop": 0.05, "take": 0.08},
    {"suffix": "h1m3_cont105", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.05, "stop": 0.05, "take": 0.08},
    {"suffix": "h1m4_cont102", "hold": 1, "max_hold": 4, "exit": 0.98, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "h2m3_exit098", "hold": 2, "max_hold": 3, "exit": 0.98, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "h2m4_exit098", "hold": 2, "max_hold": 4, "exit": 0.98, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "exit095", "hold": 1, "max_hold": 3, "exit": 0.95, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "exit100", "hold": 1, "max_hold": 3, "exit": 1.00, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "exit102", "hold": 1, "max_hold": 3, "exit": 1.02, "cont": 1.02, "stop": 0.05, "take": 0.08},
    {"suffix": "stop03", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.02, "stop": 0.03, "take": 0.08},
    {"suffix": "stop08", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.02, "stop": 0.08, "take": 0.08},
    {"suffix": "take06", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.02, "stop": 0.05, "take": 0.06},
    {"suffix": "take12", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.02, "stop": 0.05, "take": 0.12},
    {"suffix": "tight_stop03_exit100", "hold": 1, "max_hold": 3, "exit": 1.00, "cont": 1.02, "stop": 0.03, "take": 0.08},
    {"suffix": "loose_stop08_cont100", "hold": 1, "max_hold": 3, "exit": 0.98, "cont": 1.00, "stop": 0.08, "take": 0.12},
]


def load_signal(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})


def write_variant(source_name: str, source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    name = f"{source_name}_{case['suffix']}"
    df["holding_days"] = int(case["hold"])
    df["max_holding_days"] = int(case["max_hold"])
    df["score_exit_entry_ratio"] = f"{float(case['exit']):.5f}"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = f"{float(case['cont']):.5f}"
    df["signal_stop_loss_pct"] = float(case["stop"])
    df["signal_take_profit_pct"] = float(case["take"])
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["dynamic_hold_name"] = f"h{case['hold']}m{case['max_hold']}_exit{case['exit']}_cont{case['cont']}_sl{case['stop']}_tp{case['take']}"
    out = SIGNAL_DIR / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": name,
        "source": source_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "max_positions": 1,
        "holding_days": int(case["hold"]),
        "max_holding_days": int(case["max_hold"]),
        "score_exit_entry_ratio": float(case["exit"]),
        "score_continue_entry_ratio": float(case["cont"]),
        "stop_loss": float(case["stop"]),
        "take_profit": float(case["take"]),
        "mean_daily_target_sum": float(pd.to_numeric(df["target_pct"], errors="coerce").groupby(df["buy_date"]).sum().mean()),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(base_mod.RUNNER),
        "--strategy-dir", str(base_mod.STRATEGY_DIR),
        "--signal-file", row["signal_file"],
        "--log-file", str(log),
        "--score-db", str(base_mod.SCORE_DB),
        "--score-table", base_mod.SCORE_TABLE,
        "--market-db", str(base_mod.MARKET_DB),
        "--max-positions", str(row["max_positions"]),
        "--holding-days", str(row["holding_days"]),
        "--max-holding-days", str(row["max_holding_days"]),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    ind = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            ind = payload.get("indicator") or {}
            if isinstance(ind, str):
                ind = base_mod.parse_indicator_text(ind)
        except Exception:
            ind = base_mod.extract_indicator(text)
    else:
        ind = base_mod.extract_indicator(text)
    out = dict(row)
    out["returncode"] = proc.returncode
    out["log_file"] = str(log)
    if isinstance(ind, dict):
        out.update(ind)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifests = []
    for source_name, path in SOURCES.items():
        source = load_signal(path)
        for case in CASES:
            manifests.append(write_variant(source_name, source, case))
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        frame = pd.DataFrame(results)
        frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open": result.get("open_count")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").head(12).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
