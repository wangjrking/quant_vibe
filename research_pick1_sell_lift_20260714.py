from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_pick1_weight_lift_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SRC = REPORT_DIR / "signals" / "pick1_weight_lift" / "p1wl_h100_m075_l025.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "pick1_sell_lift"
LOG_DIR = REPORT_DIR / "logs" / "pick1_sell_lift"
OUT_CSV = REPORT_DIR / "pick1_sell_lift_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "pick1_sell_lift_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "pick1卖出规则复核_20260714.md"


spec = importlib.util.spec_from_file_location("weight_mod", BASE_SCRIPT)
weight_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(weight_mod)
slice_mod = weight_mod.slice_mod


CASES = [
    {"case": "p1sell_h1m1_e999_c999", "hold": 1, "max_hold": 1, "exit": 0.999, "cont": 9.99},
    {"case": "p1sell_h1m2_e990_c101", "hold": 1, "max_hold": 2, "exit": 0.990, "cont": 1.01},
    {"case": "p1sell_h1m3_e980_c102", "hold": 1, "max_hold": 3, "exit": 0.980, "cont": 1.02},
    {"case": "p1sell_h2m3_e980_c101", "hold": 2, "max_hold": 3, "exit": 0.980, "cont": 1.01},
    {"case": "p1sell_h2m4_e970_c100", "hold": 2, "max_hold": 4, "exit": 0.970, "cont": 1.00},
    {"case": "p1sell_h3m5_e960_c100", "hold": 3, "max_hold": 5, "exit": 0.960, "cont": 1.00},
]

PERIODS = [
    ("full", "2022-08-12 09:00:00", "2026-07-24 15:30:00"),
    ("from_202501", "2025-01-01 09:00:00", "2026-07-24 15:30:00"),
    ("from_202407", "2024-07-01 09:00:00", "2026-07-24 15:30:00"),
]


def build_signal(case: dict) -> dict:
    df = pd.read_csv(SRC, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["holding_days"] = int(case["hold"])
    df["max_holding_days"] = int(case["max_hold"])
    df["score_exit_entry_ratio"] = f"{float(case['exit']):.5f}"
    df["score_continue_entry_ratio"] = f"{float(case['cont']):.5f}"
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "max_positions": 1,
        "hold": int(case["hold"]),
        "max_hold": int(case["max_hold"]),
        "exit": float(case["exit"]),
        "cont": float(case["cont"]),
        "mean_daily_target_sum": float(pd.to_numeric(df["target_pct"], errors="coerce").groupby(df["buy_date"]).sum().mean()),
    }


def run_juejin(row: dict, period: str, start: str, end: str) -> dict:
    log = LOG_DIR / f"{row['case']}_{period}_slip0p0030.log"
    base = {**row, "period": period, "start": start, "end": end, "slippage": 0.003, "log_file": str(log)}
    if log.exists() and log.stat().st_size > 0:
        ind = slice_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if ind.get("pnl_ratio_annual") is not None:
            return {**base, "returncode": 0, **ind}
    cmd = [
        sys.executable,
        str(slice_mod.RUNNER),
        "--strategy-dir",
        str(slice_mod.STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log),
        "--score-db",
        str(slice_mod.SCORE_DB),
        "--score-table",
        slice_mod.SCORE_TABLE,
        "--market-db",
        str(slice_mod.MARKET_DB),
        "--backtest-start",
        start,
        "--backtest-end",
        end,
        "--backtest-slippage-ratio",
        "0.0030",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    ind = slice_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore") if log.exists() else "")
    out = {**base, "returncode": proc.returncode}
    if ind:
        out.update(ind)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def write_report(frame: pd.DataFrame) -> None:
    def num(value, default=0.0) -> float:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float(default)
        return float(value)

    lines = [
        "# pick1 卖出规则复核",
        "",
        "## 当前结论",
        "",
        "- 本轮固定选股和仓位，只调整持仓天数、最大持仓天数、分数退出和继续持有参数。",
        "- 使用 0.30% 滑点掘金回测，不发布生产策略。",
        "",
        "## 掘金结果",
        "",
        "| 候选 | 切片 | 持仓/最长 | exit/continue | 年化 | Sharpe | 最大回撤 | 开仓 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values(["period", "pnl_ratio_annual"], ascending=[True, False], na_position="last").iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('period')} | {int(num(row.get('hold')))} / {int(num(row.get('max_hold')))} | "
            f"{num(row.get('exit')):.3f} / {num(row.get('cont')):.3f} | "
            f"{num(row.get('pnl_ratio_annual')) * 100:.2f}% | {num(row.get('sharp_ratio')):.3f} | "
            f"{num(row.get('max_drawdown')) * 100:.2f}% | {int(num(row.get('open_count')))} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifests = [build_signal(case) for case in CASES]
    results = []
    for row in manifests:
        for period, start, end in PERIODS:
            result = run_juejin(row, period, start, end)
            results.append(result)
            pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "case": row["case"],
                        "period": period,
                        "annual": result.get("pnl_ratio_annual"),
                        "sharpe": result.get("sharp_ratio"),
                        "mdd": result.get("max_drawdown"),
                        "open": result.get("open_count"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
