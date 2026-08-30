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
SIGNAL_DIR = REPORT_DIR / "signals" / "core_buy_open_gap_variants"
LOG_DIR = REPORT_DIR / "logs" / "core_buy_open_gap_variants"
OUT_CSV = REPORT_DIR / "core_buy_open_gap_variants_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_buy_open_gap_variants_juejin_results_20260714.json"
SUMMARY_JSON = REPORT_DIR / "core_buy_open_gap_variants_summary_20260714.json"
REPORT_MD = REPORT_DIR / "核心买入日开盘缺口邻域复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "x115": REPORT_DIR / "signals" / "market_regime_scaling" / "mkt_panic_deep_x115.csv",
    "x125": REPORT_DIR / "signals" / "market_regime_scaling" / "mkt_panic_deep_x125.csv",
}


CASES = [
    {"suffix": "base_copy", "kind": "copy"},
    {"suffix": "gap_le_0p0", "kind": "filter", "min_gap": None, "max_gap": 0.0},
    {"suffix": "gap_le_0p5", "kind": "filter", "min_gap": None, "max_gap": 0.5},
    {"suffix": "gap_le_1p0", "kind": "filter", "min_gap": None, "max_gap": 1.0},
    {"suffix": "gap_m8_to_0p5", "kind": "filter", "min_gap": -8.0, "max_gap": 0.5},
    {"suffix": "gap_m5_to_0p5", "kind": "filter", "min_gap": -5.0, "max_gap": 0.5},
    {"suffix": "gap_m3_to_0p5", "kind": "filter", "min_gap": -3.0, "max_gap": 0.5},
    {"suffix": "gap_m5_to_1p5", "kind": "filter", "min_gap": -5.0, "max_gap": 1.5},
    {"suffix": "gap_ge_m5", "kind": "filter", "min_gap": -5.0, "max_gap": None},
    {"suffix": "drop_missing_gap", "kind": "filter", "drop_missing": True},
    {"suffix": "scale_pos05_070", "kind": "scale", "rules": [("gt", 0.5, 0.70)]},
    {"suffix": "scale_pos05_050", "kind": "scale", "rules": [("gt", 0.5, 0.50)]},
    {"suffix": "scale_pos10_050", "kind": "scale", "rules": [("gt", 1.0, 0.50)]},
    {"suffix": "scale_pos10_030", "kind": "scale", "rules": [("gt", 1.0, 0.30)]},
    {"suffix": "scale_deepm5_070", "kind": "scale", "rules": [("lt", -5.0, 0.70)]},
    {"suffix": "scale_deepm5_110", "kind": "scale", "rules": [("lt", -5.0, 1.10)]},
    {"suffix": "scale_mid_m5_0p5_105", "kind": "scale", "rules": [("between", (-5.0, 0.5), 1.05)]},
    {"suffix": "scale_mid105_pos050_deep070", "kind": "scale", "rules": [("between", (-5.0, 0.5), 1.05), ("gt", 0.5, 0.50), ("lt", -5.0, 0.70)]},
    {"suffix": "scale_mid110_pos050_deep070", "kind": "scale", "rules": [("between", (-5.0, 0.5), 1.10), ("gt", 0.5, 0.50), ("lt", -5.0, 0.70)]},
    {"suffix": "scale_mid105_pos070", "kind": "scale", "rules": [("between", (-5.0, 0.5), 1.05), ("gt", 0.5, 0.70)]},
]


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ["target_pct", "buy_open_gap_raw_pct", "buy_open_gap_pct", "exec_open_gap_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "exec_open_gap_pct" not in df.columns:
        if "buy_open_gap_raw_pct" in df.columns:
            df["exec_open_gap_pct"] = df["buy_open_gap_raw_pct"]
        elif "buy_open_gap_pct" in df.columns:
            df["exec_open_gap_pct"] = df["buy_open_gap_pct"]
        else:
            df["exec_open_gap_pct"] = pd.NA
    if "buy_open_gap_raw_pct" in df.columns and "buy_open_gap_pct" in df.columns:
        df["exec_open_gap_pct"] = df["buy_open_gap_raw_pct"].fillna(df["buy_open_gap_pct"])
    return df


def apply_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = source.copy()
    gap = pd.to_numeric(df["exec_open_gap_pct"], errors="coerce")
    if case["kind"] == "filter":
        mask = pd.Series(True, index=df.index)
        if case.get("drop_missing"):
            mask &= gap.notna()
        if case.get("min_gap") is not None:
            mask &= gap >= float(case["min_gap"])
        if case.get("max_gap") is not None:
            mask &= gap <= float(case["max_gap"])
        df = df[mask].copy()
    elif case["kind"] == "scale":
        scale = pd.Series(1.0, index=df.index)
        for op, value, factor in case["rules"]:
            if op == "gt":
                scale = scale.mask(gap > float(value), scale * float(factor))
            elif op == "lt":
                scale = scale.mask(gap < float(value), scale * float(factor))
            elif op == "between":
                lo, hi = value
                scale = scale.mask((gap >= float(lo)) & (gap <= float(hi)), scale * float(factor))
            else:
                raise ValueError(op)
        df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * scale
    elif case["kind"] == "copy":
        pass
    else:
        raise ValueError(case["kind"])
    if len(df):
        daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
        cap_ratio = (1.0 / daily_sum).clip(upper=1.0)
        df["target_pct"] = df["target_pct"] * cap_ratio
        df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    return df


def write_variant(source_name: str, source: pd.DataFrame, case: dict) -> dict:
    name = f"{source_name}_{case['suffix']}"
    df = apply_case(source, case)
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["open_gap_variant_case"] = case["suffix"]
    out = SIGNAL_DIR / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    daily_target = df.groupby("buy_date")["target_pct"].sum() if len(df) else pd.Series(dtype=float)
    return {
        "case": name,
        "source": source_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "max_positions": 1,
        "holding_days": 1,
        "max_holding_days": 3,
        "kind": case["kind"],
        "mean_daily_target_sum": float(daily_target.mean()) if len(daily_target) else 0.0,
        "max_daily_target_sum": float(daily_target.max()) if len(daily_target) else 0.0,
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


def write_report(frame: pd.DataFrame, hits: pd.DataFrame) -> None:
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").head(12)
    lines = [
        "# 核心买入日开盘涨幅邻域复核",
        "",
        "## 结论",
        "",
    ]
    if len(hits):
        lines.append(f"- 本轮有 {len(hits)} 个候选满足表面目标：年化 >= 500%、Sharpe >= 4、最大回撤 <= 40%。")
        lines.append("- 这些候选仍需继续做去月份、去股票、近期切片和参数邻域准入复核，不能直接发布生产。")
    else:
        lines.append("- 本轮买入日开盘涨幅过滤/缩放没有找到新的表面达标候选。")
        lines.append("- 结论：仅靠买入日开盘涨幅邻域，暂未解决核心策略的偶然性和收益目标兼顾问题。")
    lines.extend(
        [
            "",
            "## 回测口径",
            "",
            "- 输入信号来源：`mkt_panic_deep_x115`、`mkt_panic_deep_x125`。",
            "- 只改变买入日开盘涨幅过滤或目标仓位缩放，其余持仓与卖出参数保持核心候选口径。",
            "- 回测使用 `run_juejin_signal_backtest.py`，`max_positions=1`、`holding_days=1`、`max_holding_days=3`。",
            "- 模型输入仍为 active formal L4 DuckDB 10D 资产；交易过滤使用 L2 不复权真实价格口径。",
            "",
            "## 最优结果前 12",
            "",
            "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 平仓 | 买入日 | 行数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(row.get('open_count', 0)) if pd.notna(row.get('open_count')) else ''} | "
            f"{int(row.get('close_count', 0)) if pd.notna(row.get('close_count')) else ''} | "
            f"{int(row.get('buy_days', 0))} | {int(row.get('rows', 0))} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{OUT_CSV}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    annual = pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce")
    sharpe = pd.to_numeric(frame["sharp_ratio"], errors="coerce")
    mdd = pd.to_numeric(frame["max_drawdown"], errors="coerce")
    hits = frame[(annual >= 5.0) & (sharpe >= 4.0) & (mdd <= 0.4)].copy()
    summary = {
        "sources": {k: str(v) for k, v in SOURCES.items()},
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "report": str(REPORT_MD),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").head(12).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame, hits)
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
