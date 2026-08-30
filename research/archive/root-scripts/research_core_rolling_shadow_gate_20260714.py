from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SOURCE_DIR = REPORT_DIR / "signals" / "core_buy_open_gap_variants"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_rolling_shadow_gate"
LOG_DIR = REPORT_DIR / "logs" / "core_rolling_shadow_gate"
OUT_CSV = REPORT_DIR / "core_rolling_shadow_gate_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_rolling_shadow_gate_juejin_results_20260714.json"
SHADOW_CSV = REPORT_DIR / "core_rolling_shadow_gate_daily_shadow_20260714.csv"
SUMMARY_JSON = REPORT_DIR / "core_rolling_shadow_gate_summary_20260714.json"
REPORT_MD = REPORT_DIR / "核心滚动影子账户开关复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "x125_drop_missing_gap": SOURCE_DIR / "x125_drop_missing_gap.csv",
    "x125_gap_le_1p0": SOURCE_DIR / "x125_gap_le_1p0.csv",
    "x115_drop_missing_gap": SOURCE_DIR / "x115_drop_missing_gap.csv",
    "x115_gap_le_1p0": SOURCE_DIR / "x115_gap_le_1p0.csv",
}


CASES = [
    {"suffix": "w10_mean_gt_m2_win50", "window": 10, "mean_min": -0.002, "win_min": 0.50, "warmup": 10},
    {"suffix": "w10_mean_gt_0_win50", "window": 10, "mean_min": 0.0, "win_min": 0.50, "warmup": 10},
    {"suffix": "w10_mean_gt_2_win55", "window": 10, "mean_min": 0.002, "win_min": 0.55, "warmup": 10},
    {"suffix": "w15_mean_gt_m2_win50", "window": 15, "mean_min": -0.002, "win_min": 0.50, "warmup": 15},
    {"suffix": "w15_mean_gt_0_win50", "window": 15, "mean_min": 0.0, "win_min": 0.50, "warmup": 15},
    {"suffix": "w15_mean_gt_1_win52", "window": 15, "mean_min": 0.001, "win_min": 0.52, "warmup": 15},
    {"suffix": "w20_mean_gt_m3_win48", "window": 20, "mean_min": -0.003, "win_min": 0.48, "warmup": 20},
    {"suffix": "w20_mean_gt_m1_win50", "window": 20, "mean_min": -0.001, "win_min": 0.50, "warmup": 20},
    {"suffix": "w20_mean_gt_0_win50", "window": 20, "mean_min": 0.0, "win_min": 0.50, "warmup": 20},
    {"suffix": "w30_mean_gt_m2_win50", "window": 30, "mean_min": -0.002, "win_min": 0.50, "warmup": 30},
    {"suffix": "w30_mean_gt_0_win50", "window": 30, "mean_min": 0.0, "win_min": 0.50, "warmup": 30},
    {"suffix": "w40_mean_gt_m2_win48", "window": 40, "mean_min": -0.002, "win_min": 0.48, "warmup": 40},
    {"suffix": "w40_mean_gt_0_win50", "window": 40, "mean_min": 0.0, "win_min": 0.50, "warmup": 40},
]


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    if "sort_score" not in df.columns:
        df["sort_score"] = pd.to_numeric(df.get("pred_10d", df.get("pred_prob", 0.0)), errors="coerce").fillna(0.0)
    for col in ["target_pct", "sort_score", "rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def daily_top(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["buy_date"]
    ascending = [True]
    if "sort_score" in df.columns:
        sort_cols.append("sort_score")
        ascending.append(False)
    if "rank" in df.columns:
        sort_cols.append("rank")
        ascending.append(True)
    ranked = df.sort_values(sort_cols, ascending=ascending).copy()
    return ranked.groupby("buy_date", as_index=False).head(1).copy()


def add_shadow_returns(all_top: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        prices = con.execute(
            """
            with cal as (
              select trade_date, lead(trade_date, 2) over(order by trade_date) as d2
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select b.stock_code, b.trade_date as buy_date, b.open as buy_open, e2.open as open_d2
            from STOCK_DAILY_DATA b
            left join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    prices["buy_date"] = prices["buy_date"].astype(str)
    out = all_top.merge(prices, on=["stock_code", "buy_date"], how="left")
    out["shadow_ret"] = out["open_d2"] / out["buy_open"] - 1.0
    out["shadow_win"] = out["shadow_ret"] > 0
    return out


def gate_dates(top_with_returns: pd.DataFrame, case: dict) -> pd.DataFrame:
    rows = []
    history: list[float] = []
    for _, row in top_with_returns.sort_values("buy_date").iterrows():
        if len(history) < int(case["warmup"]):
            allowed = True
            roll_mean = None
            roll_win = None
        else:
            window_vals = history[-int(case["window"]) :]
            roll_mean = float(pd.Series(window_vals).mean())
            roll_win = float((pd.Series(window_vals) > 0).mean())
            allowed = (roll_mean >= float(case["mean_min"])) and (roll_win >= float(case["win_min"]))
        rows.append(
            {
                "source_case": row["source_case"],
                "buy_date": row["buy_date"],
                "stock_code": row["stock_code"],
                "shadow_ret": row["shadow_ret"],
                "rolling_mean": roll_mean,
                "rolling_win": roll_win,
                "allowed": bool(allowed),
                "gate_case": case["suffix"],
            }
        )
        if pd.notna(row["shadow_ret"]):
            history.append(float(row["shadow_ret"]))
    return pd.DataFrame(rows)


def write_variant(source_name: str, source: pd.DataFrame, gate: pd.DataFrame, case: dict) -> dict:
    name = f"{source_name}_{case['suffix']}"
    allow_dates = set(gate.loc[gate["allowed"], "buy_date"].astype(str))
    df = source[source["buy_date"].astype(str).isin(allow_dates)].copy()
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["rolling_shadow_gate_case"] = case["suffix"]
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
        "gate_window": int(case["window"]),
        "gate_mean_min": float(case["mean_min"]),
        "gate_win_min": float(case["win_min"]),
        "warmup": int(case["warmup"]),
        "allowed_days": int(gate["allowed"].sum()),
        "blocked_days": int((~gate["allowed"]).sum()),
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
        "# 核心滚动影子账户开关复核",
        "",
        "## 结论",
        "",
    ]
    if len(hits):
        lines.append(f"- 本轮有 {len(hits)} 个候选满足表面目标。")
        lines.append("- 这些候选还需要继续做去月份、去股票、去 2024 和最近切片复核。")
    else:
        lines.append("- 本轮滚动影子账户开关没有找到新的表面达标候选。")
        lines.append("- 该方向可以减少部分失效阶段交易，但会显著压缩交易次数和复利路径，暂未同时保留 500% 年化与 Sharpe 4。")
    lines.extend(
        [
            "",
            "## 规则口径",
            "",
            "- 对每个买入日，只使用此前已完成信号的开盘到后续开盘收益作为影子账户表现。",
            "- 达到 warmup 之前允许交易；达到 warmup 后，最近 N 笔影子交易均值和胜率同时达标才允许交易。",
            "- 该规则不使用年份、月份、未来标签或未来交易结果。",
            "",
            "## 最优结果前 12",
            "",
            "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 阻断日 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(row.get('open_count', 0)) if pd.notna(row.get('open_count')) else ''} | "
            f"{int(row.get('buy_days', 0))} | {int(row.get('blocked_days', 0))} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{OUT_CSV}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
            f"- 影子账户日表：`{SHADOW_CSV}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sources = {name: load_signal(path) for name, path in SOURCES.items()}
    tops = []
    for name, df in sources.items():
        top = daily_top(df)
        top["source_case"] = name
        tops.append(top)
    shadow = add_shadow_returns(pd.concat(tops, ignore_index=True))
    shadow.to_csv(SHADOW_CSV, index=False, encoding="utf-8-sig")

    manifests = []
    for source_name, source in sources.items():
        part = shadow[shadow["source_case"] == source_name].copy()
        for case in CASES:
            gate = gate_dates(part, case)
            manifests.append(write_variant(source_name, source, gate, case))
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
        "shadow_csv": str(SHADOW_CSV),
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
