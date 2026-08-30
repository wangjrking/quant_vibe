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
SOURCE_DIR = REPORT_DIR / "signals" / "core_buy_open_gap_variants"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_structural_decay_refine"
LOG_DIR = REPORT_DIR / "logs" / "core_structural_decay_refine"
OUT_CSV = REPORT_DIR / "core_structural_decay_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_structural_decay_refine_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "核心结构衰减修正规则复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "x125": SOURCE_DIR / "x125_drop_missing_gap.csv",
    "x115": SOURCE_DIR / "x115_drop_missing_gap.csv",
}

CASES = [
    {"suffix": "drop_shallow2", "drop_shallow_gt": -2.0},
    {"suffix": "drop_shallow3", "drop_shallow_gt": -3.0},
    {"suffix": "drop_gap1", "drop_gap_gt": 1.0},
    {"suffix": "drop_gap0", "drop_gap_gt": 0.0},
    {"suffix": "drop_shallow2_gap1", "drop_shallow_gt": -2.0, "drop_gap_gt": 1.0},
    {"suffix": "drop_shallow3_gap1", "drop_shallow_gt": -3.0, "drop_gap_gt": 1.0},
    {"suffix": "drop_bigmv_lowturn", "drop_big_mv": 1_000_000.0, "drop_turn_lt": 4.0},
    {"suffix": "drop_bigmv_lowturn_shallow2", "drop_big_mv": 1_000_000.0, "drop_turn_lt": 4.0, "drop_shallow_gt": -2.0},
    {"suffix": "scale_shallow05_gap08", "scale_rules": [("signal_gt", -2.0, 0.50), ("gap_gt", 1.0, 0.80)]},
    {"suffix": "scale_shallow07_gap07", "scale_rules": [("signal_gt", -2.0, 0.70), ("gap_gt", 1.0, 0.70)]},
    {"suffix": "scale_midboost", "scale_rules": [("signal_between", (-10.0, -5.0), 1.10), ("signal_gt", -2.0, 0.50), ("gap_gt", 1.0, 0.70)]},
    {"suffix": "scale_midboost_deep", "scale_rules": [("signal_between", (-10.0, -5.0), 1.18), ("signal_lte", -10.0, 1.05), ("signal_gt", -2.0, 0.50), ("gap_gt", 1.0, 0.70)]},
    {"suffix": "scale_gap_bucket", "scale_rules": [("gap_between", (-2.0, 0.0), 1.08), ("gap_gt", 0.0, 0.82), ("gap_gt", 1.0, 0.55)]},
    {"suffix": "scale_turn_mv", "scale_rules": [("bigmv_lowturn", (1_000_000.0, 4.0), 0.65), ("signal_between", (-10.0, -5.0), 1.08)]},
]


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in [
        "target_pct",
        "signal_pct_chg_raw",
        "buy_open_gap_raw_pct",
        "buy_open_gap_pct",
        "exec_open_gap_pct",
        "total_mv",
        "turnover_rate",
        "sort_score",
        "rank",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "exec_open_gap_pct" not in df.columns:
        df["exec_open_gap_pct"] = df.get("buy_open_gap_raw_pct", df.get("buy_open_gap_pct"))
    else:
        df["exec_open_gap_pct"] = df["exec_open_gap_pct"].fillna(df.get("buy_open_gap_raw_pct")).fillna(df.get("buy_open_gap_pct"))
    return df


def apply_case(df: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if case.get("drop_shallow_gt") is not None:
        mask &= out["signal_pct_chg_raw"] <= float(case["drop_shallow_gt"])
    if case.get("drop_gap_gt") is not None:
        mask &= out["exec_open_gap_pct"] <= float(case["drop_gap_gt"])
    if case.get("drop_big_mv") is not None:
        bad = (out["total_mv"] >= float(case["drop_big_mv"])) & (out["turnover_rate"] < float(case["drop_turn_lt"]))
        mask &= ~bad
    out = out[mask].copy()
    if case.get("scale_rules") and len(out):
        scale = pd.Series(1.0, index=out.index)
        for op, value, factor in case["scale_rules"]:
            if op == "signal_gt":
                scale = scale.mask(out["signal_pct_chg_raw"] > float(value), scale * float(factor))
            elif op == "signal_lte":
                scale = scale.mask(out["signal_pct_chg_raw"] <= float(value), scale * float(factor))
            elif op == "gap_gt":
                scale = scale.mask(out["exec_open_gap_pct"] > float(value), scale * float(factor))
            elif op == "signal_between":
                lo, hi = value
                scale = scale.mask((out["signal_pct_chg_raw"] >= float(lo)) & (out["signal_pct_chg_raw"] <= float(hi)), scale * float(factor))
            elif op == "gap_between":
                lo, hi = value
                scale = scale.mask((out["exec_open_gap_pct"] >= float(lo)) & (out["exec_open_gap_pct"] <= float(hi)), scale * float(factor))
            elif op == "bigmv_lowturn":
                mv, turn = value
                scale = scale.mask((out["total_mv"] >= float(mv)) & (out["turnover_rate"] < float(turn)), scale * float(factor))
            else:
                raise ValueError(op)
        out["target_pct"] = out["target_pct"] * scale
    if len(out):
        daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
        out["target_pct"] = out["target_pct"] * (1.0 / daily_sum).clip(upper=1.0)
        out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out


def write_variant(source_name: str, source: pd.DataFrame, case: dict) -> dict:
    name = f"{source_name}_{case['suffix']}"
    df = apply_case(source, case)
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["structural_decay_refine_case"] = case["suffix"]
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
        "--max-positions", "1",
        "--holding-days", "1",
        "--max-holding-days", "3",
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
        "# 核心结构衰减修正规则复核",
        "",
        "## 结论",
        "",
    ]
    if len(hits):
        lines.append(f"- 本轮有 {len(hits)} 个候选满足表面目标；仍需继续做去 2024、去月份、去股票复核。")
    else:
        lines.append("- 本轮没有找到新的表面达标候选。")
    lines.append("- 本轮规则只使用信号日跌幅、买入日真实开盘缺口、市值和换手率，不使用年份/月过滤。")
    lines.extend(
        [
            "",
            "## 最优结果前 12",
            "",
            "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(row.get('open_count', 0)) if pd.notna(row.get('open_count')) else ''} | {int(row.get('buy_days', 0))} |"
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
    (REPORT_DIR / "core_structural_decay_refine_summary_20260714.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame, hits)
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
