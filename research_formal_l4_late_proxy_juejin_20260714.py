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
SEARCH_SCRIPT = MAIN / "research_formal_l4_late_proxy_search_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "formal_l4_late_proxy_juejin"
LOG_DIR = REPORT_DIR / "logs" / "formal_l4_late_proxy_juejin"
OUT_CSV = REPORT_DIR / "formal_l4_late_proxy_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "formal_l4_late_proxy_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "formal_L4后段代理候选掘金验证_20260714.md"


base_spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(base_spec)
assert base_spec and base_spec.loader
base_spec.loader.exec_module(base_mod)

search_spec = importlib.util.spec_from_file_location("late_proxy_search", SEARCH_SCRIPT)
search_mod = importlib.util.module_from_spec(search_spec)
assert search_spec and search_spec.loader
search_spec.loader.exec_module(search_mod)


SELECTED = [
    "w10_100_p5_gap1_turn2_amt10_top2",
    "w10_60_w5_25_w1_15_p2_gap2_turn1_amt20_big_top3",
    "w10_80_w1_20_p1_gap1_turn2_amt10_pred1_top1",
    "w10_100_p2_gap0_turn4_amt15_top3",
]

SLICES = {
    "full": ("00000000", "99999999"),
    "from_202501": ("20250101", "99999999"),
    "from_202407": ("20240701", "99999999"),
}


def parse_case(case_name: str) -> tuple[str, dict, int]:
    results = pd.read_csv(search_mod.OUT_CSV, encoding="utf-8-sig")
    row = results[results["case"] == case_name]
    if row.empty:
        raise ValueError(f"case not found in proxy results: {case_name}")
    item = row.iloc[0].to_dict()
    weight_name = str(item["weight_name"])
    rule = {}
    for key, value in item.items():
        if str(key).startswith("rule_") and pd.notna(value):
            rule[str(key)[5:]] = value
    topn = int(float(item["topn"]))
    return weight_name, rule, topn


def score_frame(frame: pd.DataFrame, weight_name: str) -> pd.Series:
    weights = dict(search_mod.WEIGHTS)[weight_name]
    score = pd.Series(0.0, index=frame.index)
    for col, weight in weights.items():
        score += frame[col] * float(weight)
    return score


def symbol_from_code(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    exchange = {"SH": "SHSE", "SZ": "SZSE", "BJ": "BJSE"}[suffix]
    return f"{exchange}.{code}"


def build_signal(frame: pd.DataFrame, case_name: str) -> dict:
    weight_name, rule, topn = parse_case(case_name)
    work = frame.copy()
    work["sort_score"] = score_frame(work, weight_name)
    filtered = search_mod.apply_filter(work, rule)
    picked = (
        filtered.sort_values(["signal_date", "sort_score"], ascending=[True, False])
        .groupby("signal_date", group_keys=False)
        .head(topn)
        .copy()
    )
    picked = picked.sort_values(["buy_date", "sort_score"], ascending=[True, False]).copy()
    picked["rank"] = picked.groupby("buy_date").cumcount() + 1
    picked["symbol"] = picked["stock_code"].map(symbol_from_code)
    picked["pred_prob"] = picked["pred_10d"]
    picked["entry_score"] = picked["sort_score"]
    picked["target_pct"] = 0.90 / float(topn)
    picked["holding_days"] = 1
    picked["max_holding_days"] = 3
    picked["score_exit_entry_ratio"] = "0.98000"
    picked["score_continue_entry_ratio"] = "1.02000"
    picked["min_holding_days_before_score_exit"] = 1
    picked["signal_stop_loss_pct"] = 0.05
    picked["signal_take_profit_pct"] = 0.08
    picked["strategy_variant"] = case_name
    picked["source_strategy_variant"] = "active_formal_l4_rank_late_proxy_search"
    picked["filter_name"] = case_name
    picked["entry_weight_name"] = weight_name
    picked["dynamic_hold_name"] = "h1m3_exit098_cont102"
    picked["buy_day_market_available"] = True
    picked["buy_day_hard_gate_complete"] = True
    picked["buy_day_st_rejected"] = False
    picked["buy_day_open_limit_up_rejected"] = False
    picked["buy_open_gap_pct"] = picked["buy_open_gap_raw_pct"]
    picked["feature_weight_scale"] = 1.0
    picked["daily_target_sum_after_cap"] = picked.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
        "score_continue_entry_ratio",
        "signal_stop_loss_pct",
        "signal_take_profit_pct",
        "strategy_variant",
        "source_strategy_variant",
        "filter_name",
        "entry_weight_name",
        "dynamic_hold_name",
        "buy_day_market_available",
        "buy_day_hard_gate_complete",
        "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected",
        "buy_open_gap_pct",
        "buy_open_gap_raw_pct",
        "feature_weight_scale",
        "daily_target_sum_after_cap",
        "sort_score",
    ]
    out = picked[cols].copy()
    out_path = SIGNAL_DIR / f"{case_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "case": case_name,
        "signal_file": str(out_path),
        "rows": int(len(out)),
        "buy_days": int(out["buy_date"].nunique()),
        "stock_count": int(out["stock_code"].nunique()),
        "max_positions": int(max(out.groupby("buy_date")["stock_code"].count().max(), 1)),
        "mean_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().mean()),
    }


def write_slice(row: dict, slice_name: str, start: str, end: str) -> dict | None:
    df = pd.read_csv(row["signal_file"], encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df = df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()
    if df.empty:
        return None
    case = f"{row['case']}_{slice_name}"
    out = SIGNAL_DIR / f"{case}.csv"
    df["strategy_variant"] = case
    df["filter_name"] = case
    df.to_csv(out, index=False, encoding="utf-8-sig")
    result = dict(row)
    result.update(
        {
            "case": case,
            "base_case": row["case"],
            "slice": slice_name,
            "signal_file": str(out),
            "rows": int(len(df)),
            "buy_days": int(df["buy_date"].nunique()),
            "stock_count": int(df["stock_code"].nunique()),
            "max_positions": int(max(df.groupby("buy_date")["stock_code"].count().max(), 1)),
            "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        }
    )
    return result


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
    if log.exists() and log.stat().st_size > 0:
        indicator = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(indicator, dict) and indicator.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(indicator)
            return out
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(base_mod.RUNNER),
        "--strategy-dir",
        str(base_mod.STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log),
        "--score-db",
        str(base_mod.SCORE_DB),
        "--score-table",
        base_mod.SCORE_TABLE,
        "--market-db",
        str(base_mod.MARKET_DB),
        "--max-positions",
        str(row["max_positions"]),
        "--holding-days",
        "1",
        "--max-holding-days",
        "3",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    indicator = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore") if log.exists() else "")
    out = dict(row)
    out["returncode"] = proc.returncode
    out["log_file"] = str(log)
    if isinstance(indicator, dict):
        out.update(indicator)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def write_report(frame: pd.DataFrame) -> None:
    lines = [
        "# formal L4 后段代理候选掘金验证",
        "",
        "## 结论",
        "",
        "- 本轮验证从代理搜索中选出的 rank-based 候选，不使用绝对分数阈值。",
        "- 输入仍为 active formal L4 DuckDB 分数；回测结论以掘金日志为准。",
        "",
        "## 掘金结果",
        "",
        "| base_case | slice | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 平均日目标仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values(["base_case", "slice"]).iterrows():
        lines.append(
            f"| {row.get('base_case')} | {row.get('slice')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(float(row.get('open_count', 0)))} | {int(float(row.get('buy_days', 0)))} | "
            f"{float(row.get('mean_daily_target_sum', 0))*100:.2f}% |"
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
    frame = search_mod.load_frame()
    base_rows = [build_signal(frame, case) for case in SELECTED]
    manifests = []
    for base_row in base_rows:
        for slice_name, (start, end) in SLICES.items():
            item = write_slice(base_row, slice_name, start, end)
            if item is not None:
                manifests.append(item)
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        out_frame = pd.DataFrame(results)
        out_frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    out_frame = pd.DataFrame(results)
    write_report(out_frame)
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
