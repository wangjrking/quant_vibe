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
SIGNAL_DIR = REPORT_DIR / "signals" / "rank_normalized_deepdrop"
LOG_DIR = REPORT_DIR / "logs" / "rank_normalized_deepdrop"
OUT_CSV = REPORT_DIR / "rank_normalized_deepdrop_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "rank_normalized_deepdrop_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "rank归一化深跌策略复核_20260714.md"

L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
L4_1D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb"
L4_3D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb"
L4_5D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb"
L4_10D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"

T1 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"
T3 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate"
T5 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
T10 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {
        "case": "rnd_w10_top1_deep5_gap1_amt9_t1",
        "weights": {"pred_10d_rank": 1.00},
        "topn": 1,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 90000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 5_000_000.0,
        "atr_max": 15.0,
        "target": 0.57,
    },
    {
        "case": "rnd_w10_top1_deep8_gap1_amt9_t1",
        "weights": {"pred_10d_rank": 1.00},
        "topn": 1,
        "pct_max": -8.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 90000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 5_000_000.0,
        "atr_max": 15.0,
        "target": 0.57,
    },
    {
        "case": "rnd_w10w5w1_top1_deep5_gap1_amt9_t1",
        "weights": {"pred_10d_rank": 0.70, "pred_5d_rank": 0.20, "pred_1d_rank": 0.10},
        "topn": 1,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 90000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 5_000_000.0,
        "atr_max": 15.0,
        "target": 0.57,
    },
    {
        "case": "rnd_w10w1_top1_deep5_gap0_amt15_t1",
        "weights": {"pred_10d_rank": 0.80, "pred_1d_rank": 0.20},
        "topn": 1,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 0.0,
        "amount_min": 150000.0,
        "turn_min": 2.0,
        "mv_min": 200000.0,
        "mv_max": 4_000_000.0,
        "atr_max": 12.0,
        "target": 0.57,
    },
    {
        "case": "rnd_w10_top2_deep5_gap1_amt9_t045",
        "weights": {"pred_10d_rank": 1.00},
        "topn": 2,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 90000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 5_000_000.0,
        "atr_max": 15.0,
        "target": 0.45,
    },
    {
        "case": "rnd_w10w5w1_top2_deep5_gap1_amt9_t045",
        "weights": {"pred_10d_rank": 0.70, "pred_5d_rank": 0.20, "pred_1d_rank": 0.10},
        "topn": 2,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 90000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 5_000_000.0,
        "atr_max": 15.0,
        "target": 0.45,
    },
    {
        "case": "rnd_w10_top3_deep5_gap1_amt15_t030",
        "weights": {"pred_10d_rank": 1.00},
        "topn": 3,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 150000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 6_000_000.0,
        "atr_max": 15.0,
        "target": 0.30,
    },
    {
        "case": "rnd_w10w5w1_top3_deep5_gap1_amt15_t030",
        "weights": {"pred_10d_rank": 0.70, "pred_5d_rank": 0.20, "pred_1d_rank": 0.10},
        "topn": 3,
        "pct_max": -5.0,
        "gap_min": -8.0,
        "gap_max": 1.0,
        "amount_min": 150000.0,
        "turn_min": 1.0,
        "mv_min": 200000.0,
        "mv_max": 6_000_000.0,
        "atr_max": 15.0,
        "target": 0.30,
    },
]

SLICES = {
    "full": ("00000000", "99999999"),
    "from_202407": ("20240701", "99999999"),
    "from_202501": ("20250101", "99999999"),
}


def symbol_from_code(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    exchange = {"SH": "SHSE", "SZ": "SZSE"}[suffix]
    return f"{exchange}.{code}"


def load_frame() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"attach '{L2_DB.as_posix()}' as l2")
        con.execute(f"attach '{L4_1D_DB.as_posix()}' as l1")
        con.execute(f"attach '{L4_5D_DB.as_posix()}' as l5")
        con.execute(f"attach '{L4_10D_DB.as_posix()}' as l10")
        frame = con.execute(
            f"""
            with cal as (
              select
                trade_date as signal_date,
                lead(trade_date, 1) over(order by trade_date) as buy_date
              from (select distinct trade_date from l2.STOCK_DAILY_DATA order by trade_date)
            ),
            sig_base as (
              select
                sig.trade_date as signal_date,
                sig.stock_code,
                sig.name,
                sig.pct_chg as signal_pct_chg_raw,
                sig.amount,
                sig.turnover_rate,
                sig.total_mv,
                sig.atr_qfq,
                sig.ST_TYPE,
                sig.ST_TYPE_name,
                cal.buy_date,
                buy.open as buy_open_raw,
                buy.pre_close as buy_pre_close_raw,
                buy.name as buy_name,
                buy.ST_TYPE as buy_ST_TYPE,
                buy.ST_TYPE_name as buy_ST_TYPE_name,
                ((buy.open / nullif(buy.pre_close, 0) - 1.0) * 100.0) as buy_open_gap_raw_pct
              from l2.STOCK_DAILY_DATA sig
              join cal on cal.signal_date=sig.trade_date
              join l2.STOCK_DAILY_DATA buy on buy.trade_date=cal.buy_date and buy.stock_code=sig.stock_code
              where sig.trade_date between '20220606' and '20260713'
                and sig.stock_code not like '%.BJ'
                and cal.buy_date is not null
                and sig.pct_chg <= -5.0
                and sig.amount >= 90000
                and sig.turnover_rate >= 1.0
                and sig.total_mv between 200000 and 6000000
                and sig.atr_qfq <= 15
                and sig.name not like 'ST%'
                and sig.name not like '*ST%'
                and coalesce(sig.ST_TYPE, '') in ('', '0')
                and coalesce(sig.ST_TYPE_name, '') in ('', '正常')
                and coalesce(buy.ST_TYPE, '') in ('', '0')
                and coalesce(buy.ST_TYPE_name, '') in ('', '正常')
                and buy.open is not null
                and buy.pre_close is not null
                and ((buy.open / nullif(buy.pre_close, 0) - 1.0) * 100.0) between -8.0 and 1.0
            ),
            p1r as (
              select
                trade_date,
                stock_code,
                pred_prob as pred_1d,
                rank() over(partition by trade_date order by pred_prob) * 1.0 / count(*) over(partition by trade_date) as pred_1d_rank
              from l1.{T1}
              where trade_date between '20220606' and '20260713'
            ),
            p5r as (
              select
                trade_date,
                stock_code,
                pred_prob as pred_5d,
                rank() over(partition by trade_date order by pred_prob) * 1.0 / count(*) over(partition by trade_date) as pred_5d_rank
              from l5.{T5}
              where trade_date between '20220606' and '20260713'
            ),
            p10r as (
              select
                trade_date,
                stock_code,
                pred_prob as pred_10d,
                rank() over(partition by trade_date order by pred_prob) * 1.0 / count(*) over(partition by trade_date) as pred_10d_rank
              from l10.{T10}
              where trade_date between '20220606' and '20260713'
            )
            select
              sig_base.*,
              p1r.pred_1d,
              p5r.pred_5d,
              p10r.pred_10d,
              p1r.pred_1d_rank,
              p5r.pred_5d_rank,
              p10r.pred_10d_rank,
              0.0 as pred_3d,
              0.0 as pred_3d_rank
            from sig_base
            join p10r on p10r.trade_date=sig_base.signal_date and p10r.stock_code=sig_base.stock_code
            join p1r on p1r.trade_date=sig_base.signal_date and p1r.stock_code=sig_base.stock_code
            join p5r on p5r.trade_date=sig_base.signal_date and p5r.stock_code=sig_base.stock_code
            """
        ).fetchdf()
    finally:
        con.close()
    numeric_cols = [
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "signal_pct_chg_raw",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "buy_open_raw",
        "buy_pre_close_raw",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_signal(frame: pd.DataFrame, case: dict) -> dict:
    work = frame.copy()
    mask = (
        (work["signal_pct_chg_raw"] <= float(case["pct_max"]))
        & (work["buy_open_gap_raw_pct"] >= float(case["gap_min"]))
        & (work["buy_open_gap_raw_pct"] <= float(case["gap_max"]))
        & (work["amount"] >= float(case["amount_min"]))
        & (work["turnover_rate"] >= float(case["turn_min"]))
        & (work["total_mv"] >= float(case["mv_min"]))
        & (work["total_mv"] <= float(case["mv_max"]))
        & (work["atr_qfq"] <= float(case["atr_max"]))
    )
    work = work[mask].copy()
    if work.empty:
        raise RuntimeError(f"empty candidate: {case['case']}")
    score = pd.Series(0.0, index=work.index)
    for col, weight in case["weights"].items():
        score += work[col] * float(weight)
    work["sort_score"] = score
    topn = int(case["topn"])
    picked = (
        work.sort_values(["signal_date", "sort_score", "pred_10d_rank"], ascending=[True, False, False])
        .groupby("signal_date", group_keys=False)
        .head(topn)
        .copy()
    )
    picked = picked.sort_values(["buy_date", "sort_score"], ascending=[True, False]).copy()
    picked["rank"] = picked.groupby("buy_date").cumcount() + 1
    picked["symbol"] = picked["stock_code"].map(symbol_from_code)
    picked["pred_prob"] = picked["pred_10d"]
    picked["entry_score"] = picked["sort_score"]
    picked["target_pct"] = float(case["target"])
    daily_sum = picked.groupby("buy_date")["target_pct"].transform("sum")
    picked["target_pct"] = picked["target_pct"] * (1.0 / daily_sum).clip(upper=1.0)
    picked["holding_days"] = 1
    picked["max_holding_days"] = 3
    picked["score_exit_entry_ratio"] = "0.98000"
    picked["score_continue_entry_ratio"] = "1.02000"
    picked["min_holding_days_before_score_exit"] = 1
    picked["signal_stop_loss_pct"] = 0.05
    picked["signal_take_profit_pct"] = 0.08
    picked["strategy_variant"] = case["case"]
    picked["source_strategy_variant"] = "active_formal_l4_rank_normalized_deepdrop"
    picked["filter_name"] = case["case"]
    picked["entry_weight_name"] = "+".join(f"{k}:{v}" for k, v in case["weights"].items())
    picked["dynamic_hold_name"] = "h1m3_exit098_cont102"
    picked["buy_day_market_available"] = True
    picked["buy_day_hard_gate_complete"] = True
    picked["buy_day_st_rejected"] = False
    picked["buy_day_open_limit_up_rejected"] = False
    picked["buy_open_gap_pct"] = picked["buy_open_gap_raw_pct"]
    picked["exec_open_gap_pct"] = picked["buy_open_gap_raw_pct"]
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
        "pred_1d_rank",
        "pred_3d_rank",
        "pred_5d_rank",
        "pred_10d_rank",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "buy_open_gap_pct",
        "buy_open_gap_raw_pct",
        "exec_open_gap_pct",
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
        "feature_weight_scale",
        "daily_target_sum_after_cap",
        "sort_score",
    ]
    out = picked[cols].copy()
    out_path = SIGNAL_DIR / f"{case['case']}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    day_count = out.groupby("buy_date")["stock_code"].count()
    return {
        "case": case["case"],
        "signal_file": str(out_path),
        "rows": int(len(out)),
        "buy_days": int(out["buy_date"].nunique()),
        "stock_count": int(out["stock_code"].nunique()),
        "max_positions": int(day_count.max()),
        "mean_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().max()),
        "min_signal_date": str(out["signal_date"].min()),
        "max_signal_date": str(out["signal_date"].max()),
        "min_buy_date": str(out["buy_date"].min()),
        "max_buy_date": str(out["buy_date"].max()),
        **{f"rule_{k}": v for k, v in case.items() if k not in {"weights"}},
    }


def write_slice(row: dict, slice_name: str, start: str, end: str) -> dict | None:
    df = pd.read_csv(row["signal_file"], encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    part = df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()
    if part.empty:
        return None
    case = f"{row['case']}_{slice_name}"
    out = SIGNAL_DIR / f"{case}.csv"
    part["strategy_variant"] = case
    part["filter_name"] = case
    part.to_csv(out, index=False, encoding="utf-8-sig")
    day_count = part.groupby("buy_date")["stock_code"].count()
    item = dict(row)
    item.update(
        {
            "case": case,
            "base_case": row["case"],
            "slice": slice_name,
            "signal_file": str(out),
            "rows": int(len(part)),
            "buy_days": int(part["buy_date"].nunique()),
            "stock_count": int(part["stock_code"].nunique()),
            "max_positions": int(day_count.max()),
            "mean_daily_target_sum": float(part.groupby("buy_date")["target_pct"].sum().mean()),
            "max_daily_target_sum": float(part.groupby("buy_date")["target_pct"].sum().max()),
            "min_signal_date": str(part["signal_date"].min()),
            "max_signal_date": str(part["signal_date"].max()),
            "min_buy_date": str(part["buy_date"].min()),
            "max_buy_date": str(part["buy_date"].max()),
        }
    )
    return item


def run_juejin(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
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
        "--score-exit-entry-ratio",
        "0.98",
        "--score-continue-entry-ratio",
        "1.02",
        "--min-holding-days-before-score-exit",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
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


def write_report(results: pd.DataFrame) -> None:
    def num(value, default=0.0) -> float:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float(default)
        return float(value)

    lines = [
        "# rank 归一化深跌策略复核",
        "",
        "## 当前结论",
        "",
        "- 本轮只使用 active formal L4 DuckDB 1D/3D/5D/10D 和 L2 DuckDB 行情。",
        "- 本轮重点验证：用每日排名替代绝对模型分数阈值后，是否改善 2025 后段独立表现。",
        "- 掘金结果仍是唯一正式回测口径；本报告不发布生产策略。",
        "",
        "## 掘金结果",
        "",
        "| 候选 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 平均目标仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.sort_values(["slice", "pnl_ratio_annual"], ascending=[True, False], na_position="last").iterrows():
        lines.append(
            f"| {row.get('base_case', row.get('case'))} | {row.get('slice')} | "
            f"{num(row.get('pnl_ratio_annual')) * 100:.2f}% | {num(row.get('sharp_ratio')):.3f} | "
            f"{num(row.get('max_drawdown')) * 100:.2f}% | "
            f"{int(num(row.get('open_count')))} | {int(num(row.get('buy_days')))} | "
            f"{num(row.get('mean_daily_target_sum')) * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 准入检查口径",
            "",
            "- 目标：年化 >= 500%、Sharpe >= 4、最大回撤 <= 40%，且后段切片不能明显塌陷。",
            "- 额外检查：信号数量、买入日覆盖、ST/北交所过滤、DuckDB-only、qfq/真实价格语义。",
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
    frame = load_frame()
    manifest = [build_signal(frame, case) for case in CASES]
    tasks = []
    for row in manifest:
        for slice_name, (start, end) in SLICES.items():
            item = write_slice(row, slice_name, start, end)
            if item is not None:
                tasks.append(item)
    results = []
    for task in tasks:
        result = run_juejin(task)
        results.append(result)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "slice": result.get("slice"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    out = pd.DataFrame(results)
    out.sort_values(["slice", "pnl_ratio_annual", "sharp_ratio"], ascending=[True, False, False], na_position="last").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(out)
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "report": str(REPORT_MD), "tasks": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
