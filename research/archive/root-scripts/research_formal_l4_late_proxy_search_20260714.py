from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
OUT_DIR = REPORT_DIR / "formal_l4_late_proxy_search"
OUT_CSV = REPORT_DIR / "formal_l4_late_proxy_search_results_20260714.csv"
OUT_JSON = REPORT_DIR / "formal_l4_late_proxy_search_results_20260714.json"
REPORT_MD = REPORT_DIR / "formal_L4后段代理搜索_20260714.md"

L2_DB = DATA_DIR / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
L4_1D_DB = DATA_DIR / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb"
L4_3D_DB = DATA_DIR / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb"
L4_5D_DB = DATA_DIR / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb"
L4_10D_DB = DATA_DIR / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"

T1 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"
T3 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate"
T5 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
T10 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"


WEIGHTS = [
    ("w10_100", {"pred_10d_rank": 1.00}),
    ("w10_80_w1_20", {"pred_10d_rank": 0.80, "pred_1d_rank": 0.20}),
    ("w10_70_w5_20_w1_10", {"pred_10d_rank": 0.70, "pred_5d_rank": 0.20, "pred_1d_rank": 0.10}),
    ("w10_70_w3_15_w5_15", {"pred_10d_rank": 0.70, "pred_3d_rank": 0.15, "pred_5d_rank": 0.15}),
    ("w10_60_w5_25_w1_15", {"pred_10d_rank": 0.60, "pred_5d_rank": 0.25, "pred_1d_rank": 0.15}),
]

FILTERS = [
    {"name": "p2_gap1_turn2_amt10", "pct_max": -2.0, "gap_min": -6.0, "gap_max": 1.0, "turn_min": 2.0, "amount_min": 100000.0, "mv_min": 200000.0, "mv_max": 5_000_000.0, "atr_max": 12.0},
    {"name": "p3_gap1_turn2_amt10", "pct_max": -3.0, "gap_min": -6.0, "gap_max": 1.0, "turn_min": 2.0, "amount_min": 100000.0, "mv_min": 200000.0, "mv_max": 5_000_000.0, "atr_max": 12.0},
    {"name": "p5_gap1_turn2_amt10", "pct_max": -5.0, "gap_min": -8.0, "gap_max": 1.0, "turn_min": 2.0, "amount_min": 100000.0, "mv_min": 200000.0, "mv_max": 5_000_000.0, "atr_max": 15.0},
    {"name": "p2_gap0_turn4_amt15", "pct_max": -2.0, "gap_min": -6.0, "gap_max": 0.0, "turn_min": 4.0, "amount_min": 150000.0, "mv_min": 200000.0, "mv_max": 3_000_000.0, "atr_max": 12.0},
    {"name": "p5_gap0_turn4_amt15", "pct_max": -5.0, "gap_min": -8.0, "gap_max": 0.0, "turn_min": 4.0, "amount_min": 150000.0, "mv_min": 200000.0, "mv_max": 3_000_000.0, "atr_max": 15.0},
    {"name": "p2_gap2_turn1_amt20_big", "pct_max": -2.0, "gap_min": -5.0, "gap_max": 2.0, "turn_min": 1.0, "amount_min": 200000.0, "mv_min": 500000.0, "mv_max": 10_000_000.0, "atr_max": 10.0},
    {"name": "p1_gap1_turn2_amt10_pred1", "pct_max": -1.0, "gap_min": -5.0, "gap_max": 1.0, "turn_min": 2.0, "amount_min": 100000.0, "mv_min": 200000.0, "mv_max": 5_000_000.0, "atr_max": 12.0, "pred_1d_min": 0.95},
]

TOP_NS = [1, 2, 3]


def load_frame() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"attach '{L2_DB.as_posix()}' as l2")
        con.execute(f"attach '{L4_1D_DB.as_posix()}' as l1")
        con.execute(f"attach '{L4_3D_DB.as_posix()}' as l3")
        con.execute(f"attach '{L4_5D_DB.as_posix()}' as l5")
        con.execute(f"attach '{L4_10D_DB.as_posix()}' as l10")
        frame = con.execute(
            f"""
            with cal as (
              select
                trade_date as signal_date,
                lead(trade_date, 1) over(order by trade_date) as buy_date,
                lead(trade_date, 2) over(order by trade_date) as d1,
                lead(trade_date, 3) over(order by trade_date) as d2,
                lead(trade_date, 4) over(order by trade_date) as d3
              from (select distinct trade_date from l2.STOCK_DAILY_DATA order by trade_date)
            ),
            pred as (
              select
                p10.trade_date as signal_date,
                p10.stock_code,
                p1.pred_prob as pred_1d,
                p3.pred_prob as pred_3d,
                p5.pred_prob as pred_5d,
                p10.pred_prob as pred_10d
              from l10.{T10} p10
              join l1.{T1} p1 on p1.trade_date=p10.trade_date and p1.stock_code=p10.stock_code
              join l3.{T3} p3 on p3.trade_date=p10.trade_date and p3.stock_code=p10.stock_code
              join l5.{T5} p5 on p5.trade_date=p10.trade_date and p5.stock_code=p10.stock_code
              where p10.trade_date between '20240601' and '20260710'
            )
            select
              pred.*,
              sig.name,
              sig.pct_chg as signal_pct_chg_raw,
              sig.amount,
              sig.turnover_rate,
              sig.total_mv,
              sig.atr_qfq,
              sig.ST_TYPE,
              sig.ST_TYPE_name,
              cal.buy_date,
              buy.open as buy_open,
              buy.pre_close as buy_pre_close,
              buy.name as buy_name,
              buy.ST_TYPE as buy_ST_TYPE,
              buy.ST_TYPE_name as buy_ST_TYPE_name,
              buy.amount as buy_amount,
              buy.turnover_rate as buy_turnover_rate,
              d1.open as open_d1,
              d2.open as open_d2,
              d3.open as open_d3
            from pred
            join cal on cal.signal_date=pred.signal_date
            join l2.STOCK_DAILY_DATA sig on sig.trade_date=pred.signal_date and sig.stock_code=pred.stock_code
            left join l2.STOCK_DAILY_DATA buy on buy.trade_date=cal.buy_date and buy.stock_code=pred.stock_code
            left join l2.STOCK_DAILY_DATA d1 on d1.trade_date=cal.d1 and d1.stock_code=pred.stock_code
            left join l2.STOCK_DAILY_DATA d2 on d2.trade_date=cal.d2 and d2.stock_code=pred.stock_code
            left join l2.STOCK_DAILY_DATA d3 on d3.trade_date=cal.d3 and d3.stock_code=pred.stock_code
            where pred.stock_code not like '%.BJ'
              and sig.pct_chg <= -1.0
              and sig.amount >= 80000
              and sig.turnover_rate >= 1.0
              and sig.total_mv between 200000 and 10000000
              and sig.atr_qfq <= 20
              and sig.name not like 'ST%'
              and sig.name not like '*ST%'
              and coalesce(sig.ST_TYPE, '') in ('', '0')
              and coalesce(sig.ST_TYPE_name, '') in ('', '正常')
              and coalesce(buy.ST_TYPE, '') in ('', '0')
              and coalesce(buy.ST_TYPE_name, '') in ('', '正常')
              and buy.open is not null
              and buy.pre_close is not null
              and ((buy.open / nullif(buy.pre_close, 0) - 1.0) * 100.0) between -8.0 and 2.0
            """
        ).fetchdf()
    finally:
        con.close()
    for col in frame.columns:
        if col.startswith("pred_") or col in [
            "signal_pct_chg_raw",
            "amount",
            "turnover_rate",
            "total_mv",
            "atr_qfq",
            "buy_open",
            "buy_pre_close",
            "buy_amount",
            "buy_turnover_rate",
            "open_d1",
            "open_d2",
            "open_d3",
        ]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["buy_open_gap_raw_pct"] = (frame["buy_open"] / frame["buy_pre_close"] - 1.0) * 100.0
    frame["ret_o1"] = frame["open_d1"] / frame["buy_open"] - 1.0
    frame["ret_o2"] = frame["open_d2"] / frame["buy_open"] - 1.0
    frame["ret_o3"] = frame["open_d3"] / frame["buy_open"] - 1.0
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        frame[f"{col}_rank"] = frame.groupby("signal_date")[col].rank(method="average", pct=True)
    return frame


def apply_filter(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    mask = (
        (frame["signal_pct_chg_raw"] <= rule["pct_max"])
        & (frame["buy_open_gap_raw_pct"] >= rule["gap_min"])
        & (frame["buy_open_gap_raw_pct"] <= rule["gap_max"])
        & (frame["turnover_rate"] >= rule["turn_min"])
        & (frame["amount"] >= rule["amount_min"])
        & (frame["total_mv"] >= rule["mv_min"])
        & (frame["total_mv"] <= rule["mv_max"])
        & (frame["atr_qfq"] <= rule["atr_max"])
    )
    if rule.get("pred_1d_min") is not None:
        mask &= frame["pred_1d"] >= float(rule["pred_1d_min"])
    return frame[mask].copy()


def score_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weight_name, weights in WEIGHTS:
        work = frame.copy()
        score = pd.Series(0.0, index=work.index)
        for col, weight in weights.items():
            score += work[col] * float(weight)
        work["sort_score"] = score
        for rule in FILTERS:
            filtered = apply_filter(work, rule)
            if filtered.empty:
                continue
            for topn in TOP_NS:
                picked = (
                    filtered.sort_values(["signal_date", "sort_score"], ascending=[True, False])
                    .groupby("signal_date", group_keys=False)
                    .head(topn)
                    .copy()
                )
                if picked.empty:
                    continue
                picked["weight"] = 1.0 / float(topn)
                def pack(prefix: str, data: pd.DataFrame) -> dict:
                    if data.empty:
                        return {
                            f"{prefix}_rows": 0,
                            f"{prefix}_days": 0,
                            f"{prefix}_ret_o1": None,
                            f"{prefix}_ret_o2": None,
                            f"{prefix}_ret_o3": None,
                            f"{prefix}_win_o2": None,
                        }
                    return {
                        f"{prefix}_rows": int(len(data)),
                        f"{prefix}_days": int(data["buy_date"].nunique()),
                        f"{prefix}_ret_o1": float((data["ret_o1"] * data["weight"]).sum()),
                        f"{prefix}_ret_o2": float((data["ret_o2"] * data["weight"]).sum()),
                        f"{prefix}_ret_o3": float((data["ret_o3"] * data["weight"]).sum()),
                        f"{prefix}_win_o2": float((data["ret_o2"] > 0).mean()),
                    }
                full = picked
                late = picked[picked["buy_date"] >= "20250101"]
                recent = picked[picked["buy_date"] >= "20240701"]
                row = {
                    "case": f"{weight_name}_{rule['name']}_top{topn}",
                    "weight_name": weight_name,
                    "filter_name": rule["name"],
                    "topn": topn,
                    **{f"rule_{k}": v for k, v in rule.items()},
                    **pack("full", full),
                    **pack("from_202501", late),
                    **pack("from_202407", recent),
                }
                # Conservative proxy: prefer rules that have positive recent and late return with enough days.
                row["proxy_objective"] = (
                    (row["from_202501_ret_o2"] or -999.0)
                    + 0.35 * (row["from_202407_ret_o2"] or -999.0)
                    + 0.15 * (row["full_ret_o2"] or -999.0)
                    + 0.20 * (row["from_202501_win_o2"] or 0.0)
                )
                rows.append(row)
    return pd.DataFrame(rows).sort_values("proxy_objective", ascending=False)


def write_report(results: pd.DataFrame, frame: pd.DataFrame) -> None:
    top = results.head(30)
    lines = [
        "# formal L4 后段代理搜索",
        "",
        "## 结论",
        "",
        "- 本报告只做代理搜索，不发布生产，不作为正式回测结论。",
        "- 输入为当前 active formal L4 DuckDB 1D/3D/5D/10D 分数，以及 L2 不复权真实开盘价。",
        "- 规则不使用年份、月份或未来收益；`ret_o1/ret_o2/ret_o3` 仅作为搜索代理，用于决定哪些候选值得交给掘金。",
        "",
        "## 数据覆盖",
        "",
        f"- 样本行数：{len(frame)}",
        f"- signal_date：{frame['signal_date'].min()} 到 {frame['signal_date'].max()}",
        f"- buy_date：{frame['buy_date'].min()} 到 {frame['buy_date'].max()}",
        "",
        "## 代理搜索前 30",
        "",
        "| case | topN | 2025起始天数 | 2025起始ret_o2 | 2025起始胜率 | 202407起始ret_o2 | 全周期ret_o2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['case']} | {int(row['topn'])} | {int(row['from_202501_days'])} | "
            f"{float(row['from_202501_ret_o2']):.4f} | {float(row['from_202501_win_o2']):.3f} | "
            f"{float(row['from_202407_ret_o2']):.4f} | {float(row['full_ret_o2']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{OUT_CSV}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    results = score_candidates(frame)
    results.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results.head(100).to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(results, frame)
    print(json.dumps({"rows": len(results), "csv": str(OUT_CSV), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
