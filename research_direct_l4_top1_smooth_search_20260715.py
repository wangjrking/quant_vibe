from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
OUT_DIR = REPORT_DIR / "direct_l4_top1_smooth_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB_1D = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb"
DB_5D = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb"
DB_10D = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
DB_L2 = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

T1 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"
T5 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
T10 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"

POOL_CSV = OUT_DIR / "direct_l4_candidate_pool_20260715.csv"
LOCAL_CSV = OUT_DIR / "direct_l4_top1_local_screen_20260715.csv"
SIGNAL_MANIFEST = OUT_DIR / "direct_l4_top1_signal_manifest_20260715.csv"
REPORT_MD = OUT_DIR / "direct_l4_top1_smooth_search_report_20260715.md"


def load_pool() -> pd.DataFrame:
    if POOL_CSV.exists():
        return pd.read_csv(POOL_CSV, encoding="utf-8-sig", dtype={"trade_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect()
    con.execute(f"ATTACH '{DB_1D.as_posix()}' AS d1 (READ_ONLY)")
    con.execute(f"ATTACH '{DB_5D.as_posix()}' AS d5 (READ_ONLY)")
    con.execute(f"ATTACH '{DB_10D.as_posix()}' AS d10 (READ_ONLY)")
    con.execute(f"ATTACH '{DB_L2.as_posix()}' AS l2 (READ_ONLY)")
    sql = f"""
    WITH cal AS (
        SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
               lead(trade_date, 2) OVER (ORDER BY trade_date) AS next_trade_date
        FROM (SELECT DISTINCT trade_date FROM l2.STOCK_DAILY_DATA)
    ),
    scored AS (
        SELECT
            t10.trade_date,
            cal.buy_date,
            cal.next_trade_date,
            t10.stock_code,
            t1.pred_prob AS pred_1d,
            t5.pred_prob AS pred_5d,
            t10.pred_prob AS pred_10d,
            t1.pred_prob AS rank_1d,
            t5.pred_prob AS rank_5d,
            t10.pred_prob AS rank_10d
        FROM d10.{T10} t10
        JOIN d5.{T5} t5 ON t5.trade_date = t10.trade_date AND t5.stock_code = t10.stock_code
        JOIN d1.{T1} t1 ON t1.trade_date = t10.trade_date AND t1.stock_code = t10.stock_code
        JOIN cal ON cal.trade_date = t10.trade_date
        WHERE cal.buy_date IS NOT NULL
          AND cal.next_trade_date IS NOT NULL
          AND t10.stock_code NOT LIKE '%.BJ'
    )
    SELECT
        s.*,
        m.name,
        m.amount,
        m.turnover_rate,
        m.total_mv,
        m.atr_qfq,
        m.pct_chg AS signal_pct_chg_raw,
        b.open AS buy_open,
        b.pre_close AS buy_pre_close,
        (b.open / NULLIF(b.pre_close, 0) - 1.0) * 100.0 AS exec_open_gap_pct,
        n.open / NULLIF(b.open, 0) - 1.0 AS next_open_ret,
        CASE WHEN s.stock_code LIKE '300%' OR s.stock_code LIKE '301%' OR s.stock_code LIKE '688%' THEN 0.20 ELSE 0.10 END AS limit_ratio
    FROM scored s
    JOIN l2.STOCK_DAILY_DATA m ON m.trade_date = s.trade_date AND m.stock_code = s.stock_code
    JOIN l2.STOCK_DAILY_DATA b ON b.trade_date = s.buy_date AND b.stock_code = s.stock_code
    JOIN l2.STOCK_DAILY_DATA n ON n.trade_date = s.next_trade_date AND n.stock_code = s.stock_code
    WHERE s.rank_10d >= 0.88
      AND m.pct_chg <= -1.0
      AND m.amount >= 80000
      AND m.total_mv >= 150000
      AND m.atr_qfq IS NOT NULL
      AND b.open IS NOT NULL
      AND b.pre_close IS NOT NULL
      AND (b.open / NULLIF(b.pre_close, 0) - 1.0) * 100.0 BETWEEN -8.0 AND 2.0
      AND b.open < b.pre_close * (1.0 + CASE WHEN s.stock_code LIKE '300%' OR s.stock_code LIKE '301%' OR s.stock_code LIKE '688%' THEN 0.20 ELSE 0.10 END) * 0.995
      AND coalesce(m.ST_TYPE, '') IN ('', '0', '0.0')
      AND coalesce(b.ST_TYPE, '') IN ('', '0', '0.0')
      AND coalesce(m.ST_TYPE_name, '') IN ('', '0', '0.0', '正常', '无')
      AND coalesce(b.ST_TYPE_name, '') IN ('', '0', '0.0', '正常', '无')
    """
    frame = con.execute(sql).fetchdf()
    con.close()
    frame.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")
    return frame


def score_frame(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "r10":
        return frame["rank_10d"]
    if mode == "r10_5":
        return 0.75 * frame["rank_10d"] + 0.25 * frame["rank_5d"]
    if mode == "r10_5_1":
        return 0.70 * frame["rank_10d"] + 0.20 * frame["rank_5d"] + 0.10 * frame["rank_1d"]
    if mode == "r10_low5":
        return 0.85 * frame["rank_10d"] + 0.15 * (1.0 - frame["rank_5d"])
    if mode == "r10_mid1":
        return 0.80 * frame["rank_10d"] - 0.12 * (frame["rank_1d"] - 0.82).abs()
    raise ValueError(mode)


def annualize(daily: pd.Series) -> tuple[float, float, float]:
    if daily.empty:
        return 0.0, 0.0, 0.0
    equity = (1.0 + daily.fillna(0)).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(daily)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=0))
    sharpe = float(daily.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1.0).min() * -1.0)
    return annual, sharpe, mdd


def local_eval(selected: pd.DataFrame, target_pct: float) -> dict:
    work = selected.copy()
    work["daily_ret"] = pd.to_numeric(work["next_open_ret"], errors="coerce").fillna(0.0) * target_pct - target_pct * 0.003
    daily = work.groupby("buy_date")["daily_ret"].sum().sort_index()
    full = annualize(daily)
    from2025 = annualize(daily[daily.index.astype(str) >= "20250102"])
    recent60 = annualize(daily.tail(60))
    return {
        "local_full_annual": full[0],
        "local_full_sharpe": full[1],
        "local_full_mdd": full[2],
        "local_from2025_annual": from2025[0],
        "local_from2025_sharpe": from2025[1],
        "local_from2025_mdd": from2025[2],
        "local_recent60_annual": recent60[0],
        "local_recent60_sharpe": recent60[1],
        "local_recent60_mdd": recent60[2],
        "rows": int(len(work)),
        "buy_days": int(work["buy_date"].nunique()),
        "max_buy_date": str(work["buy_date"].max()),
        "mean_next_open_ret": float(work["next_open_ret"].mean()),
        "win_rate": float((work["next_open_ret"] > 0).mean()),
    }


def build_signal(selected: pd.DataFrame, case: dict) -> Path:
    out = selected.copy()
    out["signal_date"] = out["trade_date"]
    out["symbol"] = out["stock_code"].map(lambda x: ("SHSE." + x[:6]) if str(x).endswith(".SH") else ("SZSE." + x[:6]))
    out["rank"] = 1
    out["pred_prob"] = out["pred_10d"]
    out["entry_score"] = out["select_score"]
    out["target_pct"] = float(case["target_pct"])
    out["holding_days"] = int(case["hold"])
    out["max_holding_days"] = max(int(case["hold"]), 3)
    out["score_exit_entry_ratio"] = 0.98
    out["min_holding_days_before_score_exit"] = 1
    out["score_continue_entry_ratio"] = 1.02
    out["signal_stop_loss_pct"] = 0.05
    out["signal_take_profit_pct"] = 0.08
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out["entry_weight_name"] = case["mode"]
    out["dynamic_hold_name"] = "h" + str(case["hold"])
    out["buy_day_market_available"] = True
    out["buy_day_hard_gate_complete"] = True
    out["buy_day_st_rejected"] = False
    out["buy_day_open_limit_up_rejected"] = False
    out["buy_open_gap_pct"] = out["exec_open_gap_pct"]
    out["buy_open_gap_raw_pct"] = out["exec_open_gap_pct"]
    out["daily_target_sum_after_cap"] = float(case["target_pct"])
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_5d", "pred_10d", "rank_1d", "rank_5d", "rank_10d", "amount", "turnover_rate",
        "total_mv", "atr_qfq", "signal_pct_chg_raw", "buy_open_gap_pct", "buy_open_gap_raw_pct",
        "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "filter_name", "entry_weight_name",
        "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete",
        "buy_day_st_rejected", "buy_day_open_limit_up_rejected", "daily_target_sum_after_cap",
    ]
    path = OUT_DIR / f"{case['case']}.csv"
    out[cols].sort_values("buy_date").to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    pool = load_pool()
    for col in ["rank_1d", "rank_5d", "rank_10d", "signal_pct_chg_raw", "exec_open_gap_pct", "amount", "turnover_rate", "total_mv", "atr_qfq", "next_open_ret"]:
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    cases = []
    idx = 0
    for mode in ["r10", "r10_5", "r10_5_1", "r10_low5", "r10_mid1"]:
        score = score_frame(pool, mode)
        for pct_max in [-1.5, -2.5, -4.0]:
            for gap_high in [0.5, 1.5]:
                for gap_low in [-8.0, -3.0]:
                    for r10_min in [0.90, 0.95, 0.98]:
                        for target_pct in [0.45, 0.55, 0.65]:
                            idx += 1
                            mask = (
                                (pool["signal_pct_chg_raw"] <= pct_max)
                                & (pool["exec_open_gap_pct"] <= gap_high)
                                & (pool["exec_open_gap_pct"] >= gap_low)
                                & (pool["rank_10d"] >= r10_min)
                            )
                            part = pool.loc[mask].copy()
                            if len(part) < 120:
                                continue
                            part["select_score"] = score.loc[part.index]
                            selected = part.sort_values(["buy_date", "select_score", "stock_code"], ascending=[True, False, True]).groupby("buy_date", as_index=False).head(1)
                            if selected["buy_date"].nunique() < 120:
                                continue
                            case = {
                                "case": f"dl4_{idx}_{mode}_p{str(pct_max).replace('-', 'm').replace('.', 'p')}_gl{str(gap_low).replace('-', 'm').replace('.', 'p')}_gh{str(gap_high).replace('.', 'p')}_r10{str(r10_min).replace('.', 'p')}_tp{int(target_pct*100)}",
                                "mode": mode,
                                "pct_max": pct_max,
                                "gap_low": gap_low,
                                "gap_high": gap_high,
                                "r10_min": r10_min,
                                "target_pct": target_pct,
                                "hold": 1,
                            }
                            case.update(local_eval(selected, target_pct))
                            case["_selected"] = selected
                            cases.append(case)
    ranked = pd.DataFrame([{k: v for k, v in case.items() if k != "_selected"} for case in cases])
    if ranked.empty:
        raise SystemExit("No direct L4 cases generated")
    ranked["min_period_annual"] = ranked[["local_full_annual", "local_from2025_annual", "local_recent60_annual"]].min(axis=1)
    ranked["min_period_sharpe"] = ranked[["local_full_sharpe", "local_from2025_sharpe", "local_recent60_sharpe"]].min(axis=1)
    ranked = ranked.sort_values(["min_period_annual", "local_recent60_annual", "local_full_sharpe"], ascending=False)
    ranked.to_csv(LOCAL_CSV, index=False, encoding="utf-8-sig")
    top_names = set(ranked.head(12)["case"])
    manifest = []
    for case in cases:
        if case["case"] not in top_names:
            continue
        path = build_signal(case["_selected"], case)
        manifest.append({k: v for k, v in case.items() if k != "_selected"} | {"signal_file": str(path)})
    pd.DataFrame(manifest).to_csv(SIGNAL_MANIFEST, index=False, encoding="utf-8-sig")
    lines = [
        "# 直接 formal L4 Top1 平滑搜索本地粗筛 20260715",
        "",
        "## 说明",
        "",
        "- 本轮直接读取 active formal L4 1D/5D/10D DuckDB 资产和 L2 未复权执行行情。",
        "- 本地粗筛用买入日开盘到下一交易日开盘收益，只用于筛选少量掘金候选。",
        "- 已应用 no-BJ、ST/风险警示、开盘涨停不可买、买入日市场数据存在等硬约束。",
        "",
        "| 候选 | 本地full年化 | 本地2025年化 | 本地recent60年化 | full Sharpe | recent60 Sharpe | 买入日 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked.head(20).to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['local_full_annual']*100:.2f}% | {row['local_from2025_annual']*100:.2f}% | "
            f"{row['local_recent60_annual']*100:.2f}% | {row['local_full_sharpe']:.3f} | "
            f"{row['local_recent60_sharpe']:.3f} | {int(row['buy_days'])} | {row['max_buy_date']} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 候选池：`{POOL_CSV}`", f"- 本地粗筛：`{LOCAL_CSV}`", f"- 掘金候选信号清单：`{SIGNAL_MANIFEST}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pool_rows": len(pool), "cases": len(ranked), "manifest": str(SIGNAL_MANIFEST), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
