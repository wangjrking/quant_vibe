from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "active_l4_multilabel_pool"
LOG_DIR = REPORT_DIR / "logs" / "active_l4_multilabel_pool"
OUT_CSV = REPORT_DIR / "active_l4_multilabel_pool_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "active_l4_multilabel_pool_juejin_results_20260714.json"
CANDIDATE_PARQUET = REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"


L4_1D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb"
L4_3D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb"
L4_5D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb"
L4_10D_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

T1 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"
T3 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate"
T5 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
T10 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
L2_TABLE = "STOCK_DAILY_DATA"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {
        "case": "mlp_10d70_1d20_5d10_t2_deep",
        "top_n": 2,
        "weights": {"p10": 0.70, "p1": 0.20, "p5": 0.10, "p3": 0.00},
        "pct_max": -3.0,
        "pred10_min": 0.96,
        "pred1_min": 0.75,
        "amount_min": 120000,
        "atr_max": 10.0,
        "base_target": 0.30,
        "high_target": 0.55,
        "cap": 1.0,
    },
    {
        "case": "mlp_10d80_1d10_5d10_t2_deep",
        "top_n": 2,
        "weights": {"p10": 0.80, "p1": 0.10, "p5": 0.10, "p3": 0.00},
        "pct_max": -3.0,
        "pred10_min": 0.97,
        "pred1_min": 0.60,
        "amount_min": 120000,
        "atr_max": 10.0,
        "base_target": 0.30,
        "high_target": 0.58,
        "cap": 1.0,
    },
    {
        "case": "mlp_10d60_1d30_5d10_t3_deep",
        "top_n": 3,
        "weights": {"p10": 0.60, "p1": 0.30, "p5": 0.10, "p3": 0.00},
        "pct_max": -3.0,
        "pred10_min": 0.95,
        "pred1_min": 0.80,
        "amount_min": 150000,
        "atr_max": 8.0,
        "base_target": 0.22,
        "high_target": 0.42,
        "cap": 1.0,
    },
    {
        "case": "mlp_10d60_1d20_5d20_t3_liq",
        "top_n": 3,
        "weights": {"p10": 0.60, "p1": 0.20, "p5": 0.20, "p3": 0.00},
        "pct_max": -2.0,
        "pred10_min": 0.94,
        "pred1_min": 0.75,
        "amount_min": 300000,
        "atr_max": 8.0,
        "base_target": 0.20,
        "high_target": 0.38,
        "cap": 0.96,
    },
    {
        "case": "mlp_all_10d50_1d25_5d15_3d10_t3",
        "top_n": 3,
        "weights": {"p10": 0.50, "p1": 0.25, "p5": 0.15, "p3": 0.10},
        "pct_max": -2.5,
        "pred10_min": 0.94,
        "pred1_min": 0.70,
        "amount_min": 150000,
        "atr_max": 10.0,
        "base_target": 0.22,
        "high_target": 0.40,
        "cap": 1.0,
    },
    {
        "case": "mlp_10d90_t2_pure",
        "top_n": 2,
        "weights": {"p10": 0.90, "p1": 0.05, "p5": 0.05, "p3": 0.00},
        "pct_max": -2.5,
        "pred10_min": 0.985,
        "pred1_min": 0.00,
        "amount_min": 120000,
        "atr_max": 12.0,
        "base_target": 0.30,
        "high_target": 0.60,
        "cap": 1.0,
    },
]


def build_candidates() -> pd.DataFrame:
    if CANDIDATE_PARQUET.exists():
        return pd.read_parquet(CANDIDATE_PARQUET)
    con = duckdb.connect()
    con.execute(f"ATTACH '{L4_1D_DB.as_posix()}' AS l4_1d")
    con.execute(f"ATTACH '{L4_3D_DB.as_posix()}' AS l4_3d")
    con.execute(f"ATTACH '{L4_5D_DB.as_posix()}' AS l4_5d")
    con.execute(f"ATTACH '{L4_10D_DB.as_posix()}' AS l4_10d")
    con.execute(f"ATTACH '{L2_DB.as_posix()}' AS l2db")
    sql = f"""
    with calendar as (
      select trade_date,
             lead(trade_date) over(order by trade_date) as buy_date
      from (select distinct trade_date from l2db.{L2_TABLE} where trade_date between '20220606' and '20260713')
    ),
    top10 as (
      select trade_date, stock_code
      from l4_10d.{T10}
      qualify row_number() over(partition by trade_date order by pred_prob desc) <= 80
    ),
    top1 as (
      select trade_date, stock_code
      from l4_1d.{T1}
      qualify row_number() over(partition by trade_date order by pred_prob desc) <= 80
    ),
    top5 as (
      select trade_date, stock_code
      from l4_5d.{T5}
      qualify row_number() over(partition by trade_date order by pred_prob desc) <= 80
    ),
    seed as (
      select * from top10
      union
      select * from top1
      union
      select * from top5
    )
    select
      s.trade_date as signal_date,
      c.buy_date,
      case
        when s.stock_code like '%.SH' then 'SHSE.' || replace(s.stock_code, '.SH', '')
        when s.stock_code like '%.SZ' then 'SZSE.' || replace(s.stock_code, '.SZ', '')
        else s.stock_code
      end as symbol,
      s.stock_code,
      m.name,
      coalesce(p1.pred_prob, 0.0) as pred_1d,
      coalesce(p3.pred_prob, 0.0) as pred_3d,
      coalesce(p5.pred_prob, 0.0) as pred_5d,
      coalesce(p10.pred_prob, 0.0) as pred_10d,
      m.amount,
      m.turnover_rate,
      m.total_mv,
      m.atr_qfq,
      m.pct_chg as signal_pct_chg_raw,
      m.close as signal_close_raw,
      b.open as buy_open_raw,
      case when m.close is not null and m.close != 0 and b.open is not null
           then (b.open / m.close - 1.0) * 100.0 end as buy_open_gap_raw_pct,
      case when m.close_qfq is not null and m.close_qfq != 0 and b.open_qfq is not null
           then (b.open_qfq / m.close_qfq - 1.0) * 100.0 end as buy_open_gap_pct,
      m.ST_TYPE,
      m.ST_TYPE_name,
      m.market,
      m.list_date
    from seed s
    join calendar c on c.trade_date = s.trade_date and c.buy_date is not null
    left join l4_1d.{T1} p1 on p1.trade_date=s.trade_date and p1.stock_code=s.stock_code
    left join l4_3d.{T3} p3 on p3.trade_date=s.trade_date and p3.stock_code=s.stock_code
    left join l4_5d.{T5} p5 on p5.trade_date=s.trade_date and p5.stock_code=s.stock_code
    left join l4_10d.{T10} p10 on p10.trade_date=s.trade_date and p10.stock_code=s.stock_code
    join l2db.{L2_TABLE} m on m.trade_date=s.trade_date and m.stock_code=s.stock_code
    left join l2db.{L2_TABLE} b on b.trade_date=c.buy_date and b.stock_code=s.stock_code
    where s.stock_code not like '%.BJ'
      and coalesce(m.ST_TYPE, '') in ('', '0')
      and coalesce(m.ST_TYPE_name, '') not like '%ST%'
      and coalesce(m.name, '') not like 'ST%'
      and coalesce(m.name, '') not like '*ST%'
      and coalesce(m.name, '') not like '%退%'
      and m.amount is not null and m.amount > 0
      and m.total_mv is not null and m.total_mv > 0
      and m.atr_qfq is not null
      and m.trade_date >= '20220606'
      and m.trade_date <= '20260713'
    """
    df = con.execute(sql).fetchdf()
    CANDIDATE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CANDIDATE_PARQUET, index=False)
    return df


def write_signal(candidates: pd.DataFrame, case: dict) -> dict:
    df = candidates.copy()
    w = case["weights"]
    df["entry_score"] = (
        float(w["p10"]) * pd.to_numeric(df["pred_10d"], errors="coerce").fillna(0.0)
        + float(w["p1"]) * pd.to_numeric(df["pred_1d"], errors="coerce").fillna(0.0)
        + float(w["p5"]) * pd.to_numeric(df["pred_5d"], errors="coerce").fillna(0.0)
        + float(w["p3"]) * pd.to_numeric(df["pred_3d"], errors="coerce").fillna(0.0)
    )
    mask = (
        (df["signal_pct_chg_raw"] <= float(case["pct_max"]))
        & (df["pred_10d"] >= float(case["pred10_min"]))
        & (df["pred_1d"] >= float(case["pred1_min"]))
        & (df["amount"] >= float(case["amount_min"]))
        & (df["atr_qfq"] <= float(case["atr_max"]))
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise RuntimeError(f"empty candidate for {case['case']}")
    raw_gap = pd.to_numeric(df["buy_open_gap_raw_pct"], errors="coerce")
    gap_penalty = pd.Series(1.0, index=df.index)
    gap_penalty.loc[raw_gap > 1.0] = 0.0
    gap_penalty.loc[(raw_gap > 0.5) & (raw_gap <= 1.0)] = 0.60
    gap_penalty.loc[raw_gap.isna()] = 0.70
    df["sort_score"] = df["entry_score"] + 0.03 * df["pred_1d"] + 0.02 * df["pred_5d"]
    df = (
        df.sort_values(["buy_date", "sort_score", "pred_10d"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(int(case["top_n"]))
        .copy()
    )
    high = (df["entry_score"] >= 0.985) & (df["pred_10d"] >= 0.985) & (df["signal_pct_chg_raw"] <= -5.0)
    df["target_pct"] = float(case["base_target"])
    df.loc[high, "target_pct"] = float(case["high_target"])
    df["target_pct"] = df["target_pct"] * gap_penalty.reindex(df.index).fillna(0.70)
    df = df[df["target_pct"] > 0].copy()
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    df["target_pct"] = df["target_pct"] * (float(case["cap"]) / daily_sum).clip(upper=1.0)
    df["rank"] = df.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    df["pred_prob"] = df["entry_score"]
    df["holding_days"] = 1
    df["max_holding_days"] = 3
    df["score_exit_entry_ratio"] = "0.98000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "1.02000"
    df["signal_stop_loss_pct"] = 0.05
    df["signal_take_profit_pct"] = 0.08
    df["strategy_variant"] = case["case"]
    df["source_strategy_variant"] = "active_formal_l4_multilabel_pool"
    df["filter_name"] = case["case"]
    df["entry_weight_name"] = json.dumps(case["weights"], ensure_ascii=False)
    df["dynamic_hold_name"] = "h1m3_exit098_cont102"
    df["buy_day_market_available"] = df["buy_open_raw"].notna()
    df["buy_day_hard_gate_complete"] = df["buy_open_raw"].notna()
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["hybrid_source"] = "active_formal_l4_multilabel"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    out_cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "atr_qfq",
        "signal_pct_chg_raw", "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "source_strategy_variant", "filter_name",
        "entry_weight_name", "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete",
        "buy_day_st_rejected", "buy_day_open_limit_up_rejected", "buy_open_gap_pct", "hybrid_source",
        "buy_open_gap_raw_pct", "feature_weight_scale", "daily_target_sum_after_cap", "sort_score",
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
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
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    manifests = [write_signal(candidates, case) for case in CASES]
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
