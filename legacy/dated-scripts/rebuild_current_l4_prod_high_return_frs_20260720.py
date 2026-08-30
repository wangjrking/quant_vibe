from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
OUT = DATA / "reports" / "strategy_agent_prod_high_return_frs_current_l4_rebuild_20260720"
SIGNAL = OUT / "signals" / "prod_high_return_frs_current_l4_full.csv"
LOG = OUT / "logs" / "prod_high_return_frs_current_l4_juejin.log"
SUMMARY = OUT / "current_l4_rebuild_summary.json"
REPORT = OUT / "current_l4_rebuild_report.md"
CACHE = DATA / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714" / "active_l4_wide_buy_quality_base_pool_20260714.parquet"

L2 = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
L4 = {
    "p1": DATA / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb",
    "p3": DATA / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb",
    "p5": DATA / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb",
    "p10": DATA / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb",
}
TABLE = {
    "p1": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
    "p3": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate",
    "p5": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
    "p10": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
}
STRATEGY = MAIN / "strategy_library" / "production" / "prod_high_return_frs_scale090_cap090_v20260716" / "code_snapshot"
RUNNER = MAIN / "run_juejin_signal_backtest.py"


def clean(alias: str) -> str:
    return (
        f"coalesce(cast({alias}.ST_TYPE as varchar), '') in ('', '0', '0.0', 'None', 'NONE') "
        f"and coalesce(cast({alias}.ST_TYPE_name as varchar), '') not like '%ST%' "
        f"and coalesce(cast({alias}.name as varchar), '') not like 'ST%' "
        f"and coalesce(cast({alias}.name as varchar), '') not like '*ST%' "
        f"and coalesce(cast({alias}.name as varchar), '') not like '%退%'"
    )


def load_current_formal_pool() -> pd.DataFrame:
    cache_con = duckdb.connect()
    try:
        cache_con.execute(f"attach '{L4['p3'].as_posix()}' as current_p3 (read_only)")
        cached = cache_con.execute(
            f"""
            select c.* exclude(pred_3d), p.pred_prob as pred_3d
            from read_parquet('{CACHE.as_posix()}') c
            join current_p3.{TABLE['p3']} p
              on p.trade_date=c.signal_date and p.stock_code=c.stock_code
            """
        ).fetchdf()
    finally:
        cache_con.close()
    cached = cached.rename(
        columns={
            "pred_1d": "raw_pred_1d",
            "pred_3d": "raw_pred_3d",
            "pred_5d": "raw_pred_5d",
            "pred_10d": "raw_pred_10d",
        }
    )
    cached["signal_date"] = cached["signal_date"].astype(str)
    cached["buy_date"] = cached["buy_date"].astype(str)
    cache_max = str(cached["signal_date"].max())
    con = duckdb.connect()
    try:
        con.execute(f"attach '{L2.as_posix()}' as m (read_only)")
        for alias, path in L4.items():
            con.execute(f"attach '{path.as_posix()}' as {alias} (read_only)")
        sql = f"""
        with calendar as (
          select trade_date as signal_date,
                 lead(trade_date) over(order by trade_date) as buy_date
          from (select distinct trade_date from m.STOCK_DAILY_DATA)
        ), joined as (
          select
            p10.trade_date as signal_date,
            c.buy_date,
            p10.stock_code,
            sm.name,
            p1.pred_prob as raw_pred_1d,
            p3.pred_prob as raw_pred_3d,
            p5.pred_prob as raw_pred_5d,
            p10.pred_prob as raw_pred_10d,
            sm.amount,
            sm.turnover_rate,
            sm.total_mv,
            sm.atr_qfq,
            sm.pct_chg as signal_pct_chg_raw,
            bm.open as buy_open_raw,
            bm.pre_close as buy_pre_close_raw,
            ((bm.open / nullif(bm.pre_close, 0)) - 1.0) * 100.0 as buy_open_gap_raw_pct
          from p10.{TABLE['p10']} p10
          join p5.{TABLE['p5']} p5 using(trade_date, stock_code)
          join p3.{TABLE['p3']} p3 using(trade_date, stock_code)
          join p1.{TABLE['p1']} p1 using(trade_date, stock_code)
          join calendar c on c.signal_date=p10.trade_date and c.buy_date is not null
          join m.STOCK_DAILY_DATA sm on sm.trade_date=p10.trade_date and sm.stock_code=p10.stock_code
          join m.STOCK_DAILY_DATA bm on bm.trade_date=c.buy_date and bm.stock_code=p10.stock_code
          where p10.stock_code not like '%.BJ'
            and {clean('sm')}
            and {clean('bm')}
            and sm.amount is not null and sm.total_mv is not null and sm.pct_chg is not null
            and bm.open is not null and bm.pre_close is not null and bm.pre_close > 0
        )
        select * from joined where signal_date > '{cache_max}'
        """
        latest = con.execute(sql).fetchdf()
    finally:
        con.close()
    latest["signal_date"] = latest["signal_date"].astype(str)
    latest["buy_date"] = latest["buy_date"].astype(str)
    keep = [
        "signal_date", "buy_date", "stock_code", "name", "raw_pred_1d", "raw_pred_3d", "raw_pred_5d",
        "raw_pred_10d", "amount", "turnover_rate", "total_mv", "signal_pct_chg_raw", "buy_open_raw",
        "buy_pre_close_raw", "buy_open_gap_raw_pct",
    ]
    pool = pd.concat([cached.reindex(columns=keep), latest.reindex(columns=keep)], ignore_index=True)
    pool = pool.drop_duplicates(["signal_date", "stock_code"], keep="last")
    for raw, rank in [
        ("raw_pred_1d", "pred_1d"),
        ("raw_pred_3d", "pred_3d"),
        ("raw_pred_5d", "pred_5d"),
        ("raw_pred_10d", "pred_10d"),
    ]:
        r = pool.groupby("signal_date")[raw].rank(method="min", ascending=True)
        n = pool.groupby("signal_date")[raw].transform("count")
        pool[rank] = ((r - 1.0) / (n - 1.0).where(n > 1, 1.0)).fillna(0.0)
    pool["base_entry_score"] = 0.25 * pool["pred_1d"] + 0.25 * pool["pred_3d"] + 0.50 * pool["pred_10d"]
    pool["refill_entry_score"] = 0.10 * pool["pred_3d"] + 0.20 * pool["pred_5d"] + 0.70 * pool["pred_10d"]
    pool.attrs["cache_max_signal_date"] = cache_max
    pool.attrs["latest_rows_appended"] = int(len(latest))
    return pool


def enrich_atr(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keys = frame[["signal_date", "stock_code"]].drop_duplicates().copy()
    con = duckdb.connect()
    try:
        con.register("keys", keys)
        con.execute(f"attach '{L2.as_posix()}' as m (read_only)")
        atr = con.execute(
            """
            select k.signal_date, k.stock_code, d.atr_qfq
            from keys k
            left join m.STOCK_DAILY_DATA d
              on d.trade_date=k.signal_date and d.stock_code=k.stock_code
            """
        ).fetchdf()
    finally:
        con.close()
    out = frame.drop(columns=["atr_qfq"], errors="ignore").merge(atr, on=["signal_date", "stock_code"], how="left")
    return out


def build_fw_base(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    limit_rate = pd.Series(0.095, index=pool.index)
    board20 = pool["stock_code"].astype(str).str.startswith(("300", "301", "688"))
    limit_rate.loc[board20] = 0.195
    open_ret = pool["buy_open_raw"] / pool["buy_pre_close_raw"] - 1.0
    universe = pool[
        (pool["amount"] >= 90000)
        & (pool["total_mv"] >= 200000)
        & (pool["buy_open_gap_raw_pct"] <= 1.5)
        & (open_ret < limit_rate)
    ].copy()

    eligible = universe[(universe["signal_pct_chg_raw"] <= -1.75) & (universe["pred_10d"] >= 0.70)].copy()
    base = (
        eligible.sort_values(["buy_date", "base_entry_score", "stock_code"], ascending=[True, False, True])
        .groupby("buy_date", group_keys=False)
        .head(3)
        .copy()
    )
    base["source_layer"] = "base_production"
    base["entry_score"] = base["base_entry_score"]
    base["target_pct"] = 0.435

    base_counts = base.groupby("buy_date").size().to_dict()
    base_keys = set(zip(base["buy_date"].astype(str), base["stock_code"].astype(str)))
    refill_pool = universe[
        (universe["signal_pct_chg_raw"] <= -8.20237)
        & (universe["total_mv"] <= 453773)
        & (universe["refill_entry_score"] >= 0.995)
    ].copy()
    refill_pool = refill_pool[
        ~refill_pool.apply(lambda r: (str(r["buy_date"]), str(r["stock_code"])) in base_keys, axis=1)
    ]
    refill_pool = refill_pool.sort_values(["buy_date", "refill_entry_score", "stock_code"], ascending=[True, False, True])
    refill_parts = []
    for buy_date, day in refill_pool.groupby("buy_date"):
        need = max(0, 3 - int(base_counts.get(buy_date, 0)))
        if need:
            refill_parts.append(day.head(need))
    refill = pd.concat(refill_parts, ignore_index=True) if refill_parts else refill_pool.head(0).copy()
    refill["source_layer"] = "formal_l4_refill"
    refill["entry_score"] = refill["refill_entry_score"]
    refill["target_pct"] = 0.10

    fw = pd.concat([base, refill], ignore_index=True, sort=False)
    weak = (fw["signal_pct_chg_raw"] >= -2.5) | (fw["buy_open_gap_raw_pct"] >= 0)
    strong = (
        (fw["signal_pct_chg_raw"] <= -5.0)
        & (fw["buy_open_gap_raw_pct"] <= -0.8)
        & (fw["turnover_rate"] >= 4.5)
    )
    fw["feature_weight_scale"] = 1.0
    fw.loc[weak, "feature_weight_scale"] = 0.85
    fw.loc[strong, "feature_weight_scale"] = 1.05
    fw["target_pct"] *= fw["feature_weight_scale"]
    daily = fw.groupby("buy_date")["target_pct"].transform("sum")
    fw["target_pct"] *= (0.91 / daily).clip(upper=1.0)
    return fw, {
        "joined_rows": int(len(pool)),
        "fw_base_rows": int(len(base)),
        "fw_refill_rows": int(len(refill)),
        "fw_buy_days": int(fw["buy_date"].nunique()),
    }


def quality_bucket(df: pd.DataFrame) -> pd.Series:
    high = (
        df["buy_open_gap_raw_pct"].between(-5.0, 0.5)
        & (df["turnover_rate"] >= 4.0)
        & (df["atr_qfq"] <= 8.0)
        & (df["signal_pct_chg_raw"] <= -1.75)
        & (df["pred_10d"] >= 0.70)
    )
    mid = (
        df["buy_open_gap_raw_pct"].between(-6.0, 1.0)
        & (df["amount"] >= 120000)
        & (df["atr_qfq"] <= 12.0)
        & (df["signal_pct_chg_raw"] <= -1.75)
        & (df["pred_10d"] >= 0.70)
        & ~high
    )
    bucket = pd.Series(0, index=df.index, dtype="int64")
    bucket.loc[mid] = 1
    bucket.loc[high] = 2
    return bucket


def build_frs(pool: pd.DataFrame, fw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = enrich_atr(fw)
    work["quality_bucket"] = quality_bucket(work)
    work["sort_score"] = work["quality_bucket"] * 10.0 + work["pred_10d"] + work["pred_1d"] * 0.05
    core = (
        work.sort_values(["buy_date", "sort_score", "stock_code"], ascending=[True, False, True])
        .groupby("buy_date", group_keys=False)
        .head(2)
        .copy()
    )
    core["target_pct"] = core["quality_bucket"].map({2: 0.72, 1: 0.36, 0: 0.216}).astype(float)
    core.loc[core["signal_pct_chg_raw"] > -3.0, "target_pct"] *= 0.75
    core.loc[core["buy_open_gap_raw_pct"] > 0.5, "target_pct"] *= 0.50
    core.loc[core["buy_open_gap_raw_pct"] <= -3.0, "target_pct"] *= 1.05
    daily = core.groupby("buy_date")["target_pct"].transform("sum")
    core["target_pct"] *= (1.0 / daily).clip(upper=1.0)

    core["target_pct"] *= 0.92
    daily = core.groupby("buy_date")["target_pct"].transform("sum")
    core["target_pct"] *= (1.0 / daily).clip(upper=1.0)
    core["target_pct"] *= 1.04
    daily = core.groupby("buy_date")["target_pct"].transform("sum")
    core["target_pct"] *= (1.0 / daily).clip(upper=1.0)

    deep = core["signal_pct_chg_raw"] < -4.5
    q1 = (~deep) & (core["quality_bucket"] == 1)
    q0 = (~deep) & (core["quality_bucket"] == 0)
    scale = pd.Series(1.075, index=core.index)
    scale.loc[deep] = 1.075 * 0.86
    scale.loc[q1] *= 1.20
    scale.loc[q0] *= 1.50
    core["target_pct"] *= scale
    daily = core.groupby("buy_date")["target_pct"].transform("sum")
    core["target_pct"] *= (1.0 / daily).clip(upper=1.0)
    core.loc[core["signal_pct_chg_raw"] < -8.0, "target_pct"] *= 1.10
    daily = core.groupby("buy_date")["target_pct"].transform("sum")
    core["target_pct"] *= (1.0 / daily).clip(upper=1.0)
    core["target_pct"] *= 2.0
    daily = core.groupby("buy_date")["target_pct"].transform("sum")
    core["target_pct"] *= (1.0 / daily).clip(upper=1.0)
    core["layer"] = "core"

    core_days = set(core["buy_date"].astype(str))
    fallback = pool[
        (~pool["buy_date"].astype(str).isin(core_days))
        & (pool["raw_pred_10d"] >= 0.98)
        & (pool["amount"] >= 300000)
        & (pool["buy_open_gap_raw_pct"].between(-3.5, 0.5))
        & (pool["turnover_rate"] >= 2.0)
        & (pool["signal_pct_chg_raw"].between(-4.5, 2.0))
    ].copy()
    fallback = enrich_atr(fallback)
    fallback = fallback[fallback["atr_qfq"] <= 8.0].copy()
    fallback["sort_score"] = fallback["raw_pred_10d"] * 10 + fallback["raw_pred_5d"] + fallback["raw_pred_1d"] * 0.2
    fallback = (
        fallback.sort_values(["buy_date", "sort_score", "amount"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(1)
        .copy()
    )
    fallback["entry_score"] = fallback["pred_10d"]
    fallback["target_pct"] = 0.03
    fallback["quality_bucket"] = -1
    fallback["layer"] = "tiny_refill"

    out = pd.concat([core, fallback], ignore_index=True, sort=False)
    out["target_pct"] *= 0.90
    refill_mask = out["layer"].eq("tiny_refill")
    out.loc[refill_mask, "target_pct"] = out.loc[refill_mask, "target_pct"].clip(upper=0.03)
    out["target_pct"] = out["target_pct"].clip(lower=0.0, upper=0.90)
    daily = out.groupby("buy_date")["target_pct"].transform("sum")
    out["target_pct"] *= (1.0 / daily).clip(upper=1.0)
    out = out.sort_values(["buy_date", "layer", "sort_score"], ascending=[True, True, False]).copy()
    out["rank"] = out.groupby("buy_date").cumcount() + 1
    return out, {
        "core_rows": int(len(core)),
        "tiny_refill_rows": int(len(fallback)),
        "signal_rows": int(len(out)),
        "buy_days": int(out["buy_date"].nunique()),
        "stock_count": int(out["stock_code"].nunique()),
        "max_positions": int(out.groupby("buy_date").size().max()),
    }


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["stock_code"].map(
        lambda c: ("SHSE." if str(c).endswith(".SH") else "SZSE.") + str(c).split(".")[0]
    )
    out["pred_prob"] = out["entry_score"]
    out["holding_days"] = 1
    out["max_holding_days"] = 3
    out["score_exit_entry_ratio"] = "0.98000"
    out["min_holding_days_before_score_exit"] = 1
    out["score_continue_entry_ratio"] = "1.02000"
    out["signal_stop_loss_pct"] = 0.05
    out["signal_take_profit_pct"] = 0.08
    out["strategy_variant"] = "frs_scale090_cap090_current_l4_rebuild"
    out["source_strategy_variant"] = "current_active_formal_l4_full_rebuild"
    out["filter_name"] = out["strategy_variant"]
    out["entry_weight_name"] = "current_l4_rank_chain_rebuilt"
    out["dynamic_hold_name"] = "h1_mh3_exit098_cont102"
    out["buy_day_market_available"] = True
    out["buy_day_hard_gate_complete"] = True
    out["buy_day_st_rejected"] = False
    out["buy_day_open_limit_up_rejected"] = False
    out["latest_market_date"] = out["buy_date"]
    out["buy_open_gap_pct"] = out["buy_open_gap_raw_pct"]
    out["hybrid_source"] = out["layer"]
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "atr_qfq",
        "signal_pct_chg_raw", "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "source_strategy_variant", "filter_name", "entry_weight_name",
        "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete", "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected", "latest_market_date", "buy_open_gap_pct", "hybrid_source",
        "buy_open_gap_raw_pct", "feature_weight_scale", "daily_target_sum_after_cap", "quality_bucket", "sort_score", "layer",
    ]
    return out.reindex(columns=cols)


def parse_indicator(log_text: str) -> dict:
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                return {"raw": payload}
    return {}


def run_juejin() -> dict:
    cmd = [
        sys.executable, str(RUNNER), "--strategy-dir", str(STRATEGY), "--signal-file", str(SIGNAL),
        "--log-file", str(LOG), "--max-positions", "2", "--holding-days", "1", "--max-holding-days", "3",
        "--score-db", str(L4["p10"]), "--score-table", TABLE["p10"], "--market-db", str(L2),
        "--score-exit-entry-ratio", "0.98", "--min-holding-days-before-score-exit", "1",
        "--score-continue-entry-ratio", "1.02", "--stop-loss-pct", "0.05", "--take-profit-pct", "0.08",
        "--backtest-adjust", "none", "--backtest-initial-cash", "600000", "--backtest-slippage-ratio", "0.0015",
        "--backtest-start", "2022-06-06 09:00:00", "--backtest-end", "2026-07-20 15:30:00",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else ""
    return {"returncode": proc.returncode, "command": cmd, "indicator": parse_indicator(text), "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}


def main() -> None:
    SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pool = load_current_formal_pool()
    cache_audit = {
        "cache_path": str(CACHE),
        "cache_max_signal_date": pool.attrs.get("cache_max_signal_date"),
        "latest_rows_appended_from_current_l4": pool.attrs.get("latest_rows_appended"),
    }
    fw, fw_audit = build_fw_base(pool)
    frs, frs_audit = build_frs(pool, fw)
    final = finalize(frs)
    final.to_csv(SIGNAL, index=False, encoding="utf-8-sig")
    result = run_juejin()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "research_only_current_formal_l4_rebuild",
        "strategy_id": "prod_high_return_frs_scale090_cap090_v20260716",
        "input_contract": {
            "l2": str(L2),
            "l4": {k: {"db": str(v), "table": TABLE[k]} for k, v in L4.items()},
            "legacy_used": False,
            "historical_signal_used": False,
        },
        "signal": {
            "path": str(SIGNAL),
            "rows": int(len(final)),
            "signal_days": int(final["signal_date"].nunique()),
            "buy_days": int(final["buy_date"].nunique()),
            "stock_count": int(final["stock_code"].nunique()),
            "min_signal_date": str(final["signal_date"].min()),
            "max_signal_date": str(final["signal_date"].max()),
            "min_buy_date": str(final["buy_date"].min()),
            "max_buy_date": str(final["buy_date"].max()),
            "duplicate_keys": int(final.duplicated(["buy_date", "stock_code"]).sum()),
            "bj_rows": int(final["stock_code"].astype(str).str.endswith(".BJ").sum()),
        },
        "fw_audit": fw_audit,
        "frs_audit": frs_audit,
        "cache_audit": cache_audit,
        "juejin": result,
        "production_changed": False,
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ind = result.get("indicator") or {}
    lines = [
        "# 当前 formal L4 生产策略全历史重建复跑",
        "",
        "## 结论",
        f"- 信号行数：`{len(final)}`，买入日：`{final['buy_date'].nunique()}`。",
        f"- 覆盖：`{final['signal_date'].min()}` 至 `{final['signal_date'].max()}`。",
        f"- 掘金年化：`{float(ind.get('pnl_ratio_annual', 0))*100:.2f}%`。",
        f"- Sharpe：`{float(ind.get('sharp_ratio', 0)):.4f}`。",
        f"- 最大回撤：`{float(ind.get('max_drawdown', 0))*100:.2f}%`。",
        "",
        "## 口径",
        "- 1D/3D/5D/10D 全部读取当前 approved formal DuckDB 表。",
        "- 全历史信号从 L4 和 L2 重新生成，未读取任何旧生产信号 CSV。",
        "- 因子与模型排名使用 qfq 语义；交易、涨停和开盘缺口使用 L2 不复权真实价格。",
        "- 本轮只写研究报告，不修改生产策略和生产信号。",
        "",
        "## 证据",
        f"- 信号：`{SIGNAL}`",
        f"- 掘金日志：`{LOG}`",
        f"- JSON：`{SUMMARY}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "signal": str(SIGNAL), "juejin": ind, "returncode": result["returncode"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
