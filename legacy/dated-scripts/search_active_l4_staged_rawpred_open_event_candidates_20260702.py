from __future__ import annotations

import itertools
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "active_l4_staged_rawpred_open_event_search"
)
MARKET_DB_COPY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "active_l4_qfq_event_search"
    / "runtime_l2_stock_daily_data_copy.duckdb"
)
L4 = {
    "1d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_1d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
    ),
    "3d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_3d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate",
    ),
    "5d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_5d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
    ),
    "10d": (
        ROOT / "quant/data_file/production_assets/duckdb/l4_executable_10d_open_return_formal.duckdb",
        "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
    ),
}


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(ret: pd.Series) -> float | None:
    std = float(ret.std(ddof=1))
    if not std:
        return None
    return float(ret.mean()) / std * (252**0.5)


def _mdd(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float(-(nav / nav.cummax() - 1.0).min())


def _read_l4(label: str, keys: pd.DataFrame | None = None) -> pd.DataFrame:
    db, table = L4[label]
    con = duckdb.connect(str(db), read_only=True)
    try:
        if keys is None:
            return con.execute(
                f"""
                SELECT trade_date AS signal_date, stock_code, pred_prob AS pred_{label}
                FROM {table}
                WHERE trade_date BETWEEN '20240101' AND '20260630'
                  AND stock_code NOT LIKE '%.BJ'
                  AND pred_prob >= 0.97
                """
            ).fetchdf()
        con.register("keys", keys[["signal_date", "stock_code"]])
        return con.execute(
            f"""
            SELECT k.signal_date, k.stock_code, p.pred_prob AS pred_{label}
            FROM keys k
            JOIN {table} p
              ON p.trade_date = k.signal_date
             AND p.stock_code = k.stock_code
            """
        ).fetchdf()
    finally:
        con.close()


def _read_market(preds: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(MARKET_DB_COPY), read_only=True)
    try:
        cal = con.execute(
            """
            SELECT trade_date AS signal_date,
                   lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                   lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_ref_date
                    FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA WHERE trade_date BETWEEN '20240101' AND '20260702')
            """
        ).fetchdf()
        x = preds.merge(cal, on="signal_date", how="inner")
        x = x[x["buy_date"].notna() & x["sell_ref_date"].notna()].copy()
        pairs = pd.concat(
            [
                x[["signal_date", "stock_code"]].rename(columns={"signal_date": "trade_date"}).assign(role="signal"),
                x[["buy_date", "stock_code"]].rename(columns={"buy_date": "trade_date"}).assign(role="buy"),
                x[["sell_ref_date", "stock_code"]].rename(columns={"sell_ref_date": "trade_date"}).assign(role="sell"),
            ],
            ignore_index=True,
        ).drop_duplicates()
        con.register("pairs", pairs)
        md = con.execute(
            """
            SELECT
                p.role,
                p.trade_date,
                p.stock_code,
                d.name,
                d.market,
                d.industry,
                d.open_qfq,
                d.close_qfq,
                d.amount,
                d.turnover_rate,
                d.total_mv,
                d.atr_qfq,
                d.pct_chg,
                d.ST_TYPE,
                d.ST_TYPE_name
            FROM pairs p
            JOIN STOCK_DAILY_DATA d
              ON d.trade_date = p.trade_date
             AND d.stock_code = p.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    sig = md[md["role"] == "signal"].drop(columns=["role"]).rename(
        columns={
            "trade_date": "signal_date",
            "open_qfq": "signal_open_qfq",
            "close_qfq": "signal_close_qfq",
        }
    )
    buy = md[md["role"] == "buy"][["trade_date", "stock_code", "open_qfq"]].rename(
        columns={"trade_date": "buy_date", "open_qfq": "buy_open_qfq"}
    )
    sell = md[md["role"] == "sell"][["trade_date", "stock_code", "open_qfq"]].rename(
        columns={"trade_date": "sell_ref_date", "open_qfq": "sell_open_qfq"}
    )
    x = x.merge(sig, on=["signal_date", "stock_code"], how="inner")
    x = x.merge(buy, on=["buy_date", "stock_code"], how="inner")
    x = x.merge(sell, on=["sell_ref_date", "stock_code"], how="inner")
    return x


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not MARKET_DB_COPY.exists():
        raise SystemExit(f"runtime market db copy not found: {MARKET_DB_COPY}")

    preds = _read_l4("10d")
    for label in ["1d", "3d", "5d"]:
        preds = preds.merge(_read_l4(label, preds), on=["signal_date", "stock_code"], how="inner")
    base = _read_market(preds)
    base = base[
        (base["stock_code"].str.endswith(".BJ") == False)
        & (base["signal_close_qfq"].notna())
        & (base["buy_open_qfq"].notna())
        & (base["sell_open_qfq"].notna())
        & (base["atr_qfq"].notna())
        & (base["amount"] >= 150000)
        & (base["total_mv"] >= 200000)
        & (base["turnover_rate"] >= 1.0)
        & (base["ST_TYPE"].fillna("").isin(["", "0"]))
        & (~base["ST_TYPE_name"].fillna("").str.contains("ST", na=False))
        & (~base["name"].fillna("").str.startswith(("ST", "*ST")))
    ].copy()
    base["buy_open_gap_qfq"] = base["buy_open_qfq"] / base["signal_close_qfq"] - 1.0
    base["ret_open_to_next_open_qfq"] = base["sell_open_qfq"] / base["buy_open_qfq"] - 1.0
    base["atr_pct_qfq"] = base["atr_qfq"] / base["signal_close_qfq"]
    base = base[(base["buy_open_gap_qfq"] >= -0.08) & (base["buy_open_gap_qfq"] <= 0.03)].copy()
    base["entry_score"] = (
        0.10 * base["pred_1d"]
        + 0.15 * base["pred_3d"]
        + 0.25 * base["pred_5d"]
        + 0.50 * base["pred_10d"]
    )
    for col in ["amount", "turnover_rate", "total_mv", "atr_pct_qfq"]:
        base[col + "_rank"] = base.groupby("signal_date")[col].rank(pct=True)
    base.to_parquet(REPORT_DIR / "staged_rawpred_open_event_base.parquet", index=False)

    rows = []
    for p10_min, p5_min, p1_min, gap_low, gap_high, amount_rank_min, turn_min, atr_rank_max, topn, target in itertools.product(
        [0.97, 0.985],
        [None, 0.70],
        [None, 0.55],
        [-0.04, 0.0],
        [0.005, 0.015],
        [0.0, 0.30],
        [0.0, 2.0],
        [0.80, 1.00],
        [1, 2, 3],
        [0.25, 0.35, 0.50],
    ):
        if topn * target > 1.05:
            continue
        x = base[
            (base["pred_10d"] >= p10_min)
            & (base["buy_open_gap_qfq"] >= gap_low)
            & (base["buy_open_gap_qfq"] <= gap_high)
            & (base["amount_rank"] >= amount_rank_min)
            & (base["turnover_rate"] >= turn_min)
            & (base["atr_pct_qfq_rank"] <= atr_rank_max)
        ].copy()
        if p5_min is not None:
            x = x[x["pred_5d"] >= p5_min]
        if p1_min is not None:
            x = x[x["pred_1d"] >= p1_min]
        if x.empty:
            continue
        x["rank_score"] = (
            x["entry_score"]
            + 0.035 * x["amount_rank"]
            + 0.020 * x["turnover_rate_rank"]
            - 0.020 * x["atr_pct_qfq_rank"]
            - 0.010 * x["total_mv_rank"]
        )
        x = x.sort_values(["signal_date", "rank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(topn)
        daily = x.groupby("signal_date").agg(
            daily_ret=("ret_open_to_next_open_qfq", lambda s: float(s.mean()) * target),
            names=("stock_code", "count"),
        )
        if len(daily) < 60 or float(daily["names"].mean()) < min(topn, 1.3):
            continue
        ret = daily["daily_ret"].astype(float)
        rows.append(
            {
                "name": f"staged_p10{int(p10_min*100)}_p5{('x' if p5_min is None else int(p5_min*100))}_p1{('x' if p1_min is None else int(p1_min*100))}_g{int(gap_low*1000)}to{int(gap_high*1000)}_a{int(amount_rank_min*100)}_tr{int(turn_min)}_atr{int(atr_rank_max*100)}_top{topn}_pos{int(target*100)}",
                "p10_min": p10_min,
                "p5_min": p5_min,
                "p1_min": p1_min,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "amount_rank_min": amount_rank_min,
                "turn_min": turn_min,
                "atr_rank_max": atr_rank_max,
                "topn": topn,
                "target_pct": target,
                "days": int(len(daily)),
                "avg_names": float(daily["names"].mean()),
                "local_annual": _annualized(float(ret.mean())),
                "local_sharpe": _sharpe(ret),
                "local_mdd": _mdd(ret),
                "recent60": _annualized(float(ret.tail(60).mean())) if len(ret) >= 60 else None,
                "recent120": _annualized(float(ret.tail(120).mean())) if len(ret) >= 120 else None,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("no candidates")
    summary = summary.sort_values(["local_sharpe", "local_annual"], ascending=[False, False])
    summary.to_csv(REPORT_DIR / "local_staged_rawpred_open_event_summary.csv", index=False, encoding="utf-8-sig")
    selected = summary[
        (summary["local_annual"] >= 3.0)
        & (summary["local_sharpe"] >= 3.0)
        & (summary["local_mdd"] <= 0.40)
        & (summary["recent60"] > 0)
        & (summary["recent120"] > 0)
    ].head(8)
    if selected.empty:
        selected = summary.head(8)
    selected.to_csv(REPORT_DIR / "selected_for_juejin.csv", index=False, encoding="utf-8-sig")

    score_db = REPORT_DIR / "score.duckdb"
    if score_db.exists():
        score_db.unlink()
    con = duckdb.connect(str(score_db))
    try:
        con.register("base_score", base[["signal_date", "stock_code", "entry_score"]])
        con.execute(
            "CREATE TABLE score AS SELECT signal_date AS trade_date, stock_code, entry_score AS pred_prob FROM base_score"
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()

    manifest_rows = []
    for row in selected.to_dict("records"):
        x = base[
            (base["pred_10d"] >= row["p10_min"])
            & (base["buy_open_gap_qfq"] >= row["gap_low"])
            & (base["buy_open_gap_qfq"] <= row["gap_high"])
            & (base["amount_rank"] >= row["amount_rank_min"])
            & (base["turnover_rate"] >= row["turn_min"])
            & (base["atr_pct_qfq_rank"] <= row["atr_rank_max"])
        ].copy()
        if pd.notna(row["p5_min"]):
            x = x[x["pred_5d"] >= row["p5_min"]]
        if pd.notna(row["p1_min"]):
            x = x[x["pred_1d"] >= row["p1_min"]]
        x["rank_score"] = (
            x["entry_score"]
            + 0.035 * x["amount_rank"]
            + 0.020 * x["turnover_rate_rank"]
            - 0.020 * x["atr_pct_qfq_rank"]
            - 0.010 * x["total_mv_rank"]
        )
        x = x.sort_values(["signal_date", "rank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(int(row["topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x.insert(2, "symbol", x["stock_code"].map(_symbol))
        x["pred_prob"] = x["entry_score"]
        x["target_pct"] = f"{float(row['target_pct']):.5f}"
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_exit_entry_ratio"] = "0.96000"
        x["min_holding_days_before_score_exit"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["signal_stop_loss_pct"] = "0.05000"
        x["signal_take_profit_pct"] = "0.08000"
        x["strategy_variant"] = row["name"]
        x["filter_name"] = "active_l4_staged_rawpred_open_event"
        x["entry_weight_name"] = "rawpred_10d_dominant"
        x["dynamic_hold_name"] = "h1m1_staged_rawpred_open_event"
        x["buy_day_market_available"] = True
        x["buy_day_hard_gate_complete"] = True
        x["buy_day_st_rejected"] = False
        x["buy_day_open_limit_up_rejected"] = False
        x["latest_market_date"] = "20260701"
        x["buy_open_gap_pct"] = x["buy_open_gap_qfq"] * 100.0
        out_cols = [
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
            "pct_chg",
            "buy_open_gap_pct",
            "target_pct",
            "holding_days",
            "max_holding_days",
            "score_exit_entry_ratio",
            "min_holding_days_before_score_exit",
            "score_continue_entry_ratio",
            "signal_stop_loss_pct",
            "signal_take_profit_pct",
            "strategy_variant",
            "filter_name",
            "entry_weight_name",
            "dynamic_hold_name",
            "buy_day_market_available",
            "buy_day_hard_gate_complete",
            "buy_day_st_rejected",
            "buy_day_open_limit_up_rejected",
            "latest_market_date",
        ]
        out_dir = REPORT_DIR / row["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x[out_cols].to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("signal_date").size()
        manifest_rows.append(
            {
                "name": row["name"],
                "signal_file": str(signal_file),
                "score_db": str(score_db),
                "score_table": "score",
                "rows": int(len(x)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                "days_below_topn": int((counts < int(row["topn"])).sum()),
                **row,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(REPORT_DIR / "juejin_candidate_manifest.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "base_rows": int(len(base)),
                "summary_rows": int(len(summary)),
                "market_db_copy": str(MARKET_DB_COPY),
                "l4_duckdb_tables": {k: {"db": str(v[0]), "table": v[1]} for k, v in L4.items()},
                "boundary": "research-only; formal Juejin validation required",
                "qfq_semantics": "buy_open_gap and local return use *_qfq fields explicitly",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
