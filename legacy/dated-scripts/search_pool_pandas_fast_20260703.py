from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "merged_signal_pool_cache.duckdb"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT_DIR = REPORT_DIR / "pool_pandas_fast_variants"


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _metrics(df: pd.DataFrame, hold: int, topn: int, target: float, slip: float = 0.0015) -> dict | None:
    if df.empty:
        return None
    ret_col = f"ret_h{hold}"
    sel = df.sort_values(["signal_date", "select_score", "stock_code"], ascending=[True, False, True])
    sel = sel.groupby("signal_date", group_keys=False).head(topn).copy()
    if len(sel) < 60:
        return None
    sel["_net_ret"] = (1.0 + sel[ret_col]) * (1.0 - slip) / (1.0 + slip) - 1.0
    daily = []
    wins = losses = 0
    for _, g in sel.groupby("signal_date"):
        gross = target * len(g)
        scale = min(1.0, 1.0 / gross) if gross > 0 else 0.0
        daily.append(float((g["_net_ret"] * target * scale).sum()))
        wins += int((g["_net_ret"] > 0).sum())
        losses += int((g["_net_ret"] < 0).sum())
    if len(daily) < 40:
        return None
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in daily:
        equity *= max(0.0001, 1.0 + r)
        peak = max(peak, equity)
        mdd = max(mdd, 1.0 - equity / peak)
    years = len(daily) / 252.0
    annual = equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else float("nan")
    mean = sum(daily) / len(daily)
    var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1) if len(daily) > 1 else 0.0
    sharpe = mean / math.sqrt(var) * math.sqrt(252.0) if var > 0 else float("nan")
    return {
        "local_annual": annual,
        "local_sharpe": sharpe,
        "local_max_drawdown": mdd,
        "rows": int(len(sel)),
        "days": int(len(daily)),
        "win_ratio": wins / (wins + losses) if wins + losses else float("nan"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = duckdb.connect(str(CACHE_DB), read_only=True)
    pool = cache.execute("SELECT * FROM pool").fetchdf()
    cache.close()
    pool["buy_date"] = pool["buy_date"].astype(str)
    pool["signal_date"] = pool["signal_date"].astype(str)
    pool["blend_score"] = 0.15 * pool["rank_1d"] + 0.35 * pool["rank_3d"] + 0.50 * pool["rank_10d"]
    dates = sorted(pool["buy_date"].unique())
    stocks = sorted(pool["stock_code"].unique())
    market = duckdb.connect(str(MARKET_DB), read_only=True)
    cal = market.execute("SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date").fetchdf()
    tds = cal["trade_date"].astype(str).tolist()
    idx = {d: i for i, d in enumerate(tds)}
    needed_dates = set()
    for d in dates:
        i = idx.get(d)
        if i is None:
            continue
        for step in [0, 1, 2, 3, 5]:
            if i + step < len(tds):
                needed_dates.add(tds[i + step])
    tmp_dates = pd.DataFrame({"trade_date": sorted(needed_dates)})
    tmp_stocks = pd.DataFrame({"stock_code": stocks})
    market.register("tmp_dates", tmp_dates)
    market.register("tmp_stocks", tmp_stocks)
    md = market.execute(
        """
        SELECT m.trade_date, m.stock_code, m.open, m.pre_close, m.amount, m.turnover_rate
        FROM STOCK_DAILY_DATA m
        JOIN tmp_dates d USING (trade_date)
        JOIN tmp_stocks s USING (stock_code)
        WHERE m.open IS NOT NULL
        """
    ).fetchdf()
    market.close()
    md["trade_date"] = md["trade_date"].astype(str)
    key = {(r.trade_date, r.stock_code): r for r in md.itertuples(index=False)}
    ret_rows = []
    for row in pool[["buy_date", "stock_code"]].drop_duplicates().itertuples(index=False):
        d = row.buy_date
        stock = row.stock_code
        i = idx.get(d)
        buy = key.get((d, stock))
        if i is None or buy is None or not buy.open:
            continue
        out = {
            "buy_date": d,
            "stock_code": stock,
            "buy_day_open_gap": (buy.open / buy.pre_close - 1.0) if buy.pre_close else float("nan"),
            "buy_amount": buy.amount,
            "buy_turnover": buy.turnover_rate,
        }
        for hold, step in [(1, 1), (2, 2), (3, 3), (5, 5)]:
            sell_date = tds[i + step] if i + step < len(tds) else None
            sell = key.get((sell_date, stock)) if sell_date else None
            out[f"ret_h{hold}"] = sell.open / buy.open - 1.0 if sell is not None and buy.open else float("nan")
        ret_rows.append(out)
    ret = pd.DataFrame(ret_rows)
    df = pool.merge(ret, on=["buy_date", "stock_code"], how="inner")
    for col in ["amount", "total_mv", "turnover_rate", "buy_amount", "buy_turnover"]:
        df[col + "_pctile"] = df.groupby("signal_date")[col].rank(pct=True)
    rows = []
    for pct_max in [-3.0, -5.0, -7.0, -10.0]:
        for gap_low, gap_high in [(-8.0, 1.5), (-5.0, 1.5), (-3.0, 0.0), (-1.0, 0.0)]:
            for buy_gap_low, buy_gap_high in [(-0.08, 0.03), (-0.05, 0.015), (-0.03, 0.01)]:
                for r10 in [0.8, 0.9, 0.95]:
                    for r1 in [0.0, 0.8, 0.9]:
                        base = df[
                            (df["pct_chg"] <= pct_max)
                            & (df["buy_open_gap_raw_pct"].between(gap_low, gap_high))
                            & (df["buy_day_open_gap"].between(buy_gap_low, buy_gap_high))
                            & (df["rank_10d"] >= r10)
                            & (df["rank_1d"] >= r1)
                        ].copy()
                        if len(base) < 80:
                            continue
                        score_modes = {
                            "blend": base["blend_score"],
                            "liq": base["blend_score"] + 0.03 * base["amount_pctile"] + 0.02 * base["total_mv_pctile"],
                            "turnlow": base["blend_score"] - 0.04 * base["turnover_rate_pctile"] + 0.02 * base["amount_pctile"],
                            "buyliq": base["blend_score"] + 0.03 * base["buy_amount_pctile"] - 0.02 * base["buy_turnover_pctile"],
                        }
                        for mode, score in score_modes.items():
                            base["select_score"] = score
                            for topn in [1, 2, 3]:
                                target = min(0.99, 0.99 / topn)
                                for hold in [1, 2, 3, 5]:
                                    m = _metrics(base, hold, topn, target)
                                    if not m:
                                        continue
                                    name = (
                                        f"pfast_pct{pct_max}_g{gap_low}_{gap_high}_bg{buy_gap_low}_{buy_gap_high}"
                                        f"_r10{r10}_r1{r1}_{mode}_top{topn}_h{hold}"
                                    )
                                    name = re.sub(r"[^A-Za-z0-9_]+", "_", name.replace("-", "m").replace(".", "p"))
                                    rows.append(
                                        {
                                            "name": name,
                                            "pct_max": pct_max,
                                            "gap_low": gap_low,
                                            "gap_high": gap_high,
                                            "buy_gap_low": buy_gap_low,
                                            "buy_gap_high": buy_gap_high,
                                            "r10": r10,
                                            "r1": r1,
                                            "mode": mode,
                                            "topn": topn,
                                            "hold": hold,
                                            "target": target,
                                            **m,
                                        }
                                    )
    summary = pd.DataFrame(rows).sort_values(["local_annual", "local_sharpe"], ascending=[False, False])
    summary_path = REPORT_DIR / "pool_pandas_fast_local_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest = []
    for row in summary.head(10).to_dict("records"):
        base = df[
            (df["pct_chg"] <= row["pct_max"])
            & (df["buy_open_gap_raw_pct"].between(row["gap_low"], row["gap_high"]))
            & (df["buy_day_open_gap"].between(row["buy_gap_low"], row["buy_gap_high"]))
            & (df["rank_10d"] >= row["r10"])
            & (df["rank_1d"] >= row["r1"])
        ].copy()
        if row["mode"] == "liq":
            base["select_score"] = base["blend_score"] + 0.03 * base["amount_pctile"] + 0.02 * base["total_mv_pctile"]
        elif row["mode"] == "turnlow":
            base["select_score"] = base["blend_score"] - 0.04 * base["turnover_rate_pctile"] + 0.02 * base["amount_pctile"]
        elif row["mode"] == "buyliq":
            base["select_score"] = base["blend_score"] + 0.03 * base["buy_amount_pctile"] - 0.02 * base["buy_turnover_pctile"]
        else:
            base["select_score"] = base["blend_score"]
        sig = base.sort_values(["signal_date", "select_score", "stock_code"], ascending=[True, False, True])
        sig = sig.groupby("signal_date", group_keys=False).head(int(row["topn"])).copy()
        sig["rank"] = sig.groupby("signal_date").cumcount() + 1
        sig["symbol"] = sig["stock_code"].map(_symbol)
        keep = [
            "signal_date",
            "buy_date",
            "symbol",
            "stock_code",
            "name",
            "rank",
            "entry_score",
            "pred_1d",
            "pred_3d",
            "pred_5d",
            "pred_10d",
            "rank_1d",
            "rank_3d",
            "rank_5d",
            "rank_10d",
            "amount",
            "turnover_rate",
            "total_mv",
            "atr_qfq",
            "pct_chg",
            "buy_open_gap_raw_pct",
        ]
        out = sig[keep].copy()
        out.insert(6, "pred_prob", out["entry_score"])
        out["target_pct"] = f"{row['target']:.5f}"
        out["holding_days"] = int(row["hold"])
        out["max_holding_days"] = int(row["hold"])
        out["score_exit_entry_ratio"] = "9.99000"
        out["min_holding_days_before_score_exit"] = 1
        out["score_continue_entry_ratio"] = "9.99000"
        out["signal_stop_loss_pct"] = ""
        out["signal_take_profit_pct"] = ""
        out["strategy_variant"] = row["name"]
        out["filter_name"] = "pool_pandas_fast"
        out["entry_weight_name"] = row["mode"]
        out["dynamic_hold_name"] = f"h{int(row['hold'])}"
        out["buy_day_market_available"] = True
        out["buy_day_hard_gate_complete"] = True
        out["buy_day_st_rejected"] = False
        out["buy_day_open_limit_up_rejected"] = False
        out["latest_market_date"] = "20260702"
        f = OUT_DIR / f"{row['name']}.csv"
        out.to_csv(f, index=False, encoding="utf-8")
        manifest.append({"name": row["name"], "signal_file": str(f), **{k: row[k] for k in ["local_annual", "local_sharpe", "local_max_drawdown", "rows", "days"]}})
    manifest_path = OUT_DIR / "pool_pandas_fast_top10_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0].keys()) if manifest else ["name"])
        writer.writeheader()
        writer.writerows(manifest)
    (OUT_DIR / "pool_pandas_fast_top10_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))
    print(summary_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
