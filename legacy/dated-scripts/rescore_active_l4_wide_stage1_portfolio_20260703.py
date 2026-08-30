from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
STAGE1 = REPORT_DIR / "active_l4_wide_sql_stage1_summary.csv"
OUT_CSV = REPORT_DIR / "active_l4_wide_portfolio_rescore_summary.csv"
VARIANT_DIR = REPORT_DIR / "active_l4_wide_portfolio_rescore_variants"
MANIFEST = REPORT_DIR / "active_l4_wide_portfolio_rescore_variant_manifest.csv"

SLIPPAGE = 0.0015
BASELINE_ANNUAL = 0.4853

SORT_MODES = {
    "score": "pred_prob desc, buy_amount desc",
    "lowgap": "buy_open_gap_raw_pct asc, pred_prob desc",
}


def add_filter(filters: list[str], col: str, op: str, value) -> None:
    if pd.notna(value):
        filters.append(f"{col} {op} {float(value)}")


def metrics(equity_curve: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(equity_curve, dtype=float)
    rets = np.diff(arr, prepend=1.0) / np.r_[1.0, arr[:-1]]
    annual = arr[-1] ** (252.0 / len(arr)) - 1.0 if len(arr) and arr[-1] > 0 else -1.0
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = rets.mean() / std * np.sqrt(252.0) if std > 0 else 0.0
    mdd = -float((arr / np.maximum.accumulate(arr) - 1.0).min()) if len(arr) else 0.0
    return annual, sharpe, mdd, arr[-1] - 1.0


def simulate(signals: pd.DataFrame, calendar: list[str], topn: int, hold: int, target_each: float) -> dict:
    ret_col = f"ret_h{hold}"
    sig_by = {d: g.sort_values("pick_rank").to_dict("records") for d, g in signals.groupby("buy_date")}
    cash = 1.0
    positions: list[dict] = []
    equity_curve: list[float] = []
    open_count = close_count = win = lose = 0
    for i, d in enumerate(calendar):
        new_positions = []
        for p in positions:
            if i >= p["exit_idx"]:
                ret = ((1.0 + p["ret"]) * (1.0 - SLIPPAGE) / (1.0 + SLIPPAGE) - 1.0)
                cash += p["value"] * (1.0 + ret)
                close_count += 1
                if ret > 0:
                    win += 1
                else:
                    lose += 1
            else:
                new_positions.append(p)
        positions = new_positions
        nav = cash + sum(p["value"] for p in positions)
        open_slots = max(topn - len(positions), 0)
        for r in sig_by.get(d, []):
            if open_slots <= 0:
                break
            if any(p["stock_code"] == r["stock_code"] for p in positions):
                continue
            if pd.isna(r[ret_col]):
                continue
            buy_value = min(nav * target_each, cash * 0.98)
            if buy_value <= 1e-9:
                continue
            cash -= buy_value
            positions.append(
                {
                    "stock_code": r["stock_code"],
                    "value": buy_value,
                    "ret": float(r[ret_col]),
                    "exit_idx": i + hold,
                }
            )
            open_count += 1
            open_slots -= 1
        equity_curve.append(cash + sum(p["value"] for p in positions))
    annual, sharpe, mdd, total = metrics(equity_curve)
    return {
        "portfolio_annual": annual,
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": mdd,
        "portfolio_total_return": total,
        "portfolio_open_count": open_count,
        "portfolio_close_count": close_count,
        "portfolio_win_ratio": win / (win + lose) if (win + lose) else 0.0,
    }


def query_signals(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    hold = int(row["hold"])
    topn = int(row["topn"])
    ret_col = f"ret_h{hold}"
    order_expr = SORT_MODES[str(row["sort_mode"])]
    filters = [
        f"score_pct_rank >= {float(row['score_min'])}",
        f"signal_amount >= {float(row['amount_min'])}",
        f"buy_amount >= {float(row['amount_min'])}",
        f"{ret_col} is not null",
    ]
    add_filter(filters, "signal_pct_chg", ">=", row.get("pct_min"))
    add_filter(filters, "signal_pct_chg", "<=", row.get("pct_max"))
    add_filter(filters, "buy_open_gap_raw_pct", ">=", row.get("buy_gap_min"))
    add_filter(filters, "buy_open_gap_raw_pct", "<=", row.get("buy_gap_max"))
    where = " and ".join(filters)
    return con.execute(
        f"""
        with picked as (
            select
                *,
                row_number() over(partition by buy_date order by {order_expr}) as pick_rank
            from active_l4_wide
            where {where}
        )
        select
            signal_date, buy_date, stock_code, name, market, pick_rank,
            pred_prob, score_pct_rank, signal_pct_chg, signal_open_gap_raw_pct,
            buy_open_gap_raw_pct, signal_amount, buy_amount, signal_total_mv,
            buy_total_mv, ret_h1, ret_h2, ret_h3, ret_h5
        from picked
        where pick_rank <= {topn}
        order by buy_date, pick_rank
        """
    ).fetchdf()


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    stage = pd.read_csv(STAGE1)
    # Keep candidates that looked promising under the fast daily proxy plus the
    # lower-drawdown edge cases. The portfolio simulator is the gate.
    selected = pd.concat(
        [
            stage.sort_values(["annual", "sharpe"], ascending=False).head(80),
            stage[stage["max_drawdown"] < 0.60].sort_values(["annual", "sharpe"], ascending=False).head(80),
            stage.sort_values(["sharpe", "annual"], ascending=False).head(40),
        ],
        ignore_index=True,
    ).drop_duplicates("case_id")
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    calendar = con.execute(
        "select distinct buy_date from active_l4_wide where buy_date between '20220607' and '20260702' order by buy_date"
    ).fetchdf()["buy_date"].astype(str).tolist()
    rows = []
    saved = []
    for _, row in selected.iterrows():
        sig = query_signals(con, row)
        sig["buy_date"] = sig["buy_date"].astype(str)
        sim = simulate(sig, calendar, int(row["topn"]), int(row["hold"]), float(row["target_each"]))
        out = row.to_dict()
        out.update(sim)
        rows.append(out)
    summary = pd.DataFrame(rows).sort_values(["portfolio_annual", "portfolio_sharpe"], ascending=False)
    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    promotable = summary[
        (summary["portfolio_annual"] > BASELINE_ANNUAL)
        & (summary["portfolio_max_drawdown"] < 0.40)
    ].sort_values(["portfolio_annual", "portfolio_sharpe"], ascending=False).head(5)
    for _, row in promotable.iterrows():
        sig = query_signals(con, row)
        sig["symbol"] = np.where(
            sig["stock_code"].str.endswith(".SH"),
            "SHSE." + sig["stock_code"].str.split(".").str[0],
            "SZSE." + sig["stock_code"].str.split(".").str[0],
        )
        sig["rank"] = sig["pick_rank"]
        sig["target_pct"] = float(row["target_each"])
        sig["holding_days"] = int(row["hold"])
        sig["max_holding_days"] = int(row["hold"])
        sig["score_exit_entry_ratio"] = 9.99
        sig["min_holding_days_before_score_exit"] = 1
        sig["score_continue_entry_ratio"] = 9.99
        sig["buy_day_market_available"] = True
        sig["buy_day_hard_gate_complete"] = True
        sig["buy_day_st_rejected"] = False
        sig["buy_day_open_limit_up_rejected"] = False
        sig["latest_market_date"] = "20260702"
        name = str(row["name"]) + "_portfolio"
        path = VARIANT_DIR / f"{name}.csv"
        cols = [
            "signal_date", "buy_date", "symbol", "stock_code", "name", "rank",
            "pred_prob", "score_pct_rank", "signal_pct_chg", "signal_open_gap_raw_pct",
            "buy_open_gap_raw_pct", "signal_amount", "buy_amount", "signal_total_mv",
            "buy_total_mv", "target_pct", "holding_days", "max_holding_days",
            "score_exit_entry_ratio", "min_holding_days_before_score_exit",
            "score_continue_entry_ratio", "buy_day_market_available",
            "buy_day_hard_gate_complete", "buy_day_st_rejected",
            "buy_day_open_limit_up_rejected", "latest_market_date",
        ]
        sig[cols].to_csv(path, index=False, encoding="utf-8-sig")
        saved_row = row.to_dict()
        saved_row["signal_file"] = str(path)
        saved.append(saved_row)
    pd.DataFrame(saved).to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    print(summary.head(30)[[
        "name", "portfolio_annual", "portfolio_sharpe", "portfolio_max_drawdown",
        "portfolio_open_count", "portfolio_win_ratio", "annual", "sharpe", "max_drawdown"
    ]].to_string(index=False))
    print("summary", OUT_CSV)
    print("manifest", MANIFEST, "rows", len(saved))
    con.close()


if __name__ == "__main__":
    main()
