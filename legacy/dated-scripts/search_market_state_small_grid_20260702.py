from __future__ import annotations

import itertools
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "active_l4_market_state_candidates"
BASE_PARQUET = OUT_DIR / "market_state_base.parquet"
MARKET_DB = BASE_DIR / "active_l4_qfq_event_search" / "runtime_l2_stock_daily_data_copy.duckdb"


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(series: pd.Series) -> float | None:
    std = float(series.std(ddof=1))
    if std == 0:
        return None
    return float(series.mean()) / std * (252**0.5)


def _mdd(series: pd.Series) -> float:
    nav = (1.0 + series.fillna(0.0)).cumprod()
    return float(-(nav / nav.cummax() - 1.0).min())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{MARKET_DB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base0 AS
            SELECT
                *,
                r10 AS score_10d,
                0.5 * r5 + 0.5 * r10 AS score_5d10d,
                0.10 * r1 + 0.15 * r3 + 0.25 * r5 + 0.50 * r10 AS score_mix,
                0.10 * r1 + 0.15 * r3 + 0.25 * r5 + 0.50 * r10
                    + 0.04 * amount_rank + 0.02 * turnover_rank - 0.03 * atr_rank AS score_liq
            FROM read_parquet('{BASE_PARQUET.as_posix()}')
            WHERE trade_date >= '20240101'
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE base AS
            SELECT
                b.*,
                buy.open AS buy_open_raw,
                sellref.open AS sell_open_raw,
                sellref.open / NULLIF(buy.open, 0) - 1 AS ret_open_to_next_open_raw
            FROM base0 b
            JOIN marketdb.STOCK_DAILY_DATA buy
              ON buy.trade_date = b.buy_date AND buy.stock_code = b.stock_code
            JOIN marketdb.STOCK_DAILY_DATA sellref
              ON sellref.trade_date = b.sell_ref_date AND sellref.stock_code = b.stock_code
            WHERE buy.open IS NOT NULL
              AND sellref.open IS NOT NULL
            """
        )
        calendar = con.execute(
            """
            SELECT DISTINCT trade_date
            FROM base
            ORDER BY trade_date
            """
        ).fetchdf()
        calendar["daily_ret"] = 0.0
        calendar_days = int(len(calendar))
        rows = []
        configs = list(
            itertools.product(
                ["score_10d", "score_mix", "score_liq"],
                [0.50, 0.60],
                [0.03, 0.08],
                [0.0],
                [(-0.02, 0.010), (-0.005, 0.015)],
                [(-2.0, 6.0), (0.0, 8.0)],
                [0.60, 0.85],
                [0.2],
                [1, 3],
                [0.30, 0.45],
            )
        )
        for score_col, up_min, d5_max, med_min, gap_pair, pct_pair, atr_max, amount_min, topn, target in configs:
            if topn * target > 1.05:
                continue
            gap_low, gap_high = gap_pair
            pct_min, pct_max = pct_pair
            sql = f"""
            WITH filtered AS (
                SELECT *
                FROM base
                WHERE up_ratio >= {up_min}
                  AND down5_ratio <= {d5_max}
                  AND median_pct >= {med_min}
                  AND buy_open_gap_qfq BETWEEN {gap_low} AND {gap_high}
                  AND pct_chg BETWEEN {pct_min} AND {pct_max}
                  AND atr_rank <= {atr_max}
                  AND amount_rank >= {amount_min}
            ),
            ranked AS (
                SELECT
                    trade_date,
                    stock_code,
                    ret_open_to_next_open_raw,
                    row_number() OVER (PARTITION BY trade_date ORDER BY {score_col} DESC, stock_code) AS rn
                FROM filtered
            ),
            picked AS (
                SELECT * FROM ranked WHERE rn <= {int(topn)}
            )
            SELECT
                trade_date,
                avg(ret_open_to_next_open_raw) * {float(target)} AS daily_ret,
                count(*) AS names
            FROM picked
            GROUP BY trade_date
            ORDER BY trade_date
            """
            daily = con.execute(sql).fetchdf()
            if len(daily) < 50:
                continue
            if float(daily["names"].mean()) < min(topn, 1.2):
                continue
            active_ret = daily["daily_ret"].astype(float)
            calendar_ret = calendar[["trade_date", "daily_ret"]].merge(
                daily[["trade_date", "daily_ret"]],
                on="trade_date",
                how="left",
                suffixes=("_zero", ""),
            )
            ret = calendar_ret["daily_ret"].fillna(calendar_ret["daily_ret_zero"]).astype(float)
            rows.append(
                {
                    "name": f"mss_{score_col}_up{int(up_min*100)}_d5{int(d5_max*100)}_med{str(med_min).replace('.','p')}_g{int(gap_low*1000)}to{int(gap_high*1000)}_p{str(pct_min).replace('.','p').replace('-','m')}to{str(pct_max).replace('.','p')}_atr{int(atr_max*100)}_a{int(amount_min*100)}_top{topn}_pos{int(target*100)}",
                    "score_col": score_col,
                    "up_min": up_min,
                    "down5_max": d5_max,
                    "med_min": med_min,
                    "gap_low": gap_low,
                    "gap_high": gap_high,
                    "pct_min": pct_min,
                    "pct_max": pct_max,
                    "atr_max": atr_max,
                    "amount_min": amount_min,
                    "topn": topn,
                    "target_pct": target,
                    "days": int(len(daily)),
                    "calendar_days": calendar_days,
                    "avg_names": float(daily["names"].mean()),
                    "active_day_annual": _annualized(float(active_ret.mean())),
                    "active_day_sharpe": _sharpe(active_ret),
                    "local_annual": _annualized(float(ret.mean())),
                    "local_sharpe": _sharpe(ret),
                    "local_mdd": _mdd(ret),
                    "recent60": _annualized(float(ret.tail(60).mean())) if len(ret) >= 60 else None,
                }
            )

        summary = pd.DataFrame(rows)
        if summary.empty:
            raise SystemExit("no market-state candidates")
        summary = summary.sort_values(["local_sharpe", "local_annual"], ascending=[False, False])
        summary.to_csv(OUT_DIR / "local_market_state_small_grid_summary.csv", index=False, encoding="utf-8-sig")
        selected = summary[
            (summary["local_annual"] > 0)
            & (summary["local_sharpe"] > 0.5)
            & (summary["local_mdd"] < 0.60)
        ].head(4)
        if selected.empty:
            selected = summary.head(4)
        selected.to_csv(OUT_DIR / "selected_small_grid_for_juejin.csv", index=False, encoding="utf-8-sig")

        manifest = []
        for row in selected.to_dict("records"):
            signal_sql = f"""
            WITH filtered AS (
                SELECT *
                FROM base
                WHERE up_ratio >= {row['up_min']}
                  AND down5_ratio <= {row['down5_max']}
                  AND median_pct >= {row['med_min']}
                  AND buy_open_gap_qfq BETWEEN {row['gap_low']} AND {row['gap_high']}
                  AND pct_chg BETWEEN {row['pct_min']} AND {row['pct_max']}
                  AND atr_rank <= {row['atr_max']}
                  AND amount_rank >= {row['amount_min']}
            ),
            ranked AS (
                SELECT
                    *,
                    row_number() OVER (PARTITION BY trade_date ORDER BY {row['score_col']} DESC, stock_code) AS rank
                FROM filtered
            )
            SELECT *
            FROM ranked
            WHERE rank <= {int(row['topn'])}
            ORDER BY trade_date, rank
            """
            sig = con.execute(signal_sql).fetchdf()
            if sig.empty:
                continue
            out_dir = OUT_DIR / row["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            score_db = out_dir / "score.duckdb"
            if score_db.exists():
                score_db.unlink()
            con.execute(f"ATTACH '{score_db.as_posix()}' AS out_score")
            con.execute(
                f"""
                CREATE TABLE out_score.score AS
                SELECT trade_date, stock_code, {row['score_col']} AS pred_prob
                FROM base
                """
            )
            con.execute("CHECKPOINT out_score")
            con.execute("DETACH out_score")
            sig.insert(2, "symbol", sig["stock_code"].map(_symbol))
            sig["pred_prob"] = sig[row["score_col"]]
            sig["target_pct"] = f"{float(row['target_pct']):.5f}"
            sig["holding_days"] = 1
            sig["max_holding_days"] = 1
            sig["score_exit_entry_ratio"] = "9.99000"
            sig["min_holding_days_before_score_exit"] = 1
            sig["score_continue_entry_ratio"] = "9.99000"
            sig["strategy_variant"] = row["name"]
            sig["filter_name"] = "market_state_small_grid"
            sig["entry_weight_name"] = row["score_col"]
            sig["dynamic_hold_name"] = "h1m1_market_state_small"
            sig["buy_day_market_available"] = True
            sig["buy_day_hard_gate_complete"] = True
            sig["buy_day_st_rejected"] = False
            sig["buy_day_open_limit_up_rejected"] = False
            sig["latest_market_date"] = "20260701"
            sig["buy_open_gap_pct"] = sig["buy_open_gap_qfq"] * 100.0
            out_cols = [
                "trade_date",
                "buy_date",
                "symbol",
                "stock_code",
                "name",
                "rank",
                "pred_prob",
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
            signal = sig[out_cols].rename(columns={"trade_date": "signal_date"})
            signal_file = out_dir / "signals.csv"
            signal.to_csv(signal_file, index=False, encoding="utf-8")
            counts = signal.groupby("signal_date").size()
            manifest.append(
                {
                    "name": row["name"],
                    "signal_file": str(signal_file),
                    "score_db": str(score_db),
                    "score_table": "score",
                    "rows": int(len(signal)),
                    "signal_days": int(counts.size),
                    "avg_names": float(counts.mean()),
                    "days_below_topn": int((counts < int(row["topn"])).sum()),
                    **row,
                }
            )
        pd.DataFrame(manifest).to_csv(OUT_DIR / "juejin_small_grid_candidate_manifest.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(manifest).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
