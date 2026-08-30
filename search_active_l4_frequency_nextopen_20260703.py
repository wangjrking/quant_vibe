from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from prediction_manifest import load_prediction_source_manifest  # noqa: E402
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402


REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_active_l4_prod_repro_grid_20260703"
)
OUT_DIR = REPORT_DIR / "frequency_nextopen_variants"

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


@dataclass(frozen=True)
class Case:
    name: str
    weights: tuple[float, float, float, float]
    pct_low: float
    pct_high: float
    gap_low: float
    gap_high: float
    r1_min: float
    r3_min: float
    r10_min: float
    amount_min: float
    mv_min: float
    atr_max: float
    topn: int
    target_pct: float
    hold: int
    liq_bonus: float
    atr_penalty: float


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _annualized_return(daily_returns: list[float]) -> float:
    if not daily_returns:
        return float("nan")
    equity = 1.0
    for ret in daily_returns:
        equity *= max(0.0001, 1.0 + ret)
    years = len(daily_returns) / 252.0
    if years <= 0 or equity <= 0:
        return float("nan")
    return equity ** (1.0 / years) - 1.0


def _sharpe(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return float("nan")
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(var)
    return mean / std * math.sqrt(252.0) if std else float("nan")


def _mdd(daily_returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    out = 0.0
    for ret in daily_returns:
        equity *= max(0.0001, 1.0 + ret)
        peak = max(peak, equity)
        out = max(out, 1.0 - equity / peak)
    return out


def _evaluate(rows: list[dict], hold: int, target_pct: float, slippage: float = 0.0015) -> dict:
    by_day: dict[str, list[dict]] = {}
    wins = 0
    losses = 0
    ret_col = f"ret_h{hold}"
    for row in rows:
        try:
            raw_ret = float(row[ret_col])
        except Exception:
            continue
        net_ret = (1.0 + raw_ret) * (1.0 - slippage) / (1.0 + slippage) - 1.0
        row = dict(row)
        row["_net_ret"] = net_ret
        by_day.setdefault(str(row["signal_date"]), []).append(row)
        if net_ret > 0:
            wins += 1
        elif net_ret < 0:
            losses += 1
    daily: list[float] = []
    for day_rows in by_day.values():
        gross = target_pct * len(day_rows)
        scale = min(1.0, 1.0 / gross) if gross > 0 else 0.0
        daily.append(sum(target_pct * scale * row["_net_ret"] for row in day_rows))
    return {
        "local_annual": _annualized_return(daily),
        "local_sharpe": _sharpe(daily),
        "local_max_drawdown": _mdd(daily),
        "local_signal_days": len(by_day),
        "local_open_count": sum(len(v) for v in by_day.values()),
        "local_win_ratio": wins / (wins + losses) if wins + losses else float("nan"),
    }


def _cases() -> list[Case]:
    weights = [
        ("w10_20_00_70", (0.10, 0.20, 0.00, 0.70)),
        ("w15_35_00_50", (0.15, 0.35, 0.00, 0.50)),
        ("w25_25_00_50", (0.25, 0.25, 0.00, 0.50)),
    ]
    pct_ranges = [(-20.0, -7.0), (-12.0, -5.0), (-8.0, -3.0), (-5.0, -1.0)]
    gap_ranges = [(-0.05, -0.01), (-0.03, 0.0), (-0.01, 0.015), (-0.05, 0.015)]
    rank_sets = [
        (0.60, 0.0, 0.80),
        (0.80, 0.0, 0.80),
        (0.80, 0.60, 0.80),
        (0.90, 0.60, 0.85),
    ]
    amount_mins = [90_000.0]
    mv_mins = [200_000.0]
    atr_maxes = [0.18]
    topns = [1, 2, 3]
    holds = [1, 2, 5]
    out: list[Case] = []
    for w_name, w in weights:
        for pct_low, pct_high in pct_ranges:
            for gap_low, gap_high in gap_ranges:
                for r1_min, r3_min, r10_min in rank_sets:
                    for amount_min in amount_mins:
                        for mv_min in mv_mins:
                            for atr_max in atr_maxes:
                                for topn in topns:
                                    for hold in holds:
                                        target_pct = min(0.99, 0.99 / topn)
                                        name = (
                                            f"freq_{w_name}_pct{pct_low:g}to{pct_high:g}"
                                            f"_gap{gap_low:g}to{gap_high:g}"
                                            f"_r1{int(r1_min*100)}r3{int(r3_min*100)}r10{int(r10_min*100)}"
                                            f"_amt{int(amount_min/10000)}w_mv{int(mv_min/10000)}w"
                                            f"_atr{int(atr_max*100)}_top{topn}_h{hold}"
                                        )
                                        out.append(
                                            Case(
                                                name=name.replace("-", "m").replace(".", "p"),
                                                weights=w,
                                                pct_low=pct_low,
                                                pct_high=pct_high,
                                                gap_low=gap_low,
                                                gap_high=gap_high,
                                                r1_min=r1_min,
                                                r3_min=r3_min,
                                                r10_min=r10_min,
                                                amount_min=amount_min,
                                                mv_min=mv_min,
                                                atr_max=atr_max,
                                                topn=topn,
                                                target_pct=target_pct,
                                                hold=hold,
                                                liq_bonus=0.02,
                                                atr_penalty=0.04,
                                            )
                                        )
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {k: load_prediction_source_manifest(v, require_approved=True, allow_legacy=False) for k, v in MANIFESTS.items()}
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)
    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS marketdb (READ_ONLY)")
        tables = {k: v["table"] for k, v in sources.items()}
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_h1,
                    lead(trade_date, 3) OVER (ORDER BY trade_date) AS sell_h2,
                    lead(trade_date, 4) OVER (ORDER BY trade_date) AS sell_h3,
                    lead(trade_date, 6) OVER (ORDER BY trade_date) AS sell_h5
                FROM (SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA)
            ),
            preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{tables['10d']}" p10
                JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{tables['1d']}" p1 USING (trade_date, stock_code)
                WHERE p10.stock_code NOT LIKE '%.BJ'
            ),
            ranked AS (
                SELECT
                    *,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
                FROM preds
            ),
            md AS (
                SELECT
                    trade_date, stock_code, name, open, close, pre_close, open_qfq, close_qfq,
                    amount, turnover_rate, total_mv, atr_qfq, pct_chg, ST_TYPE, ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
            )
            SELECT
                r.*,
                cal.buy_date,
                sig.name,
                sig.amount,
                sig.turnover_rate,
                sig.total_mv,
                sig.atr_qfq,
                sig.pct_chg,
                buy.open AS buy_open_raw,
                buy.pre_close AS buy_pre_close_raw,
                buy.open / NULLIF(sig.close, 0) - 1 AS buy_open_gap_raw,
                sig.atr_qfq / NULLIF(sig.close_qfq, 0) AS atr_pct_qfq,
                sell1.open / NULLIF(buy.open, 0) - 1 AS ret_h1,
                sell2.open / NULLIF(buy.open, 0) - 1 AS ret_h2,
                sell3.open / NULLIF(buy.open, 0) - 1 AS ret_h3,
                sell5.open / NULLIF(buy.open, 0) - 1 AS ret_h5,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.total_mv) AS mv_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.atr_qfq / NULLIF(sig.close_qfq, 0)) AS atr_rank
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN md sig ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            JOIN md sell1 ON sell1.trade_date = cal.sell_h1 AND sell1.stock_code = r.stock_code
            JOIN md sell2 ON sell2.trade_date = cal.sell_h2 AND sell2.stock_code = r.stock_code
            JOIN md sell3 ON sell3.trade_date = cal.sell_h3 AND sell3.stock_code = r.stock_code
            JOIN md sell5 ON sell5.trade_date = cal.sell_h5 AND sell5.stock_code = r.stock_code
            WHERE coalesce(sig.ST_TYPE, '') IN ('', '0')
              AND coalesce(sig.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(sig.name, '') NOT LIKE 'ST%'
              AND coalesce(sig.name, '') NOT LIKE '*ST%'
              AND coalesce(buy.ST_TYPE, '') IN ('', '0')
              AND coalesce(buy.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND buy.open IS NOT NULL
              AND buy.pre_close IS NOT NULL
              AND sig.close IS NOT NULL
              AND sig.close_qfq IS NOT NULL
              AND sig.atr_qfq IS NOT NULL
            """
        )
        broad_df = con.execute(
            """
            SELECT *
            FROM base
            WHERE pct_chg BETWEEN -20.0 AND 1.0
              AND buy_open_gap_raw BETWEEN -0.05 AND 0.015
              AND r10 >= 0.75
              AND amount >= 90000.0
              AND total_mv >= 200000.0
              AND atr_pct_qfq <= 0.18
            """
        ).fetchdf()
        print(f"broad rows={len(broad_df)} days={broad_df['trade_date'].nunique() if not broad_df.empty else 0}")
        rows_out: list[dict] = []
        cases = _cases()
        for i, case in enumerate(cases, 1):
            w1, w3, w5, w10 = case.weights
            df = broad_df[
                (broad_df["pct_chg"] >= case.pct_low)
                & (broad_df["pct_chg"] <= case.pct_high)
                & (broad_df["buy_open_gap_raw"] >= case.gap_low)
                & (broad_df["buy_open_gap_raw"] <= case.gap_high)
                & (broad_df["r1"] >= case.r1_min)
                & (broad_df["r3"] >= case.r3_min)
                & (broad_df["r10"] >= case.r10_min)
                & (broad_df["amount"] >= case.amount_min)
                & (broad_df["total_mv"] >= case.mv_min)
                & (broad_df["atr_pct_qfq"] <= case.atr_max)
            ].copy()
            if df.empty or len(df) < 60:
                continue
            df["entry_score"] = w1 * df["r1"] + w3 * df["r3"] + w5 * df["r5"] + w10 * df["r10"]
            df["rerank_score"] = (
                df["entry_score"] + case.liq_bonus * df["amount_rank"] + 0.01 * df["mv_rank"] - case.atr_penalty * df["atr_rank"]
            )
            df = (
                df.sort_values(["trade_date", "rerank_score", "stock_code"], ascending=[True, False, True])
                .groupby("trade_date", group_keys=False)
                .head(case.topn)
                .copy()
            )
            df["pick_rank"] = df.groupby("trade_date").cumcount() + 1
            records = df.to_dict("records")
            metrics = _evaluate(records, case.hold, case.target_pct)
            if metrics["local_open_count"] < 60:
                continue
            rows_out.append(
                {
                    "name": case.name,
                    "rows": len(df),
                    "days": int(df["trade_date"].nunique()),
                    "topn": case.topn,
                    "target_pct": case.target_pct,
                    "hold": case.hold,
                    "pct_low": case.pct_low,
                    "pct_high": case.pct_high,
                    "gap_low": case.gap_low,
                    "gap_high": case.gap_high,
                    "r1_min": case.r1_min,
                    "r3_min": case.r3_min,
                    "r10_min": case.r10_min,
                    "amount_min": case.amount_min,
                    "mv_min": case.mv_min,
                    "atr_max": case.atr_max,
                    "weights": case.weights,
                    **metrics,
                }
            )
            if i % 1000 == 0:
                print(f"checked {i}/{len(cases)} kept={len(rows_out)}")
        summary = pd.DataFrame(rows_out)
        summary = summary.sort_values(
            ["local_annual", "local_sharpe", "local_max_drawdown"],
            ascending=[False, False, True],
        )
        summary_path = REPORT_DIR / "frequency_nextopen_local_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        top = summary.head(30).to_dict("records")
        for row in top:
            case = next(c for c in cases if c.name == row["name"])
            w1, w3, w5, w10 = case.weights
            out_file = OUT_DIR / f"{case.name}.csv"
            out_df = broad_df[
                (broad_df["pct_chg"] >= case.pct_low)
                & (broad_df["pct_chg"] <= case.pct_high)
                & (broad_df["buy_open_gap_raw"] >= case.gap_low)
                & (broad_df["buy_open_gap_raw"] <= case.gap_high)
                & (broad_df["r1"] >= case.r1_min)
                & (broad_df["r3"] >= case.r3_min)
                & (broad_df["r10"] >= case.r10_min)
            ].copy()
            out_df["entry_score"] = w1 * out_df["r1"] + w3 * out_df["r3"] + w5 * out_df["r5"] + w10 * out_df["r10"]
            out_df["rerank_score"] = (
                out_df["entry_score"] + case.liq_bonus * out_df["amount_rank"] + 0.01 * out_df["mv_rank"] - case.atr_penalty * out_df["atr_rank"]
            )
            out_df = (
                out_df.sort_values(["trade_date", "rerank_score", "stock_code"], ascending=[True, False, True])
                .groupby("trade_date", group_keys=False)
                .head(case.topn)
                .copy()
            )
            out_df["pick_rank"] = out_df.groupby("trade_date").cumcount() + 1
            out_df["buy_open_gap_raw_pct"] = out_df["buy_open_gap_raw"] * 100.0
            out_df = out_df.rename(columns={"trade_date": "signal_date", "pick_rank": "rank", "r1": "rank_1d", "r3": "rank_3d", "r5": "rank_5d", "r10": "rank_10d"})
            out_df["pred_prob"] = out_df["entry_score"]
            out_df = out_df[
                [
                    "signal_date", "buy_date", "stock_code", "name", "rank", "pred_prob", "entry_score",
                    "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d",
                    "rank_10d", "amount", "turnover_rate", "total_mv", "atr_qfq", "pct_chg",
                    "buy_open_gap_raw_pct",
                ]
            ]
            out_df.insert(2, "symbol", out_df["stock_code"].map(_symbol))
            out_df["target_pct"] = f"{case.target_pct:.5f}"
            out_df["holding_days"] = case.hold
            out_df["max_holding_days"] = case.hold
            out_df["score_exit_entry_ratio"] = "9.99000"
            out_df["min_holding_days_before_score_exit"] = 1
            out_df["score_continue_entry_ratio"] = "9.99000"
            out_df["signal_stop_loss_pct"] = ""
            out_df["signal_take_profit_pct"] = ""
            out_df["strategy_variant"] = case.name
            out_df["filter_name"] = "frequency_nextopen_active_l4_raw_exec"
            out_df["entry_weight_name"] = str(case.weights)
            out_df["dynamic_hold_name"] = f"h{case.hold}"
            out_df["buy_day_market_available"] = True
            out_df["buy_day_hard_gate_complete"] = True
            out_df["buy_day_st_rejected"] = False
            out_df["buy_day_open_limit_up_rejected"] = False
            out_df["latest_market_date"] = "20260702"
            out_df.to_csv(out_file, index=False, encoding="utf-8")
            row["signal_file"] = str(out_file)
        top_path = OUT_DIR / "frequency_nextopen_top30_manifest.csv"
        with top_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(top[0].keys()) if top else ["name"])
            writer.writeheader()
            writer.writerows(top)
        (OUT_DIR / "frequency_nextopen_top30_manifest.json").write_text(
            json.dumps(top, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(summary.head(20).to_string(index=False))
        print(summary_path)
        print(top_path)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
