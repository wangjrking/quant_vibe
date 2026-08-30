from __future__ import annotations

import csv
import json
import math
import os
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
    / "strategy_agent_active_l4_clean_proxy_20260703"
)
SIGNAL_DIR = REPORT_DIR / "signals"

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
    topn: int
    total_target: float
    pct_low: float | None
    pct_high: float | None
    gap_low: float | None
    gap_high: float | None
    r1_min: float
    r3_min: float
    r5_min: float
    r10_min: float
    amount_min: float
    mv_min: float
    atr_max: float | None
    liq_bonus: float
    mv_bonus: float
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
    return equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else float("nan")


def _sharpe(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return float("nan")
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((ret - mean) ** 2 for ret in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(var)
    return mean / std * math.sqrt(252.0) if std else float("nan")


def _max_drawdown(daily_returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    out = 0.0
    for ret in daily_returns:
        equity *= max(0.0001, 1.0 + ret)
        peak = max(peak, equity)
        if peak > 0:
            out = max(out, 1.0 - equity / peak)
    return out


def _evaluate(rows: pd.DataFrame, trade_dates: list[str], total_target: float, slippage: float = 0.0015) -> dict:
    if rows.empty:
        return {}
    min_buy = str(rows["buy_date"].min())
    max_buy = str(rows["buy_date"].max())
    span = [d for d in trade_dates if min_buy <= d <= max_buy]
    grouped = {str(k): v for k, v in rows.groupby("buy_date")}
    per_name_target = float(rows["target_pct"].iloc[0])
    daily: list[float] = []
    wins = 0
    losses = 0
    for date in span:
        day_rows = grouped.get(date)
        if day_rows is None:
            daily.append(0.0)
            continue
        returns = []
        for ret in day_rows["open_to_next_open_ret"].tolist():
            net = (1.0 + float(ret)) * (1.0 - slippage) / (1.0 + slippage) - 1.0
            returns.append(net)
            if net > 0:
                wins += 1
            elif net < 0:
                losses += 1
        target_sum = per_name_target * len(returns)
        scale = min(1.0, total_target / target_sum) if target_sum > 0 else 0.0
        daily.append(sum(per_name_target * scale * ret for ret in returns))
    nonzero = [x for x in daily if abs(x) > 1e-12]
    return {
        "proxy_annual": _annualized_return(daily),
        "proxy_sharpe": _sharpe(daily),
        "proxy_max_drawdown": _max_drawdown(daily),
        "proxy_trade_days": len(span),
        "proxy_active_days": len(nonzero),
        "proxy_coverage": len(nonzero) / len(span) if span else 0.0,
        "proxy_open_count": int(len(rows)),
        "proxy_win_ratio": wins / (wins + losses) if wins + losses else float("nan"),
    }


def _cases() -> list[Case]:
    small = os.environ.get("L5_PROXY_SMALL", "0").strip().lower() in {"1", "true", "yes"}
    weights = [
        ("w0000100", (0.0, 0.0, 0.0, 1.0)),
        ("w002080", (0.0, 0.0, 0.2, 0.8)),
        ("w003070", (0.0, 0.0, 0.3, 0.7)),
        ("w101080", (0.1, 0.1, 0.0, 0.8)),
        ("w2515055", (0.25, 0.15, 0.05, 0.55)),
        ("w3515050", (0.35, 0.15, 0.0, 0.50)),
        ("w25250050", (0.25, 0.25, 0.0, 0.50)),
        ("w15250060", (0.15, 0.25, 0.0, 0.60)),
    ]
    if small:
        weights = [
            ("w0000100", (0.0, 0.0, 0.0, 1.0)),
            ("w002080", (0.0, 0.0, 0.2, 0.8)),
            ("w3515050", (0.35, 0.15, 0.0, 0.50)),
        ]
    pct_bands = [
        ("pct_all", None, None),
        ("pct_m20_m5", -20.0, -5.0),
        ("pct_m12_m35", -12.0, -3.5),
        ("pct_m8_m2", -8.0, -2.0),
        ("pct_m5_2", -5.0, 2.0),
        ("pct_le_m175", None, -1.75),
    ]
    gap_bands = [
        ("gap_all", None, None),
        ("gap_m8_0", -0.08, 0.0),
        ("gap_m5_0", -0.05, 0.0),
        ("gap_m3_0", -0.03, 0.0),
        ("gap_m3_p15", -0.03, 0.015),
        ("gap_m8_m15", -0.08, -0.015),
        ("gap_m3_m15", -0.03, -0.015),
    ]
    rank_sets = [
        ("r10_70", 0.0, 0.0, 0.0, 0.70),
        ("r10_80", 0.0, 0.0, 0.0, 0.80),
        ("r10_90", 0.0, 0.0, 0.0, 0.90),
        ("r10_90_r1_80", 0.80, 0.0, 0.0, 0.90),
        ("r10_95_r1_80", 0.80, 0.0, 0.0, 0.95),
        ("r10_90_r3_70", 0.0, 0.70, 0.0, 0.90),
        ("r10_80_r5_80", 0.0, 0.0, 0.80, 0.80),
    ]
    liquidity = [
        ("liq0", 90_000.0, 200_000.0, None, 0.00, 0.00, 0.00),
        ("liq1", 150_000.0, 300_000.0, 0.18, 0.02, 0.01, 0.03),
        ("liq2", 250_000.0, 500_000.0, 0.16, 0.03, 0.02, 0.04),
    ]
    if small:
        pct_bands = [
            ("pct_m20_m5", -20.0, -5.0),
            ("pct_m12_m35", -12.0, -3.5),
            ("pct_le_m175", None, -1.75),
        ]
        gap_bands = [
            ("gap_m8_0", -0.08, 0.0),
            ("gap_m5_0", -0.05, 0.0),
            ("gap_m3_m15", -0.03, -0.015),
        ]
        rank_sets = [
            ("r10_80", 0.0, 0.0, 0.0, 0.80),
            ("r10_90", 0.0, 0.0, 0.0, 0.90),
            ("r10_95_r1_80", 0.80, 0.0, 0.0, 0.95),
        ]
        liquidity = [
            ("liq0", 90_000.0, 200_000.0, None, 0.00, 0.00, 0.00),
            ("liq1", 150_000.0, 300_000.0, 0.18, 0.02, 0.01, 0.03),
        ]
        topn_values = [1, 2, 3]
        target_values = [0.80, 0.99]
    else:
        topn_values = [1, 2, 3, 5]
        target_values = [0.60, 0.80, 0.99]
    cases: list[Case] = []
    for wname, w in weights:
        for topn in topn_values:
            for total_target in target_values:
                for pname, pct_low, pct_high in pct_bands:
                    for gname, gap_low, gap_high in gap_bands:
                        for rname, r1, r3, r5, r10 in rank_sets:
                            for lname, amount_min, mv_min, atr_max, liq_bonus, mv_bonus, atr_penalty in liquidity:
                                name = f"{wname}_top{topn}_t{int(total_target*100)}_{pname}_{gname}_{rname}_{lname}"
                                cases.append(
                                    Case(
                                        name=name,
                                        weights=w,
                                        topn=topn,
                                        total_target=total_target,
                                        pct_low=pct_low,
                                        pct_high=pct_high,
                                        gap_low=gap_low,
                                        gap_high=gap_high,
                                        r1_min=r1,
                                        r3_min=r3,
                                        r5_min=r5,
                                        r10_min=r10,
                                        amount_min=amount_min,
                                        mv_min=mv_min,
                                        atr_max=atr_max,
                                        liq_bonus=liq_bonus,
                                        mv_bonus=mv_bonus,
                                        atr_penalty=atr_penalty,
                                    )
                                )
    return cases


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)
    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS marketdb (READ_ONLY)")
        tables = {label: sources[label]["table"] for label in sources}
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_date
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
                    trade_date,
                    stock_code,
                    name,
                    open,
                    close,
                    pre_close,
                    open_qfq,
                    close_qfq,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    ST_TYPE,
                    ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
            )
            SELECT
                r.*,
                cal.buy_date,
                cal.sell_date,
                sig.name,
                sig.amount,
                sig.turnover_rate,
                sig.total_mv,
                sig.atr_qfq,
                sig.pct_chg,
                sig.close AS signal_close_raw,
                sig.close_qfq AS signal_close_qfq,
                buy.open AS buy_open_raw,
                buy.pre_close AS buy_pre_close_raw,
                sell.open AS sell_open_raw,
                buy.open / NULLIF(sig.close, 0) - 1 AS buy_open_gap_raw,
                sig.atr_qfq / NULLIF(sig.close_qfq, 0) AS atr_pct_qfq,
                sell.open / NULLIF(buy.open, 0) - 1 AS open_to_next_open_ret,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.total_mv) AS mv_rank,
                percent_rank() OVER (
                    PARTITION BY r.trade_date
                    ORDER BY sig.atr_qfq / NULLIF(sig.close_qfq, 0)
                ) AS atr_rank
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN md sig ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            JOIN md sell ON sell.trade_date = cal.sell_date AND sell.stock_code = r.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND cal.sell_date IS NOT NULL
              AND sig.close IS NOT NULL
              AND sig.close_qfq IS NOT NULL
              AND buy.open IS NOT NULL
              AND buy.pre_close IS NOT NULL
              AND sell.open IS NOT NULL
              AND coalesce(sig.ST_TYPE, '') IN ('', '0')
              AND coalesce(sig.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(sig.name, '') NOT LIKE 'ST%'
              AND coalesce(sig.name, '') NOT LIKE '*ST%'
              AND coalesce(buy.ST_TYPE, '') IN ('', '0')
              AND coalesce(buy.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND buy.open / NULLIF(buy.pre_close, 0) - 1 < CASE
                    WHEN r.stock_code LIKE '300%' OR r.stock_code LIKE '301%' OR r.stock_code LIKE '688%' THEN 0.195
                    ELSE 0.095
                  END
            """
        )
        trade_dates = [
            str(row[0])
            for row in con.execute("SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA ORDER BY trade_date").fetchall()
        ]
        manifest: list[dict] = []
        saved_cases: list[tuple[Case, pd.DataFrame, dict]] = []
        for case in _cases():
            w1, w3, w5, w10 = case.weights
            score_expr = f"({w1} * r1 + {w3} * r3 + {w5} * r5 + {w10} * r10)"
            rerank_expr = (
                f"{score_expr} + {case.liq_bonus} * amount_rank + {case.mv_bonus} * mv_rank - {case.atr_penalty} * atr_rank"
            )
            where = [
                f"r1 >= {case.r1_min}",
                f"r3 >= {case.r3_min}",
                f"r5 >= {case.r5_min}",
                f"r10 >= {case.r10_min}",
                f"amount >= {case.amount_min}",
                f"total_mv >= {case.mv_min}",
            ]
            if case.pct_low is not None:
                where.append(f"pct_chg >= {case.pct_low}")
            if case.pct_high is not None:
                where.append(f"pct_chg <= {case.pct_high}")
            if case.gap_low is not None:
                where.append(f"buy_open_gap_raw >= {case.gap_low}")
            if case.gap_high is not None:
                where.append(f"buy_open_gap_raw <= {case.gap_high}")
            if case.atr_max is not None:
                where.append(f"atr_pct_qfq <= {case.atr_max}")
            where_sql = " AND ".join(where)
            df = con.execute(
                f"""
                WITH filtered AS (
                    SELECT
                        *,
                        {score_expr}::DOUBLE AS entry_score,
                        {rerank_expr}::DOUBLE AS rerank_score
                    FROM base
                    WHERE {where_sql}
                ),
                picked AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY trade_date
                            ORDER BY rerank_score DESC, stock_code
                        ) AS pick_rank
                    FROM filtered
                )
                SELECT
                    trade_date AS signal_date,
                    buy_date,
                    stock_code,
                    name,
                    pick_rank AS rank,
                    entry_score,
                    pred_1d,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    r1 AS rank_1d,
                    r3 AS rank_3d,
                    r5 AS rank_5d,
                    r10 AS rank_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    buy_open_gap_raw,
                    open_to_next_open_ret
                FROM picked
                WHERE pick_rank <= {case.topn}
                ORDER BY signal_date, rank
                """
            ).fetchdf()
            if df.empty or len(df) < 80:
                continue
            per_target = min(0.99, case.total_target / case.topn)
            df.insert(2, "symbol", df["stock_code"].map(_symbol))
            df["pred_prob"] = df["entry_score"]
            df["target_pct"] = per_target
            df["holding_days"] = 1
            df["max_holding_days"] = 1
            df["score_exit_entry_ratio"] = "9.99000"
            df["min_holding_days_before_score_exit"] = 1
            df["score_continue_entry_ratio"] = "9.99000"
            df["signal_stop_loss_pct"] = "0.05000"
            df["signal_take_profit_pct"] = "0.08000"
            df["strategy_variant"] = case.name
            df["filter_name"] = "active_l4_clean_proxy_raw_exec"
            df["entry_weight_name"] = case.name.split("_top", 1)[0]
            df["dynamic_hold_name"] = "h1m1_raw_open_proxy"
            df["buy_day_market_available"] = True
            df["buy_day_hard_gate_complete"] = True
            df["buy_day_st_rejected"] = False
            df["buy_day_open_limit_up_rejected"] = False
            df["buy_open_gap_raw_pct"] = df["buy_open_gap_raw"] * 100.0
            metrics = _evaluate(df, trade_dates, case.total_target)
            if not metrics:
                continue
            row = {
                "name": case.name,
                "rows": int(len(df)),
                "signal_days": int(df["signal_date"].nunique()),
                "topn": case.topn,
                "total_target": case.total_target,
                "per_target": per_target,
                "weights": case.weights,
                "pct_low": case.pct_low,
                "pct_high": case.pct_high,
                "gap_low": case.gap_low,
                "gap_high": case.gap_high,
                "r1_min": case.r1_min,
                "r3_min": case.r3_min,
                "r5_min": case.r5_min,
                "r10_min": case.r10_min,
                "amount_min": case.amount_min,
                "mv_min": case.mv_min,
                "atr_max": case.atr_max,
                "liq_bonus": case.liq_bonus,
                "mv_bonus": case.mv_bonus,
                "atr_penalty": case.atr_penalty,
                **metrics,
            }
            manifest.append(row)
            if metrics["proxy_annual"] > 0.20 or len(saved_cases) < 30:
                saved_cases.append((case, df, row))
        manifest = sorted(manifest, key=lambda r: (r["proxy_annual"], r["proxy_sharpe"]), reverse=True)
        keep_names = {row["name"] for row in manifest[:80]}
        for case, df, _ in saved_cases:
            if case.name not in keep_names:
                continue
            out = df.drop(columns=["buy_open_gap_raw", "open_to_next_open_ret"])
            path = SIGNAL_DIR / f"{case.name}.csv"
            out.to_csv(path, index=False, encoding="utf-8")
            for row in manifest:
                if row["name"] == case.name:
                    row["signal_file"] = str(path)
                    break
        out_csv = REPORT_DIR / "clean_proxy_summary.csv"
        with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
            fieldnames = list(manifest[0].keys()) if manifest else ["name"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
        (REPORT_DIR / "clean_proxy_summary_top.json").write_text(
            json.dumps(manifest[:80], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (REPORT_DIR / "run_manifest.json").write_text(
            json.dumps(
                {
                    "input_manifests": {k: str(v) for k, v in MANIFESTS.items()},
                    "manifest_tables": {k: sources[k]["table"] for k in sources},
                    "market_db": str(market_db),
                    "case_count": len(_cases()),
                    "result_count": len(manifest),
                    "note": "research-only; raw open/close execution; full-calendar proxy with 0.15% buy and sell slippage",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(pd.DataFrame(manifest).head(30).to_string(index=False))
        print(out_csv)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
