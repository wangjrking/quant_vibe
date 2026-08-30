from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_forward_frequency_optimization_20260721"
    / "current_l4_pool.duckdb"
)
MARKET = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_l4_attainability_audit_20260721"
SELECTION_END = "20251231"
ROUND_TRIP_COST = 0.006
GM_ANNUAL_TARGET = 5.0
REFERENCE_START = datetime(2022, 6, 7, 9, 0, 0)
REFERENCE_END = datetime(2026, 7, 17, 15, 30, 0)
MODELS = ("1d", "3d", "5d", "10d")
TOP_NS = (1, 3, 5, 10)


def metrics(con: duckdb.DuckDBPyConnection, source: str, return_column: str, horizon: int) -> list[dict]:
    rows: list[dict] = []
    for model in MODELS:
        con.execute(
            f'''CREATE OR REPLACE TEMP TABLE ranked_metric AS
                SELECT {return_column} realized_return,
                       row_number() OVER(
                           PARTITION BY signal_date
                           ORDER BY rank_{model} DESC, stock_code
                       ) pick_rank
                FROM {source}
                WHERE signal_date <= '{SELECTION_END}'
                  AND {return_column} IS NOT NULL'''
        )
        for top_n in TOP_NS:
            count, gross, median, win_rate = con.execute(
                f'''SELECT count(*), avg(realized_return), median(realized_return),
                           avg(CASE WHEN realized_return > 0 THEN 1.0 ELSE 0.0 END)
                    FROM ranked_metric WHERE pick_rank <= {top_n}'''
            ).fetchone()
            net = float(gross) - ROUND_TRIP_COST
            naive_annual = (1.0 + max(net, -0.99)) ** (252.0 / horizon) - 1.0
            rows.append(
                {
                    "horizon_trade_days": horizon,
                    "model": model,
                    "top_n": top_n,
                    "trades": int(count),
                    "gross_mean": float(gross),
                    "net_mean_after_0p6pct_round_trip": net,
                    "gross_median": float(median),
                    "win_rate": float(win_rate),
                    "naive_cagr_proxy": float(naive_annual),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(SOURCE), read_only=False)
    try:
        con.execute("PRAGMA threads=8")
        con.execute(f"ATTACH '{MARKET.as_posix()}' AS market (READ_ONLY)")
        con.execute(
            f'''CREATE TEMP TABLE returns_10d AS
                WITH calendar AS (
                  SELECT trade_date signal_date,
                         lead(trade_date, 11) OVER(ORDER BY trade_date) exit_date
                  FROM (SELECT DISTINCT trade_date FROM market.STOCK_DAILY_DATA)
                )
                SELECT p.signal_date, p.stock_code,
                       p.rank_1d, p.rank_3d, p.rank_5d, p.rank_10d,
                       x.open / nullif(p.buy_open_raw, 0) - 1.0 open_return_10d_raw
                FROM pool p
                JOIN calendar c USING(signal_date)
                JOIN market.STOCK_DAILY_DATA x
                  ON x.trade_date=c.exit_date AND x.stock_code=p.stock_code
                WHERE p.signal_date <= '{SELECTION_END}' AND x.open > 0'''
        )
        rows = metrics(con, "pool", "open_return_5d_raw", 5)
        rows.extend(metrics(con, "returns_10d", "open_return_10d_raw", 10))
    finally:
        con.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "formal_l4_attainability_summary.csv", index=False, encoding="utf-8-sig")
    reference_years = (REFERENCE_END - REFERENCE_START).total_seconds() / (365.0 * 86400.0)
    gm_required_cumulative = GM_ANNUAL_TARGET * reference_years
    gm_equivalent_cagr = (1.0 + gm_required_cumulative) ** (1.0 / reference_years) - 1.0
    required = {}
    for horizon in (5, 10):
        net = (1.0 + gm_equivalent_cagr) ** (horizon / 252.0) - 1.0
        required[str(horizon)] = {
            "required_net_return_per_horizon": net,
            "required_gross_return_with_0p6pct_round_trip_cost": net + ROUND_TRIP_COST,
        }
    best = frame.sort_values("naive_cagr_proxy", ascending=False).iloc[0].to_dict()
    payload = {
        "status": "diagnostic_only_not_a_backtest_or_strategy_candidate",
        "selection_data_max_date": SELECTION_END,
        "source_pool": str(SOURCE),
        "market_db": str(MARKET),
        "round_trip_cost": ROUND_TRIP_COST,
        "gm_pnl_ratio_annual_target": GM_ANNUAL_TARGET,
        "reference_backtest_years_365": reference_years,
        "gm_target_required_cumulative_return": gm_required_cumulative,
        "gm_target_equivalent_cagr": gm_equivalent_cagr,
        "required_return": required,
        "best_observed_simple_horizon_case": best,
        "caveat": "naive CAGR proxy ignores overlapping cohorts, cash competition and portfolio path; it is an optimistic attainability diagnostic, not a Juejin result",
    }
    (OUT / "attainability_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_5 = frame[frame["horizon_trade_days"] == 5].sort_values("naive_cagr_proxy", ascending=False).iloc[0]
    best_10 = frame[frame["horizon_trade_days"] == 10].sort_values("naive_cagr_proxy", ascending=False).iloc[0]
    report = f"""# 当前 formal L4 策略收益目标可达性审计

## 结论

在只使用 `20220606-20251231` 观察期、当前 formal L4 排名、不复权开盘执行价格和单次往返 `0.6%` 成本的简单持有诊断下，没有证据表明当前模型排序强度足以支撑 `500%` 年化目标。

## 关键数据

- 5 个交易日持有的最好组合：`{best_5['model']}` Top{int(best_5['top_n'])}，平均毛收益 `{best_5['gross_mean']:.4%}`，成本后平均收益 `{best_5['net_mean_after_0p6pct_round_trip']:.4%}`，朴素 CAGR 代理 `{best_5['naive_cagr_proxy']:.2%}`。
- 10 个交易日持有的最好组合：`{best_10['model']}` Top{int(best_10['top_n'])}，平均毛收益 `{best_10['gross_mean']:.4%}`，成本后平均收益 `{best_10['net_mean_after_0p6pct_round_trip']:.4%}`，朴素 CAGR 代理 `{best_10['naive_cagr_proxy']:.2%}`。
- 在本次约 `{reference_years:.2f}` 年回测长度下，掘金 `pnl_ratio_annual=500%` 对应累计收益约 `{gm_required_cumulative:.2%}`，等价 CAGR 约 `{gm_equivalent_cagr:.2%}`。
- 达到该掘金指标，5 日周期每期至少需要净收益 `{required['5']['required_net_return_per_horizon']:.2%}`，加回成本后需要毛收益 `{required['5']['required_gross_return_with_0p6pct_round_trip_cost']:.2%}`。
- 达到该掘金指标，10 日周期每期至少需要净收益 `{required['10']['required_net_return_per_horizon']:.2%}`，加回成本后需要毛收益 `{required['10']['required_gross_return_with_0p6pct_round_trip_cost']:.2%}`。

当前最好毛收益与目标所需毛收益仍存在约数倍差距。继续围绕同一批评分做密集阈值搜索，更可能制造样本内尖峰，而不是形成可通过准入的稳定策略。

## 口径边界

该审计不是掘金正式回测，也不用于发布收益结论。它忽略重叠持仓、现金竞争和完整组合路径，因此已经是偏乐观的可达性诊断。正式策略结论仍以掘金为准。

掘金字段 `pnl_ratio_annual` 与 CAGR 不是同一指标。后续报告必须同时列出掘金原始年化字段、累计收益和按净值计算的 CAGR，不得再把三者混写。

本审计没有读取 2026 数据选择参数，没有使用旧信号名单、research-only 预测资产、行业或月份过滤，也没有修改生产策略。
"""
    (OUT / "attainability_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
