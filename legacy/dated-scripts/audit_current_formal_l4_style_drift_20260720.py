from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_contribution_audit_20260720"
TRADES = REPORT / "trade_contributions.csv"
SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_top3_independent_sell_refine_20260720" / "signals" / "r1ge88_top3_min12_max13_ratio100_edge0.csv"


def board(code: str) -> str:
    value = str(code)
    if value.startswith(("300", "301")):
        return "创业板"
    if value.startswith("688"):
        return "科创板"
    return "主板"


def period(date: str) -> str:
    value = str(date)
    if value < "20240101":
        return "2022H2-2023"
    return value[:4]


def summarize(group: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(group["net_pnl_proxy"], errors="coerce")
    total_abs = float(pnl.abs().sum())
    return {
        "trades": int(len(group)),
        "total_mv_median": float(group["total_mv"].median()),
        "amount_median": float(group["amount"].median()),
        "turnover_rate_median": float(group["turnover_rate"].median()),
        "atr_qfq_median": float(group["atr_qfq"].median()),
        "small_cap_share_below_500000": float((group["total_mv"] < 500000).mean()),
        "low_amount_share_below_200000": float((group["amount"] < 200000).mean()),
        "main_board_share": float((group["board"] == "主板").mean()),
        "chinext_share": float((group["board"] == "创业板").mean()),
        "star_share": float((group["board"] == "科创板").mean()),
        "pnl_proxy": float(pnl.sum()),
        "abs_pnl_share": float(pnl.abs().sum() / total_abs) if total_abs else 0.0,
    }


def main() -> None:
    trades = pd.read_csv(TRADES, dtype={"entry_date": str, "stock_code": str})
    signal = pd.read_csv(SIGNAL, dtype={"buy_date": str, "stock_code": str})
    fields = ["buy_date", "stock_code", "amount", "turnover_rate", "total_mv", "atr_qfq", "rank_1d", "rank_5d", "rank_10d"]
    signal = signal[fields].drop_duplicates(["buy_date", "stock_code"])
    joined = trades.merge(signal, left_on=["entry_date", "stock_code"], right_on=["buy_date", "stock_code"], how="left")
    joined["period"] = joined["entry_date"].map(period)
    joined["board"] = joined["stock_code"].map(board)
    joined.to_csv(REPORT / "trade_style_joined.csv", index=False, encoding="utf-8-sig")
    missing = int(joined["total_mv"].isna().sum())
    clean = joined.dropna(subset=["total_mv", "amount", "turnover_rate", "atr_qfq"]).copy()
    summaries = {name: summarize(group) for name, group in clean.groupby("period")}
    summaries["full"] = summarize(clean)
    medians = pd.DataFrame(summaries).T
    medians.to_csv(REPORT / "style_by_period.csv", encoding="utf-8-sig")
    full = summaries["full"]
    drift = {}
    for name, item in summaries.items():
        if name == "full":
            continue
        drift[name] = {
            "total_mv_median_ratio_to_full": item["total_mv_median"] / full["total_mv_median"],
            "amount_median_ratio_to_full": item["amount_median"] / full["amount_median"],
            "turnover_median_ratio_to_full": item["turnover_rate_median"] / full["turnover_rate_median"],
            "atr_median_ratio_to_full": item["atr_qfq_median"] / full["atr_qfq_median"],
            "main_board_share_delta": item["main_board_share"] - full["main_board_share"],
        }
    payload = {
        "status": "research_only_style_disclosure",
        "uses_historical_point_in_time_fields": True,
        "trade_count": int(len(trades)),
        "joined_count": int(len(clean)),
        "missing_style_rows": missing,
        "full": full,
        "periods": {name: item for name, item in summaries.items() if name != "full"},
        "drift_vs_full": drift,
        "style_warning": "市值、成交额、板块暴露存在时间变化；仅作风险提示，是否升级为强准入取决于其与参数敏感性及时间切片的共同表现。",
    }
    (REPORT / "style_drift_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
