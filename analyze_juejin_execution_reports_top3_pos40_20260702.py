from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\work\quant\quant_mcp")
ART_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "artifact_probe_top3_pos40"
)


def main() -> int:
    reports = []
    with (ART_DIR / "execution_reports.jsonl").open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            r = obj["report"]
            reports.append(
                {
                    "trade_date": obj.get("trade_date"),
                    "created_at": r.get("created_at"),
                    "symbol": r.get("symbol"),
                    "side": r.get("side"),
                    "position_effect": r.get("position_effect"),
                    "price": float(r.get("price") or 0),
                    "volume": float(r.get("volume") or 0),
                    "amount": float(r.get("amount") or 0),
                    "commission": float(r.get("commission") or 0),
                }
            )
    df = pd.DataFrame(reports)
    df.to_csv(ART_DIR / "execution_reports_flat.csv", index=False, encoding="utf-8")

    # 掘金 side=1 为买，side=2 为卖。用现金流近似重建已实现单日流。
    daily = defaultdict(lambda: {"buy_amount": 0.0, "sell_amount": 0.0, "commission": 0.0, "buy_count": 0, "sell_count": 0})
    for row in reports:
        d = row["trade_date"]
        if row["side"] == 1:
            daily[d]["buy_amount"] += row["amount"]
            daily[d]["buy_count"] += 1
        elif row["side"] == 2:
            daily[d]["sell_amount"] += row["amount"]
            daily[d]["sell_count"] += 1
        daily[d]["commission"] += row["commission"]
    daily_rows = []
    for d, v in sorted(daily.items()):
        daily_rows.append({"trade_date": d, **v, "net_cashflow": v["sell_amount"] - v["buy_amount"] - v["commission"]})
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(ART_DIR / "execution_daily_cashflow.csv", index=False, encoding="utf-8")

    summary = {
        "execution_rows": int(len(df)),
        "buy_rows": int((df["side"] == 1).sum()),
        "sell_rows": int((df["side"] == 2).sum()),
        "trade_days": int(daily_df["trade_date"].nunique()),
        "avg_buy_amount": float(daily_df["buy_amount"].mean()),
        "avg_sell_amount": float(daily_df["sell_amount"].mean()),
        "worst_net_cashflow": float(daily_df["net_cashflow"].min()),
        "best_net_cashflow": float(daily_df["net_cashflow"].max()),
    }
    pd.DataFrame([summary]).to_csv(ART_DIR / "execution_cashflow_summary.csv", index=False, encoding="utf-8")
    daily_df.sort_values("net_cashflow").head(30).to_csv(ART_DIR / "worst_execution_cashflow_days.csv", index=False, encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(daily_df.sort_values("net_cashflow").head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
