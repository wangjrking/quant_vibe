from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_contribution_audit_20260720"
LOG = REPORT / "verbose_full.log"
BASE_SIGNAL = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_current_formal_l4_5d10d_top3_independent_sell_refine_20260720" / "signals" / "r1ge88_top3_min12_max13_ratio100_edge0.csv"
PRESSURE = REPORT / "pressure_signals"

BUY_RE = re.compile(r"BUY_ATTEMPT\s+(\d{8})\s+(\S+)\s+volume=(\d+)\s+price=([0-9.]+)")
SELL_RE = re.compile(r"SELL_ATTEMPT\s+(\d{8})\s+(\S+)\s+volume=(\d+)\s+price=([0-9.]+)")


def stock_code(symbol: str) -> str:
    market, code = symbol.split(".", 1)
    return f"{code}.SH" if market == "SHSE" else f"{code}.SZ"


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    PRESSURE.mkdir(parents=True, exist_ok=True)
    queues: dict[str, deque[dict]] = defaultdict(deque)
    trades = []
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        buy = BUY_RE.search(line)
        if buy:
            date, symbol, volume, price = buy.groups()
            queues[symbol].append({"entry_date": date, "symbol": symbol, "volume": int(volume), "buy_price": float(price)})
            continue
        sell = SELL_RE.search(line)
        if not sell:
            continue
        date, symbol, volume, price = sell.groups()
        if not queues[symbol]:
            continue
        trade = queues[symbol].popleft()
        trade.update({"exit_date": date, "sell_volume": int(volume), "sell_price": float(price)})
        trade["stock_code"] = stock_code(symbol)
        trade["entry_month"] = trade["entry_date"][:6]
        trade["net_return_proxy"] = trade["sell_price"] * 0.997 / (trade["buy_price"] * 1.003) - 1.0
        trade["net_pnl_proxy"] = trade["sell_volume"] * trade["sell_price"] * 0.997 - trade["volume"] * trade["buy_price"] * 1.003
        trades.append(trade)
    frame = pd.DataFrame(trades).sort_values("net_pnl_proxy", ascending=False)
    frame.to_csv(REPORT / "trade_contributions.csv", index=False, encoding="utf-8-sig")

    by_stock = frame.groupby("stock_code", as_index=False).agg(
        trades=("stock_code", "size"), net_pnl_proxy=("net_pnl_proxy", "sum"), mean_return=("net_return_proxy", "mean")
    ).sort_values("net_pnl_proxy", ascending=False)
    by_month = frame.groupby("entry_month", as_index=False).agg(
        trades=("entry_month", "size"), net_pnl_proxy=("net_pnl_proxy", "sum"), mean_return=("net_return_proxy", "mean")
    ).sort_values("net_pnl_proxy", ascending=False)
    by_stock.to_csv(REPORT / "contribution_by_stock.csv", index=False, encoding="utf-8-sig")
    by_month.to_csv(REPORT / "contribution_by_entry_month.csv", index=False, encoding="utf-8-sig")

    signal = pd.read_csv(BASE_SIGNAL)
    best_trade = frame.iloc[0]
    top_n = max(1, math.ceil(len(frame) * 0.01))
    top_keys = set(zip(frame.head(top_n)["entry_date"].astype(str), frame.head(top_n)["stock_code"].astype(str)))
    best_stock = str(by_stock.iloc[0]["stock_code"])
    best_month = str(by_month.iloc[0]["entry_month"])
    variants = {
        "base": signal,
        "remove_best_trade": signal[~((signal["buy_date"].astype(str) == str(best_trade["entry_date"])) & (signal["stock_code"] == best_trade["stock_code"]))],
        "remove_top1pct_trades": signal[~signal.apply(lambda row: (str(row["buy_date"]), str(row["stock_code"])) in top_keys, axis=1)],
        "remove_best_stock": signal[signal["stock_code"] != best_stock],
        "remove_best_entry_month": signal[~signal["buy_date"].astype(str).str.startswith(best_month)],
    }
    variant_meta = {}
    for name, data in variants.items():
        path = PRESSURE / f"{name}.csv"
        data.to_csv(path, index=False, encoding="utf-8-sig")
        variant_meta[name] = {"rows": int(len(data)), "removed": int(len(signal) - len(data)), "path": str(path)}
    positive_total = float(frame.loc[frame["net_pnl_proxy"] > 0, "net_pnl_proxy"].sum())
    payload = {
        "status": "research_only",
        "trade_count": int(len(frame)),
        "unclosed_count": int(sum(len(queue) for queue in queues.values())),
        "best_trade": best_trade.to_dict(),
        "best_stock": by_stock.iloc[0].to_dict(),
        "best_entry_month": by_month.iloc[0].to_dict(),
        "top1pct_trade_count": top_n,
        "best_trade_share_of_positive_pnl": float(best_trade["net_pnl_proxy"] / positive_total) if positive_total else None,
        "top1pct_share_of_positive_pnl": float(frame.head(top_n)["net_pnl_proxy"].sum() / positive_total) if positive_total else None,
        "variants": variant_meta,
    }
    (REPORT / "contribution_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
