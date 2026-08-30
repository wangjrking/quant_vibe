from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
ARTIFACT_DIR = OUT_DIR / "top2_s150_cap82_w62_artifacts"
SIGNAL_FILE = OUT_DIR / "executable_top2_open_gap_market_signals/top2_s150_cap82_w62.csv"
TRADES_OUT = OUT_DIR / "top2_s150_cap82_w62_paired_trades_with_features.csv"
SEGMENT_OUT = OUT_DIR / "top2_s150_cap82_w62_volatility_segment_summary.csv"
REPORT_OUT = OUT_DIR / "top2_s150_cap82_w62_volatility_segment_report_20260704.md"


def stock_code_from_symbol(symbol: str) -> str:
    if symbol.startswith("SHSE."):
        return symbol[5:] + ".SH"
    if symbol.startswith("SZSE."):
        return symbol[5:] + ".SZ"
    return symbol


def load_exec_reports() -> list[dict]:
    rows: list[dict] = []
    path = ARTIFACT_DIR / "execution_reports.jsonl"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        report = obj["report"]
        report["trade_date"] = str(obj.get("trade_date") or "")
        rows.append(report)
    rows.sort(key=lambda r: (r.get("created_at") or "", r.get("exec_id") or ""))
    return rows


def pair_trades(reports: list[dict]) -> pd.DataFrame:
    open_lots: dict[str, deque[dict]] = defaultdict(deque)
    paired: list[dict] = []
    for report in reports:
        symbol = str(report["symbol"])
        side = int(report.get("side") or 0)
        effect = int(report.get("position_effect") or 0)
        volume = float(report.get("volume") or 0)
        amount = float(report.get("amount") or 0)
        price = float(report.get("price") or 0)
        commission = float(report.get("commission") or 0)
        if side == 1 and effect == 1:
            open_lots[symbol].append(
                {
                    "symbol": symbol,
                    "stock_code": stock_code_from_symbol(symbol),
                    "buy_date": str(report["trade_date"]),
                    "buy_time": report.get("created_at"),
                    "buy_price": price,
                    "buy_volume": volume,
                    "buy_amount_exec": amount,
                    "buy_commission": commission,
                }
            )
        elif side == 2 and effect == 4:
            remaining = volume
            while remaining > 0 and open_lots[symbol]:
                lot = open_lots[symbol][0]
                lot_volume = float(lot["buy_volume"])
                matched = min(remaining, lot_volume)
                ratio = matched / lot_volume if lot_volume else 0.0
                buy_amount = float(lot["buy_amount_exec"]) * ratio
                buy_commission = float(lot["buy_commission"]) * ratio
                sell_amount = amount * matched / volume if volume else 0.0
                sell_commission = commission * matched / volume if volume else 0.0
                pnl = sell_amount - sell_commission - buy_amount - buy_commission
                paired.append(
                    {
                        **lot,
                        "sell_date": str(report["trade_date"]),
                        "sell_time": report.get("created_at"),
                        "sell_price": price,
                        "sell_volume": matched,
                        "sell_amount": sell_amount,
                        "sell_commission": sell_commission,
                        "trade_pnl": pnl,
                        "trade_return": pnl / (buy_amount + buy_commission) if buy_amount + buy_commission else None,
                    }
                )
                remaining -= matched
                remain_ratio = (lot_volume - matched) / lot_volume if lot_volume else 0.0
                lot["buy_volume"] = lot_volume - matched
                lot["buy_amount_exec"] = float(lot["buy_amount_exec"]) * remain_ratio
                lot["buy_commission"] = float(lot["buy_commission"]) * remain_ratio
                if lot["buy_volume"] <= 0:
                    open_lots[symbol].popleft()
    return pd.DataFrame(paired)


def summarize(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    g = (
        df.groupby(feature, dropna=False)
        .agg(
            trades=("trade_return", "count"),
            avg_return=("trade_return", "mean"),
            std_return=("trade_return", "std"),
            median_return=("trade_return", "median"),
            win_rate=("trade_return", lambda x: float((x > 0).mean())),
            total_pnl=("trade_pnl", "sum"),
            avg_target=("target_pct", "mean"),
            avg_buy_gap=("buy_open_gap_raw_pct", "mean"),
            avg_up_ratio=("up_ratio", "mean"),
            avg_score=("score_pct_rank", "mean"),
        )
        .reset_index()
        .rename(columns={feature: "bucket"})
    )
    g["feature"] = feature
    g["return_to_vol"] = g["avg_return"] / g["std_return"]
    return g[
        [
            "feature",
            "bucket",
            "trades",
            "avg_return",
            "std_return",
            "return_to_vol",
            "median_return",
            "win_rate",
            "total_pnl",
            "avg_target",
            "avg_buy_gap",
            "avg_up_ratio",
            "avg_score",
        ]
    ]


def main() -> int:
    trades = pair_trades(load_exec_reports())
    sig = pd.read_csv(SIGNAL_FILE, dtype={"buy_date": str, "stock_code": str})
    keep_cols = [
        "buy_date",
        "stock_code",
        "name",
        "market",
        "rank",
        "pred_prob",
        "score_pct_rank",
        "signal_pct_chg",
        "signal_open_gap_raw_pct",
        "buy_open_gap_raw_pct",
        "signal_amount",
        "buy_amount",
        "signal_turnover_rate",
        "buy_turnover_rate",
        "signal_total_mv",
        "buy_total_mv",
        "signal_atr_qfq",
        "buy_atr_qfq",
        "target_pct",
        "up_ratio",
    ]
    merged = trades.merge(sig[keep_cols], on=["buy_date", "stock_code"], how="left")
    numeric_cols = [
        "trade_return",
        "trade_pnl",
        "buy_open_gap_raw_pct",
        "signal_open_gap_raw_pct",
        "signal_pct_chg",
        "score_pct_rank",
        "signal_amount",
        "signal_total_mv",
        "signal_atr_qfq",
        "target_pct",
        "up_ratio",
    ]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["buy_month"] = merged["buy_date"].astype(str).str[:6]
    merged["buy_year"] = merged["buy_date"].astype(str).str[:4]
    merged["buy_gap_bin"] = pd.cut(merged["buy_open_gap_raw_pct"], [-3.01, -2.5, -2.0, -1.5, -1.0, -0.5], include_lowest=True).astype(str)
    merged["up_ratio_bin"] = pd.cut(merged["up_ratio"], [0, 0.25, 0.4, 0.6, 0.8, 1.0], include_lowest=True).astype(str)
    merged["target_bin"] = pd.cut(merged["target_pct"], [0, 0.2, 0.4, 0.6, 0.75, 0.82, 1.0], include_lowest=True).astype(str)
    merged["rank_bin"] = merged["rank"].astype("Int64").astype(str)
    merged["sigchg_bin"] = pd.cut(merged["signal_pct_chg"], [-30, -8, -5, -2, 0, 2, 5, 8, 30], include_lowest=True).astype(str)
    merged["amount_bin"] = pd.cut(merged["signal_amount"], [0, 50000, 90000, 150000, 300000, 800000, 10000000], include_lowest=True).astype(str)
    merged["mv_bin_signal"] = pd.cut(merged["signal_total_mv"], [0, 200000, 300000, 500000, 800000, 1500000, 10000000], include_lowest=True).astype(str)
    merged.to_csv(TRADES_OUT, index=False, encoding="utf-8-sig")

    features = [
        "buy_year",
        "buy_month",
        "stock_code",
        "market",
        "rank_bin",
        "buy_gap_bin",
        "up_ratio_bin",
        "target_bin",
        "sigchg_bin",
        "amount_bin",
        "mv_bin_signal",
    ]
    summary = pd.concat([summarize(merged, feature) for feature in features], ignore_index=True)
    summary.to_csv(SEGMENT_OUT, index=False, encoding="utf-8-sig")

    worst = summary.sort_values(["return_to_vol", "total_pnl"], ascending=[True, True]).head(30)
    best = summary.sort_values(["return_to_vol", "total_pnl"], ascending=[False, False]).head(30)
    worst_pnl = summary.sort_values("total_pnl", ascending=True).head(30)
    report = [
        "# top2_s150_cap82_w62 波动贡献拆解",
        "",
        "## 汇总",
        "",
        f"- 配对交易数：{len(merged)}",
        f"- 总 PnL：{merged['trade_pnl'].sum():.2f}",
        f"- 平均单笔收益率：{merged['trade_return'].mean():.4%}",
        f"- 单笔收益标准差：{merged['trade_return'].std():.4%}",
        f"- 胜率：{(merged['trade_return'] > 0).mean():.2%}",
        "",
        "## return_to_vol 最弱分桶",
        "",
        "```csv",
        worst.to_csv(index=False, lineterminator="\n").strip(),
        "```",
        "",
        "## PnL 最弱分桶",
        "",
        "```csv",
        worst_pnl.to_csv(index=False, lineterminator="\n").strip(),
        "```",
        "",
        "## return_to_vol 最强分桶",
        "",
        "```csv",
        best.to_csv(index=False, lineterminator="\n").strip(),
        "```",
        "",
        "## 证据路径",
        "",
        f"- `{TRADES_OUT.as_posix()}`",
        f"- `{SEGMENT_OUT.as_posix()}`",
        f"- `{ARTIFACT_DIR.as_posix()}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT_OUT)
    print(worst_pnl.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
