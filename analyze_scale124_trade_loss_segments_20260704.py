from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
ARTIFACT_DIR = OUT_DIR / "scale124_base_artifacts"
SIGNAL_FILE = REPORT_DIR / "variants_refill_best_position_scale_fine/scale124_cap82.csv"
TRADES_OUT = OUT_DIR / "scale124_paired_trades_with_features.csv"
SEGMENT_OUT = OUT_DIR / "scale124_loss_segment_summary.csv"
REPORT_OUT = OUT_DIR / "scale124_loss_segment_report_20260704.md"


def stock_code_from_symbol(symbol: str) -> str:
    if symbol.startswith("SHSE."):
        return symbol[5:] + ".SH"
    if symbol.startswith("SZSE."):
        return symbol[5:] + ".SZ"
    return symbol


def load_exec_reports() -> list[dict]:
    path = ARTIFACT_DIR / "execution_reports.jsonl"
    rows: list[dict] = []
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
    for r in reports:
        symbol = str(r["symbol"])
        side = int(r.get("side") or 0)
        position_effect = int(r.get("position_effect") or 0)
        volume = float(r.get("volume") or 0)
        amount = float(r.get("amount") or 0)
        price = float(r.get("price") or 0)
        commission = float(r.get("commission") or 0)
        if side == 1 and position_effect == 1:
            open_lots[symbol].append(
                {
                    "symbol": symbol,
                    "stock_code": stock_code_from_symbol(symbol),
                    "buy_date": str(r["trade_date"]),
                    "buy_time": r.get("created_at"),
                    "buy_price": price,
                    "buy_volume": volume,
                    "buy_amount": amount,
                    "buy_commission": commission,
                }
            )
        elif side == 2 and position_effect == 4:
            remaining = volume
            while remaining > 0 and open_lots[symbol]:
                lot = open_lots[symbol][0]
                matched = min(remaining, float(lot["buy_volume"]))
                ratio = matched / float(lot["buy_volume"]) if lot["buy_volume"] else 0.0
                buy_amount = float(lot["buy_amount"]) * ratio
                buy_commission = float(lot["buy_commission"]) * ratio
                sell_amount = amount * matched / volume if volume else 0.0
                sell_commission = commission * matched / volume if volume else 0.0
                pnl = sell_amount - sell_commission - buy_amount - buy_commission
                ret = pnl / (buy_amount + buy_commission) if (buy_amount + buy_commission) else None
                paired.append(
                    {
                        **lot,
                        "sell_date": str(r["trade_date"]),
                        "sell_time": r.get("created_at"),
                        "sell_price": price,
                        "sell_volume": matched,
                        "sell_amount": sell_amount,
                        "sell_commission": sell_commission,
                        "trade_pnl": pnl,
                        "trade_return": ret,
                    }
                )
                remaining -= matched
                lot["buy_volume"] = float(lot["buy_volume"]) - matched
                lot["buy_amount"] = float(lot["buy_amount"]) * (lot["buy_volume"] / (lot["buy_volume"] + matched)) if (lot["buy_volume"] + matched) else 0
                lot["buy_commission"] = float(lot["buy_commission"]) * (lot["buy_volume"] / (lot["buy_volume"] + matched)) if (lot["buy_volume"] + matched) else 0
                if lot["buy_volume"] <= 0:
                    open_lots[symbol].popleft()
    return pd.DataFrame(paired)


def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["buy_year"] = out["buy_date"].astype(str).str[:4]
    out["gap_bin"] = pd.cut(
        out["buy_open_gap_raw_pct"],
        [-10, -3, -2.5, -2, -1.5, -1, -0.5, 0, 1],
        include_lowest=True,
    ).astype(str)
    out["sigchg_bin"] = pd.cut(
        out["signal_pct_chg"],
        [-30, -8, -5, -2, 0, 2, 5, 8, 30],
        include_lowest=True,
    ).astype(str)
    out["score_bin"] = pd.cut(
        out["score_pct_rank"],
        [0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
        include_lowest=True,
    ).astype(str)
    out["amount_bin"] = pd.cut(
        out["buy_amount_signal"],
        [0, 50000, 80000, 120000, 200000, 500000, 1_000_000, 10_000_000],
        include_lowest=True,
    ).astype(str)
    out["mv_bin"] = pd.cut(
        out["buy_total_mv"],
        [0, 200000, 300000, 500000, 800000, 1_500000, 10_000000],
        include_lowest=True,
    ).astype(str)
    out["atr_bin"] = pd.cut(
        out["buy_atr_qfq"],
        [0, 0.5, 1, 1.5, 2, 4, 20],
        include_lowest=True,
    ).astype(str)
    return out


def summarize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = (
        df.groupby(col, dropna=False)
        .agg(
            trades=("trade_return", "count"),
            avg_return=("trade_return", "mean"),
            median_return=("trade_return", "median"),
            win_rate=("trade_return", lambda x: float((x > 0).mean())),
            total_pnl=("trade_pnl", "sum"),
            avg_gap=("buy_open_gap_raw_pct", "mean"),
            avg_sigchg=("signal_pct_chg", "mean"),
            avg_score=("score_pct_rank", "mean"),
        )
        .reset_index()
    )
    g["feature"] = col
    g = g.rename(columns={col: "bucket"})
    return g[["feature", "bucket", "trades", "avg_return", "median_return", "win_rate", "total_pnl", "avg_gap", "avg_sigchg", "avg_score"]]


def frame_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def main() -> int:
    trades = pair_trades(load_exec_reports())
    sig = pd.read_csv(SIGNAL_FILE, dtype={"buy_date": str, "stock_code": str})
    sig = sig.rename(columns={"buy_amount": "buy_amount_signal"})
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
        "buy_amount_signal",
        "signal_turnover_rate",
        "buy_turnover_rate",
        "signal_total_mv",
        "buy_total_mv",
        "signal_atr_qfq",
        "buy_atr_qfq",
        "target_pct",
    ]
    merged = trades.merge(sig[keep_cols], on=["buy_date", "stock_code"], how="left")
    for col in [
        "trade_return",
        "trade_pnl",
        "buy_open_gap_raw_pct",
        "signal_pct_chg",
        "score_pct_rank",
        "buy_amount_signal",
        "buy_total_mv",
        "buy_atr_qfq",
        "target_pct",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = bin_features(merged)
    merged.to_csv(TRADES_OUT, index=False, encoding="utf-8-sig")

    features = ["buy_year", "market", "gap_bin", "sigchg_bin", "score_bin", "amount_bin", "mv_bin", "atr_bin"]
    summary = pd.concat([summarize(merged, f) for f in features], ignore_index=True)
    summary = summary.sort_values(["total_pnl", "avg_return"], ascending=[True, True])
    summary.to_csv(SEGMENT_OUT, index=False, encoding="utf-8-sig")

    worst = summary.head(20)
    top_losses = merged.sort_values("trade_pnl").head(20)
    report = [
        "# scale124 成交亏损分解",
        "",
        "生成时间：2026-07-04",
        "",
        f"配对成交数：{len(merged)}",
        f"总 PnL：{merged['trade_pnl'].sum():.2f}",
        f"平均单笔收益率：{merged['trade_return'].mean():.4%}",
        f"胜率：{(merged['trade_return'] > 0).mean():.2%}",
        "",
        "## 亏损最集中的字段桶",
        "",
        "```csv",
        frame_text(worst).strip(),
        "```",
        "",
        "## 单笔最大亏损",
        "",
        "```csv",
        frame_text(top_losses[
            [
                "buy_date",
                "sell_date",
                "stock_code",
                "name",
                "market",
                "trade_return",
                "trade_pnl",
                "buy_open_gap_raw_pct",
                "signal_pct_chg",
                "score_pct_rank",
                "buy_amount_signal",
                "buy_total_mv",
                "buy_atr_qfq",
            ]
        ]).strip(),
        "```",
        "",
        "## 证据路径",
        "",
        f"- `{TRADES_OUT.relative_to(ROOT)}`",
        f"- `{SEGMENT_OUT.relative_to(ROOT)}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT_OUT)
    print(worst.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
