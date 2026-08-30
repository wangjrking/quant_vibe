from __future__ import annotations

import importlib
from pathlib import Path


base = importlib.import_module("research_formal_5d10d_noncalendar_quality_refine_20260621")

ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_execution_micro_refine_20260621"
)


RANK5_PROD = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK5_HIGH = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


PROD_ARGS = {
    "post_filter_avg_pred_min": 2.05,
    "max_target_pct": 0.21,
    "rank_target_pct": RANK5_PROD,
    "extra_env": {"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
}

HIGH_ARGS = {
    "post_filter_avg_pred_min": 2.00,
    "max_target_pct": 0.23,
    "rank_target_pct": RANK5_HIGH,
    "extra_env": {"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
}


def merge_env(base_env: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    out = dict(base_env)
    out.update(override)
    return out


VARIANTS = []
for prefix, args in [("prod", PROD_ARGS), ("high", HIGH_ARGS)]:
    base_env = args["extra_env"]
    common = {k: v for k, v in args.items() if k != "extra_env"}
    VARIANTS.extend(
        [
            v(f"{prefix}_exec_baseline", **common, extra_env=base_env),
            v(f"{prefix}_exec_cash999", **common, extra_env=merge_env(base_env, {"GM_CASH_BUFFER": "0.999"})),
            v(f"{prefix}_exec_cash100", **common, extra_env=merge_env(base_env, {"GM_CASH_BUFFER": "1.0"})),
            v(
                f"{prefix}_exec_buy_market",
                **common,
                extra_env=merge_env(base_env, {"GM_FORCE_BUY_MARKET_ORDER": "1"}),
            ),
            v(
                f"{prefix}_exec_sell_market",
                **common,
                extra_env=merge_env(base_env, {"GM_FORCE_SELL_MARKET_ORDER": "1"}),
            ),
            v(
                f"{prefix}_exec_all_market",
                **common,
                extra_env=merge_env(base_env, {"GM_FORCE_MARKET_ORDER": "1"}),
            ),
            v(
                f"{prefix}_exec_sync_positions",
                **common,
                extra_env=merge_env(base_env, {"GM_SYNC_POSITIONS": "1"}),
            ),
            v(
                f"{prefix}_exec_stop06_replace",
                **common,
                extra_env=merge_env(
                    base_env,
                    {
                        "GM_INTRADAY_RISK_MODE": "1",
                        "GM_STOP_LOSS_PCT": "0.06",
                        "GM_TAKE_PROFIT_PCT": "none",
                        "GM_INTRADAY_REPLACE_BUY": "1",
                        "GM_INTRADAY_REPLACE_MAX_BUYS": "2",
                    },
                ),
            ),
            v(
                f"{prefix}_exec_stop06_tp12_replace",
                **common,
                extra_env=merge_env(
                    base_env,
                    {
                        "GM_INTRADAY_RISK_MODE": "1",
                        "GM_STOP_LOSS_PCT": "0.06",
                        "GM_TAKE_PROFIT_PCT": "0.12",
                        "GM_INTRADAY_REPLACE_BUY": "1",
                        "GM_INTRADAY_REPLACE_MAX_BUYS": "2",
                    },
                ),
            ),
        ]
    )


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            base._write_variant_signal(variant, signal_file)
        returncode = base._run_backtest(variant, signal_file, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "post_filter_avg_pred_min": variant.get("post_filter_avg_pred_min"),
            "max_positions": variant.get("max_positions"),
            "holding_days": variant.get("holding_days"),
            "max_holding_days": variant.get("max_holding_days", variant.get("holding_days")),
            "max_target_pct": variant.get("max_target_pct", variant.get("target_pct")),
            "rank_target_pct": variant.get("rank_target_pct"),
            "extra_env": variant.get("extra_env"),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(base._signal_stats(signal_file))
        row.update(base._exposure_stats(log_file))
        results.append(row)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')} signals={row.get('signal_count')}"
        )
    base._write_rows(REPORT_DIR / "summary.csv", results)
    base._write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=base._sort_by_annual, reverse=True))
    base._write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=base._sort_by_sharpe, reverse=True))
    target = [
        row
        for row in results
        if (row.get("annual") is not None and float(row["annual"]) >= 3.0)
        and (row.get("sharpe") is not None and float(row["sharpe"]) >= 4.0)
        and (row.get("avg_invested_pct") is not None and float(row["avg_invested_pct"]) >= 0.80)
    ]
    base._write_rows(REPORT_DIR / "summary_target_hits.csv", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
