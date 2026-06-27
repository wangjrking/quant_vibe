from __future__ import annotations

import csv
import importlib
import math
import sqlite3
from pathlib import Path


base = importlib.import_module("research_formal_5d10d_noncalendar_quality_refine_20260621")

ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_market_state_scale_20260621"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"


RANK5_21 = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK5_23 = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}
RANK5_25 = {1: 0.25, 2: 0.23, 3: 0.21, 4: 0.19, 5: 0.17}
RANK8_14 = {1: 0.14, 2: 0.135, 3: 0.13, 4: 0.125, 5: 0.12, 6: 0.115, 7: 0.11, 8: 0.105}


def v(name: str, **kwargs: object) -> dict:
    item = base._best_variant(name, **kwargs)
    item["market_state_scale"] = kwargs.get("market_state_scale", {})
    return item


VARIANTS = [
    v(
        "mstate_base_rank5_21_soft",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_21,
        market_state_scale={"risk_scale": 0.75, "strong_scale": 1.10, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "mstate_base_rank5_21_medium",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_21,
        market_state_scale={"risk_scale": 0.60, "strong_scale": 1.15, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "mstate_base_rank5_21_strict",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_21,
        market_state_scale={"risk_scale": 0.45, "strong_scale": 1.20, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "mstate_high_rank5_23_soft",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.27,
        rank_target_pct=RANK5_23,
        market_state_scale={"risk_scale": 0.75, "strong_scale": 1.10, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "mstate_high_rank5_23_medium",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.27,
        rank_target_pct=RANK5_23,
        market_state_scale={"risk_scale": 0.60, "strong_scale": 1.15, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "mstate_high_rank5_25_soft",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.30,
        rank_target_pct=RANK5_25,
        market_state_scale={"risk_scale": 0.75, "strong_scale": 1.10, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "mstate_high_rank5_25_medium",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.30,
        rank_target_pct=RANK5_25,
        market_state_scale={"risk_scale": 0.60, "strong_scale": 1.15, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "mstate_avg_rank8_14_soft",
        rank_max=8,
        max_positions=8,
        post_filter_avg_pred_min=1.95,
        max_target_pct=0.17,
        rank_target_pct=RANK8_14,
        market_state_scale={"risk_scale": 0.80, "strong_scale": 1.12, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "mstate_avg_rank8_14_medium",
        rank_max=8,
        max_positions=8,
        post_filter_avg_pred_min=1.95,
        max_target_pct=0.17,
        rank_target_pct=RANK8_14,
        market_state_scale={"risk_scale": 0.65, "strong_scale": 1.18, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "mstate_base_rank5_filter_risk",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_21,
        market_state_scale={"risk_scale": 0.0, "strong_scale": 1.18, "normal_scale": 1.0},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
]


def _safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _market_state_by_date() -> dict[str, dict[str, float | str]]:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            trade_date,
            MAX(index_2000_close) AS index_close,
            SUM(amount) AS market_amount,
            AVG(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio,
            AVG(pct_chg) AS avg_pct_chg
        FROM STOCK_DAILY_DATA
        WHERE trade_date >= '20240501' AND trade_date <= '20260630'
          AND index_2000_close IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
        """
    ).fetchall()
    conn.close()

    states: dict[str, dict[str, float | str]] = {}
    history: list[dict[str, float | str]] = []
    for row in rows:
        item = {
            "trade_date": str(row["trade_date"]),
            "index_close": float(row["index_close"]),
            "market_amount": float(row["market_amount"] or 0.0),
            "up_ratio": float(row["up_ratio"] or 0.0),
            "avg_pct_chg": float(row["avg_pct_chg"] or 0.0),
        }
        history.append(item)
        idx = len(history) - 1
        close = float(item["index_close"])
        amount = float(item["market_amount"])
        ret5 = close / float(history[idx - 5]["index_close"]) - 1.0 if idx >= 5 else 0.0
        ret20 = close / float(history[idx - 20]["index_close"]) - 1.0 if idx >= 20 else 0.0
        high20 = max(float(x["index_close"]) for x in history[max(0, idx - 20) : idx + 1])
        dd20 = close / high20 - 1.0 if high20 > 0 else 0.0
        amt20 = sum(float(x["market_amount"]) for x in history[max(0, idx - 20) : idx + 1]) / len(
            history[max(0, idx - 20) : idx + 1]
        )
        up5 = sum(float(x["up_ratio"]) for x in history[max(0, idx - 5) : idx + 1]) / len(
            history[max(0, idx - 5) : idx + 1]
        )
        item.update(
            {
                "ret5": ret5,
                "ret20": ret20,
                "dd20": dd20,
                "amount_ratio20": amount / amt20 if amt20 > 0 else 1.0,
                "up_ratio5": up5,
            }
        )
        states[str(item["trade_date"])] = item
    return states


def _scale_for_state(state: dict[str, float | str], rule: dict) -> float:
    risk_scale = float(rule.get("risk_scale", 1.0))
    strong_scale = float(rule.get("strong_scale", 1.0))
    normal_scale = float(rule.get("normal_scale", 1.0))
    dd20 = float(state.get("dd20") or 0.0)
    ret5 = float(state.get("ret5") or 0.0)
    ret20 = float(state.get("ret20") or 0.0)
    up_ratio5 = float(state.get("up_ratio5") or 0.0)
    amount_ratio20 = float(state.get("amount_ratio20") or 1.0)

    if dd20 <= -0.075 or (ret20 <= -0.04 and up_ratio5 <= 0.47) or (ret5 <= -0.035 and up_ratio5 <= 0.45):
        return risk_scale
    if ret20 >= 0.035 and ret5 >= 0.005 and up_ratio5 >= 0.52 and amount_ratio20 >= 0.90:
        return strong_scale
    return normal_scale


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    rows = base._load_base_rows()
    signal_features = base._signal_features(rows)
    fusion_features = base._load_fusion_features()
    market_states = _market_state_by_date()
    fieldnames = list(rows[0].keys()) if rows else []
    if "market_scale" not in fieldnames:
        fieldnames.append("market_scale")
    if "market_state_tag" not in fieldnames:
        fieldnames.append("market_state_tag")

    passed_rows = []
    passed_by_date: dict[str, list[dict]] = {}
    for row in rows:
        if not base._passes(row, variant, signal_features, fusion_features):
            continue
        passed_rows.append(row)
        passed_by_date.setdefault(str(row.get("signal_date") or ""), []).append(row)

    day_features = {}
    for signal_date, day_rows in passed_by_date.items():
        preds = [_safe_float(row.get("pred_prob"), 0.0) or 0.0 for row in day_rows]
        day_features[signal_date] = {
            "count": len(day_rows),
            "primary_count": sum(1 for value in preds if value >= 2.0),
            "avg_pred": sum(preds) / len(preds) if preds else None,
        }

    signal_file.parent.mkdir(parents=True, exist_ok=True)
    with signal_file.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in passed_rows:
            signal_date = str(row.get("signal_date") or "")
            feature = day_features.get(signal_date) or {}
            if "post_filter_min_count" in variant and int(feature.get("count") or 0) < int(variant["post_filter_min_count"]):
                continue
            if "post_filter_min_primary_count" in variant and int(feature.get("primary_count") or 0) < int(variant["post_filter_min_primary_count"]):
                continue
            if "post_filter_avg_pred_min" in variant:
                avg_pred = _safe_float(feature.get("avg_pred"))
                if avg_pred is None or avg_pred < float(variant["post_filter_avg_pred_min"]):
                    continue

            state = market_states.get(signal_date, {})
            scale = _scale_for_state(state, variant.get("market_state_scale") or {})
            if scale <= 0:
                continue
            rank = int(float(row.get("rank") or 0))
            rank_targets = variant.get("rank_target_pct") or {}
            target_pct = float(rank_targets.get(rank, variant["target_pct"])) * scale
            output = dict(row)
            output["target_pct"] = f"{target_pct:.5f}"
            output["holding_days"] = str(int(variant["holding_days"]))
            output["score_exit_entry_ratio"] = str(variant.get("score_exit_entry_ratio", "1.0"))
            output["min_holding_days_before_score_exit"] = str(variant.get("min_holding_days_before_score_exit", "3"))
            output["market_scale"] = f"{scale:.4f}"
            output["market_state_tag"] = (
                f"ret20={float(state.get('ret20') or 0.0):.4f};"
                f"dd20={float(state.get('dd20') or 0.0):.4f};"
                f"up5={float(state.get('up_ratio5') or 0.0):.4f};"
                f"amt20={float(state.get('amount_ratio20') or 1.0):.4f}"
            )
            writer.writerow(output)


def main() -> int:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            _write_variant_signal(variant, signal_file)
        returncode = base._run_backtest(variant, signal_file, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "market_state_scale": variant.get("market_state_scale"),
            "rank_max": variant.get("rank_max"),
            "max_positions": variant.get("max_positions"),
            "holding_days": variant.get("holding_days"),
            "max_holding_days": variant.get("max_holding_days", variant.get("holding_days")),
            "max_target_pct": variant.get("max_target_pct", variant.get("target_pct")),
            "rank_target_pct": variant.get("rank_target_pct"),
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
