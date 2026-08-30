from __future__ import annotations

import csv
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
    / "formal_5d10d_tiered_holding_exposure_20260621"
)


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


PROD_DD = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.90",
    "GM_EQUITY_DD_HARD_SCALE": "0.70",
}

NO_DD = {"GM_EQUITY_DD_RISK_MODE": "0"}


VARIANTS = [
    v(
        "tier_prod_fill15_dd",
        rank_max=5,
        max_positions=5,
        max_target_pct=0.27,
        rank_target_pct={1: 0.27, 2: 0.25, 3: 0.18, 4: 0.15, 5: 0.12},
        rank_holding_days={1: 7, 2: 7, 3: 15, 4: 15, 5: 15},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.85, 4: 0.85, 5: 0.85},
        rank_continue_entry_ratio={1: 1.02, 2: 1.02, 3: 0.85, 4: 0.85, 5: 0.85},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 10, 4: 10, 5: 10},
        post_filter_avg_pred_min=2.05,
        max_holding_days=20,
        extra_env=PROD_DD,
    ),
    v(
        "tier_prod_fill20_dd",
        rank_max=5,
        max_positions=5,
        max_target_pct=0.27,
        rank_target_pct={1: 0.27, 2: 0.25, 3: 0.18, 4: 0.15, 5: 0.12},
        rank_holding_days={1: 7, 2: 7, 3: 20, 4: 20, 5: 20},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.80, 4: 0.80, 5: 0.80},
        rank_continue_entry_ratio={1: 1.02, 2: 1.02, 3: 0.80, 4: 0.80, 5: 0.80},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 12, 4: 12, 5: 12},
        post_filter_avg_pred_min=2.05,
        max_holding_days=25,
        extra_env=PROD_DD,
    ),
    v(
        "tier_all_fill15_no_dd",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        max_target_pct=0.27,
        rank_target_pct={1: 0.27, 2: 0.24, 3: 0.18, 4: 0.15, 5: 0.12},
        rank_holding_days={1: 7, 2: 7, 3: 15, 4: 15, 5: 15},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.85, 4: 0.85, 5: 0.85},
        rank_continue_entry_ratio={1: 1.00, 2: 1.00, 3: 0.85, 4: 0.85, 5: 0.85},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 10, 4: 10, 5: 10},
        post_filter_avg_pred_min=1.90,
        max_holding_days=20,
        extra_env=NO_DD,
    ),
    v(
        "tier_all_fill20_no_dd",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        max_target_pct=0.27,
        rank_target_pct={1: 0.27, 2: 0.24, 3: 0.18, 4: 0.15, 5: 0.12},
        rank_holding_days={1: 7, 2: 7, 3: 20, 4: 20, 5: 20},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.80, 4: 0.80, 5: 0.80},
        rank_continue_entry_ratio={1: 1.00, 2: 1.00, 3: 0.80, 4: 0.80, 5: 0.80},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 12, 4: 12, 5: 12},
        post_filter_avg_pred_min=1.90,
        max_holding_days=25,
        extra_env=NO_DD,
    ),
    v(
        "tier_topheavy_fill15_no_dd",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        max_target_pct=0.35,
        rank_target_pct={1: 0.35, 2: 0.25, 3: 0.15, 4: 0.12, 5: 0.10},
        rank_holding_days={1: 7, 2: 7, 3: 15, 4: 15, 5: 15},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.85, 4: 0.85, 5: 0.85},
        rank_continue_entry_ratio={1: 1.00, 2: 1.00, 3: 0.85, 4: 0.85, 5: 0.85},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 10, 4: 10, 5: 10},
        post_filter_avg_pred_min=1.95,
        max_holding_days=20,
        extra_env=NO_DD,
    ),
    v(
        "tier_topheavy_fill20_no_dd",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        max_target_pct=0.35,
        rank_target_pct={1: 0.35, 2: 0.25, 3: 0.15, 4: 0.12, 5: 0.10},
        rank_holding_days={1: 7, 2: 7, 3: 20, 4: 20, 5: 20},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.80, 4: 0.80, 5: 0.80},
        rank_continue_entry_ratio={1: 1.00, 2: 1.00, 3: 0.80, 4: 0.80, 5: 0.80},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 12, 4: 12, 5: 12},
        post_filter_avg_pred_min=1.95,
        max_holding_days=25,
        extra_env=NO_DD,
    ),
    v(
        "tier_top3_fill20_no_dd",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        max_target_pct=0.40,
        rank_target_pct={1: 0.40, 2: 0.34, 3: 0.24},
        rank_holding_days={1: 7, 2: 7, 3: 20},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.80},
        rank_continue_entry_ratio={1: 1.00, 2: 1.00, 3: 0.80},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 12},
        post_filter_avg_pred_min=1.90,
        max_holding_days=25,
        extra_env=NO_DD,
    ),
    v(
        "tier_top3_fill25_no_dd",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        max_target_pct=0.42,
        rank_target_pct={1: 0.42, 2: 0.34, 3: 0.22},
        rank_holding_days={1: 7, 2: 7, 3: 25},
        rank_exit_entry_ratio={1: 1.0, 2: 1.0, 3: 0.75},
        rank_continue_entry_ratio={1: 1.00, 2: 1.00, 3: 0.75},
        rank_min_hold_before_exit={1: 3, 2: 3, 3: 15},
        post_filter_avg_pred_min=1.90,
        max_holding_days=30,
        extra_env=NO_DD,
    ),
]


def _rank_value(mapping: dict[int, object], rank: int, default: object) -> object:
    return mapping.get(rank, default)


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    rows = base._load_base_rows()
    signal_features = base._signal_features(rows)
    fusion_features = base._load_fusion_features()
    fieldnames = list(rows[0].keys()) if rows else []
    for extra_col in [
        "score_exit_entry_ratio",
        "score_continue_entry_ratio",
        "min_holding_days_before_score_exit",
    ]:
        if extra_col not in fieldnames:
            fieldnames.append(extra_col)

    passed_rows = []
    passed_by_date: dict[str, list[dict]] = {}
    for row in rows:
        if not base._passes(row, variant, signal_features, fusion_features):
            continue
        passed_rows.append(row)
        passed_by_date.setdefault(str(row.get("signal_date") or ""), []).append(row)

    post_filter_day_features = {}
    for signal_date, day_rows in passed_by_date.items():
        preds = [base._to_float(row.get("pred_prob"), 0.0) or 0.0 for row in day_rows]
        post_filter_day_features[signal_date] = {
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
            day_feature = post_filter_day_features.get(signal_date) or {}
            if "post_filter_avg_pred_min" in variant:
                avg_pred = base._to_float(day_feature.get("avg_pred"))
                if avg_pred is None or avg_pred < float(variant["post_filter_avg_pred_min"]):
                    continue
            output = dict(row)
            rank = int(float(output.get("rank") or 0))
            rank_targets = variant.get("rank_target_pct") or {}
            output["target_pct"] = f"{float(_rank_value(rank_targets, rank, variant['target_pct'])):.5f}"
            output["holding_days"] = str(int(_rank_value(variant.get("rank_holding_days") or {}, rank, variant["holding_days"])))
            output["score_exit_entry_ratio"] = str(
                _rank_value(variant.get("rank_exit_entry_ratio") or {}, rank, variant.get("score_exit_entry_ratio", 1.0))
            )
            output["score_continue_entry_ratio"] = str(
                _rank_value(variant.get("rank_continue_entry_ratio") or {}, rank, 1.0)
            )
            output["min_holding_days_before_score_exit"] = str(
                int(_rank_value(variant.get("rank_min_hold_before_exit") or {}, rank, 3))
            )
            writer.writerow(output)


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
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
            "rank_max": variant.get("rank_max"),
            "primary_count_min": variant.get("primary_count_min"),
            "post_filter_avg_pred_min": variant.get("post_filter_avg_pred_min"),
            "max_positions": variant.get("max_positions"),
            "holding_days": variant.get("holding_days"),
            "max_holding_days": variant.get("max_holding_days", variant.get("holding_days")),
            "max_target_pct": variant.get("max_target_pct", variant.get("target_pct")),
            "rank_target_pct": variant.get("rank_target_pct"),
            "rank_holding_days": variant.get("rank_holding_days"),
            "rank_exit_entry_ratio": variant.get("rank_exit_entry_ratio"),
            "rank_continue_entry_ratio": variant.get("rank_continue_entry_ratio"),
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
