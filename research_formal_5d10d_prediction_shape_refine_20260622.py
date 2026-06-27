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
    / "formal_5d10d_prediction_shape_refine_20260622"
)


RANK5_21 = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK5_23 = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}
RANK5_25 = {1: 0.25, 2: 0.23, 3: 0.21, 4: 0.19, 5: 0.17}
RANK6_18 = {1: 0.18, 2: 0.18, 3: 0.17, 4: 0.16, 5: 0.16, 6: 0.15}
RANK8_13 = {1: 0.13, 2: 0.13, 3: 0.13, 4: 0.125, 5: 0.12, 6: 0.115, 7: 0.11, 8: 0.105}

NO_DD = {"GM_EQUITY_DD_RISK_MODE": "0", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}
PROD_DD = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.90",
    "GM_EQUITY_DD_HARD_SCALE": "0.70",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
}


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


VARIANTS = [
    # 10D 预测值过高在分桶诊断中表现偏弱；先用上限而不是月份/行业规则做验证。
    v("shape_p10_le070_rank5_base", max_pred_10d=0.070, post_filter_avg_pred_min=2.00, max_target_pct=0.23, rank_target_pct=RANK5_23),
    v("shape_p10_le070_rank5_prod", max_pred_10d=0.070, post_filter_avg_pred_min=2.05, max_target_pct=0.21, rank_target_pct=RANK5_21, extra_env=PROD_DD),
    v("shape_p10_le045_rank5_base", max_pred_10d=0.045, post_filter_avg_pred_min=1.95, max_target_pct=0.23, rank_target_pct=RANK5_23),
    v("shape_p10_le035_rank5_base", max_pred_10d=0.035, post_filter_avg_pred_min=1.90, max_target_pct=0.23, rank_target_pct=RANK5_23),
    v("shape_p10_mid_rank5_base", min_pred_10d=0.0095, max_pred_10d=0.070, post_filter_avg_pred_min=1.95, max_target_pct=0.23, rank_target_pct=RANK5_23),
    v("shape_p10_mid_rank5_h10", min_pred_10d=0.0095, max_pred_10d=0.070, post_filter_avg_pred_min=1.95, holding_days=10, max_holding_days=15, max_target_pct=0.23, rank_target_pct=RANK5_23, extra_env=NO_DD),
    # 5D 中段在诊断中更好，验证它与 10D 上限的组合是否能提高收益质量。
    v("shape_p5_mid_p10_le070_rank5", min_pred_5d=0.0057, max_pred_5d=0.0345, max_pred_10d=0.070, post_filter_avg_pred_min=1.90, max_target_pct=0.25, rank_target_pct=RANK5_25),
    v("shape_p5_mid_p10_le070_rank5_h10", min_pred_5d=0.0057, max_pred_5d=0.0345, max_pred_10d=0.070, post_filter_avg_pred_min=1.90, holding_days=10, max_holding_days=15, max_target_pct=0.25, rank_target_pct=RANK5_25, extra_env=NO_DD),
    v("shape_p5_mid_p10_mid_rank5", min_pred_5d=0.0057, max_pred_5d=0.0345, min_pred_10d=0.0095, max_pred_10d=0.070, post_filter_avg_pred_min=1.90, max_target_pct=0.25, rank_target_pct=RANK5_25),
    v("shape_p5_mid_p10_mid_rank5_h10", min_pred_5d=0.0057, max_pred_5d=0.0345, min_pred_10d=0.0095, max_pred_10d=0.070, post_filter_avg_pred_min=1.90, holding_days=10, max_holding_days=15, max_target_pct=0.25, rank_target_pct=RANK5_25, extra_env=NO_DD),
    v("shape_p5_mid_rank5_h10", min_pred_5d=0.0057, max_pred_5d=0.0345, post_filter_avg_pred_min=1.90, holding_days=10, max_holding_days=15, max_target_pct=0.25, rank_target_pct=RANK5_25, extra_env=NO_DD),
    v("shape_p5_mid_rank5_prod", min_pred_5d=0.0057, max_pred_5d=0.0345, post_filter_avg_pred_min=2.00, max_target_pct=0.23, rank_target_pct=RANK5_23, extra_env=PROD_DD),
    # 尝试把平均持仓推近 80%，但只允许 6/8 票内的高质量补位，不做强制市价满仓。
    v("shape_p10_le070_rank6_18", rank_max=6, max_positions=6, max_pred_10d=0.070, post_filter_avg_pred_min=1.95, max_target_pct=0.18, rank_target_pct=RANK6_18),
    v("shape_p10_mid_rank6_18", rank_max=6, max_positions=6, min_pred_10d=0.0095, max_pred_10d=0.070, post_filter_avg_pred_min=1.95, max_target_pct=0.18, rank_target_pct=RANK6_18),
    v("shape_p5_mid_p10_le070_rank6_18", rank_max=6, max_positions=6, min_pred_5d=0.0057, max_pred_5d=0.0345, max_pred_10d=0.070, post_filter_avg_pred_min=1.85, max_target_pct=0.18, rank_target_pct=RANK6_18),
    v("shape_p5_mid_p10_mid_rank6_18", rank_max=6, max_positions=6, min_pred_5d=0.0057, max_pred_5d=0.0345, min_pred_10d=0.0095, max_pred_10d=0.070, post_filter_avg_pred_min=1.85, max_target_pct=0.18, rank_target_pct=RANK6_18),
    v("shape_p10_le070_rank8_13", rank_max=8, max_positions=8, max_pred_10d=0.070, post_filter_avg_pred_min=1.90, max_target_pct=0.13, rank_target_pct=RANK8_13),
    v("shape_p5_mid_p10_le070_rank8_13", rank_max=8, max_positions=8, min_pred_5d=0.0057, max_pred_5d=0.0345, max_pred_10d=0.070, post_filter_avg_pred_min=1.80, max_target_pct=0.13, rank_target_pct=RANK8_13),
]


def _passes_extra(row: dict, variant: dict, signal_features: dict, fusion_features: dict) -> bool:
    if not base._passes(row, variant, signal_features, fusion_features):
        return False
    key = (str(row.get("signal_date") or ""), str(row.get("stock_code") or ""))
    stock_features = fusion_features.get(key) or {}
    extra_checks = [
        ("max_pred_5d", "pred_5d", lambda actual, threshold: actual <= threshold),
        ("max_pred_10d", "pred_10d", lambda actual, threshold: actual <= threshold),
    ]
    for variant_key, feature_key, predicate in extra_checks:
        if variant_key not in variant:
            continue
        actual = stock_features.get(feature_key)
        if actual is None or not predicate(float(actual), float(variant[variant_key])):
            return False
    return True


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    rows = base._load_base_rows()
    signal_features = base._signal_features(rows)
    fusion_features = base._load_fusion_features()
    fieldnames = list(rows[0].keys()) if rows else []
    passed_rows = []
    passed_by_date: dict[str, list[dict]] = {}
    for row in rows:
        if not _passes_extra(row, variant, signal_features, fusion_features):
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
            if "post_filter_min_count" in variant and int(day_feature.get("count") or 0) < int(variant["post_filter_min_count"]):
                continue
            if "post_filter_min_primary_count" in variant and int(day_feature.get("primary_count") or 0) < int(variant["post_filter_min_primary_count"]):
                continue
            if "post_filter_avg_pred_min" in variant:
                avg_pred = base._to_float(day_feature.get("avg_pred"))
                if avg_pred is None or avg_pred < float(variant["post_filter_avg_pred_min"]):
                    continue
            output = dict(row)
            rank = int(float(output.get("rank") or 0))
            rank_targets = variant.get("rank_target_pct") or {}
            output["target_pct"] = f"{float(rank_targets.get(rank, variant['target_pct'])):.5f}"
            output["holding_days"] = str(int(variant["holding_days"]))
            output["score_exit_entry_ratio"] = str(variant.get("score_exit_entry_ratio", "1.0"))
            output["min_holding_days_before_score_exit"] = str(variant.get("min_holding_days_before_score_exit", "3"))
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
            "min_pred_5d": variant.get("min_pred_5d"),
            "max_pred_5d": variant.get("max_pred_5d"),
            "min_pred_10d": variant.get("min_pred_10d"),
            "max_pred_10d": variant.get("max_pred_10d"),
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
