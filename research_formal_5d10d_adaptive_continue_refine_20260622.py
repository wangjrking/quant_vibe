from __future__ import annotations

import csv
import importlib
import math
from pathlib import Path


base = importlib.import_module("research_formal_5d10d_noncalendar_quality_refine_20260621")

ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_adaptive_continue_refine_20260622"
)


RANK5_PROD = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK5_HIGH = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}
RANK5_FRONT = {1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.15}


def v(name: str, **kwargs: object) -> dict:
    item = base._best_variant(name, **kwargs)
    item["continue_rule"] = kwargs.get("continue_rule", {})
    return item


VARIANTS = [
    v(
        "ac_prod_h5_mh10_rank",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_PROD,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_prod_h6_mh12_rank",
        holding_days=6,
        max_holding_days=12,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_PROD,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_prod_h7_mh12_rank",
        holding_days=7,
        max_holding_days=12,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_PROD,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_prod_h5_mh10_strength",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_PROD,
        continue_rule={"mode": "strength", "very_strong": 0.94, "strong": 0.98, "normal": 1.04, "weak": 1.12},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_prod_h6_mh12_strength",
        holding_days=6,
        max_holding_days=12,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_PROD,
        continue_rule={"mode": "strength", "very_strong": 0.94, "strong": 0.98, "normal": 1.04, "weak": 1.12},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_high_h5_mh10_rank",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_HIGH,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_high_h6_mh12_rank",
        holding_days=6,
        max_holding_days=12,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_HIGH,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_high_h7_mh12_rank",
        holding_days=7,
        max_holding_days=12,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_HIGH,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_high_h5_mh10_strength",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_HIGH,
        continue_rule={"mode": "strength", "very_strong": 0.94, "strong": 0.98, "normal": 1.04, "weak": 1.12},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_high_h6_mh12_strength",
        holding_days=6,
        max_holding_days=12,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_HIGH,
        continue_rule={"mode": "strength", "very_strong": 0.94, "strong": 0.98, "normal": 1.04, "weak": 1.12},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_front_h5_mh10_rank",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.25,
        rank_target_pct=RANK5_FRONT,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.02, "tail": 1.10},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_front_h6_mh12_rank",
        holding_days=6,
        max_holding_days=12,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.25,
        rank_target_pct=RANK5_FRONT,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.02, "tail": 1.10},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20"},
    ),
    v(
        "ac_prod_h5_mh10_rank_no_stop",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_PROD,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20", "GM_STOP_LOSS_PCT": "none"},
    ),
    v(
        "ac_high_h5_mh10_rank_no_stop",
        holding_days=5,
        max_holding_days=10,
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_HIGH,
        continue_rule={"mode": "rank", "top": 0.96, "mid": 1.01, "tail": 1.08},
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.20", "GM_STOP_LOSS_PCT": "none"},
    ),
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _strength_tag(feature: dict) -> str:
    avg_pred = float(feature.get("avg_pred") or 0.0)
    primary_count = int(feature.get("primary_count") or 0)
    count = int(feature.get("count") or 0)
    if avg_pred >= 2.40 and primary_count >= 4 and count >= 4:
        return "very_strong"
    if avg_pred >= 2.25 and primary_count >= 3 and count >= 4:
        return "strong"
    if avg_pred < 2.08 or primary_count < 2:
        return "weak"
    return "normal"


def _continue_ratio(rank: int, feature: dict, rule: dict) -> float:
    mode = str(rule.get("mode") or "rank")
    if mode == "strength":
        tag = _strength_tag(feature)
        return float(rule.get(tag, rule.get("normal", 1.0)))
    if rank <= 2:
        return float(rule.get("top", 1.0))
    if rank <= 4:
        return float(rule.get("mid", 1.0))
    return float(rule.get("tail", 1.0))


def _write_variant_signal(variant: dict, signal_file: Path) -> None:
    rows = base._load_base_rows()
    signal_features = base._signal_features(rows)
    fusion_features = base._load_fusion_features()
    fieldnames = list(rows[0].keys()) if rows else []
    for col in ["score_continue_entry_ratio", "continue_rule_tag", "day_avg_pred", "day_primary_count"]:
        if col not in fieldnames:
            fieldnames.append(col)

    passed_rows = []
    passed_by_date: dict[str, list[dict]] = {}
    for row in rows:
        if not base._passes(row, variant, signal_features, fusion_features):
            continue
        passed_rows.append(row)
        passed_by_date.setdefault(str(row.get("signal_date") or ""), []).append(row)

    day_features = {}
    for signal_date, day_rows in passed_by_date.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in day_rows]
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
            if "post_filter_avg_pred_min" in variant:
                avg_pred = _to_float(feature.get("avg_pred"))
                if avg_pred is None or avg_pred < float(variant["post_filter_avg_pred_min"]):
                    continue
            output = dict(row)
            rank = int(float(output.get("rank") or 0))
            rank_targets = variant.get("rank_target_pct") or {}
            output["target_pct"] = f"{float(rank_targets.get(rank, variant['target_pct'])):.5f}"
            output["holding_days"] = str(int(variant["holding_days"]))
            output["score_exit_entry_ratio"] = str(variant.get("score_exit_entry_ratio", "1.0"))
            output["min_holding_days_before_score_exit"] = str(variant.get("min_holding_days_before_score_exit", "3"))
            output["score_continue_entry_ratio"] = f"{_continue_ratio(rank, feature, variant.get('continue_rule') or {}):.5f}"
            output["continue_rule_tag"] = _strength_tag(feature)
            output["day_avg_pred"] = f"{float(feature.get('avg_pred') or 0.0):.6f}"
            output["day_primary_count"] = str(int(feature.get("primary_count") or 0))
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
            "continue_rule": variant.get("continue_rule"),
            "rank_max": variant.get("rank_max"),
            "post_filter_avg_pred_min": variant.get("post_filter_avg_pred_min"),
            "max_positions": variant.get("max_positions"),
            "holding_days": variant.get("holding_days"),
            "max_holding_days": variant.get("max_holding_days"),
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
