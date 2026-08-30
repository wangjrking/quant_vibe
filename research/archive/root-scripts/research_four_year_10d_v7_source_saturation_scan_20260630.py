from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import research_10d_exante_score_state_gate_v65_20260627 as gate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"

REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_10d_v7_source_saturation_scan_20260630"
SUMMARY_JSON = REPORT_DIR / "source_saturation_summary.json"
SUMMARY_MD = REPORT_DIR / "source_saturation_summary.md"
SCAN_CSV = REPORT_DIR / "source_saturation_scan.csv"

FEATURE_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_v3_direct_guard_v7_20260630"
    / "v3_direct_features.csv"
)
BASE_DAILY_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_active_recent_blend_gate_v3_20260629"
    / "best_daily_eval.csv"
)
CAND_DAILY_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_lightweight_blend_v2_20260629"
    / "standard_eval"
    / "executable_10d_open_return_four_year_lightweight_blend_v2_20260629_daily_eval.csv"
)
CURRENT_BESTSET_DAILY_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_10d_v3_direct_guard_v7_20260630"
    / "standard_eval"
    / "executable_10d_open_return_four_year_v3_direct_guard_v7_20260630_daily_eval.csv"
)
CONTROL_DAILY_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_control_assets_20260627"
    / "standard_eval"
    / "executable_10d_open_return_four_year_control_daily_eval.csv"
)

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
FEATURE_COLUMNS = [
    "cand_top1_gap",
    "base_score_iqr",
    "base_top5_spread",
    "mean_abs_score_gap",
    "base_score_std",
    "cand_score_std",
    "top20_overlap",
]
QUANTILES = [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame.sort_values("trade_date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    return {metric: float(frame[metric].mean()) for metric in METRICS}


def recent_summary(frame: pd.DataFrame, size: int) -> dict[str, float]:
    return summarize(frame.tail(size).copy())


def subtract_metrics(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {metric: float(candidate[metric] - baseline[metric]) for metric in METRICS}


def monthly_delta(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    cand = candidate.copy()
    base = baseline.copy()
    cand["period"] = cand["trade_date"].str.slice(0, 6)
    base["period"] = base["trade_date"].str.slice(0, 6)
    cand_group = cand.groupby("period", as_index=False)[METRICS].mean()
    base_group = base.groupby("period", as_index=False)[METRICS].mean()
    joined = cand_group.merge(base_group, on="period", suffixes=("_candidate", "_baseline"))
    out = joined[["period"]].copy()
    for metric in METRICS:
        out[metric] = joined[f"{metric}_candidate"] - joined[f"{metric}_baseline"]
    return out


def monthly_stats(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float | int]:
    delta = monthly_delta(candidate, baseline)
    return {
        "month_count": int(len(delta)),
        "month_top5_positive": int((delta["top5"] > 0).sum()),
        "month_top5_nonnegative": int((delta["top5"] >= 0).sum()),
        "month_top5_negative": int((delta["top5"] < 0).sum()),
        "month_min_rank_ic_delta": float(delta["rank_ic"].min()),
        "month_min_top5_delta": float(delta["top5"].min()),
    }


def build_thresholds(features: pd.DataFrame) -> dict[str, list[float]]:
    thresholds: dict[str, list[float]] = {}
    for column in FEATURE_COLUMNS:
        values = sorted(
            {
                round(float(features[column].quantile(q)), 12)
                for q in QUANTILES
            }
        )
        thresholds[column] = values
    return thresholds


def format_condition(combo: tuple[str, ...], values: tuple[float, ...]) -> str:
    return " AND ".join(f"({column} >= {value:.12f})" for column, value in zip(combo, values))


def scan_same_source(
    features: pd.DataFrame,
    base: pd.DataFrame,
    cand: pd.DataFrame,
    current: pd.DataFrame,
    control: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = build_thresholds(features)
    cand_diff = cand[METRICS + ["top50"]].copy() - base[METRICS + ["top50"]].copy()

    current_full = summarize(current)
    current_recent20 = recent_summary(current, 20)
    current_recent63 = recent_summary(current, 63)
    control_full = summarize(control)
    control_recent20 = recent_summary(control, 20)
    control_recent63 = recent_summary(control, 63)

    rows: list[dict[str, Any]] = []
    for size in [1, 2, 3]:
        for combo in itertools.combinations(FEATURE_COLUMNS, size):
            for values in itertools.product(*(thresholds[column] for column in combo)):
                mask = pd.Series(True, index=features.index)
                for column, value in zip(combo, values):
                    mask &= features[column] >= value
                active_days = int(mask.sum())
                if active_days < 5 or active_days > 120:
                    continue

                candidate = gate.make_daily_from_mask(base, cand_diff, mask)
                full = summarize(candidate)
                recent20 = recent_summary(candidate, 20)
                recent63 = recent_summary(candidate, 63)
                delta_vs_current = subtract_metrics(full, current_full)
                delta_recent20_vs_current = subtract_metrics(recent20, current_recent20)
                delta_recent63_vs_current = subtract_metrics(recent63, current_recent63)
                month_vs_current = monthly_stats(candidate, current)
                delta_vs_control = subtract_metrics(full, control_full)
                delta_recent20_vs_control = subtract_metrics(recent20, control_recent20)
                delta_recent63_vs_control = subtract_metrics(recent63, control_recent63)
                month_vs_control = monthly_stats(candidate, control)

                dominated = not (
                    delta_vs_current["rank_ic"] >= 0
                    and delta_vs_current["top5"] >= 0
                    and delta_recent63_vs_current["top5"] >= 0
                    and delta_recent20_vs_current["top5"] >= 0
                    and month_vs_current["month_min_top5_delta"] >= 0
                )

                rows.append(
                    {
                        "combo": "|".join(combo),
                        "condition": format_condition(combo, values),
                        "active_days": active_days,
                        "dominated_vs_v7": dominated,
                        "full_rank_ic_delta_vs_v7": delta_vs_current["rank_ic"],
                        "full_top1_delta_vs_v7": delta_vs_current["top1"],
                        "full_top5_delta_vs_v7": delta_vs_current["top5"],
                        "recent63_top5_delta_vs_v7": delta_recent63_vs_current["top5"],
                        "recent20_top5_delta_vs_v7": delta_recent20_vs_current["top5"],
                        "month_positive_top5_vs_v7": month_vs_current["month_top5_positive"],
                        "month_nonnegative_top5_vs_v7": month_vs_current["month_top5_nonnegative"],
                        "month_min_top5_delta_vs_v7": month_vs_current["month_min_top5_delta"],
                        "full_rank_ic_delta_vs_control": delta_vs_control["rank_ic"],
                        "full_top5_delta_vs_control": delta_vs_control["top5"],
                        "recent63_top5_delta_vs_control": delta_recent63_vs_control["top5"],
                        "recent20_top5_delta_vs_control": delta_recent20_vs_control["top5"],
                        "month_nonnegative_top5_vs_control": month_vs_control["month_top5_nonnegative"],
                        "month_min_top5_delta_vs_control": month_vs_control["month_min_top5_delta"],
                        "score_vs_v7": (
                            delta_vs_current["rank_ic"] * 1000.0
                            + delta_vs_current["top5"] * 1200.0
                            + delta_recent63_vs_current["top5"] * 1400.0
                            + delta_recent20_vs_current["top5"] * 1600.0
                            + month_vs_current["month_top5_positive"] * 0.2
                            - active_days * 0.01
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values("score_vs_v7", ascending=False).reset_index(drop=True)


def render_markdown(summary: dict[str, Any]) -> str:
    best = summary["best_non_dominated_candidate"]
    lines = [
        "# 10D 同源门控饱和扫描",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        f"- 当前 bestset：`{summary['current_bestset_asset']}`",
        f"- 扫描结论：`{summary['decision']}`",
        f"- 扫描条件数：`{summary['scanned_rows']}`",
        f"- 非劣候选数：`{summary['non_dominated_count']}`",
        "",
        "## 最优非劣候选",
        "",
        f"- 条件：`{best['condition']}`",
        f"- 激活天数：`{best['active_days']}`",
        f"- 相对 v7 Full RankIC：`{best['full_rank_ic_delta_vs_v7']:+.6f}`",
        f"- 相对 v7 Full Top5：`{best['full_top5_delta_vs_v7']:+.6f}`",
        f"- 相对 v7 Recent63 Top5：`{best['recent63_top5_delta_vs_v7']:+.6f}`",
        f"- 相对 v7 Recent20 Top5：`{best['recent20_top5_delta_vs_v7']:+.6f}`",
        f"- 相对 v7 月度最差 Top5：`{best['month_min_top5_delta_vs_v7']:+.6f}`",
        f"- 相对 control 月度最差 Top5：`{best['month_min_top5_delta_vs_control']:+.6f}`",
        "",
        "## 判断",
        "",
        "- 这条 `v3 + v2` 同源组合线还能给出少量非劣门控，但提升量级非常小，且没有继续压缩相对 control 的月度负尾。",
        "- 因此当前更合理的后续动作不是继续在同一 source pair 上微调门槛，而是切到新的模型轴或新的候选来源。",
        "",
        "## 边界",
        "",
        "- 本次仅 research-only 扫描",
        "- 未训练模型",
        "- 未修改 formal / production manifest",
        "- 未生成信号",
        "- 未跑回测",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURE_PATH)
    features["trade_date"] = features["trade_date"].astype(str)
    base = load_daily(BASE_DAILY_PATH)
    cand = load_daily(CAND_DAILY_PATH)
    current = load_daily(CURRENT_BESTSET_DAILY_PATH)
    control = load_daily(CONTROL_DAILY_PATH)

    common_dates = sorted(
        set(features["trade_date"])
        & set(base["trade_date"])
        & set(cand["trade_date"])
        & set(current["trade_date"])
        & set(control["trade_date"])
    )
    features = features[features["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    base = base[base["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    cand = cand[cand["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    current = current[current["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)
    control = control[control["trade_date"].isin(common_dates)].sort_values("trade_date").reset_index(drop=True)

    scan = scan_same_source(features, base, cand, current, control)
    scan.to_csv(SCAN_CSV, index=False, encoding="utf-8-sig")

    non_dominated = scan[~scan["dominated_vs_v7"]].copy()
    best_row = non_dominated.iloc[0].to_dict() if not non_dominated.empty else scan.iloc[0].to_dict()
    decision = (
        "same_source_saturated_no_meaningful_tail_improvement"
        if float(best_row["month_min_top5_delta_vs_control"]) <= -0.04461745032422404
        and float(best_row["recent20_top5_delta_vs_v7"]) <= 0.0
        else "same_source_has_followup_space"
    )

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_four_year_10d_v7_same_source_saturation_scan",
        "current_bestset_asset": "research_10d_four_year_v3_direct_guard_v7_20260630",
        "current_bestset_daily": str(CURRENT_BESTSET_DAILY_PATH),
        "base_daily": str(BASE_DAILY_PATH),
        "candidate_daily": str(CAND_DAILY_PATH),
        "scanned_rows": int(len(scan)),
        "non_dominated_count": int(len(non_dominated)),
        "decision": decision,
        "best_non_dominated_candidate": best_row,
        "top_non_dominated_candidates": non_dominated.head(20).to_dict(orient="records"),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
