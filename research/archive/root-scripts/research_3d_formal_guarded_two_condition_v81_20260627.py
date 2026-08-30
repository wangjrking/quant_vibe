from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded
from score_model_promotion_candidate import evaluate_candidate


DATA_DIR = guarded.DATA_DIR
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_formal_guarded_two_condition_v81_20260627"
CONSTRAINTS = guarded.CONSTRAINTS

LABEL = "executable_3d_open_return"
SPEC = {
    "asset": "research_3d_formal_guarded_two_condition_v81_20260627",
    "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "candidate_table": "stock_predict_data_model_agent_3d_exante_std_gate_v69_20260627_executable_3d_open_return_research",
    "target_table": "stock_predict_data_model_agent_3d_formal_guarded_two_condition_v81_20260627_executable_3d_open_return_research",
    "rank_ic_floor": -0.0015,
}

FEATURES = guarded.FEATURES
QUANTILES = [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]
METRICS = guarded.METRICS


def summarize_daily(daily: pd.DataFrame, window: int | None = None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {metric: float(sub[metric].mean()) for metric in METRICS}


def daily_switch_eval(
    pair: pd.DataFrame,
    formal_daily: pd.DataFrame,
    active_dates: set[str],
) -> dict[str, object]:
    active = pair["trade_date"].isin(active_dates)
    daily = pd.DataFrame({"trade_date": pair["trade_date"]})
    for metric in METRICS:
        daily[metric] = np.where(active, pair[f"{metric}_candidate"], pair[f"{metric}_formal"])

    full = summarize_daily(daily)
    recent63 = summarize_daily(daily, 63)
    recent20 = summarize_daily(daily, 20)
    formal_full = summarize_daily(formal_daily)
    formal63 = summarize_daily(formal_daily, 63)
    formal20 = summarize_daily(formal_daily, 20)
    delta_full = guarded.deltas(full, formal_full)
    delta63 = guarded.deltas(recent63, formal63)
    delta20 = guarded.deltas(recent20, formal20)
    month = guarded.month_top5_delta(daily, formal_daily)
    month.pop("monthly")
    return {
        "daily": daily,
        "full": full,
        "recent63": recent63,
        "recent20": recent20,
        "delta_full": delta_full,
        "delta_recent63": delta63,
        "delta_recent20": delta20,
        "month": month,
        "objective": guarded.objective(delta_full, delta63, delta20, month),
    }


def condition_catalog(features: pd.DataFrame) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    for feature in FEATURES:
        values = features[feature].replace([np.inf, -np.inf], np.nan).dropna()
        thresholds = sorted(set(float(values.quantile(q)) for q in QUANTILES))
        for threshold in thresholds:
            for op in ["<=", ">="]:
                if op == "<=":
                    active_dates = set(features.loc[features[feature] <= threshold, "trade_date"])
                else:
                    active_dates = set(features.loc[features[feature] >= threshold, "trade_date"])
                if 3 <= len(active_dates) <= 260:
                    conditions.append(
                        {
                            "condition": f"{feature} {op} {threshold:.12g}",
                            "feature": feature,
                            "op": op,
                            "threshold": threshold,
                            "active_dates": active_dates,
                        }
                    )
    return conditions


def row_from_result(condition: str, active_dates: set[str], result: dict[str, object]) -> dict[str, object]:
    delta_full = result["delta_full"]
    delta63 = result["delta_recent63"]
    delta20 = result["delta_recent20"]
    month = result["month"]
    return {
        "condition": condition,
        "active_days": len(active_dates),
        "objective": result["objective"],
        "full_rank_ic_delta": delta_full["rank_ic"],
        "full_top1_delta": delta_full["top1"],
        "full_top5_delta": delta_full["top5"],
        "recent63_rank_ic_delta": delta63["rank_ic"],
        "recent63_top1_delta": delta63["top1"],
        "recent63_top5_delta": delta63["top5"],
        "recent20_rank_ic_delta": delta20["rank_ic"],
        "recent20_top1_delta": delta20["top1"],
        "recent20_top5_delta": delta20["top5"],
        "positive_top5_months": month["positive_top5_months"],
        "nonnegative_top5_months": month["nonnegative_top5_months"],
        "min_month_top5_delta": month["min_month_top5_delta"],
    }


def pass_hard(row: dict[str, object]) -> bool:
    return bool(
        row["full_top5_delta"] >= 0
        and row["recent63_top5_delta"] >= 0
        and row["recent20_top5_delta"] >= 0
        and row["positive_top5_months"] >= 3
        and row["min_month_top5_delta"] >= 0
        and row["full_rank_ic_delta"] >= SPEC["rank_ic_floor"]
        and row["recent63_rank_ic_delta"] >= 0
    )


def scan_variants(features: pd.DataFrame, pair: pd.DataFrame, formal_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    conditions = condition_catalog(features)
    rows: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []

    for item in conditions:
        result = daily_switch_eval(pair, formal_daily, item["active_dates"])
        row = row_from_result(str(item["condition"]), item["active_dates"], result)
        row["kind"] = "single"
        row["pass_hard"] = pass_hard(row)
        rows.append(row)
        evaluated.append({"row": row, "result": result, "active_dates": item["active_dates"]})

    seed = sorted(evaluated, key=lambda item: (item["row"]["pass_hard"], item["row"]["objective"]), reverse=True)[:48]
    for i, left in enumerate(seed):
        for right in seed[i + 1 :]:
            for joiner in ["AND", "OR"]:
                if joiner == "AND":
                    active_dates = set(left["active_dates"]) & set(right["active_dates"])
                else:
                    active_dates = set(left["active_dates"]) | set(right["active_dates"])
                if len(active_dates) < 3 or len(active_dates) > 260:
                    continue
                condition = f"({left['row']['condition']}) {joiner} ({right['row']['condition']})"
                result = daily_switch_eval(pair, formal_daily, active_dates)
                row = row_from_result(condition, active_dates, result)
                row["kind"] = "two_condition"
                row["pass_hard"] = pass_hard(row)
                rows.append(row)
                evaluated.append({"row": row, "result": result, "active_dates": active_dates})

    frame = pd.DataFrame(rows).sort_values(
        ["pass_hard", "objective", "recent63_top5_delta", "recent20_top5_delta", "full_top5_delta"],
        ascending=[False, False, False, False, False],
    )
    best_row = frame.iloc[0].to_dict()
    best = next(item for item in evaluated if item["row"]["condition"] == best_row["condition"])
    return frame, best


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_latest, expected_rows = guarded.production_factor_latest()
    scores = guarded.load_scores(SPEC)
    labels = guarded.load_labels(LABEL, str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    formal_daily = guarded.daily_eval(eval_frame, "formal_rank", LABEL)
    candidate_daily = guarded.daily_eval(eval_frame, "candidate_rank", LABEL)
    pair = formal_daily.merge(
        candidate_daily,
        on="trade_date",
        suffixes=("_formal", "_candidate"),
        validate="one_to_one",
    )
    features = guarded.daily_features(scores)
    scan_frame, best = scan_variants(features, pair, formal_daily)

    scan_frame.to_csv(REPORT_DIR / "scan_results.csv", index=False, encoding="utf-8-sig")
    features.to_csv(REPORT_DIR / "score_state_features.csv", index=False, encoding="utf-8-sig")
    best["result"]["daily"].to_csv(REPORT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")

    summary = guarded.write_asset(LABEL, SPEC, scores, best, REPORT_DIR)
    candidate = guarded.make_candidate_payload(LABEL, SPEC, best, summary, expected_latest, expected_rows, REPORT_DIR)
    candidate["prefer_simpler_formula"] = "formal fallback + two score-state gates"
    candidate["evidence"]["scan_summary"] = str(REPORT_DIR / "v81_summary.json")
    candidate_path = REPORT_DIR / "promotion_candidate.json"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = evaluate_candidate(candidate, json.loads(CONSTRAINTS.read_text(encoding="utf-8")))
    (REPORT_DIR / "promotion_gate_result.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_3d_formal_guarded_two_condition_v81",
        "asset": SPEC["asset"],
        "formal_table": SPEC["formal_table"],
        "candidate_table": SPEC["candidate_table"],
        "target_table": SPEC["target_table"],
        "best": best["row"],
        "db_summary": summary,
        "promotion_gate": gate,
        "candidate_json": str(candidate_path),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "v81_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "label": LABEL,
                "asset": SPEC["asset"],
                "table": SPEC["target_table"],
                "condition": best["row"]["condition"],
                "kind": best["row"]["kind"],
                "promotion_gate_passed": gate["hard_constraint_passed"],
                "target_approval_status": gate["target_approval_status"],
                "failed_hard_constraints": ";".join(gate["failed_hard_constraints"]),
                "latest_trade_date": summary["latest_trade_date"],
                "latest_day_rows": summary["latest_day_rows"],
                **{
                    key: best["row"][key]
                    for key in [
                        "full_rank_ic_delta",
                        "full_top1_delta",
                        "full_top5_delta",
                        "recent63_rank_ic_delta",
                        "recent63_top1_delta",
                        "recent63_top5_delta",
                        "recent20_rank_ic_delta",
                        "recent20_top1_delta",
                        "recent20_top5_delta",
                        "positive_top5_months",
                        "min_month_top5_delta",
                    ]
                },
            }
        ]
    ).to_csv(REPORT_DIR / "v81_summary_matrix.csv", index=False, encoding="utf-8-sig")

    report = f"""# 3D 双条件门控研究候选 v81

## 结论

本轮只在 research-only L4 区域扫描 `3D` formal fallback 候选，没有训练模型，没有修改 formal/production manifest，没有生成信号或回测结论。

## 最优候选

- 资产：`{SPEC['asset']}`
- 表：`{SPEC['target_table']}`
- 条件：`{best['row']['condition']}`
- 类型：`{best['row']['kind']}`
- 激活交易日：`{best['row']['active_days']}`
- 最新覆盖：`{summary['latest_trade_date']}`，最新日行数 `{summary['latest_day_rows']}`
- promotion gate：`{gate['hard_constraint_passed']}`，目标状态 `{gate['target_approval_status']}`

## 相对当前 3D formal 的增量

- 全窗口 RankIC delta：`{best['row']['full_rank_ic_delta']:.10f}`
- 全窗口 Top1 delta：`{best['row']['full_top1_delta']:.10f}`
- 全窗口 Top5 delta：`{best['row']['full_top5_delta']:.10f}`
- 近 63 日 RankIC delta：`{best['row']['recent63_rank_ic_delta']:.10f}`
- 近 63 日 Top1 delta：`{best['row']['recent63_top1_delta']:.10f}`
- 近 63 日 Top5 delta：`{best['row']['recent63_top5_delta']:.10f}`
- 近 20 日 Top1 delta：`{best['row']['recent20_top1_delta']:.10f}`
- 近 20 日 Top5 delta：`{best['row']['recent20_top5_delta']:.10f}`

## 边界

- 未训练模型
- 未发布生产资产
- 未修改 L5 任务
- 未生成交易信号
- 未运行策略回测
"""
    (REPORT_DIR / "v81_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
