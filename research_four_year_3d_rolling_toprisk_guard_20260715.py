from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "model_agent_four_year_3d_rolling_toprisk_guard_20260715"

sys.path.insert(0, str(MAIN))
import research_four_year_3d_quarterly_feature_tophit_20260714 as base  # noqa: E402
import research_four_year_3d_toprisk_guard_20260714 as guard  # noqa: E402

LABEL_NAME = base.LABEL_NAME
LABEL_COL = base.LABEL_COL
TRAIN_LOOKBACK_DAYS = 252


def _candidate_rules(features: list[str]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for feature in features:
        for direction, thresholds in [("high", [0.90, 0.95]), ("low", [0.10, 0.05])]:
            for threshold in thresholds:
                rules.append({"feature": feature, "direction": direction, "threshold": threshold})
    return rules


def _risk_mask(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    rank = frame[f"{rule['feature']}__rank"]
    if rule["direction"] == "high":
        return rank >= float(rule["threshold"])
    return rank <= float(rule["threshold"])


def _select_rule_for_quarter(
    train: pd.DataFrame,
    rules: list[dict[str, Any]],
    *,
    pool: float,
) -> dict[str, Any] | None:
    front = train[train["baseline_score"] >= pool].copy()
    if len(front) < 2_000:
        return None
    rows: list[dict[str, Any]] = []
    for rule in rules:
        mask = _risk_mask(front, rule)
        risk_count = int(mask.sum())
        safe_count = int((~mask).sum())
        if risk_count < 80 or safe_count < 80:
            continue
        risk_mean = float(front.loc[mask, LABEL_COL].mean())
        safe_mean = float(front.loc[~mask, LABEL_COL].mean())
        rows.append(
            {
                **rule,
                "risk_count": risk_count,
                "safe_count": safe_count,
                "risk_label_mean": risk_mean,
                "safe_label_mean": safe_mean,
                "risk_minus_safe": risk_mean - safe_mean,
            }
        )
    if not rows:
        return None
    scan = pd.DataFrame(rows)
    scan = scan.sort_values(["risk_minus_safe", "risk_count"], ascending=[True, False])
    best = scan.iloc[0].to_dict()
    return {k: best[k] for k in ["feature", "direction", "threshold", "risk_count", "safe_count", "risk_label_mean", "safe_label_mean", "risk_minus_safe"]}


def _rolling_score(
    frame: pd.DataFrame,
    rules: list[dict[str, Any]],
    *,
    pool: float,
    beta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = list(frame["trade_date"].drop_duplicates())
    date_index = {date: i for i, date in enumerate(dates)}
    rows: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for quarter, test in frame.groupby("quarter", sort=True):
        first_date = str(test["trade_date"].min())
        first_idx = date_index[first_date]
        if first_idx < 40:
            out = test[["trade_date", "stock_code", LABEL_COL, "baseline_score"]].copy()
            out["candidate_score"] = out["baseline_score"]
            out["risk_guard_flag"] = False
            rows.append(out)
            selections.append({"quarter": quarter, "pool": pool, "beta": beta, "selected": False, "reason": "insufficient_prior_history"})
            continue
        train_start_idx = max(0, first_idx - TRAIN_LOOKBACK_DAYS)
        train_dates = set(dates[train_start_idx:first_idx])
        train = frame[frame["trade_date"].isin(train_dates)].copy()
        selected_rule = _select_rule_for_quarter(train, rules, pool=pool)
        out = test[["trade_date", "stock_code", LABEL_COL, "baseline_score"]].copy()
        if selected_rule is None:
            out["candidate_score"] = out["baseline_score"]
            out["risk_guard_flag"] = False
            selections.append({"quarter": quarter, "pool": pool, "beta": beta, "selected": False})
        else:
            mask = _risk_mask(test, selected_rule) & (test["baseline_score"] >= pool)
            out["candidate_score"] = out["baseline_score"] - beta * mask.astype(float)
            out["risk_guard_flag"] = mask.to_numpy()
            selections.append(
                {
                    "quarter": quarter,
                    "pool": pool,
                    "beta": beta,
                    "selected": True,
                    **selected_rule,
                    "test_rows": int(len(test)),
                    "test_guarded_rows": int(mask.sum()),
                    "train_min_trade_date": str(train["trade_date"].min()),
                    "train_max_trade_date": str(train["trade_date"].max()),
                    "train_trade_days": int(train["trade_date"].nunique()),
                }
            )
        rows.append(out)
    scored = pd.concat(rows, ignore_index=True).sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    return scored, pd.DataFrame(selections)


def _evaluate(scored: pd.DataFrame, pool: float, beta: float, selections: pd.DataFrame) -> dict[str, Any]:
    _, baseline_metrics = base._metrics_for_score(scored, "baseline_score")
    _, candidate_metrics = base._metrics_for_score(scored, "candidate_score")
    deltas = {
        block: base._delta_block(candidate_metrics, baseline_metrics, block)
        for block in ["full", "train", "holdout", "recent63", "recent20", "recent126"]
    }
    passed, failed_reasons = base._gate(candidate_metrics, baseline_metrics)
    selection_score = deltas["train"]["top5"] + 0.5 * deltas["train"]["top10"] + 0.25 * deltas["train"]["top3"]
    return {
        "pool": pool,
        "beta": beta,
        "baseline_current_formal_same_dates": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "gate": {"passed": passed, "failed_reasons": failed_reasons},
        "selection_score_train_only": selection_score,
        "selection_quarters": int(len(selections)),
        "selected_rule_quarters": int(selections.get("selected", pd.Series(dtype=bool)).sum()) if len(selections) else 0,
        "guarded_rows": int(selections.get("test_guarded_rows", pd.Series(dtype=float)).fillna(0).sum()) if len(selections) else 0,
    }


def _register_result(payload: dict[str, Any], report_json: Path, report_md: Path, scan_csv: Path, selection_csv: Path) -> None:
    status_path = DATA / "reports" / "model_agent_current_research_candidate_status_20260714" / "current_research_candidate_status_20260714.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.setdefault("not_promoted_research_lines", {})["rolling_toprisk_guard_3d"] = {
        "label": LABEL_NAME,
        "decision": payload["decision"],
        "pool": payload["selected_by_train"]["pool"],
        "beta": payload["selected_by_train"]["beta"],
        "passing_count": payload["passing_count"],
        "failed_reasons": payload["selected_by_train"]["gate"]["failed_reasons"],
        "key_deltas": {
            "full_rank_ic_delta": payload["selected_by_train"]["deltas"]["full"]["rank_ic"],
            "full_top1_delta": payload["selected_by_train"]["deltas"]["full"]["top1"],
            "full_top3_delta": payload["selected_by_train"]["deltas"]["full"]["top3"],
            "full_top5_delta": payload["selected_by_train"]["deltas"]["full"]["top5"],
            "holdout_top5_delta": payload["selected_by_train"]["deltas"]["holdout"]["top5"],
            "recent63_top5_delta": payload["selected_by_train"]["deltas"]["recent63"]["top5"],
            "recent20_top5_delta": payload["selected_by_train"]["deltas"]["recent20"]["top5"],
        },
        "evidence": {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "scan_csv": str(scan_csv),
            "selection_csv": str(selection_csv),
            "script": str(Path(__file__).resolve()),
        },
        "boundary": "research_only; no formal/L5/production/signal/backtest change",
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features = guard._feature_candidates(limit=36)
    frame, route_info = guard._load_frame(features)
    frame["quarter"] = pd.PeriodIndex(pd.to_datetime(frame["trade_date"]), freq="Q").astype(str)
    rules = _candidate_rules(features)

    candidates: list[tuple[float, dict[str, Any], pd.DataFrame]] = []
    scan_rows: list[dict[str, Any]] = []
    all_selections: list[pd.DataFrame] = []
    for pool in [0.95, 0.97, 0.98]:
        for beta in [0.0025, 0.005, 0.01]:
            scored, selections = _rolling_score(frame, rules, pool=pool, beta=beta)
            item = _evaluate(scored, pool, beta, selections)
            scan_rows.append(
                {
                    "pool": pool,
                    "beta": beta,
                    "passed": item["gate"]["passed"],
                    "failed_reasons": ";".join(item["gate"]["failed_reasons"]),
                    "selection_score_train_only": item["selection_score_train_only"],
                    "selection_quarters": item["selection_quarters"],
                    "selected_rule_quarters": item["selected_rule_quarters"],
                    "guarded_rows": item["guarded_rows"],
                    "full_rank_ic_delta": item["deltas"]["full"]["rank_ic"],
                    "full_top1_delta": item["deltas"]["full"]["top1"],
                    "full_top3_delta": item["deltas"]["full"]["top3"],
                    "full_top5_delta": item["deltas"]["full"]["top5"],
                    "full_top10_delta": item["deltas"]["full"]["top10"],
                    "full_top20_delta": item["deltas"]["full"]["top20"],
                    "holdout_top5_delta": item["deltas"]["holdout"]["top5"],
                    "recent63_top5_delta": item["deltas"]["recent63"]["top5"],
                    "recent20_top5_delta": item["deltas"]["recent20"]["top5"],
                }
            )
            selections = selections.copy()
            selections["pool_scan"] = pool
            selections["beta_scan"] = beta
            all_selections.append(selections)
            candidates.append((float(item["selection_score_train_only"]), item, selections))

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "rolling_toprisk_guard_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    selection_csv = REPORT_DIR / "rolling_toprisk_guard_3d_selections.csv"
    pd.concat(all_selections, ignore_index=True).to_csv(selection_csv, index=False, encoding="utf-8-sig")

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    passing = [payload for _, payload, _ in candidates if payload["gate"]["passed"]]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_rolling_toprisk_guard",
        "method": "quarterly rolling front-pool risk guard selected from previous one-year mature-label window",
        "selection_rule": "Each quarter selects one risk rule only from prior mature-label history. Config pool/beta is selected by train-window deltas through 20251231; holdout/recent windows are validation only.",
        "route_info": route_info,
        "train_lookback_days": TRAIN_LOOKBACK_DAYS,
        "candidate_rule_count": len(rules),
        "selected_by_train": selected,
        "passing_count": len(passing),
        "best_passing_by_train_rank": passing[0] if passing else None,
        "scan_csv": str(scan_csv),
        "selection_csv": str(selection_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else ("candidate_exists_but_not_train_selected" if passing else "no_3d_candidate_passed_rolling_toprisk_guard_scan")
        ),
        "boundaries": {
            "research_only": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "rolling_toprisk_guard_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D 滚动季度前排风险过滤研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- pool：`{sel['pool']}`",
        f"- beta：`{sel['beta']}`",
        f"- 是否过门：`{sel['gate']['passed']}`",
        f"- 失败原因：`{'; '.join(sel['gate']['failed_reasons']) if sel['gate']['failed_reasons'] else '-'}`",
        f"- 选规则季度数：`{sel['selected_rule_quarters']}`",
        f"- 被过滤行数：`{sel['guarded_rows']}`",
        "",
        "## 关键增量",
        "",
        "| 指标 | 增量 |",
        "|---|---:|",
    ]
    for block in ["full", "holdout", "recent63", "recent20"]:
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
            lines.append(f"| {block}.{metric} | {sel['deltas'][block][metric]:.8f} |")
    lines.extend(
        [
            "",
            "## 前复权输入契约治理",
            "",
            "- 本实验只使用显式 `_qfq` 因子与非价格类流动性、估值、市值字段。",
            "- 排除裸价格、裸前复权技术字段，以及 `hfq/bfq` 字段。",
            "- 未要求 L3 回退字段命名，未使用 legacy 宽表。",
            "",
            "## 边界",
            "",
            "- research-only。",
            "- 未修改 formal manifest。",
            "- 未修改 approved_for_l5。",
            "- 未生成交易信号，未跑策略回测。",
        ]
    )
    report_md = REPORT_DIR / "rolling_toprisk_guard_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _register_result(payload, report_json, report_md, scan_csv, selection_csv)
    print(json.dumps({"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
