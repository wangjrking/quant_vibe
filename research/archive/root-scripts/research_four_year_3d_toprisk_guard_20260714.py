from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "model_agent_four_year_3d_toprisk_guard_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_3d_quarterly_feature_tophit_20260714 as base  # noqa: E402
from adjustment_semantics import (  # noqa: E402
    NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS,
    NAKED_MARKET_PRICE_COLUMNS,
)
from model_asset_route import (  # noqa: E402
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
)

START_DATE = "20220606"
TRAIN_END = base.TRAIN_END
LABEL_COL = base.LABEL_COL
LABEL_NAME = base.LABEL_NAME
SCREEN_CSV = DATA / "reports" / "model_agent_four_year_3d_wide_feature_screen_xgb_20260714" / "wide_feature_screen_3d.csv"

CORE_NON_QFQ_FEATURES = {
    "turnover_rate_f",
    "turnover_rate",
    "volume_ratio",
    "amount",
    "vol",
    "total_mv",
    "circ_mv",
    "free_share",
    "float_share",
    "total_share",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "industry_encode",
    "low_close_rate",
    "high_open_rate",
    "high_rate",
}


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _allowed_feature(name: str) -> bool:
    if name in NAKED_MARKET_PRICE_COLUMNS or name in NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS:
        return False
    if name.endswith("_hfq") or name.endswith("_bfq"):
        return False
    if name.endswith("_qfq"):
        return True
    return name in CORE_NON_QFQ_FEATURES


def _feature_candidates(limit: int = 36) -> list[str]:
    screen = pd.read_csv(SCREEN_CSV)
    features: list[str] = []
    for feature in screen["feature"].astype(str).tolist():
        if _allowed_feature(feature) and feature not in features:
            features.append(feature)
        if len(features) >= limit:
            break
    for feature in sorted(CORE_NON_QFQ_FEATURES):
        if feature not in features:
            features.append(feature)
    return features[:limit]


def _load_frame(features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_db = resolve_model_feature_duckdb_path(DATA, require_exists=True)
    feature_table = resolve_model_feature_duckdb_table(DATA)
    if not feature_table:
        raise RuntimeError("active L3 feature DuckDB route is missing")
    baseline = base._formal_3d_frame()
    select_cols = ",\n                ".join(f"f.{_quote(col)} AS {_quote(col)}" for col in features)
    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{feature_db.as_posix()}' AS feat (READ_ONLY)")
        frame = con.execute(
            f"""
            SELECT
                f.trade_date,
                f.stock_code,
                {select_cols}
            FROM feat.{_quote(feature_table)} f
            WHERE f.trade_date >= ?
              AND f.stock_code NOT LIKE '%.BJ'
            ORDER BY f.trade_date, f.stock_code
            """,
            [START_DATE],
        ).fetchdf()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    merged = baseline.merge(frame, on=["trade_date", "stock_code"], how="inner")
    merged["baseline_score"] = merged["pred_3d_rank"]
    for feature in features:
        merged[f"{feature}__rank"] = merged.groupby("trade_date")[feature].rank(method="average", pct=True)
    route_info = {
        "feature_asset": f"{feature_db.as_posix()}::{feature_table}",
        "selected_features": features,
        "rows": int(len(merged)),
        "min_trade_date": str(merged["trade_date"].min()),
        "max_trade_date": str(merged["trade_date"].max()),
        "trade_days": int(merged["trade_date"].nunique()),
        "stocks": int(merged["stock_code"].nunique()),
        "qfq_contract": "Only explicit *_qfq factors and non-price liquidity/valuation/size fields are used; naked front-adjusted price/indicator names are excluded.",
    }
    return merged, route_info


def _risk_mask(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    rank = frame[f"{rule['feature']}__rank"]
    if rule["direction"] == "high":
        return rank >= float(rule["threshold"])
    return rank <= float(rule["threshold"])


def _score_with_rules(frame: pd.DataFrame, rules: list[dict[str, Any]], *, pool: float, beta: float) -> pd.Series:
    front = frame["baseline_score"] >= pool
    risk = pd.Series(False, index=frame.index)
    for rule in rules:
        risk = risk | _risk_mask(frame, rule)
    penalty = front & risk
    return frame["baseline_score"] - beta * penalty.astype(float)


def _rule_training_edge(frame: pd.DataFrame, rule: dict[str, Any], pool: float) -> dict[str, Any]:
    train = frame[(frame["trade_date"] <= TRAIN_END) & (frame["baseline_score"] >= pool)].copy()
    mask = _risk_mask(train, rule)
    risk_mean = float(train.loc[mask, LABEL_COL].mean()) if int(mask.sum()) else None
    safe_mean = float(train.loc[~mask, LABEL_COL].mean()) if int((~mask).sum()) else None
    return {
        **rule,
        "pool": pool,
        "risk_count": int(mask.sum()),
        "safe_count": int((~mask).sum()),
        "risk_label_mean": risk_mean,
        "safe_label_mean": safe_mean,
        "risk_minus_safe": None if risk_mean is None or safe_mean is None else risk_mean - safe_mean,
    }


def _scan_rules(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        for pool in [0.95, 0.97, 0.98]:
            for direction, thresholds in [("high", [0.80, 0.90, 0.95]), ("low", [0.20, 0.10, 0.05])]:
                for threshold in thresholds:
                    rows.append(_rule_training_edge(frame, {"feature": feature, "direction": direction, "threshold": threshold}, pool))
    scan = pd.DataFrame(rows)
    scan = scan[scan["risk_count"] >= 200]
    scan = scan.sort_values(["risk_minus_safe", "risk_count"], ascending=[True, False])
    return scan


def _eval_candidate(frame: pd.DataFrame, rules: list[dict[str, Any]], pool: float, beta: float) -> dict[str, Any]:
    work = frame.copy()
    work["candidate_score"] = _score_with_rules(work, rules, pool=pool, beta=beta)
    _, baseline_metrics = base._metrics_for_score(work, "baseline_score")
    _, candidate_metrics = base._metrics_for_score(work, "candidate_score")
    deltas = {
        block: base._delta_block(candidate_metrics, baseline_metrics, block)
        for block in ["full", "train", "holdout", "recent63", "recent20", "recent126"]
    }
    passed, failed_reasons = base._gate(candidate_metrics, baseline_metrics)
    selection_score = deltas["train"]["top5"] + 0.5 * deltas["train"]["top10"] + 0.25 * deltas["train"]["top3"]
    return {
        "rules": rules,
        "pool": pool,
        "beta": beta,
        "baseline_current_formal_same_dates": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "gate": {"passed": passed, "failed_reasons": failed_reasons},
        "selection_score_train_only": selection_score,
    }


def _register_result(payload: dict[str, Any], report_json: Path, report_md: Path, scan_csv: Path, rule_csv: Path) -> None:
    status_path = DATA / "reports" / "model_agent_current_research_candidate_status_20260714" / "current_research_candidate_status_20260714.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.setdefault("not_promoted_research_lines", {})["toprisk_guard_3d"] = {
        "label": LABEL_NAME,
        "decision": payload["decision"],
        "selected_rules": payload["selected_by_train"]["rules"],
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
            "rule_csv": str(rule_csv),
            "script": str(Path(__file__).resolve()),
        },
        "boundary": "research_only; no formal/L5/production/signal/backtest change",
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features = _feature_candidates()
    frame, route_info = _load_frame(features)
    rule_scan = _scan_rules(frame, features)
    rule_csv = REPORT_DIR / "toprisk_guard_3d_rule_screen.csv"
    rule_scan.to_csv(rule_csv, index=False, encoding="utf-8-sig")

    top_rules = rule_scan.head(8).to_dict("records")
    rule_sets: list[dict[str, Any]] = [
        {
            "rules": [{k: rule[k] for k in ["feature", "direction", "threshold"]}],
            "pools": [float(rule["pool"])],
        }
        for rule in top_rules
    ]

    candidates: list[tuple[float, dict[str, Any]]] = []
    scan_rows: list[dict[str, Any]] = []
    for rule_set in rule_sets:
        rules = rule_set["rules"]
        for pool in rule_set["pools"]:
            for beta in [0.0025, 0.005, 0.01]:
                item = _eval_candidate(frame, rules, pool, beta)
                scan_rows.append(
                    {
                        "rules": json.dumps(rules, ensure_ascii=False),
                        "pool": pool,
                        "beta": beta,
                        "passed": item["gate"]["passed"],
                        "failed_reasons": ";".join(item["gate"]["failed_reasons"]),
                        "selection_score_train_only": item["selection_score_train_only"],
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
                candidates.append((float(item["selection_score_train_only"]), item))

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "toprisk_guard_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    passing = [payload for _, payload in candidates if payload["gate"]["passed"]]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_toprisk_guard",
        "method": "train-window selected front-pool risk guard applied to current formal 3D rank score",
        "selection_rule": "Rules are selected only from train-window front-pool risk-minus-safe label edge through 20251231 and train-window TopN deltas. Holdout/recent windows are validation only.",
        "route_info": route_info,
        "selected_by_train": selected,
        "passing_count": len(passing),
        "best_passing_by_train_rank": passing[0] if passing else None,
        "rule_screen_csv": str(rule_csv),
        "scan_csv": str(scan_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else ("candidate_exists_but_not_train_selected" if passing else "no_3d_candidate_passed_toprisk_guard_scan")
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
    report_json = REPORT_DIR / "toprisk_guard_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D 前排风险过滤研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- 选中规则：`{sel['rules']}`",
        f"- pool：`{sel['pool']}`",
        f"- beta：`{sel['beta']}`",
        f"- 是否过门：`{sel['gate']['passed']}`",
        f"- 失败原因：`{'; '.join(sel['gate']['failed_reasons']) if sel['gate']['failed_reasons'] else '-'}`",
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
    report_md = REPORT_DIR / "toprisk_guard_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _register_result(payload, report_json, report_md, scan_csv, rule_csv)
    print(json.dumps({"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
