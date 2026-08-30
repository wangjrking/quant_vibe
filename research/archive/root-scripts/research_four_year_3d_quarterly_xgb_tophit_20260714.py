from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "model_agent_four_year_3d_quarterly_xgb_tophit_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_3d_quarterly_feature_tophit_20260714 as base  # noqa: E402

TRAIN_END = base.TRAIN_END
MAX_TRAIN_ROWS = 240_000
MAX_POS_ROWS = 60_000


def _sample_training_rows(frame: pd.DataFrame, top_pct: float, neg_ratio: int) -> pd.DataFrame:
    y = base._target(frame, top_pct)
    positives = frame[y == 1]
    negatives = frame[y == 0]
    if positives.empty or negatives.empty:
        return frame.iloc[0:0].copy()
    if len(positives) > MAX_POS_ROWS:
        positives = positives.sample(n=MAX_POS_ROWS, random_state=42)
    n_neg = min(len(negatives), len(positives) * neg_ratio, max(0, MAX_TRAIN_ROWS - len(positives)))
    sampled_neg = negatives.sample(n=n_neg, random_state=42)
    sampled = pd.concat([positives, sampled_neg], ignore_index=True)
    return sampled.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _fit_predict_quarterly_xgb(
    frame: pd.DataFrame,
    *,
    top_pct: float,
    neg_ratio: int,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    subsample: float,
    colsample_bytree: float,
) -> pd.DataFrame:
    dates = list(frame["trade_date"].drop_duplicates())
    eval_start = dates[base.INITIAL_TRAIN_DAYS]
    eval_frame = frame[frame["trade_date"] >= eval_start].copy()
    quarters = list(eval_frame["quarter"].drop_duplicates())
    rows: list[pd.DataFrame] = []

    for quarter in quarters:
        test = eval_frame[eval_frame["quarter"] == quarter].copy()
        train = frame[frame["trade_date"] < str(test["trade_date"].min())].copy()
        train_sample = _sample_training_rows(train, top_pct, neg_ratio)
        if train_sample.empty or len(train_sample) < 1_000:
            continue

        x_train = train_sample[base.LEGACY_3D_FEATURES].apply(pd.to_numeric, errors="coerce")
        medians = x_train.median(numeric_only=True).fillna(0.0)
        x_train = x_train.fillna(medians).to_numpy(dtype=np.float32, copy=True)
        y_train = base._target(train_sample, top_pct)
        pos = max(float(y_train.sum()), 1.0)
        neg = max(float(len(y_train) - y_train.sum()), 1.0)
        scale_pos_weight = neg / pos

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=10,
            reg_alpha=0.0,
            reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight,
            n_jobs=8,
            random_state=42,
        )
        model.fit(x_train, y_train, verbose=False)

        x_test = test[base.LEGACY_3D_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians)
        x_test = x_test.to_numpy(dtype=np.float32, copy=True)
        out = test[["trade_date", "stock_code", base.LABEL_COL]].copy()
        out["xgb_tophit_proba"] = model.predict_proba(x_test)[:, 1]
        out["quarter_model_train_rows"] = int(len(train_sample))
        out["quarter_model_positive_rows"] = int(y_train.sum())
        rows.append(out)

    pred = pd.concat(rows, ignore_index=True)
    pred["xgb_tophit_rank"] = pred.groupby("trade_date")["xgb_tophit_proba"].rank(method="average", pct=True)
    return pred


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features, route_info = base._load_feature_label_frame()
    baseline_frame = base._formal_3d_frame()

    configs = [
        {
            "top_pct": 0.05,
            "neg_ratio": 4,
            "max_depth": 2,
            "learning_rate": 0.03,
            "n_estimators": 160,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },
        {
            "top_pct": 0.05,
            "neg_ratio": 8,
            "max_depth": 2,
            "learning_rate": 0.02,
            "n_estimators": 220,
            "subsample": 0.75,
            "colsample_bytree": 0.85,
        },
    ]
    blend_weights = [0.02, 0.05, 0.10, 0.20, 0.35]
    blend_modes = ["direct", "inverse"]

    scan_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, Any]]] = []

    for config in configs:
        pred = _fit_predict_quarterly_xgb(features, **config)
        merged = baseline_frame.merge(pred, on=["trade_date", "stock_code", base.LABEL_COL], how="inner")
        merged["baseline_score"] = merged["pred_3d_rank"]
        _, baseline_metrics = base._metrics_for_score(merged, "baseline_score")
        for mode in blend_modes:
            feature_rank = merged["xgb_tophit_rank"] if mode == "direct" else 1.0 - merged["xgb_tophit_rank"]
            for weight in blend_weights:
                merged["candidate_score"] = (1.0 - weight) * merged["pred_3d_rank"] + weight * feature_rank
                _, candidate_metrics = base._metrics_for_score(merged, "candidate_score")
                deltas = {
                    "full": base._delta_block(candidate_metrics, baseline_metrics, "full"),
                    "train": base._delta_block(candidate_metrics, baseline_metrics, "train"),
                    "holdout": base._delta_block(candidate_metrics, baseline_metrics, "holdout"),
                    "recent63": base._delta_block(candidate_metrics, baseline_metrics, "recent63"),
                    "recent20": base._delta_block(candidate_metrics, baseline_metrics, "recent20"),
                    "recent126": base._delta_block(candidate_metrics, baseline_metrics, "recent126"),
                }
                passed, failed_reasons = base._gate(candidate_metrics, baseline_metrics)
                selection_score = (
                    deltas["train"]["top5"]
                    + 0.5 * deltas["train"]["top10"]
                    + 0.25 * deltas["train"]["top3"]
                )
                row = {
                    **config,
                    "blend_mode": mode,
                    "blend_weight": weight,
                    "passed": passed,
                    "failed_reasons": ";".join(failed_reasons),
                    "selection_score_train_only": selection_score,
                    "eval_trade_days": candidate_metrics["eval_trade_days"],
                    "eval_min_trade_date": candidate_metrics["eval_min_trade_date"],
                    "eval_max_trade_date": candidate_metrics["eval_max_trade_date"],
                    "full_rank_ic_delta": deltas["full"]["rank_ic"],
                    "full_top1_delta": deltas["full"]["top1"],
                    "full_top3_delta": deltas["full"]["top3"],
                    "full_top5_delta": deltas["full"]["top5"],
                    "full_top10_delta": deltas["full"]["top10"],
                    "full_top20_delta": deltas["full"]["top20"],
                    "holdout_top5_delta": deltas["holdout"]["top5"],
                    "recent63_top5_delta": deltas["recent63"]["top5"],
                    "recent20_top5_delta": deltas["recent20"]["top5"],
                }
                scan_rows.append(row)
                candidates.append(
                    (
                        selection_score,
                        {
                            "label": base.LABEL_NAME,
                            "config": config,
                            "blend_mode": mode,
                            "blend_weight": weight,
                            "baseline_current_formal_same_dates": baseline_metrics,
                            "candidate_metrics": candidate_metrics,
                            "deltas": deltas,
                            "gate": {"passed": passed, "failed_reasons": failed_reasons},
                            "selection_score_train_only": selection_score,
                        },
                    )
                )

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "quarterly_xgb_tophit_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    passing = [payload for _, payload in candidates if payload["gate"]["passed"]]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_quarterly_xgb_tophit",
        "method": "quarterly rolling feature-level XGBClassifier for 3D top-hit probability using active L3 features",
        "selection_rule": "Only train-window deltas through 20251231 select config/blend. Holdout/recent windows are validation only.",
        "route_info": route_info,
        "max_train_rows": MAX_TRAIN_ROWS,
        "max_positive_rows": MAX_POS_ROWS,
        "selected_by_train": selected,
        "passing_count": len(passing),
        "best_passing_by_train_rank": passing[0] if passing else None,
        "scan_csv": str(scan_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else (
                "candidate_exists_but_not_train_selected" if passing else "no_3d_candidate_passed_quarterly_xgb_tophit_scan"
            )
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
    report_json = REPORT_DIR / "quarterly_xgb_tophit_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D quarterly XGBoost top-hit 研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- 训练窗口选择配置：`{sel['config']}`",
        f"- blend_mode：`{sel['blend_mode']}`",
        f"- blend_weight：`{sel['blend_weight']}`",
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
            "## 边界",
            "",
            "- research-only。",
            "- 未修改 formal manifest。",
            "- 未修改 approved_for_l5。",
            "- 未生成交易信号，未跑策略回测。",
        ]
    )
    report_md = REPORT_DIR / "quarterly_xgb_tophit_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
