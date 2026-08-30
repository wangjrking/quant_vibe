from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_3d_online_tophit_classifier_20260714"
)

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402

LABEL_COL = "label_3d"
TRAIN_END = "20251231"
INITIAL_TRAIN_DAYS = 20
FEATURE_COLS = [
    "pred_1d_rank",
    "pred_3d_rank",
    "pred_5d_rank",
    "pred_10d_rank",
    "consensus_5d10d",
    "consensus_all",
    "r3_minus_consensus_5d10d",
    "r3_x_r5",
    "r3_x_r10",
    "r5_x_r10",
    "max_rank",
    "min_rank",
]


def _prepare_frame() -> pd.DataFrame:
    frame = rb._add_rank_columns(rb._read_scores(with_labels=True))
    frame = frame[frame[LABEL_COL].notna()].copy()
    frame["consensus_5d10d"] = (frame["pred_5d_rank"] + frame["pred_10d_rank"]) / 2.0
    frame["consensus_all"] = (
        frame["pred_1d_rank"] + frame["pred_3d_rank"] + frame["pred_5d_rank"] + frame["pred_10d_rank"]
    ) / 4.0
    frame["r3_minus_consensus_5d10d"] = frame["pred_3d_rank"] - frame["consensus_5d10d"]
    frame["r3_x_r5"] = frame["pred_3d_rank"] * frame["pred_5d_rank"]
    frame["r3_x_r10"] = frame["pred_3d_rank"] * frame["pred_10d_rank"]
    frame["r5_x_r10"] = frame["pred_5d_rank"] * frame["pred_10d_rank"]
    rank_cols = ["pred_1d_rank", "pred_3d_rank", "pred_5d_rank", "pred_10d_rank"]
    frame["max_rank"] = frame[rank_cols].max(axis=1)
    frame["min_rank"] = frame[rank_cols].min(axis=1)
    frame["label_rank"] = frame.groupby("trade_date")[LABEL_COL].rank(method="average", pct=True)
    frame["month"] = frame["trade_date"].str.slice(0, 6)
    return frame.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _target(frame: pd.DataFrame, top_pct: float) -> np.ndarray:
    return (frame["label_rank"].to_numpy(dtype=np.float32) >= (1.0 - top_pct)).astype(np.int8)


def _sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(np.int8), minlength=2).astype(np.float32)
    weights = np.ones_like(y, dtype=np.float32)
    total = float(len(y))
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y == cls] = total / (2.0 * counts[cls])
    return weights


def _x(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)


def _online_proba(frame: pd.DataFrame, top_pct: float, alpha: float, eta0: float, epochs: int) -> pd.DataFrame:
    dates = list(frame["trade_date"].drop_duplicates())
    initial_dates = set(dates[:INITIAL_TRAIN_DAYS])
    initial = frame[frame["trade_date"].isin(initial_dates)].copy()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="constant",
        eta0=eta0,
        max_iter=1,
        tol=None,
        random_state=42,
        average=True,
    )
    classes = np.array([0, 1], dtype=np.int8)
    for _ in range(max(1, epochs)):
        y_initial = _target(initial, top_pct)
        model.partial_fit(_x(initial), y_initial, classes=classes, sample_weight=_sample_weight(y_initial))

    rows: list[pd.DataFrame] = []
    remaining = frame[~frame["trade_date"].isin(initial_dates)].copy()
    for _, month_frame in remaining.groupby("month", sort=True):
        proba = model.predict_proba(_x(month_frame))[:, 1]
        out = month_frame[["trade_date", "stock_code", LABEL_COL, "pred_3d_rank"]].copy()
        out["tophit_proba"] = proba
        rows.append(out)
        for _ in range(max(1, epochs)):
            y_month = _target(month_frame, top_pct)
            model.partial_fit(_x(month_frame), y_month, classes=classes, sample_weight=_sample_weight(y_month))

    pred = pd.concat(rows, ignore_index=True)
    pred["tophit_rank"] = pred.groupby("trade_date")["tophit_proba"].rank(method="average", pct=True)
    return pred


def _metrics_for_score(frame: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = rb._daily_eval(frame, score_col, LABEL_COL)
    metrics = {
        "full": rb._metrics(daily),
        "train": rb._metrics(daily[daily["trade_date"] <= TRAIN_END]),
        "holdout": rb._metrics(daily[daily["trade_date"] > TRAIN_END]),
        "recent20": rb._metrics(daily.tail(20)),
        "recent63": rb._metrics(daily.tail(63)),
        "recent126": rb._metrics(daily.tail(126)),
        "annual_stability": rb._period_stability(daily, "year"),
        "monthly_stability": rb._period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
    }
    return daily, metrics


def _delta_block(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float]:
    return {
        key: float((candidate[block].get(key) or 0.0) - (baseline[block].get(key) or 0.0))
        for key in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    }


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = _delta_block(candidate, baseline, "full")
    holdout = _delta_block(candidate, baseline, "holdout")
    recent63 = _delta_block(candidate, baseline, "recent63")
    recent20 = _delta_block(candidate, baseline, "recent20")
    if candidate["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if full["rank_ic"] < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for key in ["top1", "top3", "top5", "top10", "top20"]:
        if full[key] <= 0.0:
            reasons.append(f"full_{key}_delta_non_positive")
    if holdout["top5"] <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if recent63["top5"] <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if recent20["top5"] <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    if (candidate["annual_stability"].get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    if (candidate["monthly_stability"].get("top5_positive_period_ratio") or 0.0) < 0.50:
        reasons.append("monthly_top5_positive_ratio_below_0_50")
    return not reasons, reasons


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = _prepare_frame()
    scan_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, Any]]] = []

    configs = [
        {"top_pct": 0.005, "alpha": 1e-4, "eta0": 0.005, "epochs": 2},
        {"top_pct": 0.010, "alpha": 1e-4, "eta0": 0.005, "epochs": 2},
        {"top_pct": 0.020, "alpha": 1e-4, "eta0": 0.005, "epochs": 2},
        {"top_pct": 0.050, "alpha": 1e-4, "eta0": 0.003, "epochs": 2},
        {"top_pct": 0.020, "alpha": 5e-5, "eta0": 0.003, "epochs": 3},
    ]
    blend_weights = [0.02, 0.05, 0.1, 0.2, 0.35]

    for config in configs:
        pred = _online_proba(frame, **config)
        pred["baseline_score"] = pred["pred_3d_rank"]
        _, baseline = _metrics_for_score(pred, "baseline_score")
        for weight in blend_weights:
            pred["candidate_score"] = (1.0 - weight) * pred["pred_3d_rank"] + weight * pred["tophit_rank"]
            _, metrics = _metrics_for_score(pred, "candidate_score")
            passed, failed_reasons = _gate(metrics, baseline)
            deltas = {
                "full": _delta_block(metrics, baseline, "full"),
                "train": _delta_block(metrics, baseline, "train"),
                "holdout": _delta_block(metrics, baseline, "holdout"),
                "recent63": _delta_block(metrics, baseline, "recent63"),
                "recent20": _delta_block(metrics, baseline, "recent20"),
                "recent126": _delta_block(metrics, baseline, "recent126"),
            }
            selection_score = (
                deltas["train"]["top5"]
                + 0.5 * deltas["train"]["top10"]
                + 0.25 * deltas["train"]["top3"]
            )
            row = {
                **config,
                "blend_weight": weight,
                "passed": passed,
                "failed_reasons": ";".join(failed_reasons),
                "selection_score_train_only": selection_score,
                "eval_trade_days": metrics["eval_trade_days"],
                "eval_min_trade_date": metrics["eval_min_trade_date"],
                "eval_max_trade_date": metrics["eval_max_trade_date"],
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
                        "label": rb.LABELS["3d"],
                        "config": config,
                        "blend_weight": weight,
                        "baseline_current_formal_same_dates": baseline,
                        "candidate_metrics": metrics,
                        "deltas": deltas,
                        "gate": {"passed": passed, "failed_reasons": failed_reasons},
                        "selection_score_train_only": selection_score,
                    },
                )
            )

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "online_tophit_classifier_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][1]
    passing = [payload for _, payload in candidates if payload["gate"]["passed"]]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_online_tophit_classifier",
        "method": "monthly online SGDClassifier for 3D top-hit probability, trained only on past months, then blended with active formal 3D rank",
        "selection_rule": "Only out-of-sample train-window deltas through 20251231 select config/blend. Holdout/recent windows are validation only.",
        "initial_train_days": INITIAL_TRAIN_DAYS,
        "feature_columns": FEATURE_COLS,
        "selected_by_train": selected,
        "passing_count": len(passing),
        "best_passing_by_train_rank": passing[0] if passing else None,
        "scan_csv": str(scan_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else (
                "candidate_exists_but_not_train_selected" if passing else "no_3d_candidate_passed_online_tophit_scan"
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
    report_json = REPORT_DIR / "online_tophit_classifier_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D online top-hit classifier 研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- 训练窗口选择配置：`{sel['config']}`",
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
    report_md = REPORT_DIR / "online_tophit_classifier_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
