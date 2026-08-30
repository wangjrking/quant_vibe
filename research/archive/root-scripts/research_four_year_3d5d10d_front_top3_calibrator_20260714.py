from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_front_top3_calibrator_20260714"
MODEL_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "front_top3_calibrator_3d5d10d_20260714"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_3d5d10d_front_top5_calibrator_20260714 as ftop  # noqa: E402


LABEL_KEYS = ["3d", "5d", "10d"]
TARGET_K = 3


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def _add_daily_topk_flag(frame: pd.DataFrame, label_key: str, k: int) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    work = frame[frame[label_col].notna()].copy()
    work["_label_rank_desc"] = work.groupby("trade_date")[label_col].rank(method="first", ascending=False)
    work["target_top5"] = (work["_label_rank_desc"] <= k).astype("int8")
    return work


def _run_label(base: pd.DataFrame, label_key: str) -> dict[str, Any]:
    label_frame = _add_daily_topk_flag(base, label_key, TARGET_K)
    baseline_daily = ftop._daily_eval(label_frame, label_key, f"pred_{label_key}_rank")
    baseline_metrics = ftop._metrics_pack(baseline_daily)
    train_all = label_frame[
        (label_frame["trade_date"] <= ftop.FIT_END)
        & (label_frame[f"pred_{label_key}_rank"] >= min(ftop.POOL_THRESHOLDS))
    ]
    model, train_info = ftop._train_classifier(train_all, label_key)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"front_top3_calibrator_{label_key}_research_20260714.json"
    model.save_model(str(model_path))
    proba = model.predict_proba(label_frame[ftop.MODEL_FEATURES])[:, 1]
    scan_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for pool_threshold in ftop.POOL_THRESHOLDS:
        for beta in ftop.BETAS:
            daily = ftop._candidate_daily(label_frame, label_key, proba, beta, pool_threshold)
            metrics = ftop._metrics_pack(daily)
            deltas = ftop._deltas(metrics, baseline_metrics)
            gate = ftop._gate(deltas, metrics)
            top3_objective = float(
                1.6 * (deltas["tail"].get("top3") or -1.0)
                + 1.2 * (deltas["tail"].get("top5") or -1.0)
                + 0.8 * (deltas["full"].get("top3") or -1.0)
                + 0.5 * (deltas["full"].get("top5") or -1.0)
                + 0.5 * (deltas["recent63"].get("top3") or -1.0)
                + 0.5 * (deltas["recent20"].get("top3") or -1.0)
            )
            row = {
                "label_key": label_key,
                "pool_threshold": pool_threshold,
                "beta": beta,
                "metrics": metrics,
                "deltas": deltas,
                "gate": gate,
                "top3_objective": top3_objective,
            }
            scan_rows.append(
                {
                    "label_key": label_key,
                    "pool_threshold": pool_threshold,
                    "beta": beta,
                    "gate_passed": gate["passed"],
                    "top3_objective": top3_objective,
                    "full_rank_ic_delta": deltas["full"].get("rank_ic"),
                    "full_top1_delta": deltas["full"].get("top1"),
                    "full_top3_delta": deltas["full"].get("top3"),
                    "full_top5_delta": deltas["full"].get("top5"),
                    "full_top10_delta": deltas["full"].get("top10"),
                    "full_top20_delta": deltas["full"].get("top20"),
                    "tail_top3_delta": deltas["tail"].get("top3"),
                    "tail_top5_delta": deltas["tail"].get("top5"),
                    "holdout_top3_delta": deltas["holdout"].get("top3"),
                    "holdout_top5_delta": deltas["holdout"].get("top5"),
                    "recent63_top3_delta": deltas["recent63"].get("top3"),
                    "recent63_top5_delta": deltas["recent63"].get("top5"),
                    "recent20_top3_delta": deltas["recent20"].get("top3"),
                    "recent20_top5_delta": deltas["recent20"].get("top5"),
                    "failed_reasons": ";".join(gate["failed_reasons"]),
                }
            )
            if selected is None or (
                gate["passed"],
                top3_objective,
                deltas["full"].get("top3") or -999.0,
                deltas["full"].get("top5") or -999.0,
                deltas["recent63"].get("top5") or -999.0,
            ) > (
                selected["gate"]["passed"],
                selected["top3_objective"],
                selected["deltas"]["full"].get("top3") or -999.0,
                selected["deltas"]["full"].get("top5") or -999.0,
                selected["deltas"]["recent63"].get("top5") or -999.0,
            ):
                selected = row
    assert selected is not None
    return {
        "label": ftop.rb.LABELS[label_key],
        "model_path": str(model_path),
        "train_info": train_info,
        "model_features": ftop.MODEL_FEATURES,
        "target_topk": TARGET_K,
        "baseline_current_formal": baseline_metrics,
        "selected": selected,
        "scan_rows": scan_rows,
    }


def _update_status(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    passed = {k: v for k, v in payload["summaries"].items() if v["selected"]["gate"]["passed"]}
    decision = (
        "front_top3_calibrator_passed_for_some_labels_candidate_discussion_required"
        if passed
        else "no_3d_5d_10d_candidate_passed_front_top3_calibrator_scan"
    )
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["front_top3_calibrator_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": "Front-pool top3 classifier targets the top3 gap found in the top5 calibrator scan.",
        }
        if passed:
            status["pending_stronger_research_candidate"] = {
                "source": str(report_json),
                "labels": sorted(passed.keys()),
                "note": "Research-only pass; formal/L5/production changes require separate user authorization and audit.",
            }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_front_top3_calibrator_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# 3D / 5D / 10D 前排 Top3 校准研究",
        "",
        "## 结论",
        "",
    ]
    passed = [k for k, v in payload["summaries"].items() if v["selected"]["gate"]["passed"]]
    if passed:
        lines.append(f"通过 research-only 候选门槛的标签：`{', '.join(passed)}`。")
    else:
        lines.append("本轮没有 3D / 5D / 10D 标签通过四年观察与近期窗口门槛，不能进入生产候选讨论。")
    lines.extend(
        [
            "",
            "## 方法",
            "",
            f"- 用 `<= {ftop.FIT_END}` 的成熟标签训练前排 Top3 二分类校准器。",
            f"- 只对 formal 分数排名超过 `{ftop.POOL_THRESHOLDS}` 的前排池做小幅重排。",
            f"- 用 `{ftop.TAIL_START}-{ftop.TAIL_END}` tail guard 选择 pool threshold 和 beta，再用 `{ftop.HOLDOUT_START}` 之后验证。",
            "- 本轮只输出 research-only 评估和模型文件，不写预测资产表，不修改 formal manifest。",
            "",
            "## 指标摘要",
            "",
            "| 标签 | 通过 | pool | beta | Full Top3 Δ | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label_key, item in payload["summaries"].items():
        selected = item["selected"]
        d = selected["deltas"]
        lines.append(
            "| {label} | {passed} | {pool:.2f} | {beta:.2f} | {ft3:.6f} | {ft5:.6f} | {ht5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key,
                passed="是" if selected["gate"]["passed"] else "否",
                pool=selected["pool_threshold"],
                beta=selected["beta"],
                ft3=d["full"].get("top3") or 0.0,
                ft5=d["full"].get("top5") or 0.0,
                ht5=d["holdout"].get("top5") or 0.0,
                r63=d["recent63"].get("top5") or 0.0,
                r20=d["recent20"].get("top5") or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- research-only。",
            "- 未训练生产模型。",
            "- 未写 formal L4 预测资产。",
            "- 未修改 formal manifest。",
            "- 未修改 `approved_for_l5`。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base = ftop._prepare_base()
    summaries: dict[str, Any] = {}
    scan_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        summary = _run_label(base, label_key)
        scan_rows.extend(summary.pop("scan_rows"))
        summaries[label_key] = summary
    scan_csv = REPORT_DIR / "front_top3_calibrator_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8")
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_front_top3_calibrator",
        "target_topk": TARGET_K,
        "feature_db": str(ftop.FEATURE_DB),
        "feature_table": ftop.FEATURE_TABLE,
        "fit_window": {"start": ftop.rb.START_DATE, "end": ftop.FIT_END},
        "tail_guard_window": {"start": ftop.TAIL_START, "end": ftop.TAIL_END},
        "holdout_window": {"start": ftop.HOLDOUT_START, "end": "mature_label_cutoff_by_horizon"},
        "pool_thresholds": ftop.POOL_THRESHOLDS,
        "betas": ftop.BETAS,
        "scan_csv": str(scan_csv),
        "model_dir": str(MODEL_DIR),
        "summaries": summaries,
        "boundaries": {
            "research_only": True,
            "no_production_training": True,
            "no_prediction_asset_write": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "front_top3_calibrator_report.json"
    report_md = REPORT_DIR / "front_top3_calibrator_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
