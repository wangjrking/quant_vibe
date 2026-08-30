from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_front_combo_calibrator_20260714"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"
TOP5_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "front_top5_calibrator_3d5d10d_20260714"
TOP3_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "front_top3_calibrator_3d5d10d_20260714"

sys.path.insert(0, str(MAIN))
import research_four_year_3d5d10d_front_top5_calibrator_20260714 as ftop  # noqa: E402


LABEL_KEYS = ["3d", "5d", "10d"]
HOLDOUT_START = "20260101"
POOL_THRESHOLDS = [0.90, 0.95, 0.97]
BETA5 = [0.0, 0.01, 0.015, 0.02, 0.03, 0.04]
BETA3 = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def _metrics_pack(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "full": ftop.rb._metrics(daily),
        "fit": ftop.rb._metrics(daily[daily["trade_date"] <= ftop.FIT_END]),
        "tail": ftop.rb._metrics(daily[(daily["trade_date"] >= ftop.TAIL_START) & (daily["trade_date"] <= ftop.TAIL_END)]),
        "holdout": ftop.rb._metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": ftop.rb._metrics(daily.tail(20)),
        "recent63": ftop.rb._metrics(daily.tail(63)),
        "recent126": ftop.rb._metrics(daily.tail(126)),
        "annual_stability": ftop.rb._period_stability(daily, "year"),
        "monthly_stability": ftop.rb._period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], block: str, metric: str) -> float | None:
    cv = candidate[block].get(metric)
    bv = baseline[block].get(metric)
    if cv is None or bv is None:
        return None
    return float(cv - bv)


def _deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    blocks = ["full", "fit", "tail", "holdout", "recent20", "recent63", "recent126"]
    metrics = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    return {block: {metric: _delta(candidate, baseline, block, metric) for metric in metrics} for block in blocks}


def _gate(deltas: dict[str, dict[str, float | None]], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    full = deltas["full"]
    holdout = deltas["holdout"]
    recent63 = deltas["recent63"]
    recent20 = deltas["recent20"]
    if (full.get("rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for metric in ["top1", "top3", "top5", "top10", "top20"]:
        if (full.get(metric) or 0.0) <= 0.0:
            reasons.append(f"full_{metric}_delta_non_positive")
    if (holdout.get("top5") or 0.0) < 0.0:
        reasons.append("holdout_top5_delta_negative")
    if (recent63.get("top5") or 0.0) < 0.0:
        reasons.append("recent63_top5_delta_negative")
    if (recent20.get("top5") or 0.0) < 0.0:
        reasons.append("recent20_top5_delta_negative")
    return {"passed": not reasons, "failed_reasons": reasons}


def _objective(deltas: dict[str, dict[str, float | None]]) -> float:
    full = deltas["full"]
    holdout = deltas["holdout"]
    recent63 = deltas["recent63"]
    recent20 = deltas["recent20"]
    return float(
        2.0 * (full.get("top3") or -1.0)
        + 1.2 * (full.get("top5") or -1.0)
        + 0.8 * (full.get("top1") or -1.0)
        + 1.0 * (holdout.get("top5") or -1.0)
        + 1.0 * (recent63.get("top5") or -1.0)
        + 1.0 * (recent20.get("top5") or -1.0)
        + 0.05 * (full.get("rank_ic") or -1.0)
    )


def _score_daily(frame: pd.DataFrame, label_key: str, score: pd.Series) -> pd.DataFrame:
    work = frame.copy()
    work["candidate_score"] = score
    return ftop.rb._daily_eval(work, "candidate_score", f"label_{label_key}")


def _run_label(full_base: pd.DataFrame, label_key: str) -> dict[str, Any]:
    frame = ftop._add_daily_top5_flag(full_base, label_key)
    baseline_daily = ftop.rb._daily_eval(frame, f"pred_{label_key}_rank", f"label_{label_key}")
    baseline = _metrics_pack(baseline_daily)
    top5_model = TOP5_DIR / f"front_top5_calibrator_{label_key}_research_20260714.json"
    top3_model = TOP3_DIR / f"front_top3_calibrator_{label_key}_research_20260714.json"
    top5 = xgb.XGBClassifier()
    top5.load_model(str(top5_model))
    top3 = xgb.XGBClassifier()
    top3.load_model(str(top3_model))
    frame["_p5"] = top5.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p3"] = top3.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p5_rank"] = frame.groupby("trade_date")["_p5"].rank(method="average", pct=True)
    frame["_p3_rank"] = frame.groupby("trade_date")["_p3"].rank(method="average", pct=True)

    scan_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    base_col = f"pred_{label_key}_rank"
    for pool_threshold in POOL_THRESHOLDS:
        mask = frame[base_col] >= pool_threshold
        for beta5 in BETA5:
            for beta3 in BETA3:
                score = frame[base_col].copy()
                score.loc[mask] = (
                    frame.loc[mask, base_col]
                    + beta5 * (frame.loc[mask, "_p5_rank"] - 0.5)
                    + beta3 * (frame.loc[mask, "_p3_rank"] - 0.5)
                )
                daily = _score_daily(frame, label_key, score)
                metrics = _metrics_pack(daily)
                deltas = _deltas(metrics, baseline)
                gate = _gate(deltas, metrics)
                row = {
                    "pool_threshold": pool_threshold,
                    "beta5": beta5,
                    "beta3": beta3,
                    "metrics": metrics,
                    "deltas": deltas,
                    "gate": gate,
                    "objective": _objective(deltas),
                }
                scan_rows.append(
                    {
                        "label_key": label_key,
                        "pool_threshold": pool_threshold,
                        "beta5": beta5,
                        "beta3": beta3,
                        "gate_passed": gate["passed"],
                        "objective": row["objective"],
                        "full_rank_ic_delta": deltas["full"].get("rank_ic"),
                        "full_top1_delta": deltas["full"].get("top1"),
                        "full_top3_delta": deltas["full"].get("top3"),
                        "full_top5_delta": deltas["full"].get("top5"),
                        "full_top10_delta": deltas["full"].get("top10"),
                        "full_top20_delta": deltas["full"].get("top20"),
                        "holdout_top5_delta": deltas["holdout"].get("top5"),
                        "recent63_top5_delta": deltas["recent63"].get("top5"),
                        "recent20_top5_delta": deltas["recent20"].get("top5"),
                        "failed_reasons": ";".join(gate["failed_reasons"]),
                    }
                )
                if selected is None or (
                    gate["passed"],
                    row["objective"],
                    deltas["full"].get("top3") or -999.0,
                    deltas["full"].get("top5") or -999.0,
                ) > (
                    selected["gate"]["passed"],
                    selected["objective"],
                    selected["deltas"]["full"].get("top3") or -999.0,
                    selected["deltas"]["full"].get("top5") or -999.0,
                ):
                    selected = row
    assert selected is not None
    return {
        "label": ftop.rb.LABELS[label_key],
        "source_models": {
            "top5_model": str(top5_model),
            "top3_model": str(top3_model),
        },
        "baseline_current_formal": baseline,
        "selected": selected,
        "scan_rows": scan_rows,
    }


def _update_status(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    passed = {k: v for k, v in payload["summaries"].items() if v["selected"]["gate"]["passed"]}
    decision = (
        "front_combo_calibrator_passed_for_some_labels_candidate_discussion_required"
        if passed
        else "no_3d_5d_10d_candidate_passed_front_combo_calibrator_scan"
    )
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["front_combo_calibrator_3d5d10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": sorted(passed.keys()),
            "reason": "All-label front combo scan reuses Top5 and Top3 front calibrators.",
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_front_combo_calibrator_3d5d10d_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": sorted(passed.keys()),
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    passed = [k for k, v in payload["summaries"].items() if v["selected"]["gate"]["passed"]]
    lines = [
        "# 3D / 5D / 10D 前排组合校准复核",
        "",
        "## 结论",
        "",
        f"通过 research-only 候选门槛的标签：`{', '.join(passed)}`。" if passed else "本轮没有标签通过。",
        "",
        "| 标签 | 通过 | pool | beta5 | beta3 | Full Top3 Δ | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label_key, item in payload["summaries"].items():
        s = item["selected"]
        d = s["deltas"]
        lines.append(
            "| {label} | {passed} | {pool:.2f} | {b5:.3f} | {b3:.3f} | {t3:.6f} | {t5:.6f} | {h5:.6f} | {r63:.6f} | {r20:.6f} |".format(
                label=label_key,
                passed="是" if s["gate"]["passed"] else "否",
                pool=s["pool_threshold"],
                b5=s["beta5"],
                b3=s["beta3"],
                t3=d["full"].get("top3") or 0.0,
                t5=d["full"].get("top5") or 0.0,
                h5=d["holdout"].get("top5") or 0.0,
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
    base = ftop._prepare_base()
    summaries: dict[str, Any] = {}
    scan_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        summary = _run_label(base, label_key)
        scan_rows.extend(summary.pop("scan_rows"))
        summaries[label_key] = summary
    scan_csv = REPORT_DIR / "front_combo_calibrator_3d5d10d_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8")
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_front_combo_calibrator",
        "scan_csv": str(scan_csv),
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
    report_json = REPORT_DIR / "front_combo_calibrator_3d5d10d_report.json"
    report_md = REPORT_DIR / "front_combo_calibrator_3d5d10d_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
