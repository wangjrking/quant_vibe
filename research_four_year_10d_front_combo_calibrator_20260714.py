from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_10d_front_combo_calibrator_20260714"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"
TOP5_MODEL = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "front_top5_calibrator_3d5d10d_20260714" / "front_top5_calibrator_10d_research_20260714.json"
TOP3_MODEL = DATA_DIR / "experimental_assets" / "model-agent" / "models" / "front_top3_calibrator_3d5d10d_20260714" / "front_top3_calibrator_10d_research_20260714.json"

sys.path.insert(0, str(MAIN))
import research_four_year_3d5d10d_front_top5_calibrator_20260714 as ftop  # noqa: E402


LABEL_KEY = "10d"
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


def _score_daily(frame: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    work = frame.copy()
    work["candidate_score"] = score
    return ftop.rb._daily_eval(work, "candidate_score", f"label_{LABEL_KEY}")


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


def _update_status(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    passed = payload["selected"]["gate"]["passed"]
    decision = (
        "front_combo_calibrator_10d_passed_candidate_discussion_required"
        if passed
        else "no_10d_candidate_passed_front_combo_calibrator_scan"
    )
    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["front_combo_calibrator_10d"] = {
            "decision": decision,
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": [LABEL_KEY] if passed else [],
            "reason": "10D Top5 and Top3 front calibrators are combined to repair the Top3 gap.",
        }
        if passed:
            status["pending_stronger_research_candidate"] = {
                "source": str(report_json),
                "labels": [LABEL_KEY],
                "note": "Research-only pass; formal/L5/production changes require separate user authorization and audit.",
            }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_front_combo_calibrator_10d_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": decision,
            "passed_labels": [LABEL_KEY] if passed else [],
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    s = payload["selected"]
    d = s["deltas"]
    lines = [
        "# 10D 前排 Top5/Top3 组合校准研究",
        "",
        "## 结论",
        "",
        "本轮 10D 组合校准{}进入生产候选讨论。".format("可以" if s["gate"]["passed"] else "仍不能"),
        "",
        "## 最优组合",
        "",
        f"- pool threshold：`{s['pool_threshold']}`",
        f"- beta5：`{s['beta5']}`",
        f"- beta3：`{s['beta3']}`",
        "",
        "## 相对当前 10D formal 的改善",
        "",
        f"- Full RankIC delta：`{d['full'].get('rank_ic'):.6f}`",
        f"- Full Top1 / Top3 / Top5 / Top10 / Top20 delta：`{d['full'].get('top1'):.6f} / {d['full'].get('top3'):.6f} / {d['full'].get('top5'):.6f} / {d['full'].get('top10'):.6f} / {d['full'].get('top20'):.6f}`",
        f"- Holdout Top5 delta：`{d['holdout'].get('top5'):.6f}`",
        f"- Recent63 Top5 delta：`{d['recent63'].get('top5'):.6f}`",
        f"- Recent20 Top5 delta：`{d['recent20'].get('top5'):.6f}`",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = ftop._prepare_base()
    frame = base[base[f"label_{LABEL_KEY}"].notna()].copy()
    baseline_daily = ftop.rb._daily_eval(frame, f"pred_{LABEL_KEY}_rank", f"label_{LABEL_KEY}")
    baseline = _metrics_pack(baseline_daily)

    top5 = xgb.XGBClassifier()
    top5.load_model(str(TOP5_MODEL))
    top3 = xgb.XGBClassifier()
    top3.load_model(str(TOP3_MODEL))
    p5 = top5.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1]
    p3 = top3.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1]
    frame["_p5"] = p5.astype("float32")
    frame["_p3"] = p3.astype("float32")
    frame["_p5_rank"] = frame.groupby("trade_date")["_p5"].rank(method="average", pct=True)
    frame["_p3_rank"] = frame.groupby("trade_date")["_p3"].rank(method="average", pct=True)

    scan_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    base_col = f"pred_{LABEL_KEY}_rank"
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
                daily = _score_daily(frame, score)
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
    scan_csv = REPORT_DIR / "front_combo_calibrator_10d_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8")
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_10d_front_combo_calibrator",
        "label_key": LABEL_KEY,
        "label": ftop.rb.LABELS[LABEL_KEY],
        "source_models": {
            "top5_model": str(TOP5_MODEL),
            "top3_model": str(TOP3_MODEL),
        },
        "baseline_current_formal": baseline,
        "selected": selected,
        "scan_csv": str(scan_csv),
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
    report_json = REPORT_DIR / "front_combo_calibrator_10d_report.json"
    report_md = REPORT_DIR / "front_combo_calibrator_10d_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, report_md)
    _update_status(report_json, report_md, payload)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
