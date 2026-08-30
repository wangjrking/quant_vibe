from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d_front_agreement_calibrator_20260714"
STATUS_DIR = DATA_DIR / "reports" / "model_agent_current_research_candidate_status_20260714"
FRONTIER_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_failure_frontier_20260714"
TOP5_MODEL = (
    DATA_DIR
    / "experimental_assets"
    / "model-agent"
    / "models"
    / "front_top5_calibrator_3d5d10d_20260714"
    / "front_top5_calibrator_3d_research_20260714.json"
)
TOP3_MODEL = (
    DATA_DIR
    / "experimental_assets"
    / "model-agent"
    / "models"
    / "front_top3_calibrator_3d5d10d_20260714"
    / "front_top3_calibrator_3d_research_20260714.json"
)

sys.path.insert(0, str(MAIN))
import research_four_year_3d5d10d_front_top5_calibrator_20260714 as ftop  # noqa: E402


LABEL_KEY = "3d"
HOLDOUT_START = "20260101"
POOL_THRESHOLDS = [0.97]
P5_THRESHOLDS = [0.80]
P3_THRESHOLDS = [0.80]
BETAS = [0.0025, 0.005]
MODES = ["add_constant", "add_confidence"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def metrics_pack(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "full": ftop.rb._metrics(daily),
        "fit": ftop.rb._metrics(daily[daily["trade_date"] <= ftop.FIT_END]),
        "tail": ftop.rb._metrics(
            daily[(daily["trade_date"] >= ftop.TAIL_START) & (daily["trade_date"] <= ftop.TAIL_END)]
        ),
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


def deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    blocks = ["full", "fit", "tail", "holdout", "recent20", "recent63", "recent126"]
    metric_names = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    out: dict[str, dict[str, float | None]] = {}
    for block in blocks:
        out[block] = {}
        for metric in metric_names:
            cv = candidate[block].get(metric)
            bv = baseline[block].get(metric)
            out[block][metric] = None if cv is None or bv is None else float(cv - bv)
    return out


def gate(d: dict[str, dict[str, float | None]], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    full = d["full"]
    if (full.get("rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for metric in ["top1", "top3", "top5", "top10", "top20"]:
        if (full.get(metric) or 0.0) <= 0.0:
            reasons.append(f"full_{metric}_delta_non_positive")
    for block in ["holdout", "recent63", "recent20"]:
        if (d[block].get("top5") or 0.0) < 0.0:
            reasons.append(f"{block}_top5_delta_negative")
    return {"passed": not reasons, "failed_reasons": reasons}


def objective(d: dict[str, dict[str, float | None]]) -> float:
    full = d["full"]
    holdout = d["holdout"]
    recent63 = d["recent63"]
    recent20 = d["recent20"]
    return float(
        2.0 * (full.get("top5") or -1.0)
        + 1.0 * (full.get("top3") or -1.0)
        + 0.8 * (full.get("top10") or -1.0)
        + 1.2 * (holdout.get("top5") or -1.0)
        + 1.5 * (recent63.get("top5") or -1.0)
        + 1.5 * (recent20.get("top5") or -1.0)
        + 0.05 * (full.get("rank_ic") or -1.0)
    )


def score_daily(frame: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    work = frame.copy()
    work["candidate_score"] = score
    return ftop.rb._daily_eval(work, "candidate_score", f"label_{LABEL_KEY}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = ftop._prepare_base()
    frame = ftop._add_daily_top5_flag(base, LABEL_KEY)
    baseline_daily = ftop.rb._daily_eval(frame, f"pred_{LABEL_KEY}_rank", f"label_{LABEL_KEY}")
    baseline = metrics_pack(baseline_daily)

    top5 = xgb.XGBClassifier()
    top5.load_model(str(TOP5_MODEL))
    top3 = xgb.XGBClassifier()
    top3.load_model(str(TOP3_MODEL))
    frame["_p5"] = top5.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p3"] = top3.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p5_rank"] = frame.groupby("trade_date")["_p5"].rank(method="average", pct=True)
    frame["_p3_rank"] = frame.groupby("trade_date")["_p3"].rank(method="average", pct=True)
    frame["_agreement"] = (frame["_p5_rank"] + frame["_p3_rank"]) / 2.0

    base_col = f"pred_{LABEL_KEY}_rank"
    scan_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for pool in POOL_THRESHOLDS:
        pool_mask = frame[base_col] >= pool
        for p5_thr in P5_THRESHOLDS:
            for p3_thr in P3_THRESHOLDS:
                agree_mask = pool_mask & (frame["_p5_rank"] >= p5_thr) & (frame["_p3_rank"] >= p3_thr)
                if not bool(agree_mask.any()):
                    continue
                for beta in BETAS:
                    for mode in MODES:
                        score = frame[base_col].copy()
                        if mode == "add_constant":
                            score.loc[agree_mask] = score.loc[agree_mask] + beta
                        elif mode == "add_confidence":
                            score.loc[agree_mask] = score.loc[agree_mask] + beta * (
                                frame.loc[agree_mask, "_agreement"] - 0.5
                            )
                        elif mode == "blend_confidence":
                            score.loc[agree_mask] = (
                                (1.0 - beta) * score.loc[agree_mask]
                                + beta * frame.loc[agree_mask, "_agreement"]
                            )
                        daily = score_daily(frame, score)
                        m = metrics_pack(daily)
                        d = deltas(m, baseline)
                        g = gate(d, m)
                        obj = objective(d)
                        row = {
                            "pool_threshold": pool,
                            "p5_threshold": p5_thr,
                            "p3_threshold": p3_thr,
                            "beta": beta,
                            "mode": mode,
                            "agreement_rows": int(agree_mask.sum()),
                            "agreement_share": float(agree_mask.mean()),
                            "metrics": m,
                            "deltas": d,
                            "gate": g,
                            "objective": obj,
                        }
                        scan_rows.append(
                            {
                                "pool_threshold": pool,
                                "p5_threshold": p5_thr,
                                "p3_threshold": p3_thr,
                                "beta": beta,
                                "mode": mode,
                                "agreement_rows": row["agreement_rows"],
                                "agreement_share": row["agreement_share"],
                                "gate_passed": g["passed"],
                                "objective": obj,
                                "full_rank_ic_delta": d["full"].get("rank_ic"),
                                "full_top1_delta": d["full"].get("top1"),
                                "full_top3_delta": d["full"].get("top3"),
                                "full_top5_delta": d["full"].get("top5"),
                                "full_top10_delta": d["full"].get("top10"),
                                "full_top20_delta": d["full"].get("top20"),
                                "holdout_top5_delta": d["holdout"].get("top5"),
                                "recent63_top5_delta": d["recent63"].get("top5"),
                                "recent20_top5_delta": d["recent20"].get("top5"),
                                "failed_reasons": ";".join(g["failed_reasons"]),
                            }
                        )
                        if selected is None or (
                            g["passed"],
                            obj,
                            d["full"].get("top5") or -999.0,
                            d["recent20"].get("top5") or -999.0,
                        ) > (
                            selected["gate"]["passed"],
                            selected["objective"],
                            selected["deltas"]["full"].get("top5") or -999.0,
                            selected["deltas"]["recent20"].get("top5") or -999.0,
                        ):
                            selected = row
    assert selected is not None
    scan_csv = REPORT_DIR / "front_agreement_3d_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8")
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_3d_front_agreement_calibrator",
        "label": ftop.rb.LABELS[LABEL_KEY],
        "source_models": {"top5_model": str(TOP5_MODEL), "top3_model": str(TOP3_MODEL)},
        "baseline_current_formal": baseline,
        "selected": selected,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_prediction_asset_write": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "front_agreement_3d_report.json"
    report_md = REPORT_DIR / "front_agreement_3d_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    d = selected["deltas"]
    lines = [
        "# 3D 前排一致性校准研究报告",
        "",
        "## 结论",
        "",
        f"- 是否通过四年候选门槛：`{'是' if selected['gate']['passed'] else '否'}`",
        f"- 选中参数：pool=`{selected['pool_threshold']}`，p5=`{selected['p5_threshold']}`，p3=`{selected['p3_threshold']}`，beta=`{selected['beta']}`，mode=`{selected['mode']}`",
        f"- Full RankIC delta：`{d['full'].get('rank_ic'):.6f}`",
        f"- Full Top1/3/5/10/20 delta：`{d['full'].get('top1'):.6f} / {d['full'].get('top3'):.6f} / {d['full'].get('top5'):.6f} / {d['full'].get('top10'):.6f} / {d['full'].get('top20'):.6f}`",
        f"- Holdout / Recent63 / Recent20 Top5 delta：`{d['holdout'].get('top5'):.6f} / {d['recent63'].get('top5'):.6f} / {d['recent20'].get('top5'):.6f}`",
        "",
        "## 边界",
        "",
        "- research-only。",
        "- 未写 formal L4 资产。",
        "- 未修改 formal manifest。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_path = STATUS_DIR / "current_research_candidate_status_20260714.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.setdefault("not_promoted_research_lines", {})["front_agreement_calibrator_3d"] = {
            "decision": (
                "front_agreement_calibrator_3d_passed_candidate_discussion_required"
                if selected["gate"]["passed"]
                else "no_3d_candidate_passed_front_agreement_calibrator_scan"
            ),
            "summary": str(report_json),
            "review_md": str(report_md),
            "passed_labels": ["3d"] if selected["gate"]["passed"] else [],
            "reason": "3D front agreement scan only adjusts names where formal rank, Top5 calibrator and Top3 calibrator agree.",
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    frontier_path = FRONTIER_DIR / "failure_frontier_summary.json"
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
        frontier["latest_front_agreement_calibrator_3d_scan"] = {
            "summary": str(report_json),
            "review_md": str(report_md),
            "decision": (
                "front_agreement_calibrator_3d_passed_candidate_discussion_required"
                if selected["gate"]["passed"]
                else "no_3d_candidate_passed_front_agreement_calibrator_scan"
            ),
            "passed_labels": ["3d"] if selected["gate"]["passed"] else [],
        }
        frontier_path.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
