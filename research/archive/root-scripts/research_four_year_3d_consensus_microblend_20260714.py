from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_3d_consensus_microblend_20260714"
)

TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
LABEL_KEY = "3d"
LABEL_COL = "label_3d"

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


def _metrics_for_score(frame: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = rb._daily_eval(frame, score_col, LABEL_COL)
    metrics = {
        "full": rb._metrics(daily),
        "train": rb._metrics(daily[daily["trade_date"] <= TRAIN_END]),
        "holdout": rb._metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
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
    out: dict[str, float] = {}
    for key in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
        out[key] = float((candidate[block].get(key) or 0.0) - (baseline[block].get(key) or 0.0))
    return out


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = _delta_block(candidate, baseline, "full")
    holdout = _delta_block(candidate, baseline, "holdout")
    recent63 = _delta_block(candidate, baseline, "recent63")
    recent20 = _delta_block(candidate, baseline, "recent20")
    annual = candidate["annual_stability"]
    monthly = candidate["monthly_stability"]

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
    if (annual.get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    if (monthly.get("top5_positive_period_ratio") or 0.0) < 0.50:
        reasons.append("monthly_top5_positive_ratio_below_0_50")
    return not reasons, reasons


def _candidate_score(frame: pd.DataFrame, pool: float, beta: float, mode: str) -> pd.Series:
    base = frame["pred_3d_rank"]
    consensus = (frame["pred_5d_rank"] + frame["pred_10d_rank"]) / 2.0
    front = base >= pool
    if mode == "pull_to_consensus":
        return base.where(~front, base + beta * (consensus - base))
    if mode == "positive_consensus_bonus":
        bonus = (consensus - pool).clip(lower=0.0)
        return base + beta * bonus.where(front, 0.0)
    if mode == "strict_agreement_bonus":
        agreement = front & (frame["pred_5d_rank"] >= pool) & (frame["pred_10d_rank"] >= pool)
        return base + beta * agreement.astype(float)
    raise ValueError(f"unknown mode: {mode}")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    frame = base[base[LABEL_COL].notna()].copy()
    frame["baseline_score"] = frame["pred_3d_rank"]
    _, baseline = _metrics_for_score(frame, "baseline_score")

    scan_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []

    pools = [0.95, 0.97, 0.98, 0.99]
    betas = [0.005, 0.01, 0.02, 0.03, 0.05]
    modes = ["pull_to_consensus", "positive_consensus_bonus", "strict_agreement_bonus"]

    for pool in pools:
        for beta in betas:
            for mode in modes:
                local = frame[["trade_date", LABEL_COL, "pred_3d_rank", "pred_5d_rank", "pred_10d_rank"]].copy()
                local["candidate_score"] = _candidate_score(local, pool, beta, mode)
                _, metrics = _metrics_for_score(local, "candidate_score")
                passed, failed_reasons = _gate(metrics, baseline)
                deltas = {
                    "full": _delta_block(metrics, baseline, "full"),
                    "train": _delta_block(metrics, baseline, "train"),
                    "holdout": _delta_block(metrics, baseline, "holdout"),
                    "recent63": _delta_block(metrics, baseline, "recent63"),
                    "recent20": _delta_block(metrics, baseline, "recent20"),
                    "recent126": _delta_block(metrics, baseline, "recent126"),
                }
                train_score = (
                    deltas["train"]["top5"]
                    + 0.5 * deltas["train"]["top10"]
                    + 0.2 * deltas["train"]["rank_ic"]
                )
                row = {
                    "pool": pool,
                    "beta": beta,
                    "mode": mode,
                    "passed": passed,
                    "failed_reasons": ";".join(failed_reasons),
                    "selection_score_train_only": train_score,
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
                payload = {
                    "label": rb.LABELS[LABEL_KEY],
                    "params": {"pool": pool, "beta": beta, "mode": mode},
                    "baseline_current_formal": baseline,
                    "candidate_metrics": metrics,
                    "deltas": deltas,
                    "gate": {"passed": passed, "failed_reasons": failed_reasons},
                    "selection_score_train_only": train_score,
                }
                candidates.append((train_score, row, payload))

    scan = pd.DataFrame(scan_rows)
    scan_csv = REPORT_DIR / "consensus_microblend_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")

    # 选择规则只看训练窗口；若训练最优不过门，再额外记录全扫描里是否存在过门项。
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0][2]
    passing = [payload for _, _, payload in candidates if payload["gate"]["passed"]]
    best_passing = passing[0] if passing else None

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_consensus_microblend",
        "selection_rule": "Only train-window selection_score_train_only over 20220606-20251231 selects the candidate. Holdout and recent windows are validation only.",
        "train_window": {"start": "20220606", "end": TRAIN_END},
        "holdout_window": {"start": HOLDOUT_START, "end": "mature_label_cutoff"},
        "baseline_current_formal": baseline,
        "selected_by_train": selected,
        "best_passing_by_train_rank": best_passing,
        "passing_count": len(passing),
        "scan_csv": str(scan_csv),
        "decision": (
            "candidate_passed_by_train_selection"
            if selected["gate"]["passed"]
            else (
                "candidate_exists_but_not_train_selected"
                if best_passing is not None
                else "no_3d_candidate_passed_consensus_microblend_scan"
            )
        ),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_asset_write": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "consensus_microblend_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D consensus microblend 研究报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- 训练窗口选择参数：`{sel['params']}`",
        f"- 训练窗口选择候选是否过门：`{sel['gate']['passed']}`",
        f"- 失败原因：`{'; '.join(sel['gate']['failed_reasons']) if sel['gate']['failed_reasons'] else '-'}`",
        "",
        "## 训练选择候选相对当前 3D formal 的关键增量",
        "",
        "| 指标 | 增量 |",
        "|---|---:|",
    ]
    for block in ["full", "holdout", "recent63", "recent20"]:
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
            if metric in sel["deltas"][block]:
                lines.append(f"| {block}.{metric} | {sel['deltas'][block][metric]:.8f} |")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本轮仅为 research-only 扫描。",
            "- 未训练模型。",
            "- 未写入 formal L4 资产。",
            "- 未修改 approved_for_l5 manifest。",
            "- 未生成交易信号，未跑策略回测。",
        ]
    )
    report_md = REPORT_DIR / "consensus_microblend_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_json), "decision": payload["decision"], "passing_count": len(passing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
