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
    / "model_agent_four_year_3d5d10d_strict_train_selected_rank_blend_fast_20260714"
)

TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
LABEL_KEYS = ["3d", "5d", "10d"]

sys.path.insert(0, str(MAIN))
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402


def _weights(label_key: str) -> list[tuple[float, float, float, float]]:
    keys = ["1d", "3d", "5d", "10d"]
    target_idx = keys.index(label_key)
    out: set[tuple[float, float, float, float]] = {rb.BASELINE_WEIGHTS[label_key]}
    for aux_idx, aux_key in enumerate(keys):
        if aux_key == label_key:
            continue
        for target_weight in [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]:
            w = [0.0, 0.0, 0.0, 0.0]
            w[target_idx] = target_weight
            w[aux_idx] = round(1.0 - target_weight, 6)
            out.add(tuple(w))
    out.add((0.25, 0.25, 0.25, 0.25))
    return sorted(out)


def _score(frame: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.Series:
    return (
        weights[0] * frame["pred_1d_rank"]
        + weights[1] * frame["pred_3d_rank"]
        + weights[2] * frame["pred_5d_rank"]
        + weights[3] * frame["pred_10d_rank"]
    )


def _train_top5_mean(train_frame: pd.DataFrame, label_col: str, weights: tuple[float, float, float, float]) -> float:
    local = train_frame[["trade_date", label_col]].copy()
    local["score"] = _score(train_frame, weights)
    local["rank_desc"] = local.groupby("trade_date")["score"].rank(ascending=False, method="first")
    return float(local[local["rank_desc"] <= 5].groupby("trade_date")[label_col].mean().mean())


def _candidate_daily(base: pd.DataFrame, label_key: str, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    frame = base[base[label_col].notna()].copy()
    frame["candidate_score"] = _score(frame, weights)
    return rb._daily_eval(frame, "candidate_score", label_col)


def _metrics(daily: pd.DataFrame) -> dict[str, Any]:
    return {
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
        "train_trade_days": int((daily["trade_date"] <= TRAIN_END).sum()),
        "holdout_trade_days": int((daily["trade_date"] >= HOLDOUT_START).sum()),
    }


def _delta_block(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
        out[key] = rb._to_float((candidate[block].get(key) or 0.0) - (baseline[block].get(key) or 0.0))
    return out


def _gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    full = _delta_block(candidate, baseline, "full")
    holdout = _delta_block(candidate, baseline, "holdout")
    recent63 = _delta_block(candidate, baseline, "recent63")
    recent20 = _delta_block(candidate, baseline, "recent20")
    if candidate["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (full["rank_ic"] or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    for key in ["top1", "top3", "top5", "top10", "top20"]:
        if (full[key] or 0.0) <= 0.0:
            reasons.append(f"full_{key}_delta_non_positive")
    if (holdout["top5"] or 0.0) <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if (recent63["top5"] or 0.0) <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if (recent20["top5"] or 0.0) <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    return not reasons, reasons


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    scan_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for label_key in LABEL_KEYS:
        label_col = f"label_{label_key}"
        train_frame = base[(base[label_col].notna()) & (base["trade_date"] <= TRAIN_END)].copy()
        baseline_weights = rb.BASELINE_WEIGHTS[label_key]
        baseline_train_top5 = _train_top5_mean(train_frame, label_col, baseline_weights)
        scored_weights: list[tuple[float, tuple[float, float, float, float]]] = []
        for weights in _weights(label_key):
            train_top5 = _train_top5_mean(train_frame, label_col, weights)
            delta = train_top5 - baseline_train_top5
            scan_rows.append(
                {
                    "label_key": label_key,
                    "w1": weights[0],
                    "w3": weights[1],
                    "w5": weights[2],
                    "w10": weights[3],
                    "train_top5": train_top5,
                    "train_top5_delta": delta,
                }
            )
            scored_weights.append((delta, weights))
        scored_weights.sort(reverse=True, key=lambda item: item[0])
        selected_delta, selected_weights = scored_weights[0]
        baseline = _metrics(_candidate_daily(base, label_key, baseline_weights))
        selected = _metrics(_candidate_daily(base, label_key, selected_weights))
        gate_passed, failures = _gate(selected, baseline)
        summaries[label_key] = {
            "label": rb.LABELS[label_key],
            "selection_rule": "Only train-window Top5 delta over 20220606-20251231 is used for weight selection.",
            "selected_weights": {
                "w1": selected_weights[0],
                "w3": selected_weights[1],
                "w5": selected_weights[2],
                "w10": selected_weights[3],
            },
            "selected_train_top5_delta": selected_delta,
            "deltas": {
                "full": _delta_block(selected, baseline, "full"),
                "train": _delta_block(selected, baseline, "train"),
                "holdout": _delta_block(selected, baseline, "holdout"),
                "recent63": _delta_block(selected, baseline, "recent63"),
                "recent20": _delta_block(selected, baseline, "recent20"),
            },
            "candidate_metrics": selected,
            "baseline_metrics": baseline,
            "gate": {"passed": gate_passed, "failed_reasons": failures},
        }

    scan_csv = REPORT_DIR / "strict_train_selected_3d5d10d_fast_scan.csv"
    pd.DataFrame(scan_rows).to_csv(scan_csv, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d5d10d_strict_train_selected_rank_blend_fast",
        "train_window": {"start": "20220606", "end": TRAIN_END},
        "holdout_window": {"start": HOLDOUT_START, "end": "mature_label_cutoff_by_horizon"},
        "scan_csv": str(scan_csv),
        "summaries": summaries,
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
    report_json = REPORT_DIR / "strict_train_selected_3d5d10d_fast_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 3D/5D/10D 严格训练窗口 rank-blend 快速扫描",
        "",
        "## 结论",
        "",
        "本轮只用 `20220606-20251231` 训练窗口 Top5 Δ 选择权重；`20260101` 之后 holdout / recent 只用于验证。",
        "",
        "| 标签 | 选中权重 | 是否过门 | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ | 失败原因 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label_key, item in summaries.items():
        lines.append(
            "| {label} | `{weights}` | {passed} | {full:.6f} | {holdout:.6f} | {r63:.6f} | {r20:.6f} | {failures} |".format(
                label=label_key,
                weights=item["selected_weights"],
                passed="是" if item["gate"]["passed"] else "否",
                full=item["deltas"]["full"]["top5"] or 0.0,
                holdout=item["deltas"]["holdout"]["top5"] or 0.0,
                r63=item["deltas"]["recent63"]["top5"] or 0.0,
                r20=item["deltas"]["recent20"]["top5"] or 0.0,
                failures="; ".join(item["gate"]["failed_reasons"]) if item["gate"]["failed_reasons"] else "-",
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- research-only。",
            "- 未训练模型。",
            "- 未写预测资产。",
            "- 未修改 formal manifest。",
            "- 未生成 `approved_for_l5`。",
            "- 未生成信号，未跑回测。",
            "",
            "## 证据",
            "",
            f"- 扫描明细：`{scan_csv}`",
            f"- 报告 JSON：`{report_json}`",
        ]
    )
    (REPORT_DIR / "strict_train_selected_3d5d10d_fast_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "report_json": str(report_json), "scan_csv": str(scan_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
