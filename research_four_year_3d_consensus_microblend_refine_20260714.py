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
    / "model_agent_four_year_3d_consensus_microblend_refine_20260714"
)

sys.path.insert(0, str(MAIN))
import research_four_year_3d_consensus_microblend_20260714 as base_scan  # noqa: E402
import research_four_year_rank_blend_all_labels_20260714 as rb  # noqa: E402

TRAIN_END = base_scan.TRAIN_END
LABEL_COL = base_scan.LABEL_COL


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = rb._add_rank_columns(rb._read_scores(with_labels=True))
    frame = base[base[LABEL_COL].notna()].copy()
    frame["baseline_score"] = frame["pred_3d_rank"]
    _, baseline = base_scan._metrics_for_score(frame, "baseline_score")

    rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    pools = [0.975, 0.98, 0.985, 0.99]
    betas = [round(x / 1000, 3) for x in range(6, 31)]
    modes = ["positive_consensus_bonus", "pull_to_consensus"]

    for pool in pools:
        for beta in betas:
            for mode in modes:
                local = frame[["trade_date", LABEL_COL, "pred_3d_rank", "pred_5d_rank", "pred_10d_rank"]].copy()
                local["candidate_score"] = base_scan._candidate_score(local, pool, beta, mode)
                _, metrics = base_scan._metrics_for_score(local, "candidate_score")
                passed, failed_reasons = base_scan._gate(metrics, baseline)
                deltas = {
                    "full": base_scan._delta_block(metrics, baseline, "full"),
                    "train": base_scan._delta_block(metrics, baseline, "train"),
                    "holdout": base_scan._delta_block(metrics, baseline, "holdout"),
                    "recent63": base_scan._delta_block(metrics, baseline, "recent63"),
                    "recent20": base_scan._delta_block(metrics, baseline, "recent20"),
                    "recent126": base_scan._delta_block(metrics, baseline, "recent126"),
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
                    "fail_count": len(failed_reasons),
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
                rows.append(row)
                candidates.append(
                    (
                        len(failed_reasons),
                        train_score,
                        {
                            "label": rb.LABELS["3d"],
                            "params": {"pool": pool, "beta": beta, "mode": mode},
                            "baseline_current_formal": baseline,
                            "candidate_metrics": metrics,
                            "deltas": deltas,
                            "gate": {"passed": passed, "failed_reasons": failed_reasons},
                            "selection_score_train_only": train_score,
                        },
                    )
                )

    scan = pd.DataFrame(rows)
    scan_csv = REPORT_DIR / "consensus_microblend_refine_3d_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")
    candidates.sort(key=lambda item: (item[0], -item[1]))
    best_by_fail_count = candidates[0][2]
    passing = [payload for fail_count, _, payload in candidates if fail_count == 0]
    passing.sort(key=lambda item: item["selection_score_train_only"], reverse=True)
    selected = passing[0] if passing else best_by_fail_count

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_3d_consensus_microblend_refine",
        "selection_rule": "This is a narrow diagnostic refinement around the closest non-passing consensus microblend region. It may identify a candidate for later stricter validation, but does not promote anything by itself.",
        "baseline_current_formal": baseline,
        "selected": selected,
        "passing_count": len(passing),
        "best_by_fail_count": best_by_fail_count,
        "scan_csv": str(scan_csv),
        "decision": "candidate_passed_refine_scan" if passing else "no_3d_candidate_passed_refine_scan",
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
    report_json = REPORT_DIR / "consensus_microblend_refine_3d_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sel = selected
    lines = [
        "# 3D consensus microblend 窄范围精扫报告",
        "",
        "## 结论",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- 过门候选数量：`{len(passing)}`",
        f"- 选中参数：`{sel['params']}`",
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
            if metric in sel["deltas"][block]:
                lines.append(f"| {block}.{metric} | {sel['deltas'][block][metric]:.8f} |")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本轮仅为 research-only 窄范围诊断。",
            "- 未训练模型。",
            "- 未写入 formal L4 资产。",
            "- 未修改 approved_for_l5 manifest。",
            "- 未生成交易信号，未跑策略回测。",
        ]
    )
    report_md = REPORT_DIR / "consensus_microblend_refine_3d_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_json),
                "decision": payload["decision"],
                "passing_count": len(passing),
                "selected_params": selected["params"],
                "selected_passed": selected["gate"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
