from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
SCAN_PATH = DATA_DIR / "reports" / "model_agent_current_best_second_or_expansion_v96_20260627" / "executable_1d_open_return" / "second_or_expansion_scan.csv"
CURRENT_SNAPSHOT = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260627_v106" / "current_best_research_snapshot_v106.json"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_no_upgrade_review_v108_20260627"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scan = pd.read_csv(SCAN_PATH)
    current = json.loads(CURRENT_SNAPSHOT.read_text(encoding="utf-8"))["current_best"]["executable_1d_open_return"]
    pass_hard = scan[scan["pass_hard"] == True].copy()
    stable = scan[(scan["min_month_top5_delta"] >= 0) & (scan["full_rank_ic_delta"] >= 0)].copy()
    stronger_top5 = scan[scan["full_top5_delta"] > float(current["metrics"]["full_top5_delta"])].copy()
    stronger_top5_stable = stronger_top5[(stronger_top5["min_month_top5_delta"] >= 0) & (stronger_top5["pass_hard"] == True)].copy()
    best_pass = pass_hard.sort_values(
        ["top_objective", "full_top5_delta", "full_rank_ic_delta"],
        ascending=[False, False, False],
    ).head(10)
    best_top5_any = scan.sort_values(
        ["full_top5_delta", "full_rank_ic_delta", "top_objective"],
        ascending=[False, False, False],
    ).head(10)
    best_rankic_any = scan.sort_values(
        ["full_rank_ic_delta", "full_top5_delta", "top_objective"],
        ascending=[False, False, False],
    ).head(10)
    best_pass.to_csv(REPORT_DIR / "best_pass_hard_candidates.csv", index=False, encoding="utf-8-sig")
    best_top5_any.to_csv(REPORT_DIR / "best_top5_any_candidates.csv", index=False, encoding="utf-8-sig")
    best_rankic_any.to_csv(REPORT_DIR / "best_rankic_any_candidates.csv", index=False, encoding="utf-8-sig")

    decision = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_1d_no_upgrade_review_v108",
        "current_asset": current["asset"],
        "current_table": current["table"],
        "current_metrics": current["metrics"],
        "scan_path": str(SCAN_PATH),
        "total_candidates": int(len(scan)),
        "pass_hard_candidates": int(len(pass_hard)),
        "stable_candidates": int(len(stable)),
        "stronger_top5_candidates": int(len(stronger_top5)),
        "stronger_top5_stable_candidates": int(len(stronger_top5_stable)),
        "decision": "keep_current_1d_v96",
        "reason": "No scanned 1D candidate improves full Top5 versus v106 while also passing hard stability constraints.",
        "boundaries": {
            "research_only": True,
            "read_only_review": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "v108_no_upgrade_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 1D 研究候选不升级复核 v108",
        "",
        "## 结论",
        "",
        "本轮只读复核既有 1D v96 扫描结果，不训练模型，不写预测表，不修改 formal/production manifest，不生成信号，不运行回测。",
        "",
        f"- 当前保留资产：`{current['asset']}`",
        f"- 当前表：`{current['table']}`",
        f"- 当前全窗口 RankIC delta：`{current['metrics']['full_rank_ic_delta']:.10f}`",
        f"- 当前全窗口 Top1 delta：`{current['metrics']['full_top1_delta']:.10f}`",
        f"- 当前全窗口 Top5 delta：`{current['metrics']['full_top5_delta']:.10f}`",
        "",
        "## 判断",
        "",
        f"- 扫描候选总数：`{len(scan)}`",
        f"- 通过硬约束候选数：`{len(pass_hard)}`",
        f"- Top5 高于当前版本的候选数：`{len(stronger_top5)}`",
        f"- Top5 高于当前且通过硬约束的候选数：`{len(stronger_top5_stable)}`",
        "",
        "结论：当前不替换 1D。能提高 Top5/RankIC 的更激进候选存在负月份或硬约束失败；在硬约束通过的候选中，当前 v96 仍是最优。",
        "",
        "## 证据",
        "",
        "- `best_pass_hard_candidates.csv`",
        "- `best_top5_any_candidates.csv`",
        "- `best_rankic_any_candidates.csv`",
        "- `v108_no_upgrade_decision.json`",
    ]
    (REPORT_DIR / "v108_no_upgrade_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
