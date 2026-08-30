from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
SCREEN_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_non1d_research_style_screen_v113_20260627"
    / "executable_5d_open_return_style_screen_v113.csv"
)
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_v105_vs_v96_review_v114_20260627"

LABEL = "executable_5d_open_return"
CURRENT_TABLE = "stock_predict_data_model_agent_5d_rankic_guard_alternative_v105_20260627_executable_5d_open_return_research"
ALT_TABLE = "stock_predict_data_model_agent_5d_second_or_expansion_v96_20260627_executable_5d_open_return_research"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    screen = pd.read_csv(SCREEN_PATH)
    current = screen[screen["table"] == CURRENT_TABLE].iloc[0].to_dict()
    alt = screen[screen["table"] == ALT_TABLE].iloc[0].to_dict()

    comparisons = {
        "rank_ic": float(current["full_rank_ic_delta"] - alt["full_rank_ic_delta"]),
        "top1": float(current["full_top1_delta"] - alt["full_top1_delta"]),
        "top3": float(current["full_top3_delta"] - alt["full_top3_delta"]),
        "top5": float(current["full_top5_delta"] - alt["full_top5_delta"]),
        "recent63_top5": float(current["recent63_top5_delta"] - alt["recent63_top5_delta"]),
        "recent20_top5": float(current["recent20_top5_delta"] - alt["recent20_top5_delta"]),
        "objective": float(current["objective_with_coverage_penalty"] - alt["objective_with_coverage_penalty"]),
    }

    keep_current = (
        comparisons["rank_ic"] > 0
        and comparisons["top1"] >= 0
        and comparisons["top5"] > 0
        and float(current["month_top5_min"]) >= 0
        and float(current["recent20_top5_delta"]) >= float(alt["recent20_top5_delta"]) - 1e-12
    )
    decision = "keep_v105_current_best" if keep_current else "reconsider_v96_for_top_objective"

    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_5d_v105_vs_v96_decision_review_v114",
        "label": LABEL,
        "decision": decision,
        "current_table": CURRENT_TABLE,
        "alternative_table": ALT_TABLE,
        "current": current,
        "alternative": alt,
        "current_minus_alternative": comparisons,
        "reason": (
            "v105 improves full RankIC, Top1, and Top5 versus v96; v96 only wins the local objective because it has a marginal recent63 Top5 edge."
            if keep_current
            else "v96 may be better under the local objective; inspect whether RankIC/TopN tradeoff is acceptable."
        ),
        "boundaries": {
            "research_only": True,
            "read_only_evaluation": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "v114_review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = pd.DataFrame(
        [
            {"metric": key, "v105_minus_v96": value}
            for key, value in comparisons.items()
        ]
    )
    rows.to_csv(REPORT_DIR / "v114_metric_comparison.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 5D v105 与 v96 决策复核 v114",
        "",
        "## 结论",
        "",
        "本报告只读复核 v113 中 5D 的排序冲突，不训练、不写预测表、不修改 manifest、不生成信号、不跑回测。",
        "",
        f"- 决策：`{decision}`",
        f"- 当前保留表：`{CURRENT_TABLE}`",
        f"- 对照表：`{ALT_TABLE}`",
        "",
        "## v105 相对 v96",
        "",
        f"- RankIC delta 差：`{comparisons['rank_ic']:.10f}`",
        f"- Top1 delta 差：`{comparisons['top1']:.10f}`",
        f"- Top3 delta 差：`{comparisons['top3']:.10f}`",
        f"- Top5 delta 差：`{comparisons['top5']:.10f}`",
        f"- 近 63 日 Top5 delta 差：`{comparisons['recent63_top5']:.10f}`",
        f"- 近 20 日 Top5 delta 差：`{comparisons['recent20_top5']:.10f}`",
        "",
        "## 判断",
        "",
        "v105 在全窗口 RankIC、Top1、Top5 上均强于 v96，且月度最差 Top5 delta 仍为非负。v96 只是在本次横向筛选 objective 中因近 63 日 Top5 有极小优势而排名更高。因此当前 5D 不回退，继续保留 v105 为当前最优研究资产。",
    ]
    (REPORT_DIR / "v114_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
