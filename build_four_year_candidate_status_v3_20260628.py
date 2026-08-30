from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"

BESTSET_V11_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_latest_bestset_status_v11_20260628"
    / "latest_bestset_status_v11.json"
)
RETRAIN_V3_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_1d_fixed4y_latestfold_retrain_v3_20260628"
    / "latestfold_retrain_v3_summary.json"
)
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_four_year_candidate_status_v3_20260628"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def main() -> int:
    bestset = json.loads(BESTSET_V11_PATH.read_text(encoding="utf-8"))
    retrain = json.loads(RETRAIN_V3_PATH.read_text(encoding="utf-8"))

    one_d = dict(bestset["details"]["1d"]["record"])
    best_candidate = retrain["best_candidate"]
    baseline_bestset = next(row for row in retrain["baselines"] if row["candidate"] == "current_four_year_bestset")
    baseline_control = next(row for row in retrain["baselines"] if row["candidate"] == "current_four_year_control")

    one_d["source_note"] = (
        "1D 继续保留四年 splice_condblend。"
        "最新 fixed4y latestfold retrain v3 在 20251208-20260611 最近三折窗口下"
        "明显输给当前四年 bestset，因此不替换。"
    )
    one_d["next_action"] = (
        "1D 后续如继续优化，应停止 recent63/recent84 轻量重训同类扫描。"
        "下一轮优先考虑标签/目标分层或结构型候选，不再继续当前 fixed4y recent-tail 配方。"
    )

    status = {
        "generated_at": now_iso(),
        "scope": "four_year_candidate_status_v3_after_1d_latestfold_retrain_v3",
        "current_best_candidates": {
            "executable_1d_open_return": {
                "label": "executable_1d_open_return",
                "asset": "research_1d_four_year_splice_condblend_20260628",
                "table": one_d["table"],
                "decision": "keep_existing_bestset",
                "target_approval_status": "approved_for_l4_candidate_only_after_four_year_observation",
                "full_rank_ic_delta_vs_control": one_d["full_rank_ic_delta"],
                "full_top5_delta_vs_control": one_d["full_top5_delta"],
                "recent63_top5_delta_vs_control": one_d["recent63_top5_delta"],
                "recent20_top5_delta_vs_control": one_d["recent20_top5_delta"],
                "latestfold_v3_window": retrain["window"],
                "latestfold_v3_best_candidate": best_candidate["candidate"],
                "latestfold_v3_objective": best_candidate["objective"],
                "latestfold_v3_rank_ic": best_candidate["rank_ic"],
                "latestfold_v3_top1": best_candidate["top1"],
                "latestfold_v3_top5": best_candidate["top5"],
                "latestfold_v3_objective_delta_vs_bestset": float(best_candidate["objective"] - baseline_bestset["objective"]),
                "latestfold_v3_top1_delta_vs_bestset": float(best_candidate["top1"] - baseline_bestset["top1"]),
                "latestfold_v3_top5_delta_vs_bestset": float(best_candidate["top5"] - baseline_bestset["top5"]),
                "latestfold_v3_objective_delta_vs_control": float(best_candidate["objective"] - baseline_control["objective"]),
                "replaced_asset": "research_1d_four_year_splice_condblend_20260628",
                "replaced_table": one_d["table"],
                "candidate_json": str(
                    DATA_DIR
                    / "reports"
                    / "model_agent_four_year_1d_splice_condblend_20260628"
                    / "promotion_candidate.json"
                ),
                "latestfold_v3_summary_json": str(RETRAIN_V3_PATH),
                "note": one_d["source_note"],
            },
            "executable_3d_open_return": bestset["details"]["3d"]["record"],
            "executable_5d_open_return": bestset["details"]["5d"]["record"],
            "executable_10d_open_return": bestset["details"]["10d"]["record"],
        },
        "boundaries": {
            "research_only": True,
            "no_new_formal_manifest_change": True,
            "no_new_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "four_year_candidate_status_v3.json"
    md_path = OUTPUT_DIR / "four_year_candidate_status_v3.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 四年观察候选状态 v3",
        "",
        "## 当前结论",
        "",
        "- 1D 不替换，继续保留 `research_1d_four_year_splice_condblend_20260628`。",
        "- 原因：最新 `fixed4y latestfold retrain v3` 在最近三折窗口 `20251208-20260611` 下，",
        "  相对当前 1D 四年 bestset 的 objective、RankIC、Top1、Top5 全部明显更弱。",
        "- 3D / 5D / 10D 本轮不变，继续沿用 v11 bestset 结论。",
        "",
        "## 1D 最新对比",
        "",
        f"- 当前四年 bestset objective：`{baseline_bestset['objective']:.6f}`",
        f"- latestfold retrain v3 最优 objective：`{best_candidate['objective']:.6f}`",
        f"- objective 差值：`{best_candidate['objective'] - baseline_bestset['objective']:.6f}`",
        f"- Top1 差值：`{best_candidate['top1'] - baseline_bestset['top1']:.6f}`",
        f"- Top5 差值：`{best_candidate['top5'] - baseline_bestset['top5']:.6f}`",
        "",
        "## 边界",
        "",
        "- 本次仅更新 research-only 候选状态。",
        "- 未修改 formal manifest。",
        "- 未修改 production manifest。",
        "- 未生成交易信号。",
        "- 未运行回测。",
        "",
        "## 证据路径",
        "",
        f"- `{BESTSET_V11_PATH}`",
        f"- `{RETRAIN_V3_PATH}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
