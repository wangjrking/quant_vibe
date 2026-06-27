from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
SOURCE_DIR = DATA_DIR / "reports" / "model_agent_current_best_second_or_expansion_v96_20260627"
SOURCE_SUMMARY = SOURCE_DIR / "v96_run_summary.json"
SOURCE_MATRIX = SOURCE_DIR / "v96_summary_matrix.csv"
SNAPSHOT_DIR = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260627_v97"
ROBUST_DIR = DATA_DIR / "reports" / "model_agent_current_best_robustness_20260627_v97"
PACKET_DIR = DATA_DIR / "reports" / "model_agent_research_promotion_packet_20260627_v98"


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_summary(table: str) -> dict[str, object]:
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   sum(case when pred_prob is null then 1 else 0 end)
            from {quote(table)}
            """
        ).fetchone()
        duplicate_key_groups = conn.execute(
            f"""
            select count(*) from (
                select trade_date, stock_code, count(*) c
                from {quote(table)}
                group by trade_date, stock_code
                having c > 1
            )
            """
        ).fetchone()[0]
        latest = str(row[2])
        latest_row = conn.execute(
            f"select count(*), count(distinct stock_code) from {quote(table)} where trade_date = ?",
            (latest,),
        ).fetchone()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": latest,
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(duplicate_key_groups),
        "latest_trade_date": latest,
        "latest_day_rows": int(latest_row[0]),
        "latest_day_stocks": int(latest_row[1]),
    }


def pick_metrics(best: dict[str, object]) -> dict[str, object]:
    keys = [
        "full_rank_ic_delta",
        "full_top1_delta",
        "full_top3_delta",
        "full_top5_delta",
        "recent63_rank_ic_delta",
        "recent63_top1_delta",
        "recent63_top3_delta",
        "recent63_top5_delta",
        "recent20_rank_ic_delta",
        "recent20_top1_delta",
        "recent20_top3_delta",
        "recent20_top5_delta",
        "positive_top5_months",
        "min_month_top5_delta",
    ]
    return {key: best[key] for key in keys if key in best}


def main() -> int:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    PACKET_DIR.mkdir(parents=True, exist_ok=True)

    source = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    current_best: dict[str, object] = {}
    robustness_rows = []
    integrity: dict[str, object] = {}
    for label, result in source["results"].items():
        best = result["best"]
        table = result["target_table"]
        summary = table_summary(table)
        integrity[label] = summary
        output_dir = SOURCE_DIR / label
        current_best[label] = {
            "asset": result["asset"],
            "table": table,
            "status": "research_only_promotable_formal_candidate",
            "promotion_gate_passed": bool(result["promotion_gate"]["hard_constraint_passed"]),
            "target_approval_status": result["promotion_gate"]["target_approval_status"],
            "condition": best["condition"],
            "metrics": pick_metrics(best),
            "coverage": summary,
            "evidence": {
                "summary": str(output_dir / f"{label}_v96_summary.json"),
                "gate": str(output_dir / "promotion_gate_result.json"),
                "candidate": str(output_dir / "promotion_candidate.json"),
                "manifest": str(output_dir / "research_candidate_manifest.json"),
            },
        }
        robustness_rows.append(
            {
                "label": label,
                "asset": result["asset"],
                "table": table,
                "eval_days": None,
                "eval_max_date": None,
                "full_rank_ic_delta": best["full_rank_ic_delta"],
                "full_top1_delta": best["full_top1_delta"],
                "full_top3_delta": best["full_top3_delta"],
                "full_top5_delta": best["full_top5_delta"],
                "recent63_rank_ic_delta": best["recent63_rank_ic_delta"],
                "recent63_top1_delta": best["recent63_top1_delta"],
                "recent63_top3_delta": best["recent63_top3_delta"],
                "recent63_top5_delta": best["recent63_top5_delta"],
                "recent20_rank_ic_delta": best["recent20_rank_ic_delta"],
                "recent20_top1_delta": best["recent20_top1_delta"],
                "recent20_top3_delta": best["recent20_top3_delta"],
                "recent20_top5_delta": best["recent20_top5_delta"],
                "month_top5_positive": best["positive_top5_months"],
                "month_top5_min": best["min_month_top5_delta"],
            }
        )

    snapshot = {
        "generated_at": guarded.now_iso(),
        "scope": "current_best_research_snapshot_v97",
        "objective": "v96 second OR expansion replaced v94 after all four labels improved comparable full-window Top1 and Top5 metrics, kept promotion gates passed, and recorded Top3 for this round.",
        "source_run_summary": str(SOURCE_SUMMARY),
        "source_matrix": str(SOURCE_MATRIX),
        "current_best": current_best,
        "decision": {
            "promotable_now": list(current_best.keys()),
            "still_needs_work": [],
            "note": "本快照仅为 research-only promotable formal candidate，不是 formal/L5/production 发布。",
        },
        "boundaries": {
            "research_only": True,
            "no_training_in_this_snapshot": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    snapshot_json = SNAPSHOT_DIR / "current_best_research_snapshot_v97.json"
    snapshot_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(robustness_rows).to_csv(ROBUST_DIR / "robustness_matrix_v97.csv", index=False, encoding="utf-8-sig")

    packet = {
        "generated_at": guarded.now_iso(),
        "scope": "research_promotion_packet_v98",
        "source_snapshot": str(snapshot_json),
        "source_robustness_matrix": str(ROBUST_DIR / "robustness_matrix_v97.csv"),
        "target_status": "approved_for_l4_candidate_only",
        "not_production": True,
        "not_l5_approved": True,
        "boundaries": snapshot["boundaries"],
        "candidates": current_best,
        "robustness_records": robustness_rows,
        "table_integrity": integrity,
    }
    packet_json = PACKET_DIR / "promotion_packet_v98.json"
    packet_json.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    (PACKET_DIR / "table_integrity_check_v98.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 研究候选晋升包 v98",
        "",
        "## 结论",
        "",
        "v96 第二层 OR 扩展在四个标签上均通过研究准入硬约束，并相对 v94 提高可比口径下的全窗口 Top1 和 Top5；本轮同时补充记录 Top3。当前仅建议作为 `approved_for_l4_candidate_only` 送审候选，不是 formal/L5/production 发布。",
        "",
        "## 边界",
        "",
        "- 未训练模型。",
        "- 未修改 formal manifest。",
        "- 未修改 production manifest。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
        "",
        "## 候选摘要",
        "",
    ]
    matrix = pd.read_csv(SOURCE_MATRIX)
    for row in matrix.to_dict("records"):
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 资产：`{row['asset']}`",
                f"- 表：`{row['table']}`",
                f"- 条件：`{row['condition']}`",
                f"- 触发日：`{int(row['base_active_days'])}` -> `{int(row['active_days'])}`",
                f"- 全窗口 Top1 delta：`{float(row['full_top1_delta']):.10f}`",
                f"- 全窗口 Top3 delta：`{float(row['full_top3_delta']):.10f}`",
                f"- 全窗口 Top5 delta：`{float(row['full_top5_delta']):.10f}`",
                f"- 最新覆盖日：`{row['latest_trade_date']}`，最新日行数：`{int(row['latest_day_rows'])}`",
                "",
            ]
        )
    (PACKET_DIR / "promotion_packet_v98.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    snapshot_lines = [
        "# 当前最优研究快照 v97",
        "",
        "v97 采用 v96 第二层 OR 扩展结果作为当前研究最优。该快照只用于模型侧研究和送审准备。",
        "",
    ]
    for label, item in current_best.items():
        metrics = item["metrics"]
        coverage = item["coverage"]
        snapshot_lines.extend(
            [
                f"## {label}",
                "",
                f"- 资产：`{item['asset']}`",
                f"- 表：`{item['table']}`",
                f"- 条件：`{item['condition']}`",
                f"- 全窗口 Top1 delta：`{metrics['full_top1_delta']:.10f}`",
                f"- 全窗口 Top3 delta：`{metrics['full_top3_delta']:.10f}`",
                f"- 全窗口 Top5 delta：`{metrics['full_top5_delta']:.10f}`",
                f"- 覆盖：`{coverage['min_trade_date']}` 到 `{coverage['max_trade_date']}`，`{coverage['row_count']}` 行。",
                "",
            ]
        )
    (SNAPSHOT_DIR / "current_best_research_snapshot_v97.md").write_text("\n".join(snapshot_lines) + "\n", encoding="utf-8")

    print(json.dumps({"snapshot": str(snapshot_json), "packet": str(packet_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
