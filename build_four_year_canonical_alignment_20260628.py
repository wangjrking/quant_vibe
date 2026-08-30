from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_canonical_alignment_20260628"

LATEST_BESTSET_V10 = DATA_DIR / "reports" / "model_agent_four_year_latest_bestset_status_v10_20260628" / "latest_bestset_status_v10.json"
CANDIDATE_STATUS = DATA_DIR / "reports" / "model_agent_four_year_candidate_status_20260628" / "four_year_candidate_status.json"
PRIORITY_REVIEW = DATA_DIR / "reports" / "model_agent_four_year_candidate_status_20260628" / "candidate_priority_review_20260628.json"
SCAN_OLD = DATA_DIR / "reports" / "model_agent_four_year_10d_shortlist_scan_20260628" / "shortlist_scan_summary.json"
SCAN_CURRENT = DATA_DIR / "reports" / "model_agent_four_year_10d_shortlist_scan_vs_clear_replacement_20260628" / "shortlist_scan_summary.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_summary() -> dict:
    v10 = _load_json(LATEST_BESTSET_V10)
    candidate_status = _load_json(CANDIDATE_STATUS)
    priority = _load_json(PRIORITY_REVIEW)
    scan_old = _load_json(SCAN_OLD)
    scan_current = _load_json(SCAN_CURRENT)

    v10_map = {row["label"]: row for row in v10["best_set"]}
    current_map = candidate_status["current_best_candidates"]
    priority_map = {row["label"]: row for row in priority["rows"]}

    authoritative = []
    for label, row in current_map.items():
        old = v10_map.get(label)
        priority_row = priority_map.get(label, {})
        authoritative.append(
            {
                "label": label,
                "authoritative_asset": row["asset"],
                "authoritative_table": row["table"],
                "authoritative_decision": row["decision"],
                "target_approval_status": row["target_approval_status"],
                "full_rank_ic_delta": row["full_rank_ic_delta"],
                "full_top5_delta": row["full_top5_delta"],
                "recent63_top5_delta": row["recent63_top5_delta"],
                "recent20_top5_delta": row["recent20_top5_delta"],
                "v10_asset": None if old is None else old["asset"],
                "v10_table": None if old is None else old["table"],
                "changed_since_v10": False if old is None else old["table"] != row["table"],
                "selection_axis": priority_row.get("selection_axis"),
                "requires_formal_explanation": priority_row.get("requires_formal_explanation"),
                "note": priority_row.get("note"),
            }
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "four_year_canonical_alignment_after_candidate_status_refresh",
        "authoritative_source": str(CANDIDATE_STATUS),
        "priority_source": str(PRIORITY_REVIEW),
        "v10_source": str(LATEST_BESTSET_V10),
        "authoritative_candidates": authoritative,
        "ten_d_scan_alignment": {
            "outdated_scan_current_table": scan_old["current_table"],
            "current_scan_current_table": scan_current["current_table"],
            "current_10d_recent20_rank_ic": scan_current["current_metrics"]["recent20_rank_ic"],
            "current_10d_recent63_rank_ic": scan_current["current_metrics"]["recent63_rank_ic"],
            "current_10d_recent20_top5": scan_current["current_metrics"]["recent20_top5"],
            "current_10d_recent63_top5": scan_current["current_metrics"]["recent63_top5"],
            "best_blend_candidate": scan_current["blend_top5"][0],
            "top_shortlist_candidate": scan_current["shortlist_top5"][0],
        },
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_write": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def _build_report(summary: dict) -> str:
    lines = [
        "# 四年观察期 canonical 对齐报告",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前 authoritative 口径",
        "",
        f"- authoritative 来源：`{summary['authoritative_source']}`",
        f"- policy/priority 来源：`{summary['priority_source']}`",
        f"- 较早 bestset 快照：`{summary['v10_source']}`",
        "",
        "结论：后续四年观察期优化、短名单扫描和候选讨论，应以 `four_year_candidate_status_20260628` 为当前 authoritative canonical 集，而不是继续沿用较早的 `latest_bestset_status_v10_20260628`。",
        "",
        "## 各标签当前 canonical",
        "",
    ]
    for row in summary["authoritative_candidates"]:
        change_note = "已切换" if row["changed_since_v10"] else "未变更"
        lines.extend(
            [
                f"- `{row['label']}`",
                f"  - 当前 canonical：`{row['authoritative_asset']}`",
                f"  - 当前表：`{row['authoritative_table']}`",
                f"  - 相对 v10：`{change_note}`",
                f"  - v10 旧表：`{row['v10_table']}`",
                f"  - 选择轴：`{row['selection_axis']}`",
                f"  - Full RankIC delta：`{row['full_rank_ic_delta']:+.6f}`",
                f"  - Full Top5 delta：`{row['full_top5_delta']:+.6f}`",
                f"  - Recent63 Top5 delta：`{row['recent63_top5_delta']:+.6f}`",
                f"  - Recent20 Top5 delta：`{row['recent20_top5_delta']:+.6f}`",
                f"  - formal 解释要求：`{row['requires_formal_explanation']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 10D 对齐结论",
            "",
            f"- 旧扫描基准：`{summary['ten_d_scan_alignment']['outdated_scan_current_table']}`",
            f"- 当前扫描基准：`{summary['ten_d_scan_alignment']['current_scan_current_table']}`",
            f"- 当前 10D Recent20 RankIC：`{summary['ten_d_scan_alignment']['current_10d_recent20_rank_ic']:+.6f}`",
            f"- 当前 10D Recent63 RankIC：`{summary['ten_d_scan_alignment']['current_10d_recent63_rank_ic']:+.6f}`",
            f"- 当前 10D Recent20 Top5：`{summary['ten_d_scan_alignment']['current_10d_recent20_top5']:+.6f}`",
            f"- 当前 10D Recent63 Top5：`{summary['ten_d_scan_alignment']['current_10d_recent63_top5']:+.6f}`",
            "",
            "结论：我前一轮 10D 短名单扫描使用的是较早的 `active_recent_gate_hybrid_v2` 基准；现在已补做以 `clear_replacement_time_splice` 为基准的扫描，后续 10D 优化应统一以新基准为准。",
            "",
            "## 10D 当前可执行判断",
            "",
            f"- 短名单里最强的直接替换候选：`{summary['ten_d_scan_alignment']['top_shortlist_candidate']['candidate']}`",
            f"  - Recent20 RankIC 相对当前：`{summary['ten_d_scan_alignment']['top_shortlist_candidate']['delta_vs_current_recent20_abs_rank_ic']:+.6f}`",
            f"  - Recent20 Top5 相对当前：`{summary['ten_d_scan_alignment']['top_shortlist_candidate']['delta_vs_current_recent20_abs_top5']:+.6f}`",
            f"- 最优 blend：`{summary['ten_d_scan_alignment']['best_blend_candidate']['candidate']}` / `weight_current={summary['ten_d_scan_alignment']['best_blend_candidate']['weight_current']}`",
            f"  - Full RankIC 相对当前：`{summary['ten_d_scan_alignment']['best_blend_candidate']['delta_full_rank_ic']:+.6f}`",
            f"  - Full Top5 相对当前：`{summary['ten_d_scan_alignment']['best_blend_candidate']['delta_full_top5']:+.6f}`",
            f"  - Recent20 RankIC 相对当前：`{summary['ten_d_scan_alignment']['best_blend_candidate']['delta_recent20_rank_ic']:+.6f}`",
            f"  - Recent20 Top5 相对当前：`{summary['ten_d_scan_alignment']['best_blend_candidate']['delta_recent20_top5']:+.6f}`",
            "",
            "结论：当前仍没有发现可以直接整体替换 `clear_replacement_time_splice` 的更优上游；最有价值的下一步仍是做日级门控，而不是做全量替换。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _build_summary()
    (REPORT_DIR / "canonical_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "canonical_alignment_report.md").write_text(
        _build_report(summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "report_dir": str(REPORT_DIR),
                "summary_json": str(REPORT_DIR / "canonical_alignment_summary.json"),
                "report_md": str(REPORT_DIR / "canonical_alignment_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
