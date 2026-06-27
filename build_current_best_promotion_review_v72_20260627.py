from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from score_model_promotion_candidate import evaluate_candidate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MAIN_DIR = ROOT / "quant" / "main"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
FACTOR_DIR = DATA_DIR / "production_factor_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_current_best_promotion_review_20260627_v72"
CONSTRAINTS = MAIN_DIR / "config" / "model_promotion_constraints_v1_20260625.json"


SOURCES = {
    "executable_1d_open_return": {
        "asset": "research_1d_exante_iqr_gate_v67_20260627",
        "table": "stock_predict_data_model_agent_1d_exante_iqr_gate_v67_20260627_executable_1d_open_return_research",
        "scan": DATA_DIR / "reports" / "model_agent_1d_exante_score_state_gate_v66_20260627" / "exante_score_state_gate_summary.json",
        "asset_summary": DATA_DIR / "reports" / "model_agent_1d_exante_score_state_asset_v67_20260627" / "asset_summary.json",
        "manifest": DATA_DIR / "reports" / "model_agent_1d_exante_score_state_asset_v67_20260627" / "research_candidate_manifest.json",
    },
    "executable_3d_open_return": {
        "asset": "research_3d_exante_std_gate_v69_20260627",
        "table": "stock_predict_data_model_agent_3d_exante_std_gate_v69_20260627_executable_3d_open_return_research",
        "scan": DATA_DIR / "reports" / "model_agent_3d_exante_score_state_gate_v68_20260627" / "exante_score_state_gate_summary.json",
        "asset_summary": DATA_DIR / "reports" / "model_agent_3d_exante_score_state_asset_v69_20260627" / "asset_summary.json",
        "manifest": DATA_DIR / "reports" / "model_agent_3d_exante_score_state_asset_v69_20260627" / "research_candidate_manifest.json",
    },
    "executable_5d_open_return": {
        "asset": "research_5d_recent_top_condblend_latest_20260626",
        "table": "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research",
        "scan": DATA_DIR / "reports" / "model_agent_5d_exante_score_state_gate_v70_20260627" / "exante_score_state_gate_summary.json",
        "manifest": None,
    },
    "executable_10d_open_return": {
        "asset": "research_10d_top5_safe_gate_v58_20260627",
        "table": "stock_predict_data_model_agent_10d_top5_safe_gate_v58_20260627_executable_10d_open_return_research",
        "scan": DATA_DIR / "reports" / "model_agent_10d_secondgate_exante_scan_v71_20260627" / "exante_score_state_gate_summary.json",
        "manifest": None,
    },
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def production_factor_latest() -> tuple[str, int]:
    frame = pd.read_parquet(FACTOR_DIR, columns=["trade_date", "stock_code"])
    frame["trade_date"] = frame["trade_date"].astype(str)
    latest = str(frame["trade_date"].max())
    rows = int(frame.loc[frame["trade_date"] == latest, "stock_code"].nunique())
    return latest, rows


def db_summary(table: str) -> dict[str, object]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   sum(case when pred_prob is null then 1 else 0 end)
            from {table}
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from {table}
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]
        latest = conn.execute(
            f"""
            select trade_date, count(*) rows, count(distinct stock_code) stocks
            from {table}
            group by trade_date
            order by trade_date desc
            limit 1
            """
        ).fetchone()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_trade_date": str(latest[0]),
        "latest_day_rows": int(latest[1]),
        "latest_day_stocks": int(latest[2]),
    }


def make_candidate(label: str, source: dict[str, object], expected_latest: str, expected_rows: int) -> dict[str, object]:
    scan = json.loads(Path(source["scan"]).read_text(encoding="utf-8"))
    best = scan["best"]
    table = str(source["table"])
    db = db_summary(table)
    return {
        "label": label,
        "asset": source["asset"],
        "table": table,
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "feature_input": "quant/data_file/production_factor_parts/",
        "label_input": "quant/data_file/prediction_label_parts/",
        "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
        "forbidden_inputs_present": False,
        "min_trade_date": db["min_trade_date"],
        "latest_trade_date": db["latest_trade_date"],
        "expected_latest_trade_date": expected_latest,
        "latest_day_rows": db["latest_day_rows"],
        "expected_latest_day_rows": expected_rows,
        "null_pred_prob": db["null_pred_prob"],
        "duplicate_key_groups": db["duplicate_key_groups"],
        "is_train_candidate": False,
        "evaluation_report_present": Path(source["scan"]).exists(),
        "candidate_manifest_present": bool(source.get("manifest") and Path(source["manifest"]).exists()),
        "full_rank_ic_delta": best.get("full_rank_ic_delta"),
        "full_top1_delta": best.get("full_top1_delta"),
        "full_top5_delta": best.get("full_top5_delta"),
        "recent63_rank_ic_delta": best.get("recent63_rank_ic_delta", scan.get("raw_candidate_delta_vs_base", {}).get("recent63", {}).get("rank_ic")),
        "recent63_top1_delta": best.get("recent63_top1_delta"),
        "recent63_top5_delta": best.get("recent63_top5_delta"),
        "recent20_rank_ic_delta": best.get("recent20_rank_ic_delta", scan.get("raw_candidate_delta_vs_base", {}).get("recent20", {}).get("rank_ic")),
        "recent20_top1_delta": best.get("recent20_top1_delta"),
        "recent20_top5_delta": best.get("recent20_top5_delta"),
        "positive_top5_periods": best.get("positive_top5_months"),
        "min_period_top5_delta": best.get("min_month_top5_delta"),
        "full_top10_delta": best.get("full_top10_delta"),
        "min_period_rank_ic_delta": best.get("min_month_rank_ic_delta"),
        "prefer_simpler_formula": "score-state gate",
        "prefer_fewer_active_days": best.get("active_days"),
        "prefer_fewer_upstream_sources": "base+candidate score tables",
        "prefer_clearer_roll_back_path": "keep previous active research table unchanged",
        "evidence": {
            "scan_summary": str(source["scan"]),
            "asset_summary": str(source.get("asset_summary") or ""),
            "candidate_manifest": str(source.get("manifest") or ""),
            "db_summary": db,
        },
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_latest, expected_rows = production_factor_latest()
    config = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    rows = []
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_current_best_promotion_review_v72",
        "constraints": str(CONSTRAINTS),
        "expected_latest_trade_date": expected_latest,
        "expected_latest_day_rows": expected_rows,
        "results": {},
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
    for label, source in SOURCES.items():
        candidate = make_candidate(label, source, expected_latest, expected_rows)
        candidate_path = REPORT_DIR / f"{label}_candidate_v72.json"
        candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        result = evaluate_candidate(candidate, config)
        result_path = REPORT_DIR / f"{label}_scored_v72.json"
        result_path.write_text(
            json.dumps({"candidate": str(candidate_path), "result": result, "candidate_payload": candidate}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload["results"][label] = {
            "candidate_json": str(candidate_path),
            "scored_json": str(result_path),
            "result": result,
        }
        rows.append(
            {
                "label": label,
                "asset": candidate["asset"],
                "table": candidate["table"],
                "hard_constraint_passed": result["hard_constraint_passed"],
                "target_approval_status": result["target_approval_status"],
                "failed_hard_constraints": ";".join(result["failed_hard_constraints"]),
                "full_rank_ic_delta": candidate.get("full_rank_ic_delta"),
                "full_top1_delta": candidate.get("full_top1_delta"),
                "full_top5_delta": candidate.get("full_top5_delta"),
                "recent63_top1_delta": candidate.get("recent63_top1_delta"),
                "recent63_top5_delta": candidate.get("recent63_top5_delta"),
                "recent20_top1_delta": candidate.get("recent20_top1_delta"),
                "recent20_top5_delta": candidate.get("recent20_top5_delta"),
                "positive_top5_periods": candidate.get("positive_top5_periods"),
                "min_period_top5_delta": candidate.get("min_period_top5_delta"),
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(REPORT_DIR / "current_best_promotion_matrix_v72.csv", index=False, encoding="utf-8-sig")
    payload["ready_for_l4_candidate_review"] = matrix.loc[matrix["hard_constraint_passed"], "label"].tolist()
    payload["continue_research"] = matrix.loc[~matrix["hard_constraint_passed"], "label"].tolist()
    (REPORT_DIR / "current_best_promotion_review_v72.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 当前最优研究模型晋升复核 v72",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 可进入 `approved_for_l4_candidate_only` 送审准备：`{', '.join(payload['ready_for_l4_candidate_review']) or '无'}`",
        f"- 继续研究：`{', '.join(payload['continue_research']) or '无'}`",
        "",
        "## 矩阵",
        "",
        "| 标签 | 结果 | 失败硬约束 | full Top1 | full Top5 | recent63 Top1 | recent63 Top5 | positive Top5 periods |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {status} | {failed} | {full_top1_delta} | {full_top5_delta} | {recent63_top1_delta} | {recent63_top5_delta} | {positive_top5_periods} |".format(
                label=row["label"],
                status="通过" if row["hard_constraint_passed"] else "不通过",
                failed=row["failed_hard_constraints"] or "-",
                full_top1_delta=row["full_top1_delta"],
                full_top5_delta=row["full_top5_delta"],
                recent63_top1_delta=row["recent63_top1_delta"],
                recent63_top5_delta=row["recent63_top5_delta"],
                positive_top5_periods=row["positive_top5_periods"],
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- research-only",
            "- 未训练模型",
            "- 未写入新预测表",
            "- 未修改 formal manifest",
            "- 未修改 production manifest",
            "- 未生成交易信号",
            "- 未运行策略回测",
        ]
    )
    (REPORT_DIR / "current_best_promotion_review_v72.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
