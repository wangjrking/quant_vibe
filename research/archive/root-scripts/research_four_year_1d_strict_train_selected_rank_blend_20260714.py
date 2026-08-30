from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

import research_four_year_top_consensus_holdout_20260714 as base


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_1d_strict_train_selected_rank_blend_20260714"
)
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_1d_strict_train_selected_rank_blend_20260714.duckdb"
)

LABEL_KEY = "1d"
LABEL_NAME = base.LABELS[LABEL_KEY]
TRAIN_END = "20251231"
HOLDOUT_START = "20260101"
WEIGHTS = [i / 100 for i in range(0, 101)]


def _score(frame: pd.DataFrame, w1: float) -> pd.Series:
    return w1 * frame["pred_1d_rank"] + (1.0 - w1) * frame["pred_10d_rank"]


def _evaluate(frame: pd.DataFrame, w1: float) -> dict[str, Any]:
    work = frame[frame[f"label_{LABEL_KEY}"].notna()].copy()
    work["candidate_score"] = _score(work, w1)
    daily = base._daily_eval(work, "candidate_score", f"label_{LABEL_KEY}")
    summary = base._summarize_daily(daily)
    summary["weight"] = {"w1": w1, "w10": 1.0 - w1}
    summary["train_objective_top5"] = summary["selection"].get("top5")
    return summary


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return base._delta(candidate, baseline)


def _strict_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    passed, reasons = base._improvement_gate(candidate, baseline)
    delta = _delta(candidate, baseline)
    if (delta.get("holdout_rank_ic") or 0.0) < -0.0015:
        reasons.append("holdout_rank_ic_delta_below_-0_0015")
    if (delta.get("recent63_rank_ic") or 0.0) < -0.006:
        reasons.append("recent63_rank_ic_delta_below_-0_006")
    return not reasons, reasons


def _materialize(full_frame: pd.DataFrame, w1: float) -> dict[str, Any]:
    table = "stock_predict_data_model_agent_four_year_1d_strict_train_selected_rank_blend_20260714_research"
    out = full_frame[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = _score(full_frame, w1).groupby(full_frame["trade_date"]).rank(method="average", pct=True)
    out["score_source"] = "strict_train_selected_rank_blend_20260714_research"
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_frame", out)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM candidate_frame")
        latest_date = con.execute(f"SELECT max(trade_date) FROM {table}").fetchone()[0]
        quality = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(distinct trade_date) AS trade_days,
                count(distinct stock_code) AS stock_count,
                sum(case when pred_prob is null then 1 else 0 end) AS null_pred_prob,
                sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows,
                (
                    SELECT count(*)
                    FROM (
                        SELECT trade_date, stock_code, count(*) c
                        FROM {table}
                        GROUP BY 1,2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups,
                sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
                count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {table}
            """,
            [latest_date, latest_date],
        ).fetchdf().iloc[0].to_dict()
        digest = hashlib.sha256()
        for trade_date, stock_code, pred_prob in con.execute(
            f"SELECT trade_date, stock_code, pred_prob FROM {table} ORDER BY trade_date, stock_code"
        ).fetchall():
            digest.update(f"{trade_date}|{stock_code}|{float(pred_prob):.17g}\n".encode("utf-8"))
    return {
        "db_path": str(EXPERIMENT_DB),
        "table": table,
        "quality": {key: base._to_float(value) for key, value in quality.items()},
        "pred_prob_sha256": digest.hexdigest(),
    }


def _flat_scan_row(item: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    delta = _delta(item, baseline)
    passed, failures = _strict_gate(item, baseline)
    return {
        "w1": item["weight"]["w1"],
        "w10": item["weight"]["w10"],
        "train_top5": item["selection"].get("top5"),
        "strict_gate_passed": passed,
        "strict_gate_failures": ";".join(failures),
        **{f"delta_{key}": value for key, value in delta.items()},
        "candidate_full_rank_ic": item["full"].get("rank_ic"),
        "candidate_full_top5": item["full"].get("top5"),
        "candidate_holdout_rank_ic": item["holdout"].get("rank_ic"),
        "candidate_holdout_top5": item["holdout"].get("top5"),
        "candidate_recent63_rank_ic": item["recent63"].get("rank_ic"),
        "candidate_recent63_top5": item["recent63"].get("top5"),
        "candidate_recent20_rank_ic": item["recent20"].get("rank_ic"),
        "candidate_recent20_top5": item["recent20"].get("top5"),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = base._add_rank_columns(base._read_scores(with_labels=True))
    full = base._add_rank_columns(base._read_scores(with_labels=False))

    baseline = _evaluate(labeled, 1.0)
    candidates = [_evaluate(labeled, w1) for w1 in WEIGHTS]
    # Strict selection: only use the pre-2026 selection window objective.
    selected = sorted(candidates, key=lambda item: item["selection"].get("top5") or -1.0, reverse=True)[0]
    selected_passed, selected_failures = _strict_gate(selected, baseline)
    selected_delta = _delta(selected, baseline)
    materialized = _materialize(full, selected["weight"]["w1"])

    scan_csv = REPORT_DIR / "strict_train_selected_weight_scan.csv"
    pd.DataFrame([_flat_scan_row(item, baseline) for item in candidates]).to_csv(
        scan_csv, index=False, encoding="utf-8-sig"
    )

    payload = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "actor": "model-agent",
        "scope": "research_only_1d_strict_train_selected_rank_blend",
        "label": LABEL_NAME,
        "selection_rule": "Select w1/w10 only by selection-window Top5 over 20220606-20251231. Holdout/recent metrics are not used for selection.",
        "selection_window": {"min_trade_date": base.START_DATE, "max_trade_date": TRAIN_END},
        "holdout_window": {"min_trade_date": HOLDOUT_START, "max_trade_date": "20260611"},
        "baseline_current_formal": baseline,
        "selected_weight": selected["weight"],
        "selected_metrics": selected,
        "selected_delta_vs_baseline": selected_delta,
        "strict_gate": {"passed": selected_passed, "failed_reasons": selected_failures},
        "materialized_candidate": materialized,
        "scan_csv": str(scan_csv),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "strict_train_selected_rank_blend_report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D 严格训练窗口 rank-blend 复核",
        "",
        "## 结论",
        "",
        "本轮只用 `20220606-20251231` 训练窗口 Top5 选择权重，不使用 2026 holdout 或近期窗口参与选择。",
        "",
        f"- 选中权重：`w1={selected['weight']['w1']:.2f}` / `w10={selected['weight']['w10']:.2f}`",
        f"- 严格改进门：{'通过' if selected_passed else '未通过'}",
        f"- Full RankIC Δ：`{selected_delta.get('full_rank_ic'):.6f}`",
        f"- Full Top5 Δ：`{selected_delta.get('full_top5'):.6f}`",
        f"- Holdout RankIC Δ：`{selected_delta.get('holdout_rank_ic'):.6f}`",
        f"- Holdout Top5 Δ：`{selected_delta.get('holdout_top5'):.6f}`",
        f"- Recent63 Top5 Δ：`{selected_delta.get('recent63_top5'):.6f}`",
        f"- Recent20 Top5 Δ：`{selected_delta.get('recent20_top5'):.6f}`",
        "",
        "## 边界",
        "",
        "- 未训练模型。",
        "- 未修改 formal manifest。",
        "- 未生成 `approved_for_l5` 资产。",
        "- 未生成策略信号，未跑回测。",
        "",
        "## 证据",
        "",
        f"- JSON：`{report_json}`",
        f"- 权重扫描 CSV：`{scan_csv}`",
        f"- 实验库：`{EXPERIMENT_DB}`",
    ]
    report_md = REPORT_DIR / "strict_train_selected_rank_blend_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "report_json": str(report_json), "candidate_db": str(EXPERIMENT_DB)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
