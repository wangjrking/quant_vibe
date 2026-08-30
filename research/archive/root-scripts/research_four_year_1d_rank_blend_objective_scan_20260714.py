from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_1d_rank_blend_objective_scan_20260714"
)
SOURCE_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "model_agent_four_year_1d_rank_blend_holdout_20260714"
)
EXPERIMENT_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_1d_rank_blend_objective_scan_20260714.duckdb"
)
LABEL_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
START_DATE = "20220606"
TRAIN_END_DATE = "20251231"
HOLDOUT_START_DATE = "20260101"
LABEL = "executable_1d_open_return"
TOP_K = [1, 3, 5, 10, 20]
TARGET_TABLE = "stock_predict_data_model_agent_four_year_1d_rank_blend_objective_scan_20260714_research"
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def load_manifest(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(MAIN))
    from prediction_manifest import load_prediction_source_manifest

    return load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)


def read_scores(*, with_label: bool) -> pd.DataFrame:
    manifests = {key: load_manifest(path) for key, path in MANIFESTS.items()}
    with duckdb.connect(database=":memory:") as con:
        con.execute(f"ATTACH '{Path(manifests['1d']['db_path']).as_posix()}' AS p1 (READ_ONLY)")
        con.execute(f"ATTACH '{Path(manifests['10d']['db_path']).as_posix()}' AS p10 (READ_ONLY)")
        if with_label:
            con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
            label_select = f", labels.{LABEL} AS label_value"
            label_join = f"JOIN labels.{LABEL_TABLE} labels USING (trade_date, stock_code)"
        else:
            label_select = ""
            label_join = ""
        return con.execute(
            f"""
            SELECT
                p1.trade_date,
                p1.stock_code,
                p1.pred_prob AS pred_1d,
                p10.pred_prob AS pred_10d
                {label_select}
            FROM p1.{manifests['1d']['table']} p1
            JOIN p10.{manifests['10d']['table']} p10 USING (trade_date, stock_code)
            {label_join}
            WHERE p1.trade_date >= ?
              AND p1.stock_code NOT LIKE '%.BJ'
              AND p1.pred_prob IS NOT NULL
              AND p10.pred_prob IS NOT NULL
            ORDER BY p1.trade_date, p1.stock_code
            """,
            [START_DATE],
        ).fetchdf()


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rank_1d"] = out.groupby("trade_date")["pred_1d"].rank(method="average", pct=True)
    out["rank_10d"] = out.groupby("trade_date")["pred_10d"].rank(method="average", pct=True)
    return out


def daily_eval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group[group["label_value"].notna()]
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort")
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rank_ic": to_float(ordered[score_col].corr(ordered["label_value"], method="spearman")),
        }
        for top_k in TOP_K:
            row[f"top{top_k}"] = to_float(ordered.head(top_k)["label_value"].mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise RuntimeError("no daily evaluation rows")
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    return daily


def metrics(daily: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "trade_days": int(len(daily)),
        "rank_ic": to_float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": to_float((daily["rank_ic"] > 0).mean()),
    }
    for top_k in TOP_K:
        result[f"top{top_k}"] = to_float(daily[f"top{top_k}"].mean())
        result[f"top{top_k}_positive_ratio"] = to_float((daily[f"top{top_k}"] > 0).mean())
    return result


def period_stability(daily: pd.DataFrame, by: str) -> dict[str, Any]:
    cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    grouped = daily.groupby(by, sort=True)[cols].mean().reset_index()
    out = {"periods": int(len(grouped))}
    for col in cols:
        out[f"{col}_positive_period_ratio"] = to_float((grouped[col] > 0).mean())
        out[f"min_{col}"] = to_float(grouped[col].min())
    return out


def summarize(daily: pd.DataFrame) -> dict[str, Any]:
    out = {
        "eval_min_trade_date": str(daily["trade_date"].min()),
        "eval_max_trade_date": str(daily["trade_date"].max()),
        "eval_trade_days": int(len(daily)),
        "full": metrics(daily),
        "selection": metrics(daily[daily["trade_date"] <= TRAIN_END_DATE]),
        "holdout": metrics(daily[daily["trade_date"] >= HOLDOUT_START_DATE]),
        "recent20": metrics(daily.tail(20)),
        "recent63": metrics(daily.tail(63)),
        "annual_stability": period_stability(daily, "year"),
        "monthly_stability": period_stability(daily, "month"),
    }
    out["four_year_gate"] = four_year_gate(out)
    return out


def four_year_gate(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if summary["eval_trade_days"] < 950:
        reasons.append("eval_trade_days_below_950")
    if (summary["full"].get("rank_ic") or 0.0) <= 0.0:
        reasons.append("full_rank_ic_non_positive")
    if (summary["full"].get("top5") or 0.0) <= 0.0:
        reasons.append("full_top5_non_positive")
    if (summary["recent63"].get("top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_non_positive")
    if (summary["recent20"].get("top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_non_positive")
    if (summary["monthly_stability"].get("top5_positive_period_ratio") or 0.0) < 0.5:
        reasons.append("monthly_top5_positive_ratio_below_0_5")
    if (summary["annual_stability"].get("top5_positive_period_ratio") or 0.0) < 0.75:
        reasons.append("annual_top5_positive_ratio_below_0_75")
    return {"passed": not reasons, "failed_reasons": reasons}


def delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in ["full", "selection", "holdout", "recent63", "recent20"]:
        for field in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]:
            out[f"{window}_{field}"] = to_float(
                (candidate[window].get(field) or 0.0) - (baseline[window].get(field) or 0.0)
            )
    return out


def improvement_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    d = delta(candidate, baseline)
    reasons: list[str] = []
    if not candidate["four_year_gate"]["passed"]:
        reasons.append("candidate_four_year_gate_failed")
    if (d.get("full_rank_ic") or 0.0) < 0.0:
        reasons.append("full_rank_ic_delta_negative")
    for field in ["full_top5", "selection_top5", "holdout_top5", "recent63_top5", "recent20_top5"]:
        if (d.get(field) or 0.0) <= 0.0:
            reasons.append(f"{field}_delta_non_positive")
    return {"passed": not reasons, "failed_reasons": reasons}


def score_objectives(summary: dict[str, Any]) -> dict[str, float]:
    s = summary["selection"]
    return {
        "train_top5_original": (s["top5"] or -1.0) + 0.5 * (s["top10"] or -1.0) + 0.05 * (s["rank_ic"] or -1.0),
        "train_top5_top3": (s["top5"] or -1.0) + 0.4 * (s["top3"] or -1.0) + 0.1 * (s["rank_ic"] or -1.0),
        "train_balanced_top5_rankic": (s["top5"] or -1.0) + 0.3 * (s["rank_ic"] or -1.0),
        "train_top5_stability": (s["top5"] or -1.0) + 0.2 * (s["top5_positive_ratio"] or -1.0),
    }


def materialize_candidate(full_base: pd.DataFrame, w1: float) -> dict[str, Any]:
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    frame = full_base[["trade_date", "stock_code"]].copy()
    frame["pred_prob"] = w1 * full_base["rank_1d"] + (1.0 - w1) * full_base["rank_10d"]
    frame["score_source"] = "rank_blend_objective_scan_1d_10d_research"
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_frame", frame)
        con.execute(f"CREATE OR REPLACE TABLE {TARGET_TABLE} AS SELECT * FROM candidate_frame")
        latest = con.execute(f"SELECT max(trade_date) FROM {TARGET_TABLE}").fetchone()[0]
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
                        FROM {TARGET_TABLE}
                        GROUP BY 1,2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups,
                sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
                count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {TARGET_TABLE}
            """,
            [latest, latest],
        ).fetchdf().iloc[0].to_dict()
    return {key: to_float(value) for key, value in quality.items()}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = add_ranks(read_scores(with_label=True))
    full_base = add_ranks(read_scores(with_label=False))
    baseline_frame = labeled[labeled["label_value"].notna()].copy()
    baseline_frame["score"] = baseline_frame["rank_1d"]
    baseline_summary = summarize(daily_eval(baseline_frame, "score"))

    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for i in range(0, 101):
        w1 = round(i / 100, 2)
        frame = labeled[labeled["label_value"].notna()].copy()
        frame["score"] = w1 * frame["rank_1d"] + (1.0 - w1) * frame["rank_10d"]
        summary = summarize(daily_eval(frame, "score"))
        d = delta(summary, baseline_summary)
        gate = improvement_gate(summary, baseline_summary)
        objectives = score_objectives(summary)
        item = {
            "w1": w1,
            "w10": round(1.0 - w1, 2),
            "summary": summary,
            "delta_vs_baseline": d,
            "improvement_gate": gate,
            "objectives": objectives,
        }
        candidates.append(item)
        rows.append(
            {
                "w1": w1,
                "w10": round(1.0 - w1, 2),
                **objectives,
                "gate_passed": gate["passed"],
                "gate_failed_reasons": ";".join(gate["failed_reasons"]),
                "full_rank_ic_delta": d["full_rank_ic"],
                "full_top5_delta": d["full_top5"],
                "selection_top5_delta": d["selection_top5"],
                "holdout_rank_ic_delta": d["holdout_rank_ic"],
                "holdout_top5_delta": d["holdout_top5"],
                "recent63_rank_ic_delta": d["recent63_rank_ic"],
                "recent63_top5_delta": d["recent63_top5"],
                "recent20_rank_ic_delta": d["recent20_rank_ic"],
                "recent20_top5_delta": d["recent20_top5"],
                "candidate_full_rank_ic": summary["full"]["rank_ic"],
                "candidate_full_top5": summary["full"]["top5"],
                "candidate_recent63_rank_ic": summary["recent63"]["rank_ic"],
                "candidate_recent63_top5": summary["recent63"]["top5"],
                "candidate_recent20_rank_ic": summary["recent20"]["rank_ic"],
                "candidate_recent20_top5": summary["recent20"]["top5"],
            }
        )

    scan = pd.DataFrame(rows)
    scan_csv = REPORT_DIR / "objective_weight_scan.csv"
    scan.to_csv(scan_csv, index=False, encoding="utf-8-sig")

    selected_by_objective: dict[str, Any] = {}
    for objective in ["train_top5_original", "train_top5_top3", "train_balanced_top5_rankic", "train_top5_stability"]:
        best_row = scan.sort_values(objective, ascending=False).iloc[0]
        selected_by_objective[objective] = {
            "w1": to_float(best_row["w1"]),
            "w10": to_float(best_row["w10"]),
            "gate_passed": bool(best_row["gate_passed"]),
            "full_rank_ic_delta": to_float(best_row["full_rank_ic_delta"]),
            "full_top5_delta": to_float(best_row["full_top5_delta"]),
            "holdout_rank_ic_delta": to_float(best_row["holdout_rank_ic_delta"]),
            "holdout_top5_delta": to_float(best_row["holdout_top5_delta"]),
            "recent63_rank_ic_delta": to_float(best_row["recent63_rank_ic_delta"]),
            "recent63_top5_delta": to_float(best_row["recent63_top5_delta"]),
            "recent20_rank_ic_delta": to_float(best_row["recent20_rank_ic_delta"]),
            "recent20_top5_delta": to_float(best_row["recent20_top5_delta"]),
        }

    gate_passed = scan[scan["gate_passed"] == True].copy()
    selected = None
    quality = None
    if not gate_passed.empty:
        # Selection is still training-window based: choose the passable weight with best train stability objective.
        selected_row = gate_passed.sort_values("train_top5_stability", ascending=False).iloc[0]
        selected = {
            "w1": to_float(selected_row["w1"]),
            "w10": to_float(selected_row["w10"]),
            "selection_rule": "among weights that pass four-year/improvement gates, maximize train_top5_stability",
        }
        quality = materialize_candidate(full_base, float(selected_row["w1"]))

    report = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_rank_blend_objective_sensitivity",
        "label": LABEL,
        "baseline": baseline_summary,
        "selected_by_train_objective": selected_by_objective,
        "eligible_candidate_count": int(len(gate_passed)),
        "selected_candidate": selected,
        "candidate_quality": quality,
        "scan_csv": str(scan_csv),
        "source_previous_candidate": str(SOURCE_DIR / "promotion_candidate_holdout.json"),
        "candidate_db": str(EXPERIMENT_DB) if selected else None,
        "candidate_table": TARGET_TABLE if selected else None,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_approved_for_l5_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    report_json = REPORT_DIR / "objective_scan_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D rank-blend 选择目标函数敏感性研究",
        "",
        "## 当前结论",
        "",
        f"- 通过四年准入和 improvement gate 的权重数量：`{len(gate_passed)}`。",
        f"- 是否落实验候选表：`{'是' if selected else '否'}`。",
        "- 本轮 research-only；未训练、未修改 formal manifest、未生成信号、未回测。",
        "",
        "## 训练目标函数选择结果",
        "",
        "| 目标函数 | w1 | w10 | Full Top5 Δ | Holdout Top5 Δ | Recent63 Top5 Δ | Recent20 Top5 Δ | Recent63 RankIC Δ | Recent20 RankIC Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in selected_by_objective.items():
        lines.append(
            f"| {name} | {item['w1']:.2f} | {item['w10']:.2f} | {item['full_top5_delta']:.6f} | "
            f"{item['holdout_top5_delta']:.6f} | {item['recent63_top5_delta']:.6f} | {item['recent20_top5_delta']:.6f} | "
            f"{item['recent63_rank_ic_delta']:.6f} | {item['recent20_rank_ic_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- `{report_json}`",
            f"- `{scan_csv}`",
        ]
    )
    if selected:
        lines.extend(["", "## 实验资产", "", f"- `{EXPERIMENT_DB}::{TARGET_TABLE}`"])
    report_md = REPORT_DIR / "objective_scan_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": str(report_json), "eligible_candidate_count": len(gate_passed), "selected": selected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
