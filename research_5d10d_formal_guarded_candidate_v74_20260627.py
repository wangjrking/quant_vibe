from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from score_model_promotion_candidate import evaluate_candidate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MAIN_DIR = ROOT / "quant" / "main"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
FACTOR_DIR = DATA_DIR / "production_factor_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d10d_formal_guarded_v74_20260627"
CONSTRAINTS = MAIN_DIR / "config" / "model_promotion_constraints_v1_20260625.json"


SPECS = {
    "executable_5d_open_return": {
        "asset": "research_5d_formal_guarded_current_v74_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research",
        "target_table": "stock_predict_data_model_agent_5d_formal_guarded_current_v74_20260627_executable_5d_open_return_research",
        "rank_ic_floor": -0.001,
    },
    "executable_10d_open_return": {
        "asset": "research_10d_formal_guarded_monthly_v74_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_10d_monthly_top5_double_veto_v62_20260627_executable_10d_open_return_research",
        "target_table": "stock_predict_data_model_agent_10d_formal_guarded_monthly_v74_20260627_executable_10d_open_return_research",
        "rank_ic_floor": -0.0005,
    },
}

METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
FEATURES = [
    "formal_std",
    "candidate_std",
    "formal_iqr",
    "candidate_iqr",
    "formal_top20_gap",
    "candidate_top20_gap",
    "rank_corr",
    "top20_overlap",
    "mean_abs_rank_gap",
]
QUANTILES = [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def production_factor_latest() -> tuple[str, int]:
    frame = pd.read_parquet(FACTOR_DIR, columns=["trade_date", "stock_code"])
    frame["trade_date"] = frame["trade_date"].astype(str)
    latest = str(frame["trade_date"].max())
    rows = int(frame.loc[frame["trade_date"] == latest, "stock_code"].nunique())
    return latest, rows


def read_scores(conn: sqlite3.Connection, table: str, alias: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    ).rename(columns={"pred_prob": f"{alias}_score"})
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def load_scores(spec: dict[str, object]) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        formal = read_scores(conn, str(spec["formal_table"]), "formal")
        candidate = read_scores(conn, str(spec["candidate_table"]), "candidate")
    scores = formal.merge(candidate, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    scores["formal_rank"] = scores.groupby("trade_date")["formal_score"].rank(method="average", pct=True)
    scores["candidate_rank"] = scores.groupby("trade_date")["candidate_score"].rank(method="average", pct=True)
    return scores


def load_labels(label: str, min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        try:
            part = pd.read_parquet(path, columns=["trade_date", "stock_code", label])
        except Exception:
            continue
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[label])
        if not part.empty:
            chunks.append(part)
    if not chunks:
        return pd.DataFrame(columns=["trade_date", "stock_code", label])
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, score_col: str, label: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), score_col)[label].mean()) if len(group) else 0.0


def daily_eval(frame: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    valid = frame.dropna(subset=[score_col, label])
    for trade_date, group in valid.groupby("trade_date", sort=True):
        if len(group) < 100:
            continue
        score_rank = group[score_col].rank(method="average", pct=True)
        label_rank = group[label].rank(method="average", pct=True)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(score_rank.corr(label_rank)),
                "top1": top_mean(group, score_col, label, 1),
                "top3": top_mean(group, score_col, label, 3),
                "top5": top_mean(group, score_col, label, 5),
                "top10": top_mean(group, score_col, label, 10),
                "top20": top_mean(group, score_col, label, 20),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summarize(daily: pd.DataFrame, window: int | None = None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {metric: float(sub[metric].mean()) for metric in METRICS}


def deltas(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {metric: float(left[metric] - right[metric]) for metric in METRICS}


def month_top5_delta(candidate_daily: pd.DataFrame, formal_daily: pd.DataFrame) -> dict[str, object]:
    frame = candidate_daily[["trade_date", "top5"]].merge(
        formal_daily[["trade_date", "top5"]],
        on="trade_date",
        suffixes=("_candidate", "_formal"),
        validate="one_to_one",
    )
    frame["month"] = frame["trade_date"].str.slice(0, 6)
    monthly = (
        frame.groupby("month")
        .apply(lambda group: float((group["top5_candidate"] - group["top5_formal"]).mean()), include_groups=False)
        .reset_index(name="top5_delta")
    )
    return {
        "positive_top5_months": int((monthly["top5_delta"] > 0).sum()),
        "nonnegative_top5_months": int((monthly["top5_delta"] >= 0).sum()),
        "min_month_top5_delta": float(monthly["top5_delta"].min()) if len(monthly) else None,
        "months": int(len(monthly)),
        "monthly": monthly,
    }


def daily_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = scores.dropna(subset=["candidate_rank"])
    for trade_date, group in valid.groupby("trade_date", sort=True):
        formal_top20 = set(group.nlargest(min(20, len(group)), "formal_rank")["stock_code"])
        candidate_top20 = set(group.nlargest(min(20, len(group)), "candidate_rank")["stock_code"])
        rows.append(
            {
                "trade_date": trade_date,
                "formal_std": float(group["formal_score"].std()),
                "candidate_std": float(group["candidate_score"].std()),
                "formal_iqr": float(group["formal_score"].quantile(0.75) - group["formal_score"].quantile(0.25)),
                "candidate_iqr": float(group["candidate_score"].quantile(0.75) - group["candidate_score"].quantile(0.25)),
                "formal_top20_gap": float(group["formal_rank"].nlargest(20).mean() - group["formal_rank"].nlargest(50).mean()),
                "candidate_top20_gap": float(group["candidate_rank"].nlargest(20).mean() - group["candidate_rank"].nlargest(50).mean()),
                "rank_corr": float(group["formal_rank"].corr(group["candidate_rank"])),
                "top20_overlap": float(len(formal_top20 & candidate_top20) / max(1, len(formal_top20 | candidate_top20))),
                "mean_abs_rank_gap": float((group["formal_rank"] - group["candidate_rank"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_score(scores: pd.DataFrame, active_dates: set[str]) -> pd.Series:
    use_candidate = scores["trade_date"].isin(active_dates) & scores["candidate_rank"].notna()
    return scores["formal_rank"].where(~use_candidate, scores["candidate_rank"])


def objective(delta_full: dict[str, float], delta_63: dict[str, float], delta_20: dict[str, float], month: dict[str, object]) -> float:
    penalty = 0.0
    min_month = month["min_month_top5_delta"]
    if min_month is not None and min_month < 0:
        penalty += 30.0 * abs(float(min_month))
    return float(
        1.0 * delta_full["top5"]
        + 0.8 * delta_full["top1"]
        + 0.5 * delta_full["rank_ic"]
        + 1.5 * delta_63["top5"]
        + 1.0 * delta_63["top1"]
        + 2.2 * delta_20["top5"]
        + 1.2 * delta_20["top1"]
        + 0.05 * int(month["positive_top5_months"])
        - penalty
    )


def evaluate_variant(eval_frame: pd.DataFrame, formal_daily: pd.DataFrame, active_dates: set[str], label: str) -> dict[str, object]:
    scored = eval_frame.copy()
    scored["_guarded_score"] = build_score(scored, active_dates)
    daily = daily_eval(scored, "_guarded_score", label)
    full = summarize(daily)
    recent63 = summarize(daily, 63)
    recent20 = summarize(daily, 20)
    formal_full = summarize(formal_daily)
    formal_63 = summarize(formal_daily, 63)
    formal_20 = summarize(formal_daily, 20)
    delta_full = deltas(full, formal_full)
    delta_63 = deltas(recent63, formal_63)
    delta_20 = deltas(recent20, formal_20)
    month = month_top5_delta(daily, formal_daily)
    month.pop("monthly")
    return {
        "daily": daily,
        "full": full,
        "recent63": recent63,
        "recent20": recent20,
        "delta_full": delta_full,
        "delta_recent63": delta_63,
        "delta_recent20": delta_20,
        "month": month,
        "objective": objective(delta_full, delta_63, delta_20, month),
    }


def scan(label: str, scores: pd.DataFrame, labels: pd.DataFrame, rank_ic_floor: float) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    formal_daily = daily_eval(eval_frame, "formal_rank", label)
    features = daily_features(scores)

    rows = []
    best: dict[str, object] | None = None
    for feature in FEATURES:
        values = features[feature].replace([np.inf, -np.inf], np.nan).dropna()
        thresholds = sorted(set(float(values.quantile(q)) for q in QUANTILES))
        for threshold in thresholds:
            for op in ["<=", ">="]:
                active_dates = set(features.loc[features[feature] <= threshold, "trade_date"] if op == "<=" else features.loc[features[feature] >= threshold, "trade_date"])
                if len(active_dates) < 3 or len(active_dates) > 260:
                    continue
                result = evaluate_variant(eval_frame, formal_daily, active_dates, label)
                row = {
                    "condition": f"{feature} {op} {threshold:.12g}",
                    "feature": feature,
                    "op": op,
                    "threshold": threshold,
                    "active_days": len(active_dates),
                    "objective": result["objective"],
                    "full_rank_ic_delta": result["delta_full"]["rank_ic"],
                    "full_top1_delta": result["delta_full"]["top1"],
                    "full_top5_delta": result["delta_full"]["top5"],
                    "recent63_rank_ic_delta": result["delta_recent63"]["rank_ic"],
                    "recent63_top1_delta": result["delta_recent63"]["top1"],
                    "recent63_top5_delta": result["delta_recent63"]["top5"],
                    "recent20_rank_ic_delta": result["delta_recent20"]["rank_ic"],
                    "recent20_top1_delta": result["delta_recent20"]["top1"],
                    "recent20_top5_delta": result["delta_recent20"]["top5"],
                    "positive_top5_months": result["month"]["positive_top5_months"],
                    "nonnegative_top5_months": result["month"]["nonnegative_top5_months"],
                    "min_month_top5_delta": result["month"]["min_month_top5_delta"],
                }
                row["pass_hard"] = bool(
                    row["full_top5_delta"] >= 0
                    and row["recent63_top5_delta"] >= 0
                    and row["recent20_top5_delta"] >= 0
                    and row["positive_top5_months"] >= 3
                    and row["min_month_top5_delta"] >= 0
                    and row["full_rank_ic_delta"] >= rank_ic_floor
                )
                rows.append(row)
                if best is None or (row["pass_hard"], row["objective"]) > (best["row"]["pass_hard"], best["row"]["objective"]):
                    best = {"row": row, "result": result, "active_dates": active_dates}

    result_frame = pd.DataFrame(rows).sort_values(
        ["pass_hard", "objective", "recent63_top5_delta", "full_top5_delta"],
        ascending=[False, False, False, False],
    )
    if best is None:
        raise RuntimeError(f"no scan rows for {label}")
    return result_frame, best, features


def db_summary(table: str) -> dict[str, object]:
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   sum(case when pred_prob is null then 1 else 0 end)
            from {quote(table)}
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from {quote(table)}
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]
        latest = conn.execute(
            f"""
            select trade_date, count(*) rows, count(distinct stock_code) stocks
            from {quote(table)}
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


def write_asset(label: str, spec: dict[str, object], scores: pd.DataFrame, best: dict[str, object], output_dir: Path) -> dict[str, object]:
    out = scores[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = build_score(scores, best["active_dates"])
    out["score_formula"] = np.where(
        out["trade_date"].isin(best["active_dates"]) & scores["candidate_rank"].notna(),
        str(best["row"]["condition"]) + ": candidate_rank",
        "formal_rank_fallback",
    )
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        out.to_sql(str(spec["target_table"]), conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{str(spec['target_table'])[:40]}_date_code on {quote(str(spec['target_table']))}(trade_date, stock_code)")
        conn.commit()

    summary = db_summary(str(spec["target_table"]))
    manifest = {
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "asset_role": "research_l4_candidate",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": str(spec["target_table"]),
        "label": label,
        "generated_at": now_iso(),
        "allowed_for_main_workflow": False,
        "feature_input": "quant/data_file/production_factor_parts/",
        "label_input": "quant/data_file/prediction_label_parts/",
        "formula": "formal rank fallback with score-state guarded candidate substitution",
        "condition": best["row"]["condition"],
        "formal_table": str(spec["formal_table"]),
        "candidate_table": str(spec["candidate_table"]),
        "notes": "研究候选资产，仅用于模型侧评价与送审准备；未批准进入 formal 或 L5。",
    }
    (output_dir / f"{label}_research_candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_candidate_payload(label: str, spec: dict[str, object], best: dict[str, object], summary: dict[str, object], expected_latest: str, expected_rows: int, output_dir: Path) -> dict[str, object]:
    row = best["row"]
    return {
        "label": label,
        "asset": spec["asset"],
        "table": spec["target_table"],
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "feature_input": "quant/data_file/production_factor_parts/",
        "label_input": "quant/data_file/prediction_label_parts/",
        "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
        "forbidden_inputs_present": False,
        "min_trade_date": summary["min_trade_date"],
        "latest_trade_date": summary["latest_trade_date"],
        "expected_latest_trade_date": expected_latest,
        "latest_day_rows": summary["latest_day_rows"],
        "expected_latest_day_rows": expected_rows,
        "null_pred_prob": summary["null_pred_prob"],
        "duplicate_key_groups": summary["duplicate_key_groups"],
        "is_train_candidate": False,
        "evaluation_report_present": True,
        "candidate_manifest_present": True,
        "full_rank_ic_delta": row["full_rank_ic_delta"],
        "full_top1_delta": row["full_top1_delta"],
        "full_top5_delta": row["full_top5_delta"],
        "recent63_rank_ic_delta": row["recent63_rank_ic_delta"],
        "recent63_top1_delta": row["recent63_top1_delta"],
        "recent63_top5_delta": row["recent63_top5_delta"],
        "recent20_rank_ic_delta": row["recent20_rank_ic_delta"],
        "recent20_top1_delta": row["recent20_top1_delta"],
        "recent20_top5_delta": row["recent20_top5_delta"],
        "positive_top5_periods": row["positive_top5_months"],
        "min_period_top5_delta": row["min_month_top5_delta"],
        "full_top10_delta": None,
        "min_period_rank_ic_delta": None,
        "prefer_simpler_formula": "formal fallback + one score-state gate",
        "prefer_fewer_active_days": row["active_days"],
        "prefer_fewer_upstream_sources": "formal+candidate score tables",
        "prefer_clearer_roll_back_path": "drop research-only v74 table and keep current formal unchanged",
        "evidence": {
            "scan_summary": str(output_dir / f"{label}_v74_summary.json"),
            "candidate_manifest": str(output_dir / f"{label}_research_candidate_manifest.json"),
            "db_summary": summary,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research-only formal-guarded 5D/10D model candidates.")
    parser.add_argument("--labels", default=",".join(SPECS), help="Comma separated labels to run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in selected_labels if label not in SPECS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_latest, expected_rows = production_factor_latest()
    config = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    all_rows = []
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_5d10d_formal_guarded_candidate_v74",
        "expected_latest_trade_date": expected_latest,
        "expected_latest_day_rows": expected_rows,
        "results": {},
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    for label in selected_labels:
        spec = SPECS[label]
        output_dir = REPORT_DIR / label
        output_dir.mkdir(parents=True, exist_ok=True)
        scores = load_scores(spec)
        labels = load_labels(label, str(scores["trade_date"].min()), str(scores["trade_date"].max()))
        scan_frame, best, features = scan(label, scores, labels, float(spec["rank_ic_floor"]))
        scan_frame.to_csv(output_dir / "scan_results.csv", index=False, encoding="utf-8-sig")
        features.to_csv(output_dir / "score_state_features.csv", index=False, encoding="utf-8-sig")
        best["result"]["daily"].to_csv(output_dir / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
        summary = write_asset(label, spec, scores, best, output_dir)
        candidate = make_candidate_payload(label, spec, best, summary, expected_latest, expected_rows, output_dir)
        candidate_path = output_dir / "promotion_candidate.json"
        candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        gate = evaluate_candidate(candidate, config)
        (output_dir / "promotion_gate_result.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        result_payload = {
            "asset": spec["asset"],
            "formal_table": spec["formal_table"],
            "candidate_table": spec["candidate_table"],
            "target_table": spec["target_table"],
            "best": best["row"],
            "db_summary": summary,
            "promotion_gate": gate,
            "candidate_json": str(candidate_path),
        }
        (output_dir / f"{label}_v74_summary.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["results"][label] = result_payload
        all_rows.append(
            {
                "label": label,
                "asset": spec["asset"],
                "table": spec["target_table"],
                "condition": best["row"]["condition"],
                "pass_scan_hard": best["row"]["pass_hard"],
                "promotion_gate_passed": gate["hard_constraint_passed"],
                "failed_hard_constraints": ";".join(gate["failed_hard_constraints"]),
                "latest_trade_date": summary["latest_trade_date"],
                "latest_day_rows": summary["latest_day_rows"],
                "full_rank_ic_delta": best["row"]["full_rank_ic_delta"],
                "full_top1_delta": best["row"]["full_top1_delta"],
                "full_top5_delta": best["row"]["full_top5_delta"],
                "recent63_top1_delta": best["row"]["recent63_top1_delta"],
                "recent63_top5_delta": best["row"]["recent63_top5_delta"],
                "recent20_top1_delta": best["row"]["recent20_top1_delta"],
                "recent20_top5_delta": best["row"]["recent20_top5_delta"],
                "positive_top5_months": best["row"]["positive_top5_months"],
                "min_month_top5_delta": best["row"]["min_month_top5_delta"],
            }
        )
    matrix = pd.DataFrame(all_rows)
    matrix.to_csv(REPORT_DIR / "v74_summary_matrix.csv", index=False, encoding="utf-8-sig")
    payload["summary_matrix"] = str(REPORT_DIR / "v74_summary_matrix.csv")
    (REPORT_DIR / "v74_run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 5D/10D formal fallback guarded 研究候选 v74",
        "",
        "## 结论",
        "",
        "本轮只生成 research-only L4 候选资产；未训练模型，未修改 formal manifest，未修改 production task，未生成信号，未运行回测。",
        "",
        "## 摘要",
        "",
    ]
    for row in all_rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 资产：`{row['asset']}`",
                f"- 表：`{row['table']}`",
                f"- 条件：`{row['condition']}`",
                f"- 最新覆盖：`{row['latest_trade_date']}`，最新日行数 `{row['latest_day_rows']}`",
                f"- promotion gate：`{row['promotion_gate_passed']}`，失败项：`{row['failed_hard_constraints'] or '无'}`",
                f"- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`",
                f"- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`",
                f"- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 证据",
            "",
            "- `v74_run_summary.json`",
            "- `v74_summary_matrix.csv`",
            "- 每个标签目录下的 `promotion_candidate.json`、`promotion_gate_result.json`、`scan_results.csv`、`best_daily_eval.csv`。",
        ]
    )
    (REPORT_DIR / "v74_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
