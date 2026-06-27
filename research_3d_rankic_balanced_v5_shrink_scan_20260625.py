from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_rankic_balanced_v5_shrink_scan_20260625"
TARGET_TABLE = (
    "stock_predict_data_model_agent_3d_rankic_balanced_v5_shrink_20260625_"
    "executable_3d_open_return_research"
)
LABEL = "executable_3d_open_return"

SOURCES = {
    "base3": "stock_predict_data_model_agent_3d_rankic_balanced_v5_20260625_executable_3d_open_return_research",
    "formal3": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "cross3": "stock_predict_data_model_agent_3d_cross_horizon_guard_v3_20260625_executable_3d_open_return_research",
    "recent3": "stock_predict_data_model_agent_3d_recent_enhance_v4_20260625_executable_3d_open_return_research",
    "risk3": "stock_predict_data_model_agent_3d_risk_balanced_v7_20260625_executable_3d_open_return_research",
    "best1": "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research",
    "best5": "stock_predict_data_model_agent_5d_condition_gate_v6_shrink_20260625_executable_5d_open_return_research",
    "best10": "stock_predict_data_model_agent_10d_risk_balanced_v5_shrink_20260625_executable_10d_open_return_research",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_source(conn: sqlite3.Connection, alias: str, table: str) -> pd.DataFrame:
    exists = conn.execute(
        "select count(*) from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()[0]
    if not exists:
        raise RuntimeError(f"missing source table: {table}")
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    if frame.empty:
        raise RuntimeError(f"empty source table: {table}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        out = read_source(conn, "base3", SOURCES["base3"])
        for alias, table in SOURCES.items():
            if alias == "base3":
                continue
            out = out.merge(
                read_source(conn, alias, table),
                on=["trade_date", "stock_code"],
                how="left",
                validate="one_to_one",
            )
    for alias in SOURCES:
        score_col = f"{alias}_score"
        out[score_col] = out[score_col].fillna(out["base3_score"])
        out[f"{alias}_rank"] = out.groupby("trade_date")[score_col].rank(method="average", pct=True)
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    labels = pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])
    if labels.empty:
        raise RuntimeError("no non-null labels for evaluation")
    return labels


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def build_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        base_top10 = top_set(group, "base3_rank", 10)
        base_top20 = top_set(group, "base3_rank", 20)
        base_top50 = top_set(group, "base3_rank", 50)
        row: dict[str, float | str] = {"trade_date": trade_date}
        row["base3_top1_top5_gap"] = float(
            group.nlargest(1, "base3_rank")["base3_rank"].mean()
            - group.nlargest(5, "base3_rank")["base3_rank"].min()
        )
        row["base3_top5_top10_gap"] = float(
            group.nlargest(5, "base3_rank")["base3_rank"].mean()
            - group.nlargest(10, "base3_rank")["base3_rank"].mean()
        )
        for alias in SOURCES:
            score_col = f"{alias}_score"
            rank_col = f"{alias}_rank"
            top10 = group.nlargest(min(10, len(group)), rank_col)
            top20 = group.nlargest(min(20, len(group)), rank_col)
            top50 = group.nlargest(min(50, len(group)), rank_col)
            row[f"{alias}_score_std"] = float(group[score_col].std())
            row[f"{alias}_top10_score_mean"] = float(top10[score_col].mean())
            row[f"{alias}_top20_score_mean"] = float(top20[score_col].mean())
            row[f"{alias}_top50_score_mean"] = float(top50[score_col].mean())
            row[f"{alias}_top10_rank_mean"] = float(top10[rank_col].mean())
            row[f"{alias}_top20_rank_mean"] = float(top20[rank_col].mean())
            row[f"{alias}_top50_rank_mean"] = float(top50[rank_col].mean())
            if alias != "base3":
                row[f"corr_base3_{alias}"] = float(group["base3_rank"].corr(group[rank_col]))
                row[f"overlap10_base3_{alias}"] = len(base_top10 & top_set(group, rank_col, 10))
                row[f"overlap20_base3_{alias}"] = len(base_top20 & top_set(group, rank_col, 20))
                row[f"overlap50_base3_{alias}"] = len(base_top50 & top_set(group, rank_col, 50))
        rows.append(row)
    return pd.DataFrame(rows)


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def source_daily_metrics(eval_frame: pd.DataFrame, alias: str) -> pd.DataFrame:
    rank_col = f"{alias}_rank"
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "source": alias,
            "rank_ic": float(group[rank_col].corr(label_rank)),
            "pearson_ic": float(group[rank_col].corr(group[LABEL])),
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(group, rank_col, n)
        row["top_bottom"] = top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_daily(daily: pd.DataFrame, dates: list[str], n: int | None) -> dict[str, float]:
    use = set(dates if n is None else dates[-n:])
    sub = daily[daily["trade_date"].isin(use)]
    return {
        col: float(sub[col].mean())
        for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    }


def period_stats(merged: pd.DataFrame) -> dict[str, float | int]:
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    out: dict[str, float | int] = {}
    top5s = []
    rankics = []
    for name, (lo, hi) in periods.items():
        sub = merged[(merged["trade_date"] >= lo) & (merged["trade_date"] <= hi)]
        top5 = float(sub["top5_delta"].mean())
        rank_ic = float(sub["rank_ic_delta"].mean())
        out[f"{name}_top5_delta"] = top5
        out[f"{name}_rank_ic_delta"] = rank_ic
        top5s.append(top5)
        rankics.append(rank_ic)
    out["positive_top5_periods"] = int(sum(v > 0 for v in top5s))
    out["min_period_top5_delta"] = float(min(top5s))
    out["min_period_rank_ic_delta"] = float(min(rankics))
    out["avg_period_top5_delta"] = float(np.mean(top5s))
    out["avg_period_rank_ic_delta"] = float(np.mean(rankics))
    return out


def evaluate_condition(
    *,
    base_daily: pd.DataFrame,
    alt_daily: pd.DataFrame,
    dates: list[str],
    feature_idx: pd.DataFrame,
    active_dates: set[str],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    active_eval_dates = [d for d in active_dates if d in alt_daily.index]
    candidate = base_daily.copy()
    if active_eval_dates:
        candidate.loc[active_eval_dates, :] = alt_daily.loc[active_eval_dates, candidate.columns]
    candidate = candidate.reset_index()
    merged = candidate.merge(base_daily.reset_index(), on="trade_date", suffixes=("", "_base"))
    for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
        merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_base"]

    full = summarize_daily(candidate, dates, None)
    base_full = summarize_daily(base_daily.reset_index(), dates, None)
    recent63 = summarize_daily(candidate, dates, 63)
    base63 = summarize_daily(base_daily.reset_index(), dates, 63)
    recent20 = summarize_daily(candidate, dates, 20)
    base20 = summarize_daily(base_daily.reset_index(), dates, 20)

    row: dict[str, float | int] = {
        "active_days": len(active_dates),
        "active_ratio": len(active_dates) / len(feature_idx),
        "recent63_active_days": len(set(dates[-63:]) & active_dates),
        "recent20_active_days": len(set(dates[-20:]) & active_dates),
    }
    for prefix, current, base in [
        ("full", full, base_full),
        ("recent63", recent63, base63),
        ("recent20", recent20, base20),
    ]:
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
            row[f"{prefix}_{metric}_delta_vs_base3"] = current[metric] - base[metric]
    row.update(period_stats(merged))
    return row, merged


def scan(scores: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    source_metrics = pd.concat([source_daily_metrics(eval_frame, alias) for alias in SOURCES], ignore_index=True)
    dates = sorted(source_metrics["trade_date"].unique().tolist())
    base_daily = source_metrics[source_metrics["source"] == "base3"].set_index("trade_date")
    feature_idx = features.set_index("trade_date")
    rows = []
    daily_rows = []
    alt_sources = ["formal3", "cross3", "recent3", "risk3", "best1", "best5", "best10"]
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.85, 0.90, 0.95]
    for alt in alt_sources:
        alt_daily = source_metrics[source_metrics["source"] == alt].set_index("trade_date")
        candidate_features = [
            col
            for col in feature_idx.columns
            if col.startswith("base3_")
            or col.startswith(f"{alt}_")
            or col.endswith(f"_{alt}")
            or col.startswith("cross3_")
            or col.startswith("recent3_")
            or col.startswith("risk3_")
            or col.startswith("best1_")
            or col.startswith("best5_")
            or col.startswith("best10_")
        ]
        for feature in sorted(set(candidate_features)):
            vals = feature_idx[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.nunique() < 3:
                continue
            thresholds = sorted(set(float(vals.quantile(q)) for q in quantiles))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active = feature_idx[feature] <= threshold if op == "<=" else feature_idx[feature] >= threshold
                    active_dates = set(active[active].index.astype(str))
                    if len(active_dates) < 6 or len(active_dates) > int(0.40 * len(feature_idx)):
                        continue
                    row, merged = evaluate_condition(
                        base_daily=base_daily,
                        alt_daily=alt_daily,
                        dates=dates,
                        feature_idx=feature_idx,
                        active_dates=active_dates,
                    )
                    row.update({"alt_source": alt, "feature": feature, "op": op, "threshold": threshold})
                    row["pass_basic"] = bool(
                        row["full_rank_ic_delta_vs_base3"] >= 0.0
                        and row["min_period_rank_ic_delta"] >= -0.0038
                        and row["recent63_top1_delta_vs_base3"] >= -0.004
                        and row["recent63_top5_delta_vs_base3"] >= 0.0
                        and row["recent20_top5_delta_vs_base3"] >= 0.0
                        and row["full_top5_delta_vs_base3"] >= -0.0002
                        and row["min_period_top5_delta"] >= 0.0
                        and row["positive_top5_periods"] >= 4
                    )
                    row["objective"] = (
                        1.4 * row["full_rank_ic_delta_vs_base3"]
                        + 1.0 * row["recent63_rank_ic_delta_vs_base3"]
                        + 0.8 * row["recent20_rank_ic_delta_vs_base3"]
                        + 1.0 * row["full_top5_delta_vs_base3"]
                        + 0.8 * row["recent63_top5_delta_vs_base3"]
                        + 0.5 * row["recent20_top5_delta_vs_base3"]
                        + 0.3 * row["full_top1_delta_vs_base3"]
                        + 0.006 * row["positive_top5_periods"]
                        - 0.004 * row["active_ratio"]
                    )
                    rows.append(row)
                    if row["pass_basic"]:
                        daily = merged[["trade_date", "rank_ic_delta", "top1_delta", "top5_delta", "top10_delta"]].copy()
                        daily["alt_source"] = alt
                        daily["feature"] = feature
                        daily["op"] = op
                        daily["threshold"] = threshold
                        daily_rows.append(daily)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["pass_basic", "objective"], ascending=[False, False]).reset_index(drop=True)
    daily_deltas = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return result, daily_deltas, source_metrics


def write_candidate(scores: pd.DataFrame, features: pd.DataFrame, best: pd.Series) -> dict[str, object]:
    alt = str(best["alt_source"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    active = features.set_index("trade_date")[feature]
    active = active <= threshold if op == "<=" else active >= threshold
    active_dates = set(active[active].index.astype(str))
    out = scores[
        ["trade_date", "stock_code", "base3_score", "base3_rank", f"{alt}_score", f"{alt}_rank"]
    ].copy()
    out["guard_active"] = out["trade_date"].isin(active_dates)
    out["pred_prob"] = np.where(out["guard_active"], out[f"{alt}_rank"], out["base3_rank"])
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then {alt}_rank else base3_rank"
    out = out.rename(columns={"base3_score": "base3", f"{alt}_score": alt})
    keep = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "base3",
        alt,
        "base3_rank",
        f"{alt}_rank",
        "guard_active",
        "score_formula",
    ]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[keep].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        short = hashlib.sha1(TARGET_TABLE.encode("utf-8")).hexdigest()[:12]
        conn.execute(f"create index if not exists idx_{short}_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
        conn.commit()
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   count(distinct stock_code), sum(case when pred_prob is null then 1 else 0 end)
            from {quote(TARGET_TABLE)}
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from {quote(TARGET_TABLE)}
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]
        latest = conn.execute(
            f"""
            select trade_date, count(*), count(distinct stock_code)
            from {quote(TARGET_TABLE)}
            group by trade_date
            order by trade_date desc
            limit 5
            """
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
        "active_days_full": int(out.groupby("trade_date")["guard_active"].first().sum()),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(scan_result: pd.DataFrame, daily: pd.DataFrame, source_metrics: pd.DataFrame, stats: dict[str, object] | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scan_path = REPORT_DIR / "rankic_balanced_v5_shrink_scan_results.csv"
    daily_path = REPORT_DIR / "rankic_balanced_v5_shrink_daily_deltas.csv"
    source_path = REPORT_DIR / "rankic_balanced_v5_shrink_source_daily_metrics.csv"
    scan_result.to_csv(scan_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    source_metrics.to_csv(source_path, index=False, encoding="utf-8-sig")
    best = scan_result.iloc[0].to_dict() if not scan_result.empty else None
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset" if stats else "l4_research_rejected_scan_artifact",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5" if stats else "research_rejected_no_improvement",
        "source_type": "sqlite_table" if stats else "scan_report_only",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db" if stats else None,
        "table": TARGET_TABLE if stats else None,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": "candidate" if stats else "rejected",
        "baseline_for_scan": "research_3d_rankic_balanced_v5",
        "research_hypothesis": "shrink_rankic_risk_days_of_current_best_3d_candidate_while_preserving_topn",
        "best_scan": best,
        "asset_stats": stats,
        "promotion_requires_user_confirmation": True,
        "audit_required_before_formal": True,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_tuning_training_parameters": True,
            "no_production_manifest_change": True,
            "no_formal_l4_write": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    manifest_path = REPORT_DIR / "rankic_balanced_v5_shrink_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 3D rankic_balanced_v5 收缩扫描报告（20260625）",
        "",
        "## 当前结论",
        "",
    ]
    if stats:
        lines.extend(
            [
                "本轮在 research-only 范围内找到一个相对 `rankic_balanced_v5` 更稳的收缩候选。",
                "",
                "候选规则：",
                "",
                "```text",
                str(stats["formula"]),
                "```",
                "",
                f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
                f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
                f"- 总行数：`{stats['row_count']}`",
                f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
                f"- 触发交易日数：`{stats['active_days_full']}`",
                f"- `pred_prob` 空值：`{stats['null_pred_prob']}`",
                f"- 重复键组：`{stats['duplicate_key_groups']}`",
                "",
                "## 相对 3D 当前最强研究基线的扫描摘要",
                "",
                f"- 全样本 RankIC 增量：`{best['full_rank_ic_delta_vs_base3']:.6f}`",
                f"- 近 63 日 RankIC 增量：`{best['recent63_rank_ic_delta_vs_base3']:.6f}`",
                f"- 全样本 Top5 增量：`{best['full_top5_delta_vs_base3']:.6f}`",
                f"- 近 63 日 Top5 增量：`{best['recent63_top5_delta_vs_base3']:.6f}`",
                f"- 近 20 日 Top5 增量：`{best['recent20_top5_delta_vs_base3']:.6f}`",
                f"- 最差自然阶段 RankIC 增量：`{best['min_period_rank_ic_delta']:.6f}`",
                f"- 正 Top5 自然阶段数：`{int(best['positive_top5_periods'])}/4`",
            ]
        )
    else:
        lines.append("本轮没有找到相对 `rankic_balanced_v5` 更优、且满足收缩约束的候选，因此不写入新的研究预测表。")
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 扫描明细：`{scan_path.as_posix()}`",
            f"- 日度增量：`{daily_path.as_posix()}`",
            f"- 源日度指标：`{source_path.as_posix()}`",
            f"- 研究 manifest：`{manifest_path.as_posix()}`",
            "",
            "## 边界声明",
            "",
            "- 未训练模型。",
            "- 未调整训练参数。",
            "- 未修改 formal / production manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    (REPORT_DIR / "rankic_balanced_v5_shrink_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    scores = load_scores()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    features = build_features(scores)
    scan_result, daily, source_metrics = scan(scores, labels, features)
    stats = None
    if not scan_result.empty and bool(scan_result.iloc[0]["pass_basic"]) and float(scan_result.iloc[0]["objective"]) > 0:
        stats = write_candidate(scores, features, scan_result.iloc[0])
    write_outputs(scan_result, daily, source_metrics, stats)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "scan_rows": int(len(scan_result)),
                "daily_rows": int(len(daily)),
                "best": scan_result.iloc[0].to_dict() if not scan_result.empty else None,
                "asset_stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
