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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_dynamic_blend_research_20260625"
TARGET_TABLE = "stock_predict_data_model_agent_5d_dynamic_blend_20260625_executable_5d_open_return_research"
LABEL = "executable_5d_open_return"

SOURCES = {
    "formal5": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
    "dyn5": "stock_predict_data_model_agent_5d_dynamic_guard_20260625_executable_5d_open_return_research",
    "strict5": "stock_predict_data_model_agent_5d_dynamic_guard_strict_20260625_executable_5d_open_return_research",
    "formal10": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
    "dyn10": "stock_predict_data_model_agent_10d_dynamic_guard_20260625_executable_10d_open_return_research",
    "cross10": "stock_predict_data_model_agent_10d_cross_horizon_guard_20260625_executable_10d_open_return_research",
    "formal3": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "cross3": "stock_predict_data_model_agent_3d_cross_horizon_guard_20260625_executable_3d_open_return_research",
    "cross3v2": "stock_predict_data_model_agent_3d_cross_horizon_guard_v2_20260625_executable_3d_open_return_research",
    "cross3v3": "stock_predict_data_model_agent_3d_cross_horizon_guard_v3_20260625_executable_3d_open_return_research",
    "formal1": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
    "strict1": "stock_predict_data_model_agent_1d_dynamic_guard_strict_20260625_executable_1d_open_return_research",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_source(conn: sqlite3.Connection, alias: str, table: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        out = read_source(conn, "formal5", SOURCES["formal5"])
        for alias, table in SOURCES.items():
            if alias == "formal5":
                continue
            out = out.merge(
                read_source(conn, alias, table),
                on=["trade_date", "stock_code"],
                how="left",
                validate="one_to_one",
            )
    for alias in SOURCES:
        score_col = f"{alias}_score"
        out[score_col] = out[score_col].fillna(out["formal5_score"])
        out[f"{alias}_rank"] = out.groupby("trade_date")[score_col].rank(method="average", pct=True)
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in LABEL_DIR.glob("*.parquet"):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def build_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        base10 = top_set(group, "formal5_rank", 10)
        base20 = top_set(group, "formal5_rank", 20)
        base50 = top_set(group, "formal5_rank", 50)
        row: dict[str, float | str] = {"trade_date": trade_date}
        for alias in SOURCES:
            score_col = f"{alias}_score"
            rank_col = f"{alias}_rank"
            row[f"{alias}_score_std"] = float(group[score_col].std())
            row[f"{alias}_top10_score_mean"] = float(group.nlargest(min(10, len(group)), rank_col)[score_col].mean())
            row[f"{alias}_top50_score_mean"] = float(group.nlargest(min(50, len(group)), rank_col)[score_col].mean())
            if alias != "formal5":
                row[f"corr_formal5_{alias}"] = float(group["formal5_rank"].corr(group[rank_col]))
                row[f"overlap10_formal5_{alias}"] = len(base10 & top_set(group, rank_col, 10))
                row[f"overlap20_formal5_{alias}"] = len(base20 & top_set(group, rank_col, 20))
                row[f"overlap50_formal5_{alias}"] = len(base50 & top_set(group, rank_col, 50))
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


def summarize_daily(daily: pd.DataFrame, dates: list[str], n: int | None) -> dict:
    use = set(dates if n is None else dates[-n:])
    sub = daily[daily["trade_date"].isin(use)]
    return {
        c: float(sub[c].mean())
        for c in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    }


def period_stats(merged: pd.DataFrame) -> dict:
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


def scan(scores: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    metric_by_source = pd.concat([source_daily_metrics(eval_frame, alias) for alias in SOURCES], ignore_index=True)
    dates = sorted(metric_by_source["trade_date"].unique().tolist())
    base = metric_by_source[metric_by_source["source"] == "formal5"].set_index("trade_date")
    feature_idx = features.set_index("trade_date")
    rows = []
    daily_rows = []
    alt_sources = ["dyn5", "strict5", "formal10", "dyn10", "cross10", "formal3", "cross3", "cross3v2", "strict1"]
    for alt in alt_sources:
        alt_daily = metric_by_source[metric_by_source["source"] == alt].set_index("trade_date")
        candidate_features = [c for c in feature_idx.columns if c.startswith(f"{alt}_") or c.endswith(f"_{alt}")]
        for feature in candidate_features:
            vals = feature_idx[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.nunique() < 3:
                continue
            thresholds = sorted(
                set(float(vals.quantile(q)) for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95])
            )
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active = feature_idx[feature] <= threshold if op == "<=" else feature_idx[feature] >= threshold
                    active_dates = set(active[active].index.astype(str))
                    if len(active_dates) < 10 or len(active_dates) > int(0.55 * len(feature_idx)):
                        continue
                    cand = base.copy()
                    active_eval_dates = [d for d in active_dates if d in alt_daily.index]
                    cand.loc[active_eval_dates, :] = alt_daily.loc[active_eval_dates, cand.columns]
                    cand = cand.reset_index()
                    merged = cand.merge(base.reset_index(), on="trade_date", suffixes=("", "_base"))
                    for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
                        merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_base"]
                    full = summarize_daily(cand, dates, None)
                    base_full = summarize_daily(base.reset_index(), dates, None)
                    r63 = summarize_daily(cand, dates, 63)
                    b63 = summarize_daily(base.reset_index(), dates, 63)
                    r20 = summarize_daily(cand, dates, 20)
                    b20 = summarize_daily(base.reset_index(), dates, 20)
                    row = {
                        "alt_source": alt,
                        "feature": feature,
                        "op": op,
                        "threshold": threshold,
                        "active_days": len(active_dates),
                        "active_ratio": len(active_dates) / len(feature_idx),
                        "recent63_active_days": len(set(dates[-63:]) & active_dates),
                        "recent20_active_days": len(set(dates[-20:]) & active_dates),
                    }
                    for prefix, cur, base_sum in [("full", full, base_full), ("recent63", r63, b63), ("recent20", r20, b20)]:
                        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
                            row[f"{prefix}_{metric}_delta"] = cur[metric] - base_sum[metric]
                    row.update(period_stats(merged))
                    row["pass_basic"] = bool(
                        row["recent63_active_days"] >= 5
                        and row["recent63_top1_delta"] > 0
                        and row["recent63_top5_delta"] > 0
                        and row["recent63_top10_delta"] > 0
                        and row["recent20_top5_delta"] >= 0
                        and row["full_top5_delta"] >= -0.001
                        and row["full_rank_ic_delta"] >= -0.0015
                        and row["positive_top5_periods"] >= 2
                    )
                    row["objective"] = (
                        3 * row["recent63_top1_delta"]
                        + 2 * row["recent63_top5_delta"]
                        + row["recent63_top10_delta"]
                        + row["recent20_top5_delta"]
                        + 0.5 * row["avg_period_top5_delta"]
                        + 0.01 * row["positive_top5_periods"]
                        - 0.01 * row["active_ratio"]
                    )
                    rows.append(row)
                    if row["pass_basic"]:
                        tmp = merged[["trade_date", "rank_ic_delta", "top1_delta", "top5_delta", "top10_delta"]].copy()
                        tmp["alt_source"] = alt
                        tmp["feature"] = feature
                        tmp["op"] = op
                        tmp["threshold"] = threshold
                        daily_rows.append(tmp)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["pass_basic", "objective"], ascending=[False, False]).reset_index(drop=True)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return result, daily, metric_by_source


def write_candidate(scores: pd.DataFrame, features: pd.DataFrame, best: pd.Series) -> dict:
    alt = str(best["alt_source"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    active = features.set_index("trade_date")[feature]
    active = active <= threshold if op == "<=" else active >= threshold
    active_dates = set(active[active].index.astype(str))
    out = scores[["trade_date", "stock_code", "formal5_score", "formal5_rank", f"{alt}_score", f"{alt}_rank"]].copy()
    out["guard_active"] = out["trade_date"].isin(active_dates)
    out["pred_prob"] = np.where(out["guard_active"], out[f"{alt}_rank"], out["formal5_rank"])
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then {alt}_rank else formal5_rank"
    out = out.rename(columns={"formal5_score": "formal5", f"{alt}_score": alt})
    keep = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal5",
        alt,
        "formal5_rank",
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
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 10"
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
        "active_days_full": int(out.groupby("trade_date")["guard_active"].first().sum()),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(scan_result: pd.DataFrame, daily: pd.DataFrame, source_metrics: pd.DataFrame, stats: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scan_path = REPORT_DIR / "cross_horizon_guard_scan_results.csv"
    daily_path = REPORT_DIR / "cross_horizon_guard_daily_deltas.csv"
    source_path = REPORT_DIR / "cross_horizon_source_daily_metrics.csv"
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
    manifest_path = REPORT_DIR / "cross_horizon_guard_research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 5D 动态混合权重研究报告（20260625）",
        "",
        "## 当前结论",
        "",
    ]
    if stats:
        lines.extend(
            [
                "本轮找到一个通过基础模型评价门槛的 5D cross-horizon research-only 候选。",
                "",
                "候选规则：",
                "",
                "```text",
                stats["formula"],
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
                "## 评价摘要",
                "",
                f"- 近 63 日 RankIC 增量：`{best['recent63_rank_ic_delta']:.6f}`",
                f"- 近 63 日 Top1 增量：`{best['recent63_top1_delta']:.6f}`",
                f"- 近 63 日 Top5 增量：`{best['recent63_top5_delta']:.6f}`",
                f"- 近 63 日 Top10 增量：`{best['recent63_top10_delta']:.6f}`",
                f"- 全样本 Top5 增量：`{best['full_top5_delta']:.6f}`",
                f"- 正 Top5 自然阶段数：`{int(best['positive_top5_periods'])}/4`",
                f"- 最差自然阶段 Top5 增量：`{best['min_period_top5_delta']:.6f}`",
            ]
        )
    else:
        lines.append("本轮未找到通过基础模型评价门槛的 5D cross-horizon 候选。")
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
            "## 边界说明",
            "",
            "- 未训练模型。",
            "- 未调参训练参数。",
            "- 未修改 production/formal manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    (REPORT_DIR / "cross_horizon_guard_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def blended_daily_metrics(eval_frame: pd.DataFrame, rank_col: str, label: str = LABEL) -> pd.DataFrame:
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        label_rank = group[label].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "source": "blend",
            "rank_ic": float(group[rank_col].corr(label_rank)),
            "pearson_ic": float(group[rank_col].corr(group[label])),
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(group, rank_col, n)
        row["top_bottom"] = top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50)
        rows.append(row)
    return pd.DataFrame(rows)


def scan(scores: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    metric_by_source = pd.concat([source_daily_metrics(eval_frame, alias) for alias in SOURCES], ignore_index=True)
    dates = sorted(metric_by_source["trade_date"].unique().tolist())
    base_daily = metric_by_source[metric_by_source["source"] == "formal5"].set_index("trade_date")
    feature_idx = features.set_index("trade_date")
    eval_idx = eval_frame.set_index("trade_date", drop=False)
    rows = []
    daily_rows = []
    alt_sources = ["cross3v3"]
    alpha_grid = [0.25, 0.50, 0.75, 1.00]
    feature_whitelist = {
        "cross3v3": ["corr_formal5_cross3v3", "overlap50_formal5_cross3v3"],
    }
    base_reset = base_daily.reset_index()
    base_full = summarize_daily(base_reset, dates, None)
    b63 = summarize_daily(base_reset, dates, 63)
    b20 = summarize_daily(base_reset, dates, 20)
    for alt in alt_sources:
        for feature in feature_whitelist[alt]:
            vals = feature_idx[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.nunique() < 3:
                continue
            thresholds = sorted(set(float(vals.quantile(q)) for q in [0.10, 0.25, 0.33, 0.50]))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active = feature_idx[feature] <= threshold if op == "<=" else feature_idx[feature] >= threshold
                    active_dates = set(active[active].index.astype(str))
                    if len(active_dates) < 10 or len(active_dates) > int(0.55 * len(feature_idx)):
                        continue
                    active_eval_dates = sorted(set(dates) & active_dates)
                    if not active_eval_dates:
                        continue
                    active_frame = eval_idx.loc[active_eval_dates].reset_index(drop=True)
                    for alpha in alpha_grid:
                        active_frame["blend_rank"] = (1.0 - alpha) * active_frame["formal5_rank"] + alpha * active_frame[f"{alt}_rank"]
                        active_daily = blended_daily_metrics(active_frame, "blend_rank").set_index("trade_date")
                        cand = base_daily.copy()
                        cand.loc[active_daily.index, :] = active_daily.loc[active_daily.index, cand.columns]
                        cand = cand.reset_index()
                        merged = cand.merge(base_reset, on="trade_date", suffixes=("", "_base"))
                        for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
                            merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_base"]
                        full = summarize_daily(cand, dates, None)
                        r63 = summarize_daily(cand, dates, 63)
                        r20 = summarize_daily(cand, dates, 20)
                        row = {
                            "alt_source": alt,
                            "feature": feature,
                            "op": op,
                            "threshold": threshold,
                            "alpha": alpha,
                            "active_days": len(active_dates),
                            "active_ratio": len(active_dates) / len(feature_idx),
                            "recent63_active_days": len(set(dates[-63:]) & active_dates),
                            "recent20_active_days": len(set(dates[-20:]) & active_dates),
                        }
                        for prefix, cur, base_sum in [("full", full, base_full), ("recent63", r63, b63), ("recent20", r20, b20)]:
                            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
                                row[f"{prefix}_{metric}_delta"] = cur[metric] - base_sum[metric]
                        row.update(period_stats(merged))
                        row["pass_basic"] = bool(
                            row["recent63_active_days"] >= 5
                            and row["recent63_top1_delta"] > 0
                            and row["recent63_top5_delta"] > 0
                            and row["recent63_top10_delta"] > 0
                            and row["recent20_top5_delta"] >= 0
                            and row["full_top5_delta"] >= -0.001
                            and row["full_rank_ic_delta"] >= -0.0015
                            and row["positive_top5_periods"] >= 2
                        )
                        row["objective"] = (
                            3 * row["recent63_top1_delta"]
                            + 2 * row["recent63_top5_delta"]
                            + row["recent63_top10_delta"]
                            + row["recent20_top5_delta"]
                            + 0.5 * row["avg_period_top5_delta"]
                            + 0.01 * row["positive_top5_periods"]
                            - 0.01 * row["active_ratio"]
                        )
                        rows.append(row)
                        if row["pass_basic"]:
                            tmp = merged[["trade_date", "rank_ic_delta", "top1_delta", "top5_delta", "top10_delta"]].copy()
                            tmp["alt_source"] = alt
                            tmp["feature"] = feature
                            tmp["op"] = op
                            tmp["threshold"] = threshold
                            tmp["alpha"] = alpha
                            daily_rows.append(tmp)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["pass_basic", "objective"], ascending=[False, False]).reset_index(drop=True)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return result, daily, metric_by_source


def write_candidate(scores: pd.DataFrame, features: pd.DataFrame, best: pd.Series) -> dict:
    alt = str(best["alt_source"])
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    alpha = float(best["alpha"])
    active = features.set_index("trade_date")[feature]
    active = active <= threshold if op == "<=" else active >= threshold
    active_dates = set(active[active].index.astype(str))
    out = scores[["trade_date", "stock_code", "formal5_score", "formal5_rank", f"{alt}_score", f"{alt}_rank"]].copy()
    out["guard_active"] = out["trade_date"].isin(active_dates)
    blend_rank = (1.0 - alpha) * out["formal5_rank"] + alpha * out[f"{alt}_rank"]
    out["pred_prob"] = np.where(out["guard_active"], blend_rank, out["formal5_rank"])
    out["alpha"] = alpha
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then (1-{alpha:.2f})*formal5_rank + {alpha:.2f}*{alt}_rank else formal5_rank"
    out = out.rename(columns={"formal5_score": "formal5", f"{alt}_score": alt})
    keep = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal5",
        alt,
        "formal5_rank",
        f"{alt}_rank",
        "alpha",
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
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 10"
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
        "active_days_full": int(out.groupby("trade_date")["guard_active"].first().sum()),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def main() -> int:
    scores = load_scores()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    features = build_features(scores)
    scan_result, daily, source_metrics = scan(scores, labels, features)
    stats = None
    if not scan_result.empty and bool(scan_result.iloc[0]["pass_basic"]):
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
