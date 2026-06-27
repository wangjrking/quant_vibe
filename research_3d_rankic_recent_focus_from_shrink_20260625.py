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
SCAN_DIR = DATA_DIR / "reports" / "model_agent_3d_rankic_balanced_v5_shrink_scan_20260625"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_rankic_recent_focus_20260625"
TARGET_TABLE = (
    "stock_predict_data_model_agent_3d_rankic_recent_focus_20260625_"
    "executable_3d_open_return_research"
)
LABEL = "executable_3d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_3d_rankic_balanced_v5_20260625_executable_3d_open_return_research"

SCAN_CSV = SCAN_DIR / "rankic_balanced_v5_shrink_scan_results.csv"

SOURCES = {
    "base3": BASE_TABLE,
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
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
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


def build_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        row: dict[str, float | str] = {"trade_date": trade_date}
        for alias in SOURCES:
            score_col = f"{alias}_score"
            rank_col = f"{alias}_rank"
            top20 = group.nlargest(min(20, len(group)), rank_col)
            top50 = group.nlargest(min(50, len(group)), rank_col)
            row[f"{alias}_top20_rank_mean"] = float(top20[rank_col].mean())
            row[f"{alias}_top20_score_mean"] = float(top20[score_col].mean())
            row[f"{alias}_top50_rank_mean"] = float(top50[rank_col].mean())
            row[f"{alias}_top50_score_mean"] = float(top50[score_col].mean())
            row[f"{alias}_score_std"] = float(group[score_col].std())
        rows.append(row)
    return pd.DataFrame(rows)


def pick_recent_focus_candidate() -> pd.Series:
    frame = pd.read_csv(SCAN_CSV)
    eligible = frame[
        (frame["full_rank_ic_delta_vs_base3"] >= 0.0)
        & (frame["recent63_rank_ic_delta_vs_base3"] >= 0.003)
        & (frame["recent20_rank_ic_delta_vs_base3"] >= 0.01)
        & (frame["recent63_top5_delta_vs_base3"] >= 0.001)
        & (frame["recent20_top5_delta_vs_base3"] >= 0.003)
        & (frame["full_top5_delta_vs_base3"] >= 0.0)
        & (frame["full_top1_delta_vs_base3"] >= -0.0015)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no recent-focus candidate satisfies relaxed constraints")
    eligible["recent_focus_score"] = (
        1.6 * eligible["recent20_rank_ic_delta_vs_base3"].astype(float)
        + 1.2 * eligible["recent63_rank_ic_delta_vs_base3"].astype(float)
        + 1.0 * eligible["full_rank_ic_delta_vs_base3"].astype(float)
        + 0.8 * eligible["recent20_top5_delta_vs_base3"].astype(float)
        + 0.6 * eligible["recent63_top5_delta_vs_base3"].astype(float)
        + 0.3 * eligible["full_top5_delta_vs_base3"].astype(float)
        + 0.1 * eligible["full_top1_delta_vs_base3"].astype(float)
        - 0.01 * eligible["active_ratio"].astype(float)
    )
    eligible = eligible.sort_values("recent_focus_score", ascending=False).reset_index(drop=True)
    return eligible.iloc[0]


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


def write_outputs(best: pd.Series, stats: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": TARGET_TABLE,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": "candidate",
        "baseline_for_scan": "research_3d_rankic_balanced_v5",
        "research_hypothesis": "recent_rankic_focused_shrink_candidate",
        "best_scan": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best.to_dict().items()},
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
    manifest_path = REPORT_DIR / "rankic_recent_focus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 3D recent-focus 研究候选报告（20260625）",
        "",
        "## 当前结论",
        "",
        "该候选来自对 `rankic_balanced_v5` 收缩扫描结果的二次筛选。它不是全阶段一致优化，而是偏向近期 RankIC 修复和近期 Top5 改善的 research-only 分支。",
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
        "",
        "## 相对 3D 当前基线的核心变化",
        "",
        f"- 全样本 RankIC 增量：`{float(best['full_rank_ic_delta_vs_base3']):.6f}`",
        f"- 近63日 RankIC 增量：`{float(best['recent63_rank_ic_delta_vs_base3']):.6f}`",
        f"- 近20日 RankIC 增量：`{float(best['recent20_rank_ic_delta_vs_base3']):.6f}`",
        f"- 全样本 Top5 增量：`{float(best['full_top5_delta_vs_base3']):.6f}`",
        f"- 近63日 Top5 增量：`{float(best['recent63_top5_delta_vs_base3']):.6f}`",
        f"- 近20日 Top5 增量：`{float(best['recent20_top5_delta_vs_base3']):.6f}`",
        f"- 全样本 Top1 增量：`{float(best['full_top1_delta_vs_base3']):.6f}`",
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
    (REPORT_DIR / "rankic_recent_focus_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    best = pick_recent_focus_candidate()
    scores = load_scores()
    features = build_features(scores)
    stats = write_candidate(scores, features, best)
    write_outputs(best, stats)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "best": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best.to_dict().items()},
                "asset_stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
