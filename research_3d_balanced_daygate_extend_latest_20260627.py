from __future__ import annotations

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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_balanced_daygate_latest_20260627"
LABEL = "executable_3d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_3d_rankic_recent_focus_20260625_executable_3d_open_return_research"
CANDIDATE_TABLE = "stock_predict_data_model_agent_3d_proxy10d5d_bestmix_20260623_executable_3d_open_return_research"
REFERENCE_AGGRESSIVE_TABLE = "stock_predict_data_model_agent_3d_recent_top_daygate_best_20260627_executable_3d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_3d_balanced_daygate_latest_20260627_executable_3d_open_return_research"

THRESHOLD = 0.0027327382943808654


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "base_score"})
        cand = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(CANDIDATE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "cand_score"})
        aggressive = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(REFERENCE_AGGRESSIVE_TABLE)} order by trade_date, stock_code",
            conn,
        ).rename(columns={"pred_prob": "aggressive_score"})
    for frame in [base, cand, aggressive]:
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["stock_code"] = frame["stock_code"].astype(str)
    scores = base.merge(cand, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    scores = scores.merge(aggressive, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    scores["cand_rank"] = scores.groupby("trade_date")["cand_score"].rank(method="average", pct=True)
    scores["aggressive_rank"] = scores.groupby("trade_date")["aggressive_score"].rank(method="average", pct=True)
    return scores


def build_daily_base_gap(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        rows.append(
            {
                "trade_date": trade_date,
                "base_top20_gap": float(group["base_rank"].nlargest(20).mean() - group["base_rank"].nlargest(50).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_balanced_scores(scores: pd.DataFrame, daily_gap: pd.DataFrame) -> pd.DataFrame:
    active_dates = set(daily_gap.loc[daily_gap["base_top20_gap"] <= THRESHOLD, "trade_date"])
    out = scores[["trade_date", "stock_code", "base_rank", "cand_rank"]].copy()
    use_candidate = out["trade_date"].isin(active_dates) & out["cand_rank"].notna()
    out["pred_prob"] = np.where(use_candidate, out["cand_rank"], out["base_rank"])
    out["score_formula"] = np.where(
        use_candidate,
        f"base_top20_gap <= {THRESHOLD:.16g}: proxy_bestmix_rank",
        "base_rank_fallback",
    )
    return out[["trade_date", "stock_code", "pred_prob", "base_rank", "cand_rank", "score_formula"]]


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[LABEL])
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else 0.0


def eval_scores(scores: pd.DataFrame, labels: pd.DataFrame, rank_col: str = "pred_prob") -> dict[str, object]:
    ev = scores[["trade_date", "stock_code", rank_col]].rename(columns={rank_col: "pred_prob"})
    ev = ev.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    rows = []
    for trade_date, group in ev.groupby("trade_date", sort=True):
        rank = group["pred_prob"].rank(method="average", pct=True)
        label_rank = group[LABEL].rank(method="average", pct=True)
        group = group.assign(_rank=rank)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(rank.corr(label_rank)),
                "top1": top_mean(group, "_rank", 1),
                "top5": top_mean(group, "_rank", 5),
                "top10": top_mean(group, "_rank", 10),
                "top20": top_mean(group, "_rank", 20),
            }
        )
    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    metrics = ["rank_ic", "top1", "top5", "top10", "top20"]
    return {
        "days": int(len(daily)),
        "min_eval_date": str(daily["trade_date"].min()),
        "max_eval_date": str(daily["trade_date"].max()),
        "full": daily[metrics].mean().to_dict(),
        "recent63": daily.tail(63)[metrics].mean().to_dict(),
        "recent20": daily.tail(20)[metrics].mean().to_dict(),
    }


def delta(left: dict[str, object], right: dict[str, object]) -> dict[str, dict[str, float]]:
    return {
        scope: {metric: float(left[scope][metric] - right[scope][metric]) for metric in left[scope]}
        for scope in ["full", "recent63", "recent20"]
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    daily_gap = build_daily_base_gap(scores)
    out = build_balanced_scores(scores, daily_gap)

    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_3d_balanced_daygate_latest_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 5"
        ).fetchall()

    labels = load_labels(str(out["trade_date"].min()), str(out["trade_date"].max()))
    balanced_eval = eval_scores(out, labels)
    base_eval = eval_scores(scores.rename(columns={"base_rank": "pred_prob"}), labels)
    aggressive_eval = eval_scores(scores.rename(columns={"aggressive_rank": "pred_prob"}), labels)

    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_3d_balanced_daygate_latest",
        "target_table": TARGET_TABLE,
        "base_table": BASE_TABLE,
        "candidate_table": CANDIDATE_TABLE,
        "reference_aggressive_table": REFERENCE_AGGRESSIVE_TABLE,
        "formula": f"candidate dates: if base_top20_gap <= {THRESHOLD:.16g} then proxy_bestmix_rank else base_rank; candidate-missing dates: base_rank",
        "coverage": {
            "row_count": int(row[0]),
            "min_trade_date": str(row[1]),
            "max_trade_date": str(row[2]),
            "trade_days": int(row[3]),
            "null_pred_prob": int(row[4] or 0),
            "duplicate_key_groups": int(dup),
            "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
        },
        "active_candidate_days": int((daily_gap["base_top20_gap"] <= THRESHOLD).sum()),
        "candidate_coverage_max": str(scores.loc[scores["cand_rank"].notna(), "trade_date"].max()),
        "eval": {
            "balanced": balanced_eval,
            "base": base_eval,
            "aggressive_reference": aggressive_eval,
            "balanced_vs_base_delta": delta(balanced_eval, base_eval),
            "balanced_vs_aggressive_delta": delta(balanced_eval, aggressive_eval),
        },
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "research_prediction_table_write_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    daily_gap.to_csv(REPORT_DIR / "base_daily_gap.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "balanced_daygate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 3D 稳健条件门控研究资产",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 结论",
        "",
        "本资产是 research-only 备选，不是 formal / production 资产。",
        "",
        "它相对当前 3D 激进版降低 RankIC 损失，但会牺牲部分近期 Top1；用途是保留一条稳健备选线。",
        "",
        "## 边界",
        "",
        "- 未训练模型",
        "- 未修改 formal manifest",
        "- 未修改 production manifest",
        "- 未生成交易信号",
        "- 未运行回测",
        "",
        "## 证据",
        "",
        "- `balanced_daygate_summary.json`",
        "- `base_daily_gap.csv`",
    ]
    (REPORT_DIR / "balanced_daygate_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
