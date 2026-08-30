from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "quant/main/config/model_research_10d_liquidity_meta_probe_20260721.json"
MANIFEST = ROOT / "quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json"
FEATURE_DB = ROOT / "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb"
LABEL_DB = ROOT / "quant/data_file/production_assets/duckdb/l3_label_current.duckdb"
OUT = ROOT / "quant/data_file/reports/model_agent_10d_liquidity_meta_probe_20260721"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def evaluate(frame: pd.DataFrame, score_col: str, start: int, end: int) -> dict:
    part = frame[(frame.trade_date >= start) & (frame.trade_date <= end)].copy()
    part["score_rank"] = part.groupby("trade_date")[score_col].rank(pct=True, method="first")
    part["position"] = part.groupby("trade_date")[score_col].rank(
        ascending=False, method="first"
    )
    daily = part.groupby("trade_date", sort=True).apply(
        lambda day: pd.Series(
            {
                "rank_ic": day["score_rank"].corr(day["label_rank"]),
                "top1": day.loc[day.position <= 1, "label_value"].mean(),
                "top3": day.loc[day.position <= 3, "label_value"].mean(),
                "top5": day.loc[day.position <= 5, "label_value"].mean(),
                "top1_amount_pct": day.loc[day.position <= 1, "rank_amount"].mean(),
                "top3_amount_pct": day.loc[day.position <= 3, "rank_amount"].mean(),
            }
        )
    )
    return {
        "trade_days": int(part.trade_date.nunique()),
        **{column: float(daily[column].mean()) for column in daily.columns},
    }


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if cfg["status"] != "frozen_research_only" or manifest["source_type"] != "duckdb_table":
        raise RuntimeError("research config or formal source contract failed")
    pred_db = (MANIFEST.parent / manifest["db_path"]).resolve()
    pred_table = str(manifest["table"])
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute(f"ATTACH '{pred_db.as_posix()}' AS pred (READ_ONLY)")
    con.execute(f"ATTACH '{FEATURE_DB.as_posix()}' AS feat (READ_ONLY)")
    con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS lab (READ_ONLY)")
    frame = con.execute(
        f"""
        SELECT
            p.trade_date::INTEGER AS trade_date,
            percent_rank() OVER (PARTITION BY p.trade_date ORDER BY p.pred_prob)::FLOAT AS rank_score,
            percent_rank() OVER (PARTITION BY p.trade_date ORDER BY f.amount)::FLOAT AS rank_amount,
            l.executable_10d_open_return::FLOAT AS label_value,
            percent_rank() OVER (
                PARTITION BY p.trade_date ORDER BY l.executable_10d_open_return
            )::FLOAT AS label_rank
        FROM pred.{pred_table} p
        JOIN feat.prod_l3_production_factor_parts_20260625 f USING (trade_date, stock_code)
        JOIN lab.prod_l3_prediction_label_parts_current l USING (trade_date, stock_code)
        WHERE p.trade_date BETWEEN ? AND ?
          AND p.stock_code NOT LIKE '%.BJ'
          AND p.pred_prob IS NOT NULL
          AND f.amount IS NOT NULL
          AND l.executable_10d_open_return IS NOT NULL
        ORDER BY p.trade_date, p.stock_code
        """,
        [cfg["train_start"], cfg["evaluation_end"]],
    ).fetchdf()
    con.close()
    for column in ("rank_score", "rank_amount", "label_value", "label_rank"):
        frame[column] = frame[column].astype("float32")
    frame["score_x_amount"] = frame.rank_score * frame.rank_amount
    frame["rank_score_sq"] = frame.rank_score * frame.rank_score
    frame["rank_amount_sq"] = frame.rank_amount * frame.rank_amount

    train = frame[frame.trade_date <= int(cfg["train_end"])]
    feature_columns = list(cfg["features"])
    weight_cfg = cfg["sample_weight"]
    weights = (
        float(weight_cfg["base"])
        + float(weight_cfg["daily_top_1pct_bonus"]) * (train.label_rank >= 0.99).astype("float32")
        + float(weight_cfg["amount_rank_multiplier"]) * train.rank_amount
    ).to_numpy(dtype="float32", copy=False)
    model = xgb.XGBRegressor(**cfg["model_params"])
    model.fit(
        train[feature_columns].to_numpy(dtype="float32", copy=False),
        train.label_value.to_numpy(dtype="float32", copy=False),
        sample_weight=weights,
        verbose=False,
    )
    model_path = OUT / "research_10d_liquidity_meta_probe_model.json"
    model.save_model(model_path)
    frame["candidate_score"] = model.predict(
        frame[feature_columns].to_numpy(dtype="float32", copy=False)
    ).astype("float32")

    evaluations = {}
    for period, bounds in cfg["frozen_evaluation_periods"].items():
        start, end = map(int, bounds)
        base = evaluate(frame, "rank_score", start, end)
        candidate = evaluate(frame, "candidate_score", start, end)
        evaluations[period] = {
            "baseline": base,
            "candidate": candidate,
            "delta": {key: candidate[key] - base[key] for key in base if key != "trade_days"},
        }

    gate_cfg = cfg["probe_gate"]
    rank_deltas = [item["delta"]["rank_ic"] for item in evaluations.values()]
    amount_deltas = [item["delta"]["top1_amount_pct"] for item in evaluations.values()]
    checks = {
        "validation_2024_top3_delta": evaluations["validation_2024"]["delta"]["top3"],
        "validation_2025_top3_delta": evaluations["validation_2025"]["delta"]["top3"],
        "recent60_top3_delta": evaluations["recent60"]["delta"]["top3"],
        "mean_rank_ic_delta": float(np.mean(rank_deltas)),
        "mean_top1_amount_pct_improvement": float(np.mean(amount_deltas)),
    }
    passed = (
        checks["validation_2024_top3_delta"] >= gate_cfg["validation_2024_top3_delta_min"]
        and checks["validation_2025_top3_delta"] >= gate_cfg["validation_2025_top3_delta_min"]
        and checks["recent60_top3_delta"] >= gate_cfg["recent60_top3_delta_min"]
        and checks["mean_rank_ic_delta"] >= gate_cfg["mean_rank_ic_delta_min"]
        and checks["mean_top1_amount_pct_improvement"]
        >= gate_cfg["mean_top1_amount_pct_improvement_min"]
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "actor": "model-agent",
        "status": "probe_passed_for_quarterly_rolling_followup" if passed else "probe_failed_no_followup_candidate",
        "probe_passed": passed,
        "selection_contract": "single frozen model; 2024-2026 never used for training or tuning",
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(frame) - len(train)),
        "evaluations": evaluations,
        "gate_checks": checks,
        "lineage": {
            "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": digest(CONFIG),
            "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "script_sha256": digest(Path(__file__).resolve()),
            "model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": digest(model_path),
            "formal_manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "formal_manifest_sha256": digest(MANIFEST),
            "feature_db": str(FEATURE_DB.relative_to(ROOT)).replace("\\", "/"),
            "label_db": str(LABEL_DB.relative_to(ROOT)).replace("\\", "/"),
        },
        "boundaries": cfg["boundaries"],
    }
    (OUT / "probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "研究报告.md").write_text(
        "# 10D 流动性元模型探针\n\n"
        f"- 状态：`{report['status']}`\n"
        f"- 是否进入季度滚动复核：`{passed}`\n"
        "- 训练仅使用 2022H2-2023，2024-2026 仅冻结评价。\n"
        "- 未修改 formal、未生成信号、未运行策略回测。\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
