"""One frozen strict-PIT 5D HistGradient research-only candidate build."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_expanding_pit_oof_baselines_20260829 import (
    CALENDAR_2021_2024, FEATURE_DB, FEATURE_TABLE, HORIZONS, LABEL_DB,
    LABEL_TABLE, POLICY_PATH, canonical_frame_hash, daily_weights, evaluate,
    query_features, query_labels, quote, sha256_file,
)
from build_10d_total_mv_feature_expansion_candidate_20260830 import compare_metrics, dump_json
from incremental_formal_l4_duckdb_mainline import resolve_savedmodel_feature_aliases

CANDIDATE_ID = "v261_5d_hist_gradient_regressor_v1"
CONTRACT = Path("quant/data_file/reports/model_agent_5d_hist_gradient_structure_20260830/training_contract.json")
BASELINE_ROOT = Path("quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3")
OUT = Path("quant/data_file/reports/model_agent_5d_hist_gradient_structure_20260830/build_r1")
END = "20241231"


def official_dates() -> list[str]:
    con = duckdb.connect(str(CALENDAR_2021_2024), read_only=True)
    try:
        values = [str(row[0]) for row in con.execute("SELECT cal_date FROM official_trade_cal WHERE is_open=1 AND cal_date<=? ORDER BY cal_date", [END]).fetchall()]
    finally:
        con.close()
    if not values or values[-1] != END or values != sorted(set(values)):
        raise RuntimeError("blocked_official_calendar")
    return values


def make_folds(dates: list[str], train_start: str, settle: int) -> list[dict[str, str]]:
    result=[]
    for year in ("2022", "2023", "2024"):
        period=[x for x in dates if x.startswith(year)]
        start,end=period[0],period[-1]; index=dates.index(start)
        result.append({"fold_id":f"fold{year}","train_start":train_start,"train_end":dates[index-settle],"test_start":start,"test_end":dates[dates.index(end)-settle],"embargo_rule":f"{settle} official open sessions"})
    return result


def baseline(fold: dict[str, str]) -> pd.DataFrame:
    con=duckdb.connect()
    try:
        return con.execute(f"SELECT trade_date,stock_code,pred_prob AS baseline_pred_prob FROM read_parquet('{BASELINE_ROOT.as_posix()}/5d_oof.parquet') WHERE fold_id=? AND label_mature_within_dev AND trade_date BETWEEN ? AND ? ORDER BY trade_date,stock_code",[fold['fold_id'],fold['test_start'],fold['test_end']]).fetchdf()
    finally:
        con.close()


def score_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.rename(columns={column:"pred_prob"})[["trade_date","stock_code","target","pred_prob"]]


def run() -> int:
    if OUT.exists(): raise RuntimeError("blocked_existing_output")
    OUT.mkdir(parents=True)
    contract=json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["candidate_id"] != CANDIDATE_ID: raise RuntimeError("blocked_contract")
    policy=json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy["training_window_hard_constraints"]["mode"] != "expanding_available_history": raise RuntimeError("blocked_training_window_policy")
    horizon=next(x for x in HORIZONS if x.key=="5d")
    metadata=json.loads(horizon.metadata_path.read_text(encoding="utf-8")); dates=official_dates()
    fcon=duckdb.connect(str(FEATURE_DB),read_only=True); lcon=duckdb.connect(str(LABEL_DB),read_only=True)
    try:
        fmin=str(fcon.execute(f"SELECT min(trade_date) FROM {quote(FEATURE_TABLE)} WHERE trade_date<=?",[END]).fetchone()[0]); lmin=str(lcon.execute(f"SELECT min(trade_date) FROM {quote(LABEL_TABLE)} WHERE trade_date<=?",[END]).fetchone()[0]); train_start=max(fmin,lmin)
        available={str(r[1]) for r in fcon.execute(f"PRAGMA table_info({quote(FEATURE_TABLE)})").fetchall()}; requested=[str(x) for x in metadata["feature_columns"]]; aliases=resolve_savedmodel_feature_aliases(requested,available)
        if aliases["missing"]: raise RuntimeError(f"blocked_feature_aliases:{aliases['missing']}")
        alias_map=dict(aliases["alias_pairs"]); features=[alias_map.get(x,x) for x in requested]
        if len(features)!=120 or len(set(features))!=len(features): raise RuntimeError("blocked_feature_contract")
        folds=make_folds(dates,train_start,horizon.maturity_sessions)
        dump_json(OUT/"preflight.json",{"candidate_id":CANDIDATE_ID,"runtime_agent_route":{"model":"gpt-5.6-terra","thinking":"medium"},"development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"feature_db_sha256":sha256_file(FEATURE_DB),"label_db_sha256":sha256_file(LABEL_DB),"calendar_sha256":sha256_file(CALENDAR_2021_2024),"production_metadata_sha256":sha256_file(horizon.metadata_path),"features":features,"feature_alias_pairs":aliases["alias_pairs"],"folds":folds,"production_unchanged":True})
        rows=[]; results={}
        for fold in folds:
            tx=query_features(fcon,features,fold["train_start"],fold["train_end"]); ty=query_labels(lcon,horizon.label,fold["train_start"],fold["train_end"]); train=tx.merge(ty,on=["trade_date","stock_code"],how="inner",validate="one_to_one").dropna(subset=["target"])
            weights=daily_weights(train[["trade_date","stock_code","target"]],metadata.get("sample_weight_config"))
            vx=query_features(fcon,features,fold["test_start"],fold["test_end"]); vy=query_labels(lcon,horizon.label,fold["test_start"],fold["test_end"]); test=vx.merge(vy,on=["trade_date","stock_code"],how="inner",validate="one_to_one").dropna(subset=["target"])
            model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=300,max_leaf_nodes=31,l2_regularization=8.,early_stopping=False,random_state=42)
            a=train[features].astype("float32"); b=test[features].astype("float32"); model.fit(a,train.target.astype("float32"),sample_weight=weights)
            preds=[model.predict(b).astype("float64") for _ in range(3)]
            if not (np.array_equal(preds[0],preds[1]) and np.array_equal(preds[0],preds[2])): raise RuntimeError(f"blocked_deterministic_prediction:{fold['fold_id']}")
            emitted=test[["trade_date","stock_code","target"]].copy(); emitted["candidate_pred_prob"]=preds[0]; emitted=emitted.merge(baseline(fold),on=["trade_date","stock_code"],how="inner",validate="one_to_one")
            if len(emitted)!=len(test) or emitted.duplicated(["trade_date","stock_code"]).any(): raise RuntimeError(f"blocked_same_key:{fold['fold_id']}")
            if emitted.stock_code.astype(str).str.endswith(".BJ").any() or not np.isfinite(emitted[["target","baseline_pred_prob","candidate_pred_prob"]].to_numpy(dtype="float64")).all(): raise RuntimeError(f"blocked_quality:{fold['fold_id']}")
            bm,_=evaluate(score_frame(emitted,"baseline_pred_prob")); cm,_=evaluate(score_frame(emitted,"candidate_pred_prob")); results[fold["fold_id"]]={"fold":fold,"train_rows":int(len(train)),"test_rows":int(len(emitted)),"baseline_metrics":bm,"candidate_metrics":cm,"failed_gates":compare_metrics(bm,cm),"candidate_prediction_sha256":canonical_frame_hash(score_frame(emitted,"candidate_pred_prob"),["trade_date","stock_code","pred_prob"])}; emitted["fold_id"]=fold["fold_id"]; rows.append(emitted)
    finally:
        fcon.close(); lcon.close()
    oof=pd.concat(rows,ignore_index=True); bm,_=evaluate(score_frame(oof,"baseline_pred_prob")); cm,_=evaluate(score_frame(oof,"candidate_pred_prob")); failures=compare_metrics(bm,cm)+[z for r in results.values() for z in r["failed_gates"]]; failures=sorted(set(failures)); passed=not failures
    oof.to_parquet(OUT/"baseline_candidate_same_key_oof.parquet",index=False)
    summary={"candidate_id":CANDIDATE_ID,"status":"completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search","development_window":["20220101","20241231"],"sealed_windows":{"2025":"not_read","2026_plus":"not_read"},"same_key_rows":int(len(oof)),"same_key_duplicate_groups":int(oof.duplicated(["trade_date","stock_code"]).sum()),"bj_rows":int(oof.stock_code.astype(str).str.endswith(".BJ").sum()),"null_or_nonfinite_rows":int((~np.isfinite(oof[["target","baseline_pred_prob","candidate_pred_prob"]].to_numpy(dtype="float64")).all(axis=1)).sum()),"aggregate":{"baseline_metrics":bm,"candidate_metrics":cm,"failed_gates":compare_metrics(bm,cm)},"folds":results,"hard_gate_passed":passed,"failed_gates":failures,"baseline_score_sha256":canonical_frame_hash(score_frame(oof,"baseline_pred_prob"),["trade_date","stock_code","pred_prob"]),"candidate_score_sha256":canonical_frame_hash(score_frame(oof,"candidate_pred_prob"),["trade_date","stock_code","pred_prob"]),"production_unchanged":True,"allow_next_layer_continue":False}
    dump_json(OUT/"evaluation_summary.json",summary); dump_json(OUT/"research_candidate_manifest.json",{"candidate_id":CANDIDATE_ID,"approval_status":"research_only_not_for_l5","decision":summary["status"],"production_unchanged":True,"allow_next_layer_continue":False}); dump_json(OUT/"hash_inventory.json",{"script_sha256":sha256_file(Path(__file__)),"contract_sha256":sha256_file(CONTRACT),"preflight_sha256":sha256_file(OUT/"preflight.json"),"oof_sha256":sha256_file(OUT/"baseline_candidate_same_key_oof.parquet"),"summary_sha256":sha256_file(OUT/"evaluation_summary.json")}); return 0


if __name__=="__main__": raise SystemExit(run())
