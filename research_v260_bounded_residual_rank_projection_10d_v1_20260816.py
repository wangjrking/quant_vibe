"""Single frozen research-only v260 bounded residual rank-projection Build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from research_v260_pit_oof_excess_logit_10d_v1_20260816 import (
    DEVELOPMENT_END, FEATURE_DB, FEATURE_TABLE, LABEL_COLUMN, METADATA_PATH,
    canonical_frame_hash, feature_aliases, fit_baseline, json_dump, label_end_map,
    load_calendar, load_data, production_params, quote, sha256_bytes, sha256_file,
    strict_monthly_baseline_state,
)

CONTRACT = Path("quant/data_file/runtime/agent_workspaces/research-agent/work/v260_bounded_residual_training_contract_20260816_r1/executable_training_contract_v2.json")
OUT = "quant/data_file/reports/model_agent_v260_bounded_residual_rank_projection_10d_v1_20260816_r1"
START = "20220104"


def rank_fields(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    out = frame.copy()
    order = out.sort_values(["trade_date", score, "stock_code"], ascending=[True, False, True], kind="mergesort")
    order["baseline_oof_rank_current"] = order.groupby("trade_date", sort=False).cumcount() + 1
    order["baseline_oof_rank_pct_current"] = 1 - (order["baseline_oof_rank_current"] - 1) / order.groupby("trade_date", sort=False)["stock_code"].transform("size")
    return out.join(order[["baseline_oof_rank_current", "baseline_oof_rank_pct_current"]])


def binary_domain(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["target", "baseline_oof_score_current", "baseline_oof_rank_current", "baseline_oof_rank_pct_current"]
    out = frame.loc[np.isfinite(frame[required]).all(axis=1)].copy()
    out = out.loc[out.groupby("trade_date", sort=False)["stock_code"].transform("size") >= 30].copy()
    median = out.groupby("trade_date", sort=False)["target"].transform("median")
    out["executable_10d_net_excess_event"] = (out["target"] > median).astype("int8")
    return out


def project(frame: pd.DataFrame) -> pd.DataFrame:
    pieces=[]
    for _, group in frame.groupby("trade_date", sort=True):
        g=group.sort_values(["baseline_oof_score_current", "stock_code"], ascending=[False, True], kind="mergesort").copy()
        residual=g["raw_residual_margin"].to_numpy(dtype="float64")
        med=float(np.median(residual)); scale=float(1.4826*np.median(np.abs(residual-med)))
        clipped=np.zeros(len(g)) if scale <= 1e-12 else np.clip((residual-med)/scale, -1.0, 1.0)
        g["clipped_residual"] = clipped
        g["preference_score"] = g["baseline_oof_score_current"] + clipped
        desired=g.sort_values(["preference_score","stock_code"], ascending=[False,True], kind="mergesort")["stock_code"].tolist()
        desired_rank={code:i+1 for i,code in enumerate(desired)}
        work=g["stock_code"].tolist()
        for phase in range(6):
            for pos in range(phase % 2, len(work)-1, 2):
                left,right=work[pos],work[pos+1]
                if (desired_rank[right], right) < (desired_rank[left], left): work[pos],work[pos+1]=right,left
        final={code:i+1 for i,code in enumerate(work)}
        g["final_rank"] = g["stock_code"].map(final).astype("int32")
        g["candidate_raw_score"] = (len(g)-g["final_rank"]).astype("float64")
        g["rank_changed"] = (g["final_rank"] != g["baseline_oof_rank_current"]).astype("int8")
        if (np.abs(g["final_rank"]-g["baseline_oof_rank_current"]) > 6).any(): raise RuntimeError("blocked_rank_projection_displacement")
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def daily_metrics(frame: pd.DataFrame) -> dict[str,float]:
    corr=[]; overlap=[]; base_sets=[]; cand_sets=[]
    for _,g in frame.groupby("trade_date",sort=True):
        if len(g)<30: continue
        corr.append(float(g["candidate_raw_score"].rank().corr(g["baseline_oof_score_current"].rank())))
        b=set(g.sort_values(["baseline_oof_score_current","stock_code"],ascending=[False,True],kind="mergesort").head(10)["stock_code"])
        c=set(g.sort_values(["candidate_raw_score","stock_code"],ascending=[False,True],kind="mergesort").head(10)["stock_code"])
        overlap.append(len(b&c)/10); base_sets.append(b); cand_sets.append(c)
    if not corr or len(base_sets)<2: raise RuntimeError("blocked_metric_domain")
    bt=float(np.mean([1-len(base_sets[i]&base_sets[i-1])/10 for i in range(1,len(base_sets))]))
    ct=float(np.mean([1-len(cand_sets[i]&cand_sets[i-1])/10 for i in range(1,len(cand_sets))]))
    return {"rank_corr":float(np.mean(corr)),"top10_overlap":float(np.mean(overlap)),"baseline_turnover_proxy":bt,"candidate_turnover_proxy":ct,"metric_days":len(corr)}


def ret_spearman(frame:pd.DataFrame,col:str)->float:
    values=[]
    for _,g in frame.groupby("trade_date",sort=True):
        if len(g)>=30 and g[col].nunique()>1 and g["target"].nunique()>1: values.append(float(g[col].rank().corr(g["target"].rank())))
    if not values: raise RuntimeError("blocked_return_spearman")
    return float(np.mean(values))


def run(replay:int,data:pd.DataFrame,features:list[str],calendar:list[str],ends:dict[str,str],base_params:dict[str,Any],params:dict[str,Any],folds:list[dict[str,str]],root:Path)->tuple[pd.DataFrame,dict[str,Any]]:
    rdir=root/"replays"/f"replay{replay}"; rdir.mkdir(parents=True)
    state, monthly = strict_monthly_baseline_state(data,features,calendar,ends,base_params,rdir/"monthly_baseline_models")
    state=rank_fields(state.rename(columns={"baseline_state_score":"baseline_oof_score_current"}),"baseline_oof_score_current")
    rows=[]; metrics=[]
    for f in folds:
        train=data.loc[(data.trade_date>=START)&(data.trade_date<f["test_start"])&(data.trade_date.map(ends)<f["test_start"])&np.isfinite(data.target)].copy()
        ctrain=binary_domain(train.merge(state[["trade_date","stock_code","baseline_oof_score_current","baseline_oof_rank_current","baseline_oof_rank_pct_current"]],on=["trade_date","stock_code"],how="inner",validate="one_to_one"))
        test=data.loc[(data.trade_date>=f["test_start"])&(data.trade_date<=f["test_end"])&data.trade_date.map(ends).notna()&np.isfinite(data.target)].copy()
        if train.empty or ctrain.empty or test.empty: raise RuntimeError(f"blocked_empty_{f['fold_id']}")
        base, bh=fit_baseline(train,test,features,base_params); test["baseline_oof_score_current"]=base; test=rank_fields(test,"baseline_oof_score_current"); test=binary_domain(test)
        clf=xgb.XGBClassifier(**params); clf.fit(ctrain[features].astype("float32"),ctrain["executable_10d_net_excess_event"],base_margin=ctrain["baseline_oof_score_current"].astype("float32"),verbose=False)
        dm=xgb.DMatrix(test[features].astype("float32"),base_margin=test["baseline_oof_score_current"].astype("float32"))
        full=clf.get_booster().predict(dm,output_margin=True).astype("float64")
        test["raw_residual_margin"]=full-test["baseline_oof_score_current"].to_numpy(); test=project(test); test["fold_id"]=f["fold_id"]
        if not np.isfinite(test[["raw_residual_margin","candidate_raw_score"]]).all().all(): raise RuntimeError("blocked_nonfinite_output")
        m=daily_metrics(test); m.update({"fold_id":f["fold_id"],"train_rows":len(ctrain),"test_rows":len(test),"baseline_auc":float(roc_auc_score(test.executable_10d_net_excess_event,test.baseline_oof_score_current)),"candidate_auc":float(roc_auc_score(test.executable_10d_net_excess_event,test.candidate_raw_score)),"baseline_spearman":ret_spearman(test,"baseline_oof_score_current"),"candidate_spearman":ret_spearman(test,"candidate_raw_score"),"changed_rank_row_share":float(test.rank_changed.mean()),"baseline_model_sha256":bh,"candidate_model_sha256":sha256_bytes(clf.get_booster().save_raw(raw_format="json"))})
        metrics.append(m); rows.append(test)
    out=pd.concat(rows,ignore_index=True).sort_values(["trade_date","stock_code"],kind="mergesort")
    agg=daily_metrics(out); agg.update({"baseline_auc":float(roc_auc_score(out.executable_10d_net_excess_event,out.baseline_oof_score_current)),"candidate_auc":float(roc_auc_score(out.executable_10d_net_excess_event,out.candidate_raw_score)),"baseline_spearman":ret_spearman(out,"baseline_oof_score_current"),"candidate_spearman":ret_spearman(out,"candidate_raw_score"),"changed_rank_row_share":float(out.rank_changed.mean())})
    checks=[agg,*metrics]
    gates={"same_key":not out.duplicated(["trade_date","stock_code"]).any(),"pit":True,"finite":bool(np.isfinite(out[["baseline_oof_score_current","candidate_raw_score"]]).all().all()),"auc_not_worse":all(x["candidate_auc"]>=x["baseline_auc"] for x in checks),"date_spearman_not_worse":all(x["candidate_spearman"]>=x["baseline_spearman"] for x in checks),"rank_corr_floor":all(x["rank_corr"]>=.70 for x in checks),"top10_overlap_floor":all(x["top10_overlap"]>=.30 for x in checks),"turnover_proxy_not_up":all(x["candidate_turnover_proxy"]<=x["baseline_turnover_proxy"] for x in checks),"nonzero_effective_change":all(x["changed_rank_row_share"]>0 for x in checks)}
    hashes={"baseline":canonical_frame_hash(out,["trade_date","stock_code","baseline_oof_score_current"]),"candidate":canonical_frame_hash(out,["trade_date","stock_code","candidate_raw_score"]),"final_rank":canonical_frame_hash(out,["trade_date","stock_code","final_rank"]),"models":sha256_bytes("".join(sorted(monthly.values())+[x["baseline_model_sha256"]+x["candidate_model_sha256"] for x in metrics]).encode())}
    summary={"aggregate":agg,"fold_metrics":metrics,"gates":gates,"hashes":hashes}; json_dump(rdir/"replay_summary.json",summary); return out,summary


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default=OUT); args=ap.parse_args(); root=Path(args.output_dir)
    if root.exists(): raise RuntimeError("refusing_existing_output")
    root.mkdir(parents=True)
    try:
        c=json.loads(CONTRACT.read_text()); assert c["candidate_count"]==c["budget"]==1
        assert xgb.__version__=="2.1.4"
        meta=json.loads(METADATA_PATH.read_text())
        con=duckdb.connect(str(FEATURE_DB),read_only=True); available={r[1] for r in con.execute(f"pragma table_info({quote(FEATURE_TABLE)})").fetchall()}; con.close()
        requested,features,aliases=feature_aliases(meta,available); assert requested==c["input_fields"]["residual_features_fixed_order"] and len(features)==40
        calendar=load_calendar(); ends=label_end_map(calendar); data=load_data(features); assert data.trade_date.max()<=DEVELOPMENT_END
        json_dump(root/"preflight.json",{"candidate_id":c["candidate_id"],"contract_sha256":sha256_file(CONTRACT),"calendar":"official_trade_cal_only","dates":"2021-2024_only","features":features,"aliases":aliases,"production_unchanged":True})
        params=dict(c["loss_and_model"]["fixed_params"]); params["missing"]=np.nan
        reps=[run(i,data,features,calendar,ends,production_params(meta),params,c["fixed_folds"]["folds"],root) for i in (1,2,3)]
        out,first=reps[0]; exact=all((s["hashes"],s["aggregate"],s["gates"])==(first["hashes"],first["aggregate"],first["gates"]) for _,s in reps[1:]); gates=dict(first["gates"]); gates["deterministic_3_of_3"]=exact; passed=all(gates.values())
        out.rename(columns={"target":"executable_10d_open_return"}).to_parquet(root/"baseline_candidate_same_key_oof.parquet",index=False)
        decision="completed_waiting_for_fixed_readonly_audit" if passed else "reject_no_further_search"
        json_dump(root/"evaluation_summary.json",{"candidate_id":c["candidate_id"],"decision":decision,"aggregate":first["aggregate"],"fold_metrics":first["fold_metrics"],"gates":gates,"deterministic": {"passed":exact,"hashes":[s["hashes"] for _,s in reps]},"same_key_rows":len(out),"allow_next_layer_continue":False,"sealed_2025_2026_not_read":True})
        json_dump(root/"research_candidate_manifest.json",{"candidate_id":c["candidate_id"],"approval_status":"research_only_not_for_l5","decision":decision,"production_unchanged":True,"allow_next_layer_continue":False})
        return 0
    except Exception as e:
        json_dump(root/"blocked_or_rejected.json",{"status":"blocked_fail_closed","error":str(e),"production_unchanged":True,"allow_next_layer_continue":False}); return 2
if __name__=="__main__": raise SystemExit(main())
