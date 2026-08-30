import pandas as pd
import tushare as ts
import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score, mean_squared_error
from datetime import datetime
import numpy as np
import shap
import pickle
import logging
import json
import os
from pathlib import Path

import sqlite3
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from adjustment_semantics import (
    NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS,
    NAKED_MARKET_PRICE_COLUMNS,
    explicit_qfq_column_name,
    require_front_adjusted_column,
)

from model_asset_route import (
    require_legacy_model_asset_chain_opt_in,
    resolve_legacy_mixed_factor_path,
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_legacy_prediction_db_path,
    resolve_model_label_duckdb_path,
    resolve_model_label_duckdb_table,
    resolve_model_feature_path,
    resolve_model_label_path,
    resolve_model_prediction_db_path,
    resolve_prediction_run_dir,
    use_legacy_mixed_features,
    use_legacy_prediction_db,
    write_prediction_manifest,
)
from l4_duckdb_sync import l4_duckdb_sync_enabled, sync_prediction_table_full_to_duckdb
from l3_duckdb_sync import GTJA_QFQ_COLUMN_MAP
from project_paths import resolve_data_dir


try:
    from dl_model_module import add_fttransformer_features_simple
except ModuleNotFoundError as exc:
    _dl_model_import_error = exc

    def add_fttransformer_features_simple(*args, **kwargs):
        raise ModuleNotFoundError(
            "dl_model_module depends on optional torch, which is not installed. "
            "The XGBoost/light-factor pipeline does not need it."
        ) from _dl_model_import_error

from leakage_guard import validate_no_leakage
from light_factor_module import get_light_factor_data
from stock_pool_module import filter_frame_by_stock_pool, load_stock_pool
from sklearn.model_selection import TimeSeriesSplit


LEGACY_FRONT_ADJUSTED_ALIAS_BASES = set(NAKED_MARKET_PRICE_COLUMNS) | set(NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS) | {
    "macdsignal",
    "macdhist",
    "macd_dea",
    "macd_dif",
    "rsi",
    "tema",
    "dema",
    "dema_10",
    "t3",
}


def resolve_selected_feature_alias(feature_name, available_columns):
    feature = str(feature_name)
    available = {str(column) for column in available_columns}
    if feature in available:
        return feature
    qfq_gtja = GTJA_QFQ_COLUMN_MAP.get(feature)
    if qfq_gtja and qfq_gtja in available:
        return qfq_gtja
    if feature in LEGACY_FRONT_ADJUSTED_ALIAS_BASES:
        qfq_name = explicit_qfq_column_name(feature)
        if qfq_name in available:
            return qfq_name
    return None


def resolve_selected_feature_aliases(requested_columns, available_columns):
    query_columns = []
    alias_pairs = []
    missing = []
    seen = set()

    for column in requested_columns:
        resolved = resolve_selected_feature_alias(column, available_columns)
        if resolved is None:
            missing.append(str(column))
            continue
        if resolved not in seen:
            query_columns.append(resolved)
            seen.add(resolved)
        if resolved != column:
            alias_pairs.append((str(column), str(resolved)))

    return {
        "query_columns": query_columns,
        "alias_pairs": alias_pairs,
        "missing": missing,
    }




def get_pro(ts_token):
	ts_pro = ts.pro_api(ts_token)
	return ts_pro

def huber_loss_obj(y_true, y_pred, delta=0.051):
    """
    XGBoost Sklearn版 Huber Loss 目标函数
    :param y_true: 真实收益率 (numpy数组)
    :param y_pred: 预测收益率 (numpy数组)
    :param delta: 阈值，建议设1.0（适配收益率百分比）
    :return: grad (一阶导数), hess (二阶导数)
    """
    # 计算预测误差
    error = y_pred - y_true
    abs_error = np.abs(error)

    # 一阶导数（梯度）：小误差用MSE的导，大误差用MAE的导
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))

    # 二阶导数（Hessian）：XGBoost要求非0，小误差为1，大误差设极小值
    hess = np.where(abs_error <= delta, np.ones_like(error), np.ones_like(error) * 1e-6)

    return grad, hess

def custom_mae(y_true, y_pred):
    global group_dates
    dates = group_dates

    if dates is None or len(dates) == 0:
        return 0.0

    # ??????????????0,1,2...
    _, group_idx = np.unique(dates, return_inverse=True)

    keep_pred = []
    keep_true = []
    # ????????
    for g in np.unique(group_idx):
        # ?????????????????
        mask = group_idx == g

        day_p = y_pred[mask]
        day_t = y_true[mask]

        # ??????10
        n = min(10, len(day_p))
        top10_idx = np.argpartition(day_p, -n)[-n:]

        keep_pred.append(day_p[top10_idx])
        keep_true.append(day_t[top10_idx])
    # ???????????10
    if not keep_pred:
        return 0.0
    all_p = np.concatenate(keep_pred)
    all_t = np.concatenate(keep_true)
    # print(all_t)
    return np.mean(all_t)
    # return np.mean(np.abs(all_p - all_t))


def top_return_loss(y_true, y_pred):
    global group_dates
    dates = group_dates
    if dates is None or len(dates) == 0:
        return 0.0
    top_k = int(os.getenv("XGB_TOP_RETURN_EVAL_K", "10"))
    if top_k <= 0:
        raise ValueError("XGB_TOP_RETURN_EVAL_K must be positive")
    _, group_idx = np.unique(dates, return_inverse=True)
    keep_true = []
    for group in np.unique(group_idx):
        mask = group_idx == group
        day_p = y_pred[mask]
        day_t = y_true[mask]
        n = min(top_k, len(day_p))
        top_idx = np.argpartition(day_p, -n)[-n:]
        keep_true.append(day_t[top_idx])
    if not keep_true:
        return 0.0
    return -float(np.mean(np.concatenate(keep_true)))


def _resolve_reg_eval_metric():
    mode = str(os.getenv("XGB_REG_EVAL_METRIC", "custom_mae")).strip().lower()
    if mode in {"custom_mae", "top_return"}:
        return custom_mae
    if mode == "top_return_loss":
        return top_return_loss
    if mode in {"rmse", "mae"}:
        return mode
    raise ValueError(f"unsupported XGB_REG_EVAL_METRIC: {mode}")


def _valid_eval_data(test_x, test_y):
    valid_mask = ~pd.isna(test_y)
    if hasattr(valid_mask, "any") and not valid_mask.any():
        return None
    eval_x = test_x.loc[valid_mask]
    eval_y = test_y.loc[valid_mask]
    if len(eval_y) == 0:
        return None
    return eval_x, eval_y


def _optional_int_env(name):
    value = os.getenv(name)
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _sample_weight_config():
    mode = str(os.getenv("XGB_SAMPLE_WEIGHT_MODE", "")).strip().lower()
    if mode in {"", "none", "off", "false", "0"}:
        return None
    if mode != "daily_top_quantile":
        raise ValueError(f"unsupported XGB_SAMPLE_WEIGHT_MODE: {mode}")
    top_pct = float(os.getenv("XGB_SAMPLE_WEIGHT_TOP_PCT", "0.02"))
    multiplier = float(os.getenv("XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER", "3.0"))
    if not 0 < top_pct <= 1:
        raise ValueError("XGB_SAMPLE_WEIGHT_TOP_PCT must be in (0, 1]")
    if multiplier < 1:
        raise ValueError("XGB_SAMPLE_WEIGHT_TOP_MULTIPLIER must be >= 1")
    return {"mode": mode, "top_pct": top_pct, "top_multiplier": multiplier}


def _trade_dates_from_index(index):
    names = list(getattr(index, "names", []) or [])
    if "trade_date" in names:
        return np.asarray(index.get_level_values("trade_date").astype(str))
    if getattr(index, "nlevels", 1) >= 2:
        return np.asarray(index.get_level_values(1).astype(str))
    return None


def _build_xgb_sample_weight(train_y):
    config = _sample_weight_config()
    if config is None:
        return None

    y = pd.Series(train_y).copy()
    values = pd.to_numeric(y, errors="coerce")
    weights = np.ones(len(values), dtype="float32")
    valid_mask = values.notna().to_numpy()
    if not valid_mask.any():
        return weights

    dates = _trade_dates_from_index(values.index)
    if dates is None:
        valid_values = values[valid_mask]
        top_count = max(1, int(np.floor(len(valid_values) * config["top_pct"])))
        top_index = valid_values.sort_values(ascending=False).head(top_count).index
        weights[values.index.isin(top_index)] = config["top_multiplier"]
        return weights

    frame = pd.DataFrame({"value": values.to_numpy(), "date": dates})
    valid = frame["value"].notna()
    ranks = frame.loc[valid].groupby("date")["value"].rank(method="first", ascending=False)
    counts = frame.loc[valid].groupby("date")["value"].transform("count")
    limits = np.floor(counts.astype(float) * config["top_pct"]).clip(lower=1)
    top_mask = ranks <= limits
    valid_positions = np.flatnonzero(valid.to_numpy())
    weights[valid_positions[top_mask.to_numpy()]] = config["top_multiplier"]
    return weights


def _validation_config():
    mode = str(os.getenv("XGB_VALIDATION_MODE", "")).strip().lower()
    if mode in {"", "none", "off", "false", "0"}:
        return None
    if mode != "train_tail_days":
        raise ValueError(f"unsupported XGB_VALIDATION_MODE: {mode}")
    tail_days = int(os.getenv("XGB_VALIDATION_TAIL_DAYS", "63"))
    if tail_days <= 0:
        raise ValueError("XGB_VALIDATION_TAIL_DAYS must be positive")
    return {"mode": mode, "tail_days": tail_days}


def _split_train_validation_by_tail_trade_days(train_x, train_y, tail_days):
    dates = _trade_dates_from_index(train_x.index)
    if dates is None:
        return train_x, train_y, None
    unique_dates = pd.Series(dates).drop_duplicates().sort_values().to_list()
    if len(unique_dates) <= int(tail_days):
        return train_x, train_y, None
    validation_dates = set(unique_dates[-int(tail_days):])
    validation_mask = pd.Series(dates).isin(validation_dates).to_numpy()
    if validation_mask.all() or not validation_mask.any():
        return train_x, train_y, None
    fit_x = train_x.loc[~validation_mask]
    fit_y = train_y.loc[~validation_mask]
    validation_x = train_x.loc[validation_mask]
    validation_y = train_y.loc[validation_mask]
    if len(fit_y) == 0 or len(validation_y) == 0:
        return train_x, train_y, None
    return fit_x, fit_y, (validation_x, validation_y)

def get_model(type):
	xgb_n_estimators = int(os.getenv("XGB_N_ESTIMATORS", "14000"))
	xgb_learning_rate = float(os.getenv("XGB_LEARNING_RATE", "0.003"))
	xgb_max_depth = int(os.getenv("XGB_MAX_DEPTH", "3"))
	xgb_subsample = float(os.getenv("XGB_SUBSAMPLE", "1"))
	xgb_colsample_bytree = float(os.getenv("XGB_COLSAMPLE_BYTREE", "1"))
	xgb_reg_alpha = float(os.getenv("XGB_REG_ALPHA", "0"))
	xgb_reg_lambda = float(os.getenv("XGB_REG_LAMBDA", "1"))
	xgb_device = os.getenv("XGB_DEVICE", "cuda")
	xgb_n_jobs = int(os.getenv("XGB_N_JOBS", "0"))
	xgb_early_stopping_rounds = _optional_int_env("XGB_EARLY_STOPPING_ROUNDS")
	class_model = xgb.XGBClassifier(
    use_label_encoder=False,  # 禁用旧版标签编码器
    eval_metric='logloss',    # 显式指定评估指标
	learning_rate=0.001,         # 学习率
	max_depth=3,               # 树的最大深度
	n_estimators=6000,          # 弱学习器数量
	subsample=1,             # 样本采样比例
	colsample_bytree=1,      # 特征采样比例
	reg_alpha=0,               # L1正则化
	reg_lambda=1,             # L2正则化
	random_state=42,
	device=xgb_device,  # 使用 GPU 加速
	n_jobs=xgb_n_jobs,
	)
	class_model.set_params(
		learning_rate=float(os.getenv("XGB_CLASS_LEARNING_RATE", str(xgb_learning_rate))),
		max_depth=int(os.getenv("XGB_CLASS_MAX_DEPTH", str(xgb_max_depth))),
		n_estimators=int(os.getenv("XGB_CLASS_N_ESTIMATORS", str(xgb_n_estimators))),
		subsample=float(os.getenv("XGB_CLASS_SUBSAMPLE", str(xgb_subsample))),
		colsample_bytree=float(os.getenv("XGB_CLASS_COLSAMPLE_BYTREE", str(xgb_colsample_bytree))),
		reg_alpha=float(os.getenv("XGB_CLASS_REG_ALPHA", str(xgb_reg_alpha))),
		reg_lambda=float(os.getenv("XGB_CLASS_REG_LAMBDA", str(xgb_reg_lambda))),
		early_stopping_rounds=_optional_int_env("XGB_CLASS_EARLY_STOPPING_ROUNDS") or xgb_early_stopping_rounds,
	)
	reg_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    #  objective="reg:pseudohubererror",   # 损失函数（抗极端收益，量化首选）
	# objective="reg:quantileerror", quantile_alpha=0.7,
    eval_metric=custom_mae,           # 回归评估指标[3,6](@ref)
    # eval_metric='mae',
    learning_rate=xgb_learning_rate,          # 可保持或适当增大(0.01-0.1)[2,6](@ref)
    max_depth = xgb_max_depth,                  # 可保持(3-10)[2,6](@ref)
    n_estimators= xgb_n_estimators,             # 可增加至200-1000[2,7](@ref)
    subsample=xgb_subsample,                # 可保持(0.5-1)[2,6](@ref)
    colsample_bytree=xgb_colsample_bytree,         # 可保持(0.5-1)[2,6](@ref)
    reg_alpha=xgb_reg_alpha,                  # L1正则化[2,6](@ref)
    reg_lambda=xgb_reg_lambda,                 # L2正则化[2,6](@ref)
    random_state=42,
	device=xgb_device,       # GPU加速[6](@ref)
	n_jobs=xgb_n_jobs,
	# early_stopping_rounds=500,
)
	if type == 'reg':
		return reg_model
	else:
 		return class_model

def get_model(type):
    xgb_n_estimators = int(os.getenv("XGB_N_ESTIMATORS", "14000"))
    xgb_learning_rate = float(os.getenv("XGB_LEARNING_RATE", "0.003"))
    xgb_max_depth = int(os.getenv("XGB_MAX_DEPTH", "3"))
    xgb_subsample = float(os.getenv("XGB_SUBSAMPLE", "1"))
    xgb_colsample_bytree = float(os.getenv("XGB_COLSAMPLE_BYTREE", "1"))
    xgb_reg_alpha = float(os.getenv("XGB_REG_ALPHA", "0"))
    xgb_reg_lambda = float(os.getenv("XGB_REG_LAMBDA", "1"))
    xgb_device = os.getenv("XGB_DEVICE", "cuda")
    xgb_n_jobs = int(os.getenv("XGB_N_JOBS", "0"))
    xgb_early_stopping_rounds = _optional_int_env("XGB_EARLY_STOPPING_ROUNDS")

    class_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        learning_rate=0.001,
        max_depth=3,
        n_estimators=6000,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        device=xgb_device,
        n_jobs=xgb_n_jobs,
    )
    class_model.set_params(
        learning_rate=float(os.getenv("XGB_CLASS_LEARNING_RATE", str(xgb_learning_rate))),
        max_depth=int(os.getenv("XGB_CLASS_MAX_DEPTH", str(xgb_max_depth))),
        n_estimators=int(os.getenv("XGB_CLASS_N_ESTIMATORS", str(xgb_n_estimators))),
        subsample=float(os.getenv("XGB_CLASS_SUBSAMPLE", str(xgb_subsample))),
        colsample_bytree=float(os.getenv("XGB_CLASS_COLSAMPLE_BYTREE", str(xgb_colsample_bytree))),
        reg_alpha=float(os.getenv("XGB_CLASS_REG_ALPHA", str(xgb_reg_alpha))),
        reg_lambda=float(os.getenv("XGB_CLASS_REG_LAMBDA", str(xgb_reg_lambda))),
        early_stopping_rounds=_optional_int_env("XGB_CLASS_EARLY_STOPPING_ROUNDS") or xgb_early_stopping_rounds,
    )

    reg_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric=_resolve_reg_eval_metric(),
        learning_rate=xgb_learning_rate,
        max_depth=xgb_max_depth,
        n_estimators=xgb_n_estimators,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        reg_alpha=xgb_reg_alpha,
        reg_lambda=xgb_reg_lambda,
        random_state=42,
        device=xgb_device,
        n_jobs=xgb_n_jobs,
        early_stopping_rounds=xgb_early_stopping_rounds,
    )
    if type == "reg":
        return reg_model
    return class_model


def store_feature_importance(x, y, data_file_url, filename):
    model = get_model('reg')
    model.fit(x, y)

    importance_df = pd.DataFrame({
        'Feature': x.columns,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    importance_df.index = importance_df.index + 1
    importance_df.index.name = 'Rank'
    importance_path = data_file_url + '/' + filename
    importance_df.to_csv(importance_path, encoding='utf-8-sig')
    return importance_df


def _coerce_predict_matrix(test_x):
    if isinstance(test_x, pd.DataFrame):
        return test_x.apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy(copy=False)
    array_x = np.asarray(test_x)
    if array_x.dtype == object:
        return pd.DataFrame(array_x).apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy(copy=False)
    return array_x.astype("float32", copy=False)


def _gpu_predict_matrix_or_none(array_x):
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    return cp.asarray(array_x)


def _predict_model_outputs(model, test_x, proba=False):
    device = ""
    for getter_name in ("get_xgb_params", "get_params"):
        getter = getattr(model, getter_name, None)
        if getter is None:
            continue
        try:
            params = getter() or {}
        except Exception:
            continue
        device = str(params.get("device") or "")
        if device:
            break

    booster = None
    if device.lower().startswith("cuda") and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
        except Exception:
            booster = None

    if booster is not None and hasattr(booster, "inplace_predict"):
        array_x = _coerce_predict_matrix(test_x)
        predict_x = array_x
        if device.lower().startswith("cuda"):
            gpu_x = _gpu_predict_matrix_or_none(array_x)
            if gpu_x is not None:
                predict_x = gpu_x
            elif hasattr(booster, "set_param"):
                # Explicit CPU prediction avoids XGBoost's noisy cuda-vs-numpy
                # fallback path. The trained model has already been saved.
                booster.set_param({"device": "cpu"})
        pred = np.asarray(booster.inplace_predict(predict_x))
        if proba:
            if pred.ndim == 2:
                if pred.shape[1] == 1:
                    return pred[:, 0]
                return pred[:, 1]
            return pred
        return pred

    if proba and hasattr(model, "predict_proba"):
        return model.predict_proba(test_x)[:, 1]
    return model.predict(test_x)

def incre_fit(model, train_x, train_y, test_x, test_y, data_file_url, save_shap=True):
    # print(test_x)
    global group_dates
    validation_config = _validation_config()
    fit_train_x = train_x
    fit_train_y = train_y
    eval_data = None
    if validation_config is not None:
        fit_train_x, fit_train_y, eval_data = _split_train_validation_by_tail_trade_days(
            train_x,
            train_y,
            validation_config["tail_days"],
        )
    if eval_data is None:
        eval_data = _valid_eval_data(test_x, test_y)
    if eval_data is not None:
        group_dates = eval_data[0].index.get_level_values(level=1).values
    else:
        group_dates = np.array([])

    # store_feature_importance(train_x, train_y, data_file_url, 'train_feature_importance.csv')
    # print('????????????????????')
    # store_feature_importance(test_x[~np.isnan(test_y)], test_y[~np.isnan(test_y)], data_file_url, 'test_feature_importance.csv')
    # print('????????????????????')

    sample_weight = _build_xgb_sample_weight(fit_train_y)
    fit_kwargs = {"verbose": 100}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    original_early_stopping_rounds = None
    early_stopping_temporarily_disabled = False
    if eval_data is None and hasattr(model, "get_params") and hasattr(model, "set_params"):
        model_params = model.get_params()
        original_early_stopping_rounds = model_params.get("early_stopping_rounds")
        if original_early_stopping_rounds is not None:
            early_stopping_temporarily_disabled = True
            model.set_params(early_stopping_rounds=None)
    try:
        if eval_data is not None:
            model.fit(fit_train_x, fit_train_y, eval_set=[eval_data], **fit_kwargs)
        else:
            model.fit(fit_train_x, fit_train_y, **fit_kwargs)
    finally:
        if early_stopping_temporarily_disabled:
            model.set_params(early_stopping_rounds=original_early_stopping_rounds)
    _save_model_artifacts(model, fit_train_x.columns)
    print('?????????')

    import gc
    gc.collect()

    pred_y_proba = _predict_model_outputs(model, test_x, proba=hasattr(model, "predict_proba"))

    if save_shap:
        explainer = shap.Explainer(model)
        shap_values = explainer(test_x)
        test_index = test_x.index.tolist()
        with open(data_file_url + '/shap_values.pkl', 'wb') as file:
            pickle.dump(shap_values, file)

        with open(data_file_url + '/test_index.pkl', 'wb') as file:
            pickle.dump(test_index, file)
    return test_y, pred_y_proba


def _save_model_artifacts(model, feature_columns):
    model_path = os.getenv("XGB_MODEL_SAVE_PATH")
    if not model_path:
        return

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    metadata_path = os.getenv("XGB_MODEL_METADATA_PATH")
    if metadata_path:
        params = {}
        for key, value in model.get_params().items():
            try:
                json.dumps(value)
                params[key] = value
            except TypeError:
                params[key] = repr(value)
        metadata = {
            "model_path": str(path),
            "model_class": type(model).__name__,
            "feature_columns": [str(column) for column in feature_columns],
            "feature_count": len(feature_columns),
            "xgb_params": params,
            "sample_weight_config": _sample_weight_config(),
            "validation_config": _validation_config(),
            "best_iteration": getattr(model, "best_iteration", None),
            "best_score": getattr(model, "best_score", None),
        }
        meta_path = Path(metadata_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# 691
def model_assess(train_x, train_y, test_x, test_y, train_data, test_data, type='class', data_file_url=None, save_shap=True):

    if type == 'class':
        model = get_model(type)
        eval_data = _valid_eval_data(test_x, test_y)
        if eval_data is not None:
            model.fit(train_x, train_y, eval_set=[(train_x, train_y), eval_data])
        else:
            model.fit(train_x, train_y, eval_set=[(train_x, train_y)])
        importance = model.feature_importances_
        feature_names = train_x.columns

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv('test.csv')
        importance_df = importance_df[importance_df['Importance'] > 0]
        top_n = int(len(feature_names) * 0.6)

        top_features = importance_df.head(top_n)['Feature'].tolist()


        train_x = train_x[top_features]
        test_x = test_x[top_features]
        test_y, pred_y_proba = incre_fit(model, train_x, train_y, test_x, test_y, data_file_url, save_shap=save_shap)
        pred_y = (pred_y_proba > 0.5).astype(int)
        try:
            auc = roc_auc_score(test_y[~np.isnan(test_y)], pred_y_proba[~np.isnan(test_y)])
            accuracy = accuracy_score(test_y[~np.isnan(test_y)], pred_y[~np.isnan(test_y)])
            logging.info(f"AUC: {auc:.8f}, Accuracy: {accuracy:.4f}")
        except:pass

        test_data['pred_prob'] = pred_y_proba
    else:
        model = get_model(type)
        test_y, pred_y = incre_fit(model, train_x, train_y, test_x, test_y, data_file_url, save_shap=save_shap)
        try:
            mse = mean_squared_error(test_y[~np.isnan(test_y)], pred_y[~np.isnan(test_y)])
            logging.info(f"MSE: {mse:.8f}")
        except : pass
        logging.info(test_data.shape)
        test_data['pred_prob'] = pred_y.astype('float64')
        logging.info(test_data.shape)
    output_columns = ['trade_date', 'name', 'stock_code', 'pred_prob','std_his_high', '10d_yield_rate', '2d_yield_rate', 'st_type','post_high','post_close','post2_close','post2_high',
                            'open3_yield_rate', 'open2_yield_rate', 'limit_times', 'close','pre_close','post_open', 'post2_open', 'post3_open', 'post4_open', 'post5_open', 'post6_open', 'post12_open',
                            'industry', 'industry_encode','atr_qfq','close_rate', 'amount', 'vol', 'turnover_rate', 'turnover_rate_f', 'circ_mv', 'total_mv', 'volume_ratio']
    label = getattr(test_y, "name", None)
    if label not in output_columns and label in test_data.columns:
        output_columns.append(label)
    test_data = test_data[[col for col in output_columns if col in test_data.columns]]
    return test_data

def sort_within_group(group):
    rank_data = group.rank(method='min', ascending=False)
    rank_data_min = rank_data.min()
    rank_data_max = rank_data.max()
    return (rank_data_max-rank_data)/(rank_data_max-rank_data_min)


def _add_top_quantile_label(frame, label, base_label, quantile):
    if base_label not in frame.columns:
        frame = prepare_training_label(frame, base_label)
    base = pd.to_numeric(frame[base_label], errors="coerce")
    valid = base.notna()
    ranks = base[valid].groupby(frame.loc[valid, "trade_date"].astype(str)).rank(pct=True, ascending=False, method="first")
    frame[label] = np.nan
    frame.loc[valid, label] = (ranks <= quantile).astype(int)
    return frame


def prepare_training_label(factor_data, label):
    if label in factor_data.columns:
        factor_data[label] = pd.to_numeric(factor_data[label], errors="coerce")
        return factor_data
    if label == "risk_adjusted_10d_yield_rate":
        base_return = pd.to_numeric(factor_data["10d_yield_rate"], errors="coerce")
        atr = pd.to_numeric(factor_data["atr_qfq"], errors="coerce")
        close_column = require_front_adjusted_column(
            factor_data.columns,
            "close",
            context="risk_adjusted_10d_yield_rate",
        )
        close = pd.to_numeric(factor_data[close_column], errors="coerce")
        atr_ratio = (atr / close).clip(lower=0.01, upper=0.20)
        factor_data[label] = base_return / atr_ratio
    elif label == "executable_10d_open_return":
        buy = pd.to_numeric(factor_data["post_open"], errors="coerce")
        sell = pd.to_numeric(factor_data["post12_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        factor_data[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_5d_open_return":
        buy = pd.to_numeric(factor_data["post_open"], errors="coerce")
        sell = pd.to_numeric(factor_data["post6_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        factor_data[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_3d_open_return":
        buy = pd.to_numeric(factor_data["post_open"], errors="coerce")
        sell = pd.to_numeric(factor_data["post4_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        factor_data[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_1d_open_return":
        buy = pd.to_numeric(factor_data["post_open"], errors="coerce")
        sell = pd.to_numeric(factor_data["post2_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        factor_data[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_1d_open_top10":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_1d_open_return", 0.10)
    elif label == "executable_1d_open_top05":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_1d_open_return", 0.05)
    elif label == "executable_1d_open_top20":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_1d_open_return", 0.20)
    elif label == "executable_3d_open_top10":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_3d_open_return", 0.10)
    elif label == "executable_3d_open_top05":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_3d_open_return", 0.05)
    elif label == "executable_3d_open_top20":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_3d_open_return", 0.20)
    elif label == "executable_5d_open_top10":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_5d_open_return", 0.10)
    elif label == "executable_5d_open_top05":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_5d_open_return", 0.05)
    elif label == "executable_5d_open_top20":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_5d_open_return", 0.20)
    elif label == "executable_10d_open_top10":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_10d_open_return", 0.10)
    elif label == "executable_10d_open_top05":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_10d_open_return", 0.05)
    elif label == "executable_10d_open_top20":
        factor_data = _add_top_quantile_label(factor_data, label, "executable_10d_open_return", 0.20)
    elif label == "excess_10d_yield_rate":
        if "adjust_10d_yield_rate" not in factor_data.columns:
            raise ValueError("adjust_10d_yield_rate is required for excess_10d_yield_rate")
        factor_data[label] = pd.to_numeric(factor_data["adjust_10d_yield_rate"], errors="coerce")
    elif label not in factor_data.columns:
        raise ValueError(f"label column not found: {label}")
    return factor_data


def load_selected_features(data_file_url, label):
    candidates = [
        Path(data_file_url) / f"selected_features_{label}.json",
        Path(data_file_url) / "selected_features.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features", payload if isinstance(payload, list) else [])
        return [str(feature) for feature in features]
    return None


def label_required_columns(label):
    if label == "risk_adjusted_10d_yield_rate":
        return ["10d_yield_rate", "atr_qfq", "close_qfq"]
    if label == "executable_10d_open_return":
        return ["post_open", "post12_open"]
    if label == "executable_5d_open_return":
        return ["post_open", "post6_open"]
    if label == "executable_3d_open_return":
        return ["post_open", "post4_open"]
    if label == "executable_1d_open_return":
        return ["post_open", "post2_open"]
    if label.startswith("executable_") and "_top" in label:
        if "_10d_" in label:
            return ["post_open", "post12_open"]
        if "_5d_" in label:
            return ["post_open", "post6_open"]
        if "_3d_" in label:
            return ["post_open", "post4_open"]
        if "_1d_" in label:
            return ["post_open", "post2_open"]
    if label == "excess_10d_yield_rate":
        return ["adjust_10d_yield_rate"]
    return [label]


def split_label_required_columns(label):
    if label == "risk_adjusted_10d_yield_rate":
        return ["10d_yield_rate"]
    if label == "excess_10d_yield_rate":
        return ["adjust_10d_yield_rate"]
    if label.startswith("executable_") and "_top" in label:
        if "_10d_" in label:
            return ["executable_10d_open_return"]
        if "_5d_" in label:
            return ["executable_5d_open_return"]
        if "_3d_" in label:
            return ["executable_3d_open_return"]
        if "_1d_" in label:
            return ["executable_1d_open_return"]
    return [label]


def split_feature_required_columns(label):
    if label == "risk_adjusted_10d_yield_rate":
        return ["atr_qfq", "close_qfq"]
    return []


def _schema_columns(path: Path) -> set[str]:
    return set(ds.dataset(path, format="parquet").schema.names)


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _read_parquet_with_fragment_fallback(path: Path, columns: list[str], filters):
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except pa.ArrowInvalid as exc:
        if not path.is_dir():
            raise
        logging.warning(
            "parquet dataset read failed for %s; falling back to per-file reads: %s",
            path,
            exc,
        )
        frames = [
            pd.read_parquet(part, columns=columns, filters=filters)
            for part in sorted(path.glob("*.parquet"))
        ]
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)


def _duckdb_table_columns(db_path: Path, table: str) -> set[str]:
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as conn:
        return {str(row[0]) for row in conn.execute(f"DESCRIBE {_quote_ident(table)}").fetchall()}


def _read_duckdb_frame(db_path: Path, table: str, columns: list[str], *, data_start_dt: str) -> pd.DataFrame:
    import duckdb

    selected = ", ".join(_quote_ident(column) for column in columns)
    sql = (
        f"SELECT {selected} "
        f"FROM {_quote_ident(table)} "
        "WHERE trade_date >= ?"
    )
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(sql, [str(data_start_dt)]).fetchdf()


def _read_duckdb_factor_data(data_start_dt, label, data_file_url, selected_features=None):
    feature_db_path = resolve_model_feature_duckdb_path(data_file_url, require_exists=True)
    label_db_path = resolve_model_label_duckdb_path(data_file_url, require_exists=True)
    feature_table = resolve_model_feature_duckdb_table(data_file_url)
    label_table = resolve_model_label_duckdb_table(data_file_url)
    if not feature_table or not label_table:
        raise RuntimeError("active L3 DuckDB assets are missing bound feature/label table names")

    feature_columns_available = _duckdb_table_columns(feature_db_path, feature_table)
    label_columns_available = _duckdb_table_columns(label_db_path, label_table)
    requested_selected_features = list(selected_features or [])
    resolved_selected = resolve_selected_feature_aliases(
        requested_selected_features,
        feature_columns_available,
    )

    feature_columns = list(
        dict.fromkeys(
            ["stock_code", "trade_date"]
            + split_feature_required_columns(label)
            + model_output_columns(label)
            + resolved_selected["query_columns"]
        )
    )
    feature_columns = [column for column in feature_columns if column in feature_columns_available]

    label_columns = list(
        dict.fromkeys(
            ["stock_code", "trade_date", "5d_yield_rate", "open6_yield_rate", "10d_yield_rate"]
            + split_label_required_columns(label)
        )
    )
    label_columns = [column for column in label_columns if column in label_columns_available]
    missing_label_columns = [
        column for column in split_label_required_columns(label) if column not in label_columns
    ]
    if missing_label_columns:
        raise ValueError(
            f"prediction_label_parts DuckDB table missing required columns for {label}: {missing_label_columns}"
        )

    feature_frame = _read_duckdb_frame(
        feature_db_path,
        feature_table,
        feature_columns,
        data_start_dt=data_start_dt,
    )
    for requested_name, source_name in resolved_selected["alias_pairs"]:
        feature_frame[requested_name] = feature_frame[source_name]
    label_frame = _read_duckdb_frame(
        label_db_path,
        label_table,
        label_columns,
        data_start_dt=data_start_dt,
    )
    return feature_frame.merge(label_frame, on=["stock_code", "trade_date"], how="inner")


def _read_split_factor_data(data_start_dt, label, data_file_url, selected_features=None):
    feature_path = resolve_model_feature_path(data_file_url, require_exists=True)
    label_path = resolve_model_label_path(data_file_url, require_exists=True)
    feature_columns_available = _schema_columns(feature_path)
    label_columns_available = _schema_columns(label_path)
    filters = [("trade_date", ">=", str(data_start_dt))]
    requested_selected_features = list(selected_features or [])
    resolved_selected = resolve_selected_feature_aliases(
        requested_selected_features,
        feature_columns_available,
    )

    feature_columns = list(
        dict.fromkeys(
            ["stock_code", "trade_date"]
            + split_feature_required_columns(label)
            + model_output_columns(label)
            + resolved_selected["query_columns"]
        )
    )
    feature_columns = [column for column in feature_columns if column in feature_columns_available]

    label_columns = list(
        dict.fromkeys(
            ["stock_code", "trade_date", "5d_yield_rate", "open6_yield_rate", "10d_yield_rate"]
            + split_label_required_columns(label)
        )
    )
    label_columns = [column for column in label_columns if column in label_columns_available]
    missing_label_columns = [
        column for column in split_label_required_columns(label) if column not in label_columns
    ]
    if missing_label_columns:
        raise ValueError(
            f"prediction_label_parts missing required columns for {label}: {missing_label_columns}"
        )

    feature_frame = _read_parquet_with_fragment_fallback(
        feature_path, columns=feature_columns, filters=filters
    )
    for requested_name, source_name in resolved_selected["alias_pairs"]:
        feature_frame[requested_name] = feature_frame[source_name]
    label_frame = _read_parquet_with_fragment_fallback(
        label_path, columns=label_columns, filters=filters
    )
    return feature_frame.merge(label_frame, on=["stock_code", "trade_date"], how="inner")


def model_output_columns(label=None):
    columns = ['trade_date', 'name', 'stock_code', 'pred_prob','std_his_high', '10d_yield_rate', '2d_yield_rate', 'st_type','post_high','post_close','post2_close','post2_high',
                            'open3_yield_rate', 'open2_yield_rate', 'limit_times', 'close','pre_close','post_open', 'post2_open', 'post3_open', 'post4_open', 'post5_open', 'post6_open', 'post12_open',
                            'industry', 'industry_encode','atr_qfq','close_rate', 'amount', 'vol', 'turnover_rate', 'turnover_rate_f', 'circ_mv', 'total_mv', 'volume_ratio']
    if label and label not in columns:
        columns.append(label)
    return columns


def get_factor_data(
    data_start_dt,
    data_test_dt,
    label,
    data_file_url,
    stock_pool_path=None,
    selected_features=None,
    use_light_factor_data=False,
    feature_source=None,
):
    '''factor_data = pd.read_sql('SELECT * FROM CDB.stock_factor_data ORDER BY stock_code, trade_date', engine)

    factor_data = factor_data[factor_data['stock_code'].isin(stock_code_lst)]
    factor_data = factor_data[(factor_data['trade_date'] >= data_start_dt) & (factor_data['trade_date'] <= data_end_dt)]'''


    if use_light_factor_data:
        features_path = None
        if selected_features:
            features_path = Path(data_file_url) / f"selected_features_runtime_{label}.json"
            payload = {"label": label, "features": list(selected_features)}
            features_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            return get_light_factor_data(
                data_dir=data_file_url,
                features_path=features_path or (Path(data_file_url) / f"selected_features_{label}.json"),
                train_start=data_start_dt,
                test_start=data_test_dt,
                label=label,
                stock_pool_path=stock_pool_path,
                allow_missing_features=True,
            )
        finally:
            if features_path and features_path.exists():
                features_path.unlink()

    selected_features = selected_features or load_selected_features(data_file_url, label)
    if use_legacy_mixed_features(feature_source):
        require_legacy_model_asset_chain_opt_in(reason="legacy mixed factor parquet input")
        read_columns = None
        parquet_path = resolve_legacy_mixed_factor_path(data_file_url, require_exists=True)
        if selected_features:
            available_columns = set(pq.ParquetFile(parquet_path).schema_arrow.names)
            needed_columns = list(
                dict.fromkeys(
                    [
                        "stock_code",
                        "trade_date",
                        "name",
                        "st_type",
                        "limit_times",
                        "5d_yield_rate",
                        "open6_yield_rate",
                        "10d_yield_rate",
                    ]
                    + label_required_columns(label)
                    + model_output_columns(label)
                    + list(selected_features)
                )
            )
            read_columns = [col for col in needed_columns if col in available_columns]
        read_filters = [("trade_date", ">=", str(data_start_dt))]
        factor_data = pd.read_parquet(
            parquet_path,
            columns=read_columns,
            filters=read_filters,
        )
    else:
        feature_duckdb_table = resolve_model_feature_duckdb_table(data_file_url)
        label_duckdb_table = resolve_model_label_duckdb_table(data_file_url)
        if feature_duckdb_table and label_duckdb_table:
            factor_data = _read_duckdb_factor_data(
                data_start_dt,
                label,
                data_file_url,
                selected_features=selected_features,
            )
        else:
            factor_data = _read_split_factor_data(
                data_start_dt,
                label,
                data_file_url,
                selected_features=selected_features,
            )
    if stock_pool_path:
        stock_pool = load_stock_pool(stock_pool_path)
        factor_data = filter_frame_by_stock_pool(factor_data, stock_pool)

    if "name" in factor_data.columns:
        factor_data = factor_data[factor_data['name'].notna()]
        factor_data = factor_data[~factor_data['name'].astype(str).str.contains('ST', na=False)]
    if "st_type" in factor_data.columns:
        factor_data = factor_data[factor_data['st_type'] != 'ST']
    print(factor_data.shape)
    if "limit_times" in factor_data.columns:
        factor_data = factor_data[factor_data['limit_times'].isnull()]
    # print(factor_data.shape)

    # logging.info(factor_data['trade_date'].max(),factor_data['trade_date'].min(),'1')
    factor_data = prepare_training_label(factor_data, label)
    # logging.info(factor_data['trade_date'].max(),factor_data['trade_date'].min(),'3')

    if "5d_yield_rate" in factor_data.columns:
        factor_data['5d_yield_rate_rank']= factor_data.groupby('trade_date')['5d_yield_rate'].transform(sort_within_group)
    if "open6_yield_rate" in factor_data.columns:
        factor_data['open6_yield_rate_rank']= factor_data.groupby('trade_date')['open6_yield_rate'].transform(sort_within_group)
    if "10d_yield_rate" in factor_data.columns:
        factor_data['10d_yield_rate_rank']= factor_data.groupby('trade_date')['10d_yield_rate'].transform(sort_within_group)
	# 使用 np.select 计算收益率

    logging.info(factor_data.shape)

    transformer_factor_list = [
        'close_rate', 'open_rate', 'high_rate', 'low_rate', 'vol', 'amount', 'stock_encode','st_type',
        'high_open_rate', 'high_close_rate', 'low_open_rate', 'low_close_rate',
        'macd', 'macdsignal', 'macdhist', 'dema', 'ema', 'kama', 'ma', 'midpoint', 'midprice',
        'diff_close_low', 'diff_close_high', 'diff_high_low',
        'std_his_low', 'std_cost_5pct', 'std_cost_15pct', 'std_cost_50pct', 'std_cost_85pct', 'std_cost_95pct',
        'pre_close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb',
        'total_share', 'float_share', 'total_mv', 'circ_mv',
        'industry_encode',
        'limit_times',
        'open6_yield_rate_rank', '10d_yield_rate_rank'
    ]
    # train_data_new, test_data_new, model = add_fttransformer_features_simple(
    #     train_data, test_data, transformer_factor_list, label
    # )

    factor_list = [
    # 'close','open','high','low','vol','amount','stock_encode',
	'close_rate','open_rate','high_rate','low_rate','vol','amount','stock_encode', 'high_open_rate', 'high_close_rate', 'low_open_rate', 'low_close_rate','st_type',
    'macd', 'macdsignal', 'macdhist', 'dema', 'ema', 'kama','ma', 'midpoint', 'midprice', 't3', 'tema', 'trima', 'wma','dema_10', 'ema_10', 'kama_10',
	'diff_close_low','diff_close_high','diff_high_low','ema_close_low','ema_close_high','ema_high_low','ema_close_low_10','ema_close_high_10','ema_high_low_10',
	'std_his_low','std_cost_5pct','std_cost_15pct','std_cost_50pct','std_cost_85pct','std_cost_95pct','std_weight_avg','winner_rate', 'std_his_high',
	'pre_close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio','pre_yield_rate',
    'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
    'industry_encode', 'act_ent_type_encode',
	'fd_amount','fd_amount_rate','fd_vol_rate','open_times','up_stat_nom','up_stat_denom','up_stat_rate','limit_times','first_time_int' ,'last_time_int',
	'l_sell','l_buy','l_amount','net_amount','net_rate','amount_rate','float_values',

    # 'ths_rank','ths_hot','dc_rank',


    # 'alpha158_kmid', 'alpha158_klen', 'alpha158_kmid2', 'alpha158_kup',
    # 'alpha158_kup2', 'alpha158_klow', 'alpha158_klow2', 'alpha158_ksft', 'alpha158_ksft2',
    # # 趋势类因子（15个）
    # 'alpha158_roc5', 'alpha158_roc10', 'alpha158_roc20', 'alpha158_roc30', 'alpha158_roc60',
    # 'alpha158_ma5', 'alpha158_ma10', 'alpha158_ma20', 'alpha158_ma30', 'alpha158_ma60',
    # 'alpha158_std5', 'alpha158_std10', 'alpha158_std20', 'alpha158_std30', 'alpha158_std60',
    # # 波动类因子（25个）
    # 'alpha158_max5', 'alpha158_max10', 'alpha158_max20', 'alpha158_max30', 'alpha158_max60',
    # 'alpha158_min5', 'alpha158_min10', 'alpha158_min20', 'alpha158_min30', 'alpha158_min60',
    # 'alpha158_qtlu5', 'alpha158_qtlu10', 'alpha158_qtlu20', 'alpha158_qtlu30', 'alpha158_qtlu60',
    # 'alpha158_qtld5', 'alpha158_qtld10', 'alpha158_qtld20', 'alpha158_qtld30', 'alpha158_qtld60',
    # 'alpha158_rsv5', 'alpha158_rsv10', 'alpha158_rsv20', 'alpha158_rsv30', 'alpha158_rsv60',

    # # 极值位置因子（15个）
    # 'alpha158_imax5', 'alpha158_imax10', 'alpha158_imax20', 'alpha158_imax30', 'alpha158_imax60',
    # 'alpha158_imin5', 'alpha158_imin10', 'alpha158_imin20', 'alpha158_imin30', 'alpha158_imin60',
    # 'alpha158_imxd5', 'alpha158_imxd10', 'alpha158_imxd20', 'alpha158_imxd30', 'alpha158_imxd60',

    # # 价量统计类因子（25个）
    # 'alpha158_corr5', 'alpha158_corr10', 'alpha158_corr20', 'alpha158_corr30', 'alpha158_corr60',
    # 'alpha158_cord5', 'alpha158_cord10', 'alpha158_cord20', 'alpha158_cord30', 'alpha158_cord60',
    # 'alpha158_cntp5', 'alpha158_cntp10', 'alpha158_cntp20', 'alpha158_cntp30', 'alpha158_cntp60',
    # 'alpha158_cntn5', 'alpha158_cntn10', 'alpha158_cntn20', 'alpha158_cntn30', 'alpha158_cntn60',
    # 'alpha158_cntd5', 'alpha158_cntd10', 'alpha158_cntd20', 'alpha158_cntd30', 'alpha158_cntd60',

    # # RSI类因子（6个）
    # 'alpha158_sump6', 'alpha158_sump12', 'alpha158_sump24',
    # 'alpha158_sumd6', 'alpha158_sumd12', 'alpha158_sumd24',

    # # 复合技术指标（11个）
    # 'alpha158_boll', 'alpha158_adx',
    # 'alpha158_k6', 'alpha158_k12', 'alpha158_k24',
    # 'alpha158_d6', 'alpha158_d12', 'alpha158_d24',
    # 'alpha158_j6', 'alpha158_j12', 'alpha158_j24',

    # # 其他因子（3个）
    # 'alpha158_rank5', 'alpha158_rank10', 'alpha158_rank20',


    # 'st_code',# 'days_from_20200101' , # 'weekday', 'quarter',  'week_of_year','month',# 'days_from_20200101' , 'year', 'days',
    # 'season_eps', 'season_dt_eps', 'season_total_revenue_ps', 'season_revenue_ps', 'season_capital_rese_ps', 'season_surplus_rese_ps', 'season_undist_profit_ps',
    # 'season_extra_item', 'season_profit_dedt', 'season_gross_margin', 'season_current_ratio', 'season_quick_ratio', 'season_cash_ratio', 'season_ar_turn', 'season_ca_turn',
    # 'season_fa_turn', 'season_assets_turn', 'season_op_income', 'season_ebit', 'season_ebitda', 'season_fcff', 'season_fcfe', 'season_current_exint', 'season_noncurrent_exint', 'season_interestdebt',
    # 'season_netdebt', 'season_tangible_asset', 'season_working_capital', 'season_networking_capital', 'season_invest_capital', 'season_retained_earnings', 'season_diluted2_eps', 'season_bps',
    # 'season_ocfps', 'season_retainedps', 'season_cfps', 'season_ebit_ps', 'season_fcff_ps', 'season_fcfe_ps', 'season_netprofit_margin', 'season_grossprofit_margin', 'season_cogs_of_sales',
    # 'season_expense_of_sales', 'season_profit_to_gr', 'season_saleexp_to_gr', 'season_adminexp_of_gr', 'season_finaexp_of_gr', 'season_impai_ttm', 'season_gc_of_gr', 'season_op_of_gr',
    # 'season_ebit_of_gr', 'season_roe', 'season_roe_waa', 'season_roe_dt', 'season_roa', 'season_npta', 'season_roic', 'season_roe_yearly', 'season_roa2_yearly', 'season_debt_to_assets', 'season_assets_to_eqt',
    # 'season_dp_assets_to_eqt', 'season_ca_to_assets', 'season_nca_to_assets', 'season_tbassets_to_totalassets', 'season_int_to_talcap', 'season_eqt_to_talcapital', 'season_currentdebt_to_debt',
    # 'season_longdeb_to_debt', 'season_ocf_to_shortdebt', 'season_debt_to_eqt', 'season_eqt_to_debt', 'season_eqt_to_interestdebt', 'season_tangibleasset_to_debt', 'season_tangasset_to_intdebt',
    # 'season_tangibleasset_to_netdebt', 'season_ocf_to_debt', 'season_turn_days', 'season_roa_yearly', 'season_roa_dp', 'season_fixed_assets', 'season_profit_to_op', 'season_q_saleexp_to_gr', 'season_q_gc_to_gr',
    # 'season_q_roe', 'season_q_dt_roe', 'season_q_npta', 'season_q_ocf_to_sales', 'season_basic_eps_yoy', 'season_dt_eps_yoy', 'season_cfps_yoy', 'season_op_yoy', 'season_ebt_yoy', 'season_netprofit_yoy',
    # 'season_dt_netprofit_yoy', 'season_ocf_yoy', 'season_roe_yoy', 'season_bps_yoy', 'season_assets_yoy', 'season_eqt_yoy', 'season_tr_yoy', 'season_or_yoy', 'season_q_sales_yoy', 'season_q_op_qoq', 'season_equity_yoy',
    # 'year_eps', 'year_dt_eps', 'year_total_revenue_ps', 'year_revenue_ps', 'year_capital_rese_ps', 'year_surplus_rese_ps', 'year_undist_profit_ps', 'year_extra_item', 'year_profit_dedt',
    # 'year_gross_margin', 'year_current_ratio', 'year_quick_ratio', 'year_cash_ratio', 'year_ar_turn', 'year_ca_turn', 'year_fa_turn', 'year_assets_turn', 'year_op_income', 'year_ebit', 'year_ebitda',
    # 'year_fcff', 'year_fcfe', 'year_current_exint', 'year_noncurrent_exint', 'year_interestdebt', 'year_netdebt', 'year_tangible_asset', 'year_working_capital', 'year_networking_capital',
    # 'year_invest_capital', 'year_retained_earnings', 'year_diluted2_eps', 'year_bps', 'year_ocfps', 'year_retainedps', 'year_cfps', 'year_ebit_ps', 'year_fcff_ps', 'year_fcfe_ps', 'year_netprofit_margin',
    # 'year_grossprofit_margin', 'year_cogs_of_sales', 'year_expense_of_sales', 'year_profit_to_gr', 'year_saleexp_to_gr', 'year_adminexp_of_gr', 'year_finaexp_of_gr', 'year_impai_ttm',
    # 'year_gc_of_gr', 'year_op_of_gr', 'year_ebit_of_gr', 'year_roe', 'year_roe_waa', 'year_roe_dt', 'year_roa', 'year_npta', 'year_roic', 'year_roe_yearly', 'year_roa2_yearly', 'year_debt_to_assets',
    # 'year_assets_to_eqt', 'year_dp_assets_to_eqt', 'year_ca_to_assets', 'year_nca_to_assets', 'year_tbassets_to_totalassets', 'year_int_to_talcap', 'year_eqt_to_talcapital',
    # 'year_currentdebt_to_debt', 'year_longdeb_to_debt', 'year_ocf_to_shortdebt', 'year_debt_to_eqt', 'year_eqt_to_debt', 'year_eqt_to_interestdebt', 'year_tangibleasset_to_debt',
    # 'year_tangasset_to_intdebt', 'year_tangibleasset_to_netdebt', 'year_ocf_to_debt', 'year_turn_days', 'year_roa_yearly', 'year_roa_dp', 'year_fixed_assets', 'year_profit_to_op',
    # 'year_q_saleexp_to_gr', 'year_q_gc_to_gr', 'year_q_roe', 'year_q_dt_roe', 'year_q_npta', 'year_q_ocf_to_sales', 'year_basic_eps_yoy', 'year_dt_eps_yoy', 'year_cfps_yoy', 'year_op_yoy',
    # 'year_ebt_yoy', 'year_netprofit_yoy', 'year_dt_netprofit_yoy', 'year_ocf_yoy', 'year_roe_yoy', 'year_bps_yoy', 'year_assets_yoy', 'year_eqt_yoy', 'year_tr_yoy', 'year_or_yoy', 'year_q_sales_yoy',
    # 'year_q_op_qoq', 'year_equity_yoy',
	'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount','buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount','buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount','buy_elg_vol',
	'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount','net_mf_vol', 'net_mf_amount',
	'asi_bfq','asi_hfq','asi_qfq','asit_bfq','asit_hfq','asit_qfq','atr_bfq','atr_hfq','atr_qfq','bbi_bfq','bbi_hfq','bbi_qfq','bias1_bfq','bias1_hfq','bias1_qfq','bias2_bfq','bias2_hfq','bias2_qfq',
	'bias3_bfq','bias3_hfq','bias3_qfq','boll_lower_bfq','boll_lower_hfq','boll_lower_qfq','boll_mid_bfq','boll_mid_hfq','boll_mid_qfq','boll_upper_bfq','boll_upper_hfq','boll_upper_qfq','brar_ar_bfq',
	'brar_ar_hfq','brar_ar_qfq','brar_br_bfq','brar_br_hfq','brar_br_qfq','cci_bfq','cci_hfq','cci_qfq','cr_bfq','cr_hfq','cr_qfq','dfma_dif_bfq','dfma_dif_hfq','dfma_dif_qfq','dfma_difma_bfq','dfma_difma_hfq',
	'dfma_difma_qfq','dmi_adx_bfq','dmi_adx_hfq','dmi_adx_qfq','dmi_adxr_bfq','dmi_adxr_hfq','dmi_adxr_qfq','dmi_mdi_bfq','dmi_mdi_hfq','dmi_mdi_qfq','dmi_pdi_bfq','dmi_pdi_hfq','dmi_pdi_qfq','downdays','updays',
	'dpo_bfq','dpo_hfq','dpo_qfq','madpo_bfq','madpo_hfq','madpo_qfq','ema_bfq_10','ema_bfq_20','ema_bfq_250','ema_bfq_30','ema_bfq_5','ema_bfq_60','ema_bfq_90','ema_hfq_10','ema_hfq_20','ema_hfq_250','ema_hfq_30','ema_hfq_5',
	'ema_hfq_60','ema_hfq_90','ema_qfq_10','ema_qfq_20','ema_qfq_250','ema_qfq_30','ema_qfq_5','ema_qfq_60','ema_qfq_90','emv_bfq','emv_hfq','emv_qfq','maemv_bfq','maemv_hfq','maemv_qfq','expma_12_bfq','expma_12_hfq','expma_12_qfq',
	'expma_50_bfq','expma_50_hfq','expma_50_qfq','kdj_bfq','kdj_hfq','kdj_qfq','kdj_d_bfq','kdj_d_hfq','kdj_d_qfq','kdj_k_bfq','kdj_k_hfq','kdj_k_qfq','ktn_down_bfq','ktn_down_hfq','ktn_down_qfq','ktn_mid_bfq','ktn_mid_hfq','ktn_mid_qfq',
	'ktn_upper_bfq','ktn_upper_hfq','ktn_upper_qfq','lowdays','topdays','ma_bfq_10','ma_bfq_20','ma_bfq_250','ma_bfq_30','ma_bfq_5','ma_bfq_60','ma_bfq_90','ma_hfq_10','ma_hfq_20','ma_hfq_250','ma_hfq_30','ma_hfq_5','ma_hfq_60',
	'ma_hfq_90','ma_qfq_10','ma_qfq_20','ma_qfq_250','ma_qfq_30','ma_qfq_5','ma_qfq_60','ma_qfq_90','macd_bfq','macd_hfq','macd_qfq','macd_dea_bfq','macd_dea_hfq','macd_dea_qfq','macd_dif_bfq','macd_dif_hfq','macd_dif_qfq','mass_bfq',
	'mass_hfq','mass_qfq','ma_mass_bfq','ma_mass_hfq','ma_mass_qfq','mfi_bfq','mfi_hfq','mfi_qfq','mtm_bfq','mtm_hfq','mtm_qfq','mtmma_bfq','mtmma_hfq','mtmma_qfq','obv_bfq','obv_hfq','obv_qfq','psy_bfq','psy_hfq','psy_qfq','psyma_bfq',
	'psyma_hfq','psyma_qfq','roc_bfq','roc_hfq','roc_qfq','maroc_bfq','maroc_hfq','maroc_qfq','rsi_bfq_12','rsi_bfq_24','rsi_bfq_6','rsi_hfq_12','rsi_hfq_24','rsi_hfq_6','rsi_qfq_12','rsi_qfq_24','rsi_qfq_6','taq_down_bfq','taq_down_hfq',
	'taq_down_qfq','taq_mid_bfq','taq_mid_hfq','taq_mid_qfq','taq_up_bfq','taq_up_hfq','taq_up_qfq','trix_bfq','trix_hfq','trix_qfq','trma_bfq','trma_hfq','trma_qfq','vr_bfq','vr_hfq','vr_qfq','wr_bfq','wr_hfq','wr_qfq','wr1_bfq','wr1_hfq',
	'wr1_qfq','xsii_td1_bfq','xsii_td1_hfq','xsii_td1_qfq','xsii_td2_bfq','xsii_td2_hfq','xsii_td2_qfq','xsii_td3_bfq','xsii_td3_hfq','xsii_td3_qfq','xsii_td4_bfq','xsii_td4_hfq','xsii_td4_qfq',
	# 'tdx_close','tdx_open','tdx_high','tdx_low','tdx_pre_close',
	# 'tdx_change','tdx_pct_change','tdx_vol','tdx_amount',
	# 'tdx_rise','tdx_vol_ratio','tdx_turnover_rate',
    # 'tdx_swing','tdx_up_num','tdx_down_num','tdx_limit_up_num','tdx_limit_down_num','tdx_lu_days','tdx_3day','tdx_5day','tdx_10day','tdx_20day','tdx_60day','tdx_mtd',
    # 'tdx_ytd','tdx_1year','tdx_pe','tdx_pb','tdx_float_mv','tdx_ab_total_mv','tdx_float_share','tdx_total_share','tdx_bm_buy_net','tdx_bm_buy_ratio','tdx_bm_net','tdx_bm_ratio',
	'gtja_alpha001', 'gtja_alpha002', 'gtja_alpha003', 'gtja_alpha004', 'gtja_alpha005', 'gtja_alpha006', 'gtja_alpha007', 'gtja_alpha008', 'gtja_alpha009', 'gtja_alpha010', 'gtja_alpha011', 'gtja_alpha012', 'gtja_alpha013', 'gtja_alpha014', 'gtja_alpha015', 'gtja_alpha016', 'gtja_alpha017', 'gtja_alpha018', 'gtja_alpha019', 'gtja_alpha020', 'gtja_alpha021', 'gtja_alpha022', 'gtja_alpha023', 'gtja_alpha024', 'gtja_alpha025', 'gtja_alpha026', 'gtja_alpha027', 'gtja_alpha028', 'gtja_alpha029', 'gtja_alpha030', 'gtja_alpha031', 'gtja_alpha032', 'gtja_alpha033', 'gtja_alpha034', 'gtja_alpha035', 'gtja_alpha036', 'gtja_alpha037', 'gtja_alpha038', 'gtja_alpha039', 'gtja_alpha040', 'gtja_alpha041', 'gtja_alpha042', 'gtja_alpha043', 'gtja_alpha044', 'gtja_alpha045', 'gtja_alpha046', 'gtja_alpha047', 'gtja_alpha048', 'gtja_alpha049', 'gtja_alpha050', 'gtja_alpha051', 'gtja_alpha052', 'gtja_alpha053', 'gtja_alpha054', 'gtja_alpha055', 'gtja_alpha056', 'gtja_alpha057', 'gtja_alpha058', 'gtja_alpha059', 'gtja_alpha060', 'gtja_alpha061', 'gtja_alpha062', 'gtja_alpha063', 'gtja_alpha064', 'gtja_alpha065', 'gtja_alpha066', 'gtja_alpha067', 'gtja_alpha068', 'gtja_alpha069', 'gtja_alpha070', 'gtja_alpha071', 'gtja_alpha072', 'gtja_alpha073', 'gtja_alpha074', 'gtja_alpha075', 'gtja_alpha076', 'gtja_alpha077', 'gtja_alpha078', 'gtja_alpha079', 'gtja_alpha080', 'gtja_alpha081', 'gtja_alpha082', 'gtja_alpha083', 'gtja_alpha084', 'gtja_alpha085', 'gtja_alpha086', 'gtja_alpha087', 'gtja_alpha088', 'gtja_alpha089', 'gtja_alpha090', 'gtja_alpha091', 'gtja_alpha092', 'gtja_alpha093', 'gtja_alpha094', 'gtja_alpha095', 'gtja_alpha096', 'gtja_alpha097', 'gtja_alpha098', 'gtja_alpha099', 'gtja_alpha100', 'gtja_alpha101', 'gtja_alpha102', 'gtja_alpha103', 'gtja_alpha104', 'gtja_alpha105', 'gtja_alpha106', 'gtja_alpha107', 'gtja_alpha108', 'gtja_alpha109', 'gtja_alpha110', 'gtja_alpha111', 'gtja_alpha112', 'gtja_alpha113', 'gtja_alpha114', 'gtja_alpha115', 'gtja_alpha116', 'gtja_alpha117', 'gtja_alpha118', 'gtja_alpha119', 'gtja_alpha120', 'gtja_alpha121', 'gtja_alpha122', 'gtja_alpha123', 'gtja_alpha124', 'gtja_alpha125', 'gtja_alpha126', 'gtja_alpha127', 'gtja_alpha128', 'gtja_alpha129', 'gtja_alpha130', 'gtja_alpha131', 'gtja_alpha132', 'gtja_alpha133', 'gtja_alpha134', 'gtja_alpha135', 'gtja_alpha136', 'gtja_alpha137', 'gtja_alpha138', 'gtja_alpha139', 'gtja_alpha140', 'gtja_alpha141', 'gtja_alpha142', 'gtja_alpha143', 'gtja_alpha144', 'gtja_alpha145', 'gtja_alpha146', 'gtja_alpha147', 'gtja_alpha148', 'gtja_alpha149', 'gtja_alpha150', 'gtja_alpha151', 'gtja_alpha152', 'gtja_alpha153', 'gtja_alpha154', 'gtja_alpha155', 'gtja_alpha156', 'gtja_alpha157', 'gtja_alpha158', 'gtja_alpha159', 'gtja_alpha160', 'gtja_alpha161', 'gtja_alpha162', 'gtja_alpha163', 'gtja_alpha164', 'gtja_alpha165', 'gtja_alpha166', 'gtja_alpha167', 'gtja_alpha168', 'gtja_alpha169', 'gtja_alpha170', 'gtja_alpha171', 'gtja_alpha172', 'gtja_alpha173', 'gtja_alpha174', 'gtja_alpha175', 'gtja_alpha176', 'gtja_alpha177', 'gtja_alpha178', 'gtja_alpha179', 'gtja_alpha180', 'gtja_alpha181', 'gtja_alpha182', 'gtja_alpha183', 'gtja_alpha184', 'gtja_alpha185', 'gtja_alpha186', 'gtja_alpha187', 'gtja_alpha188', 'gtja_alpha189', 'gtja_alpha190', 'gtja_alpha191',
    label
    ]

#     factor_list = [
#     "rsi_bfq_6", "gtja_alpha077", "gtja_alpha083", "ktn_down_bfq", "ema_hfq_250", "sell_lg_vol", "ma", "xsii_td3_bfq",
#     "gtja_alpha059", "gtja_alpha186", "bias3_bfq", "gtja_alpha093", "bias1_bfq", "ktn_upper_bfq", "trix_hfq", "gtja_alpha079",
#     "gtja_alpha175", "ema_bfq_5", "boll_lower_bfq", "maemv_hfq", "ma_qfq_90", "kdj_k_hfq", "mass_hfq", "brar_br_bfq",
#     "gtja_alpha074", "expma_12_bfq", "macdhist", "sell_lg_amount", "buy_sm_vol", "ma_bfq_10", "asi_bfq", "net_mf_amount",
#     "gtja_alpha095", "ema_close_low", "low_rate", "bias3_hfq", "macd_dif_bfq", "xsii_td3_qfq", "trima", "ma_bfq_30",
#     "dpo_hfq", "dmi_pdi_bfq", "ps_ttm", "ema_qfq_250", "ema_hfq_30", "dmi_adxr_hfq", "boll_upper_qfq", "trma_hfq",
#     "ema_close_high", "asi_hfq", "madpo_hfq", "buy_sm_amount", "gtja_alpha160", "kdj_hfq", "gtja_alpha067", "dfma_difma_bfq",
#     "trma_bfq", "bias2_hfq", "macd_dea_qfq", "dmi_mdi_hfq", "dmi_adxr_qfq", "xsii_td3_hfq", "taq_down_qfq", "trix_bfq",
#     "taq_down_bfq", "gtja_alpha024", "dmi_mdi_bfq", "boll_upper_bfq", "std_cost_5pct", "ema_qfq_90", "wma", "bias1_hfq",
#     "boll_lower_qfq", "gtja_alpha179", "expma_50_qfq", "madpo_qfq", "pb", "gtja_alpha058", "atr_qfq", "std_cost_15pct",
#     "gtja_alpha140", "gtja_alpha031", "ma_qfq_250", "ema_bfq_250", "free_share", "gtja_alpha080", "gtja_alpha124", "gtja_alpha007",
#     "roc_bfq", "volume_ratio", "gtja_alpha064", "dmi_adxr_bfq", "taq_mid_qfq", "std_weight_avg", "kama", "gtja_alpha152",
#     "gtja_alpha148", "bbi_qfq", "ema", "gtja_alpha139", "open_rate", "ma_bfq_250", "gtja_alpha030", "macdsignal",
#     "winner_rate", "bbi_bfq", "boll_mid_bfq", "maroc_hfq", "gtja_alpha052", "maroc_bfq", "sell_sm_vol", "gtja_alpha137",
#     "ema_close_low_10", "circ_mv", "ma_hfq_30", "gtja_alpha102", "taq_mid_bfq", "gtja_alpha162", "dmi_adx_hfq", "industry_encode",
#     "asit_qfq", "atr_hfq", "taq_up_bfq", "turnover_rate_f", "gtja_alpha013", "sell_md_amount", "gtja_alpha113", "maemv_bfq",
#     "dv_ratio", "gtja_alpha182", "std_his_high", "dmi_pdi_hfq", "gtja_alpha100", "ema_10", "mtm_bfq", "gtja_alpha068",
#     "gtja_alpha063", "atr_bfq", "ema_bfq_90", "emv_bfq", "gtja_alpha157", "dmi_pdi_qfq", "gtja_alpha185", "gtja_alpha110",
#     "sell_elg_amount", "buy_lg_amount", "gtja_alpha099", "gtja_alpha008", "std_cost_50pct", "taq_down_hfq", "gtja_alpha155", "obv_bfq",
#     "gtja_alpha002", "high_rate", "pe", "madpo_bfq", "gtja_alpha154", "gtja_alpha076", "gtja_alpha046", "topdays",
#     "gtja_alpha150", "brar_br_hfq", "boll_lower_hfq", "gtja_alpha092", "gtja_alpha161", "gtja_alpha053", "ema_hfq_60", "ma_mass_qfq",
#     "ma_hfq_250", "gtja_alpha025", "ma_bfq_90", "gtja_alpha069", "rsi_hfq_24", "gtja_alpha020", "ps", "std_his_low",
#     "buy_lg_vol", "gtja_alpha191", "mass_bfq", "mtmma_qfq", "gtja_alpha061", "buy_md_amount", "gtja_alpha112", "brar_ar_bfq",
#     "cr_hfq", "gtja_alpha017", "gtja_alpha090", "total_share", "dfma_difma_hfq", "std_cost_85pct", "taq_up_hfq", "ema_high_low",
#     "ma_qfq_5", "dfma_dif_bfq", "gtja_alpha062", "macd_dif_qfq", "gtja_alpha022", "dpo_qfq", "gtja_alpha057", "gtja_alpha029",
#     "expma_50_hfq", "brar_ar_hfq", "gtja_alpha159", "bias2_bfq", "dmi_adx_bfq", "high_open_rate", "gtja_alpha146", "mtmma_bfq",
#     "ema_close_high_10", "gtja_alpha050", "gtja_alpha147", "midpoint", "dema", "gtja_alpha151", "buy_md_vol", "rsi_bfq_24",
#     "gtja_alpha131", "asi_qfq", "gtja_alpha003", "std_cost_95pct", "gtja_alpha027", "low_open_rate", "gtja_alpha010", "gtja_alpha006",
#     "gtja_alpha142", "amount_rate", "gtja_alpha037", label
# ]

    # train_factor_data = pd.concat([train_data[factor_list] , train_data_new], axis=1)
    # test_factor_data = pd.concat([test_data[factor_list] , test_data_new], axis=1)

    factor_list = [feature for feature in factor_list if feature in factor_data.columns]
    if label in factor_data.columns and label not in factor_list:
        factor_list.append(label)

    if selected_features:
        selected_features = [feature for feature in selected_features if feature in factor_data.columns and feature != label]
        factor_list = selected_features + [label]
        print(f"selected_feature_file_loaded count={len(selected_features)}")

    output_factor_columns = [col for col in model_output_columns(label) if col in factor_data.columns]
    if label in factor_data.columns and label not in output_factor_columns:
        output_factor_columns.append(label)

    train_mask = factor_data['trade_date'] < data_test_dt
    test_mask = factor_data['trade_date'] >= data_test_dt
    train_label_mask = factor_data[label].notna()

    train_data = factor_data.loc[train_mask & train_label_mask, output_factor_columns].copy()
    test_data = factor_data.loc[test_mask, output_factor_columns].copy()
    train_factor_data = factor_data.loc[train_mask & train_label_mask, factor_list].copy()
    test_factor_data = factor_data.loc[test_mask, factor_list].copy()
    logging.info(test_data.shape)


    for col in train_factor_data.select_dtypes(include=['object']).columns:
        train_factor_data[col] = pd.to_numeric(train_factor_data[col], errors='coerce')
        test_factor_data[col] = pd.to_numeric(test_factor_data[col], errors='coerce')



    train_y = train_factor_data.loc[:, label]
    train_x = train_factor_data.drop(columns=[label])
    test_y = test_factor_data.loc[:, label]
    test_x = test_factor_data.drop(columns=[label])
    train_index = pd.MultiIndex.from_frame(train_data[["stock_code", "trade_date"]])
    test_index = pd.MultiIndex.from_frame(test_data[["stock_code", "trade_date"]])
    train_x.index = train_index
    train_y.index = train_index
    test_x.index = test_index
    test_y.index = test_index
    validate_no_leakage(train_x.columns, label=label)
    validate_no_leakage(test_x.columns, label=label)


    return train_x, train_y, test_x, test_y, train_data, test_data



def model_adjust(train_x, train_y, test_x, test_y):
	model = get_model('reg')
	x = pd.concat([train_x, test_x])
	y = pd.concat([train_y, test_y])
	tscv = TimeSeriesSplit(n_splits=10)
	mse_lst = []
	for train_index, test_index in tscv.split(x):
		train_x, test_x = x.iloc[train_index], x.iloc[test_index]
		train_y, test_y = y.iloc[train_index], y.iloc[test_index]
		eval_data = _valid_eval_data(test_x, test_y)
		if eval_data is not None:
			model.fit(train_x, train_y, eval_set=[eval_data], verbose=1000)
		else:
			model.fit(train_x, train_y, verbose=1000)
		logging.info(f"?????????? {model.best_iteration}")
		pred_y = model.predict(test_x)
		mse = mean_squared_error(test_y[~np.isnan(test_y)], pred_y[~np.isnan(test_y)])
		mse_lst.append(mse)

	logging.info('??????????????????',mse_lst, '????????????', sum(mse_lst) / len(mse_lst))


def download_pred_data(data, label, data_file_url, output_table=None, prediction_output_mode=None):
	table_name = output_table or ('stock_predict_data_'+label)
	if use_legacy_prediction_db(prediction_output_mode):
		require_legacy_model_asset_chain_opt_in(reason="legacy odb prediction output")
		db_path = resolve_legacy_prediction_db_path(data_file_url)
		duckdb_sync = None
	else:
		db_path = resolve_model_prediction_db_path(data_file_url, create_parent=True)
		duckdb_sync = None
	with sqlite3.connect(db_path) as conn:
		data.to_sql(table_name, con=conn, if_exists='replace', index=False)
	if not use_legacy_prediction_db(prediction_output_mode) and l4_duckdb_sync_enabled():
		duckdb_sync = sync_prediction_table_full_to_duckdb(
			data_dir=data_file_url,
			sqlite_db_path=db_path,
			table_name=table_name,
		)
	if not use_legacy_prediction_db(prediction_output_mode):
		run_dir = resolve_prediction_run_dir(data_file_url, label=label, output_table=table_name, create=True)
		write_prediction_manifest(
			run_dir / "prediction_manifest.json",
			{
				"label": label,
				"prediction_mode": "independent",
				"prediction_db": str(db_path),
				"prediction_table": table_name,
				"row_count": int(len(data)),
				"duckdb_sync": duckdb_sync,
			},
		)

def download_pdb_data(data_start_dt, data_test_dt, label, type, data_file_url, stock_pool_path=None, output_table=None, prediction_output_mode=None, feature_source=None):

	logging.info(f'#--------------------------------------------AI模块启动--------------------------------------------#')


	logging.info(f'AI模块：1.MYSQL数据库连接成功')
	train_x, train_y, test_x, test_y, train_data, test_data = get_factor_data(data_start_dt, data_test_dt, label, data_file_url, stock_pool_path, feature_source=feature_source)
	logging.info(f'AI模块：2.数据集划分完成')


	logging.info(f'AI模块：3.模型建立完成')
	# model_adjust(train_x, train_y, test_x, test_y)
	data = model_assess(train_x, train_y, test_x, test_y, train_data, test_data, type, data_file_url)
	logging.info(f'AI模块：4.模型预测和评估完成')
	download_pred_data(data, label, data_file_url, output_table=output_table, prediction_output_mode=prediction_output_mode)
	logging.info(f'AI模块：5.预测数据入仓完成')

	logging.info(f'#--------------------------------------------AI模块完成--------------------------------------------#')




if __name__ == '__main__':
    data_start_dt = '20100101'
    data_test_dt = '20260101'
    download_pdb_data(data_start_dt, data_test_dt, label='10d_yield_rate', type='reg', data_file_url=str(resolve_data_dir()))
    # download_pdb_data(data_start_dt, data_test_dt, label='22d_yield_rate', type='reg', data_file_url=str(resolve_data_dir()))
    # 0.00109
	# download_pdb_data( data_test_dt, label='day2_tag', type='class')
