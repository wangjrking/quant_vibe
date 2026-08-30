from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/"
    "strategy_agent_pre2026_strict_retune_20260730"
)
PROTOCOL = OUT / "preregistered_protocol.json"
BASE_PROTOCOL = (
    ROOT
    / "quant/main/strategy_library/production/"
    "prod_v260_10d_regime_warmup_all4key_v20260724/"
    "inputs/preregistered_protocol.json"
)
FROZEN = OUT / "frozen_candidate_before_2026.json"
MANIFESTS = [
    ROOT / "quant/main/config/prediction_manifests/executable_1d_open_return_l4_formal_20260619.json",
    ROOT / "quant/main/config/prediction_manifests/executable_3d_open_return_l4_formal_20260617.json",
    ROOT / "quant/main/config/prediction_manifests/executable_5d_open_return_l4_formal_20260620.json",
    ROOT / "quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_research_only":
        raise RuntimeError("研究协议未冻结")
    return protocol


def lock_inputs() -> dict:
    rows = []
    for path in MANIFESTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("approval_status") != "approved_for_l5":
            raise RuntimeError(f"manifest 未获 L5 批准: {path}")
        if payload.get("source_type") != "duckdb_table":
            raise RuntimeError(f"manifest 不是 DuckDB: {path}")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "table": payload["table"],
                "min_trade_date": payload["min_trade_date"],
                "max_trade_date": payload["max_trade_date"],
            }
        )
    result = {
        "protocol_sha256": sha256(PROTOCOL),
        "base_protocol_sha256": sha256(BASE_PROTOCOL),
        "manifests": rows,
        "production_change_allowed": False,
    }
    (OUT / "input_lock.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def slice_arrays(arrays: dict[str, np.ndarray], end: str) -> dict[str, np.ndarray]:
    dates = arrays["dates"].astype(str)
    count = int(np.searchsorted(dates, str(end), side="right"))
    if count <= 0 or dates[count - 1] != str(end):
        raise RuntimeError(f"开发截止信号日不存在: {end}")
    result = {}
    for key, value in arrays.items():
        if value.ndim >= 1 and value.shape[0] == len(dates):
            result[key] = value[:count].copy()
        else:
            result[key] = value.copy()
    if np.any(result["dates"].astype(str) > str(end)):
        raise RuntimeError("开发数组意外包含截止日后的日期")
    return result


def stable_id(values: dict) -> str:
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return "pre26_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def definition_for(base_protocol: dict, values: dict) -> dict:
    definition = v260.definition_for(base_protocol, 50)
    definition["min_hold_policy"] = str(values["min_hold_policy"])
    definition["sell_score_below"] = float(values["sell_score_below"])
    definition["replacement_advantage"] = float(
        values["replacement_advantage"]
    )
    return definition


def metric_row(daily: pd.DataFrame, start: str, end: str) -> dict:
    values = core.metrics(daily, start, end)
    return {
        "cumulative_return": float(values.get("cumulative_return", 0.0)),
        "linear_annual_proxy": float(
            values.get("linear_annual_proxy", 0.0)
        ),
        "sharpe": float(values.get("sharpe", 0.0)),
        "max_drawdown": float(values.get("max_drawdown", 0.0)),
        "trades": int(values.get("trades", 0)),
    }


def scale_actions(
    actions: pd.DataFrame, target_scale: float, cap: float
) -> pd.DataFrame:
    result = actions.copy()
    buy = result["action"].eq("BUY")
    result.loc[buy, "target_pct"] = (
        result.loc[buy, "target_pct"].astype(float) * float(target_scale)
    ).clip(upper=float(cap))
    return result


def prepare_development() -> None:
    protocol = load_protocol()
    lock_inputs()
    base_protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    full_arrays = v260.v258.v252.fresh_arrays()
    dev_end = str(protocol["development"]["signal_end"])
    arrays = slice_arrays(full_arrays, dev_end)
    grid = protocol["stage1_grid"]
    rows = []
    actions_by_case: dict[str, pd.DataFrame] = {}
    definitions: dict[str, dict] = {}
    for window, current_weight, policy, below, advantage in product(
        grid["smooth_window"],
        grid["current_weight"],
        grid["min_hold_policy"],
        grid["sell_score_below"],
        grid["replacement_advantage"],
    ):
        values = {
            "smooth_window": int(window),
            "current_weight": float(current_weight),
            "min_hold_policy": str(policy),
            "sell_score_below": float(below),
            "replacement_advantage": float(advantage),
        }
        case_id = stable_id(values)
        definition = definition_for(base_protocol, values)
        score, order = v95.score_pair(
            arrays,
            0.0,
            int(window),
            float(current_weight),
        )
        daily, actions = v260.run_case(
            arrays,
            score,
            order,
            definition,
            base_protocol,
            dev_end,
            record_actions=True,
        )
        full = metric_row(
            daily,
            str(protocol["development"]["signal_start"]),
            dev_end,
        )
        year_metrics = {}
        for fold in protocol["development"]["year_folds"]:
            year_metrics[fold["id"]] = metric_row(
                daily, fold["start"], fold["end"]
            )
        min_year_return = min(
            item["cumulative_return"] for item in year_metrics.values()
        )
        min_year_sharpe = min(
            item["sharpe"] for item in year_metrics.values()
        )
        rows.append(
            {
                "case_id": case_id,
                **values,
                "full_cumulative_return": full["cumulative_return"],
                "full_linear_annual_proxy": full["linear_annual_proxy"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "full_trades": full["trades"],
                "min_year_return": min_year_return,
                "min_year_sharpe": min_year_sharpe,
                **{
                    f"{year}_{name}": value
                    for year, metrics in year_metrics.items()
                    for name, value in metrics.items()
                },
            }
        )
        actions_by_case[case_id] = actions
        definitions[case_id] = {
            "score": values,
            "strategy_definition": definition,
        }
    frame = pd.DataFrame(rows)
    gate = protocol["selection_gate"]
    frame["eligible_local"] = (
        (frame["full_max_drawdown"] <= gate["full_max_drawdown_max"])
        & (
            frame["min_year_return"]
            >= gate["calendar_year_2023_2025_return_min"]
        )
        & (
            frame["min_year_sharpe"]
            >= gate["calendar_year_2023_2025_sharpe_min"]
        )
    )
    frame = frame.sort_values(
        [
            "eligible_local",
            "min_year_return",
            "min_year_sharpe",
            "full_sharpe",
            "full_linear_annual_proxy",
            "case_id",
        ],
        ascending=[False, False, False, False, False, True],
    )
    frame.to_csv(
        OUT / "stage1_pre2026_local_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    promoted = frame.head(int(grid["promote_count"]))
    actions_dir = OUT / "development_actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    stage2_rows = []
    stage2_definitions = {}
    for _, row in promoted.iterrows():
        source_id = str(row["case_id"])
        for target_scale, cap in product(
            protocol["stage2_grid"]["target_scale"],
            protocol["stage2_grid"]["single_target_cap"],
        ):
            values = {
                **definitions[source_id]["score"],
                "target_scale": float(target_scale),
                "single_target_cap": float(cap),
            }
            case_id = stable_id(values)
            actions = scale_actions(
                actions_by_case[source_id],
                float(target_scale),
                float(cap),
            )
            if (
                actions["signal_date"].astype(str).max()
                > protocol["hard_guards"]["development_action_signal_date_max"]
            ):
                raise RuntimeError("开发动作包含 2026 信号日")
            if (
                actions["buy_date"].astype(str).max()
                > protocol["hard_guards"]["development_action_buy_date_max"]
            ):
                raise RuntimeError("开发动作包含 2026 执行日")
            path = actions_dir / f"{case_id}.csv"
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            stage2_rows.append(
                {
                    "case_id": case_id,
                    "source_case_id": source_id,
                    **values,
                    "rows": int(len(actions)),
                    "buy_rows": int(actions["action"].eq("BUY").sum()),
                    "sell_rows": int(actions["action"].eq("SELL").sum()),
                    "max_signal_date": str(
                        actions["signal_date"].astype(str).max()
                    ),
                    "max_buy_date": str(
                        actions["buy_date"].astype(str).max()
                    ),
                    "action_path": str(path.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                    "action_sha256": sha256(path),
                }
            )
            stage2_definitions[case_id] = {
                **definitions[source_id],
                "target_scale": float(target_scale),
                "single_target_cap": float(cap),
            }
    pd.DataFrame(stage2_rows).to_csv(
        OUT / "stage2_development_action_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "stage2_candidate_definitions.json").write_text(
        json.dumps(stage2_definitions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage1_cases": int(len(frame)),
                "stage1_promoted": int(len(promoted)),
                "stage2_cases": int(len(stage2_rows)),
                "development_signal_end": dev_end,
                "development_buy_date_max": max(
                    row["max_buy_date"] for row in stage2_rows
                ),
                "validation_read": False,
            },
            ensure_ascii=False,
        )
    )


def parse_indicator(text: str) -> dict:
    marker = "GM_BACKTEST_INDICATOR:"
    lines = [line for line in text.splitlines() if marker in line]
    if not lines:
        raise RuntimeError("掘金日志缺少指标")
    raw = lines[-1].split(marker, 1)[1].strip()
    result = {}
    for key in (
        "pnl_ratio",
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "open_count",
        "close_count",
        "win_ratio",
    ):
        match = re.search(rf"'{key}': ([^,}}]+)", raw)
        if match:
            result[key] = ast.literal_eval(match.group(1).strip())
    if not result:
        raise RuntimeError("无法解析掘金指标")
    return result


def parse_nav(text: str) -> pd.DataFrame:
    rows = [
        (match.group(1), float(match.group(2)))
        for match in re.finditer(
            r"GM_DAILY_NAV\s+(\d{8})\s+([0-9.eE+-]+)", text
        )
    ]
    if not rows:
        raise RuntimeError("掘金日志缺少每日净值")
    frame = pd.DataFrame(rows, columns=["date", "nav"])
    return frame.drop_duplicates("date", keep="last").sort_values("date")


def nav_period(
    nav: pd.DataFrame, start: str, end: str, initial_cash: float
) -> dict:
    prior = nav[nav["date"] < start]
    base = float(prior.iloc[-1]["nav"]) if not prior.empty else initial_cash
    part = nav[(nav["date"] >= start) & (nav["date"] <= end)].copy()
    if part.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "days": 0,
        }
    values = np.r_[base, part["nav"].to_numpy(dtype=float)]
    returns = values[1:] / values[:-1] - 1.0
    std = float(np.std(returns, ddof=0))
    equity = values[1:] / base
    peak = np.maximum.accumulate(equity)
    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "sharpe": (
            float(np.mean(returns) / std * math.sqrt(252.0))
            if std > 0
            else 0.0
        ),
        "max_drawdown": float(abs(np.min(equity / peak - 1.0))),
        "days": int(len(part)),
    }


def select_from_development_logs() -> None:
    protocol = load_protocol()
    manifest = pd.read_csv(
        OUT / "stage2_development_action_manifest.csv",
        dtype={"case_id": str},
    )
    rows = []
    for item in manifest.to_dict("records"):
        log_path = OUT / "development_juejin" / f"{item['case_id']}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Traceback" in text or "GmError" in text:
            raise RuntimeError(f"掘金运行失败: {log_path}")
        indicator = parse_indicator(text)
        nav = parse_nav(text)
        years = {
            fold["id"]: nav_period(
                nav,
                fold["start"],
                fold["end"],
                float(protocol["execution"]["initial_cash"]),
            )
            for fold in protocol["development"]["year_folds"]
        }
        rows.append(
            {
                **item,
                "full_cumulative_return": float(indicator["pnl_ratio"]),
                "full_platform_linear_annual": float(
                    indicator["pnl_ratio_annual"]
                ),
                "full_sharpe": float(indicator["sharp_ratio"]),
                "full_max_drawdown": float(indicator["max_drawdown"]),
                "full_open_count": int(indicator["open_count"]),
                "full_close_count": int(indicator["close_count"]),
                "full_win_ratio": float(indicator["win_ratio"]),
                "min_year_return": min(
                    value["cumulative_return"] for value in years.values()
                ),
                "min_year_sharpe": min(
                    value["sharpe"] for value in years.values()
                ),
                **{
                    f"{year}_{name}": value
                    for year, metrics in years.items()
                    for name, value in metrics.items()
                },
                "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
                "log_sha256": sha256(log_path),
            }
        )
    frame = pd.DataFrame(rows)
    gate = protocol["selection_gate"]
    frame["eligible"] = (
        (frame["full_max_drawdown"] <= gate["full_max_drawdown_max"])
        & (
            frame["min_year_return"]
            >= gate["calendar_year_2023_2025_return_min"]
        )
        & (
            frame["min_year_sharpe"]
            >= gate["calendar_year_2023_2025_sharpe_min"]
        )
        & (frame["full_open_count"] >= gate["full_open_count_min"])
    )
    frame = frame.sort_values(
        [
            "eligible",
            "min_year_return",
            "min_year_sharpe",
            "full_sharpe",
            "full_platform_linear_annual",
            "case_id",
        ],
        ascending=[False, False, False, False, False, True],
    )
    frame.to_csv(
        OUT / "stage2_pre2026_juejin_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eligible = frame[frame["eligible"]]
    if eligible.empty:
        raise RuntimeError("开发期没有候选通过预注册门禁")
    selected = eligible.iloc[0].to_dict()
    definitions = json.loads(
        (OUT / "stage2_candidate_definitions.json").read_text(
            encoding="utf-8"
        )
    )
    case_id = str(selected["case_id"])
    frozen = {
        "status": "frozen_before_any_2026_validation_read",
        "case_id": case_id,
        "definition": definitions[case_id],
        "development_metrics": {
            key: value
            for key, value in selected.items()
            if key.startswith("full_")
            or key.startswith("min_year_")
            or key.startswith("2023_")
            or key.startswith("2024_")
            or key.startswith("2025_")
        },
        "development_action_sha256": str(selected["action_sha256"]),
        "development_log_sha256": str(selected["log_sha256"]),
        "protocol_sha256": sha256(PROTOCOL),
        "selection_feedback_from_2026": False,
        "production_change_allowed": False,
    }
    FROZEN.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(frozen, ensure_ascii=False, default=str))


def export_validation_actions() -> None:
    if not FROZEN.exists():
        raise RuntimeError("唯一候选尚未冻结，禁止导出验证动作")
    protocol = load_protocol()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_any_2026_validation_read":
        raise RuntimeError("候选冻结状态异常")
    base_protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    arrays = v260.v258.v252.fresh_arrays()
    values = frozen["definition"]["score"]
    definition = frozen["definition"]["strategy_definition"]
    score, order = v95.score_pair(
        arrays,
        0.0,
        int(values["smooth_window"]),
        float(values["current_weight"]),
    )
    end = str(arrays["dates"][-2])
    _, actions = v260.run_case(
        arrays,
        score,
        order,
        definition,
        base_protocol,
        end,
        record_actions=True,
    )
    actions = scale_actions(
        actions,
        float(frozen["definition"]["target_scale"]),
        float(frozen["definition"]["single_target_cap"]),
    )
    path = OUT / "validation_actions_frozen_candidate.csv"
    actions.to_csv(path, index=False, encoding="utf-8-sig")
    result = {
        "status": "validation_actions_exported_after_candidate_freeze",
        "case_id": frozen["case_id"],
        "signal_end": end,
        "rows": int(len(actions)),
        "buy_rows": int(actions["action"].eq("BUY").sum()),
        "sell_rows": int(actions["action"].eq("SELL").sum()),
        "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "action_sha256": sha256(path),
        "validation_may_not_change_candidate": True,
        "production_change_allowed": False,
    }
    (OUT / "validation_action_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["prepare-development", "select-development", "export-validation"],
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-development":
        prepare_development()
    elif args.command == "select-development":
        select_from_development_logs()
    else:
        export_validation_actions()


if __name__ == "__main__":
    main()
