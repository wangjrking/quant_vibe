from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project_paths import resolve_data_dir


REGISTRY_FILE = "asset_registry/production_assets.json"
ACTIVE_STATUSES = {"production_active", "approved_for_l5"}


def registry_path(data_dir: str | Path | None = None) -> Path:
    return resolve_data_dir(data_dir) / REGISTRY_FILE


def load_production_registry(data_dir: str | Path | None = None) -> dict[str, Any] | None:
    path = registry_path(data_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_registry_layer(layer: str) -> str:
    aliases = {
        "l1": "l1_raw_data",
        "l1_raw_data": "l1_raw_data",
        "l2": "l2_stock_daily_base",
        "l2_stock_daily_base": "l2_stock_daily_base",
        "l3_features": "l3_features",
        "l3_labels": "l3_labels",
        "l4": "l4_model_prediction",
        "l4_model_prediction": "l4_model_prediction",
        "l5": "l5_strategy_signal",
        "l5_strategy_signal": "l5_strategy_signal",
        "l6": "l6_backtest",
        "l6_backtest": "l6_backtest",
        "l7": "l7_trading_delivery",
        "l7_trading_delivery": "l7_trading_delivery",
    }
    key = str(layer).strip().lower()
    return aliases.get(key, key)


def active_main_workflow_assets(
    layer: str,
    *,
    data_dir: str | Path | None = None,
    asset_type_contains: str | None = None,
) -> list[dict[str, Any]]:
    registry = load_production_registry(data_dir)
    if not registry:
        return []
    normalized = _normalize_registry_layer(layer)
    candidates: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        if str(asset.get("layer", "")).strip().lower() != normalized:
            continue
        if asset.get("allowed_for_main_workflow") is not True:
            continue
        if str(asset.get("status", "")).strip().lower() not in ACTIVE_STATUSES:
            continue
        if asset_type_contains and asset_type_contains.lower() not in str(asset.get("asset_type", "")).lower():
            continue
        candidates.append(asset)
    return candidates


def active_main_workflow_asset(
    layer: str,
    *,
    data_dir: str | Path | None = None,
    asset_type_contains: str | None = None,
) -> dict[str, Any] | None:
    assets = active_main_workflow_assets(
        layer,
        data_dir=data_dir,
        asset_type_contains=asset_type_contains,
    )
    return assets[0] if assets else None


def active_main_workflow_asset_for_table(
    layer: str,
    table: str,
    *,
    data_dir: str | Path | None = None,
    asset_type_contains: str | None = None,
) -> dict[str, Any] | None:
    normalized_table = str(table or "").strip()
    if not normalized_table:
        return active_main_workflow_asset(
            layer,
            data_dir=data_dir,
            asset_type_contains=asset_type_contains,
        )

    assets = active_main_workflow_assets(
        layer,
        data_dir=data_dir,
        asset_type_contains=asset_type_contains,
    )
    for asset in assets:
        _path, asset_table = split_asset_path(asset.get("asset_path"))
        if str(asset_table or "").strip() == normalized_table:
            return asset
    return assets[0] if len(assets) == 1 else None


def split_asset_path(asset_path: str | Path | None) -> tuple[Path | None, str | None]:
    if asset_path in (None, ""):
        return None, None
    raw_value = str(asset_path)
    path_text, table = raw_value.split("::", 1) if "::" in raw_value else (raw_value, None)
    path = Path(path_text)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    return path, table
