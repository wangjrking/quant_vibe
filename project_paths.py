from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_file"


def resolve_project_path(value: str | Path | None, default: str | Path | None = None) -> Path:
    raw = value if value not in (None, "") else default
    if raw in (None, ""):
        return PROJECT_ROOT
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_data_dir(value: str | Path | None = None) -> Path:
    env_value = os.environ.get("QUANT_DATA_DIR")
    if value not in (None, ""):
        return resolve_project_path(value)
    if env_value:
        return resolve_project_path(env_value)
    return DEFAULT_DATA_DIR


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_project_path(config_path or os.environ.get("QUANT_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    config = deepcopy(config)
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        config.setdefault("datasource", {})["tushare_token"] = token
    return config


def config_base_dir(config: dict[str, Any]) -> Path:
    file_url = config.get("filecatalog", {}).get("file_url")
    return resolve_project_path(file_url, default=".")
