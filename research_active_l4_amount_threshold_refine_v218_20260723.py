# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import research_active_l4_amount_threshold_v217_20260723 as base


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_amount_threshold_refine_v218_20260723"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v218_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


base.OUT = OUT
base.PROTOCOL = OUT / "preregistered_protocol.json"
base.FROZEN = OUT / "frozen_candidates_before_known_2026.json"
base.stable_id = stable_id


if __name__ == "__main__":
    base.main()
