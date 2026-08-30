from pathlib import Path

import pytest

from quant.main.model_input_contract_v2_candidate import (
    ModelInputContractError,
    build_contract,
    build_projection_sql,
    validate_contract,
    validate_runtime_matrix_columns,
)


def test_contract_rejects_forbidden_future_feature(tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"feature_columns":["a","index_2000_post10_close"]}', encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    with pytest.raises(ModelInputContractError, match="forbidden"):
        build_contract(
            label_key="x",
            production_model_asset_id="asset",
            metadata_path=metadata,
            model_path=model,
            available_l3_columns={"a", "index_2000_post10_close"},
        )


def test_contract_sql_is_explicit_projection_and_runtime_order_is_fail_closed(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"feature_columns":["close","factor_a"]}', encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")

    class FakeBooster:
        feature_names = ["close", "factor_a"]

        def load_model(self, _path):
            return None

    monkeypatch.setattr("quant.main.model_input_contract_v2_candidate.xgb.Booster", FakeBooster)
    contract = build_contract(
        label_key="x",
        production_model_asset_id="asset",
        metadata_path=metadata,
        model_path=model,
        available_l3_columns={"close_qfq", "factor_a", "index_2000_post10_close"},
    )
    validate_contract(contract)
    sql = build_projection_sql(factor_table="features", contract=contract)
    assert "SELECT *" not in sql.upper()
    assert '"close_qfq" AS "close"' in sql
    assert '"index_2000_post10_close"' not in sql
    validate_runtime_matrix_columns(["close", "factor_a"], contract)
    with pytest.raises(ModelInputContractError, match="wide-frame"):
        validate_runtime_matrix_columns(["close", "factor_a", "index_2000_post10_close"], contract)


def test_contract_rejects_metadata_booster_order_mismatch(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"feature_columns":["a","b"]}', encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")

    class FakeBooster:
        feature_names = ["b", "a"]

        def load_model(self, _path):
            return None

    monkeypatch.setattr("quant.main.model_input_contract_v2_candidate.xgb.Booster", FakeBooster)
    with pytest.raises(ModelInputContractError, match="differ"):
        build_contract(
            label_key="x",
            production_model_asset_id="asset",
            metadata_path=metadata,
            model_path=model,
            available_l3_columns={"a", "b"},
        )
