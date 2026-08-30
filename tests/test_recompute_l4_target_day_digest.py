import hashlib

import duckdb

from quant.main.recompute_l4_target_day_digest import DIGEST_ALGORITHM_ID, canonical_digest


def test_canonical_digest_uses_declared_key_order_and_float17g(tmp_path):
    db_path = tmp_path / "target_day.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("create table score(trade_date varchar, stock_code varchar, pred_prob double)")
        connection.execute(
            "insert into score values ('20260804','000002.SZ',0.1), ('20260804','000001.SZ',0.2)"
        )
        digest = canonical_digest(connection, "score", "20260804")
    expected = hashlib.sha256(
        b"20260804|000001.SZ|0.20000000000000001\n20260804|000002.SZ|0.10000000000000001\n"
    ).hexdigest()
    assert DIGEST_ALGORITHM_ID.endswith("_v1")
    assert digest == expected
