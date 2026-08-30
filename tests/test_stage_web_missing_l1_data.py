from pathlib import Path
import sys

import pandas as pd
import pytest

MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

import stage_web_missing_l1_data as subject


STAMP = pd.Timestamp("2026-07-12 12:00:00", tz="Asia/Shanghai")


def test_url_normalization_and_sha_are_deterministic():
    first = subject.normalize_url("HTTPS://Example.COM:443/a?utm_source=x&b=2&a=1#frag")
    second = subject.normalize_url("https://example.com/a?a=1&b=2")
    assert first == second == "https://example.com/a?a=1&b=2"
    assert subject.digest_text(first) == subject.digest_text(second)


def test_news_hourly_windows_cover_boundaries_without_overlap():
    windows = subject.hourly_windows("2026-07-10 23:30:00", "2026-07-11 01:00:00")
    assert windows == [
        ("2026-07-10 23:30:00", "2026-07-11 00:29:59"),
        ("2026-07-11 00:30:00", "2026-07-11 01:00:00"),
    ]


def test_disclosure_preserves_revisions():
    base = pd.DataFrame([{"ts_code": "600000.SH", "ann_date": "20260630", "end_date": "20260630", "pre_date": "20260716", "actual_date": None}])
    changed = base.assign(pre_date="20260718")
    one = subject.transform_disclosure(base, "20260712", STAMP)
    two = subject.transform_disclosure(changed, "20260713", STAMP)
    merged = subject.merge_versions(one, two, subject.CONTRACTS["company_disclosure_calendar"].key)
    assert len(merged) == 2
    assert merged.revision_hash.nunique() == 2


def test_disclosure_explicitly_excludes_bj_scope():
    source = pd.DataFrame([
        {"ts_code": "600000.SH", "ann_date": "20260630", "end_date": "20260630", "pre_date": "20260716", "actual_date": None},
        {"ts_code": "920001.BJ", "ann_date": "20260630", "end_date": "20260630", "pre_date": "20260716", "actual_date": None},
    ])
    result = subject.transform_disclosure(source, "20260712", STAMP)
    assert result.ts_code.tolist() == ["600000.SH"]


def test_staging_write_is_idempotent_and_matches_duckdb(tmp_path):
    source = pd.DataFrame([{"exchange": "SSE", "cal_date": "20260710", "is_open": 1, "pretrade_date": "20260709"}])
    frame = subject.transform_trade_calendar(source, STAMP)
    first = subject.write_staging_table(tmp_path, "trade_calendar", frame)
    second = subject.write_staging_table(tmp_path, "trade_calendar", frame)
    assert first["rows"] == second["rows"] == 1
    assert first["content_sha256"] == second["content_sha256"]
    assert second["duplicate_keys"] == 0
    assert second["parquet_minus_duckdb"] == second["duckdb_minus_parquet"] == 0


def test_production_l1_output_is_blocked():
    source = pd.DataFrame([{"exchange": "SSE", "cal_date": "20260710", "is_open": 1, "pretrade_date": "20260709"}])
    with pytest.raises(ValueError, match="production L1"):
        subject.write_staging_table(subject.PRODUCTION_L1_ROOT, "trade_calendar", subject.transform_trade_calendar(source, STAMP))


def test_stock_minutes_enforces_no_bj_and_ohlc_contract():
    valid = pd.DataFrame([{"ts_code": "600000.SH", "trade_time": "2026-07-10 09:31:00", "close": 10.1, "open": 10.0, "high": 10.2, "low": 9.9, "vol": 100, "amount": 1010}])
    result = subject.transform_stock_mins(valid, STAMP)
    assert result.loc[0, "freq"] == "1min"
    assert result.loc[0, "adj"] == "none"
    with pytest.raises(ValueError, match=".BJ"):
        subject.transform_stock_mins(valid.assign(ts_code="920001.BJ"), STAMP)
