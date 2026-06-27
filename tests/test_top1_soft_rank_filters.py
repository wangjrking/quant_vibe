import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tune_top1_soft_rank_latest_formal_20260623 as top1


def test_delisting_names_are_not_tradable():
    assert top1._is_st_like({"name": "退市太和", "st_type": None, "st_type_name": None})
    assert top1._is_st_like({"name": "退市国化", "st_type": None, "st_type_name": None})


def test_blocked_name_helper_covers_delisting_marker():
    assert top1._has_blocked_name("退市太和")
    assert top1._has_blocked_name("退市国化")
    assert not top1._has_blocked_name("线上线下")
