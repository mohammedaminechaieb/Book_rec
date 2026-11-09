"""
==============================================================================
File      : tests/test_split_and_gate.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead, tests Amine's code)  |  Amine (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/data/split.py (no leakage: every test date >= every train date of
  the same user) and src/models/hybrid_gate.py (alpha in [0, 1], alpha(0) = 0,
  alpha increasing with n).
Input:  tiny synthetic DataFrame / arrays
Output: pytest pass / fail
==============================================================================
"""
import numpy as np
import pandas as pd

from src.data.split import per_user_time_split
from src.models.hybrid_gate import gate_alpha


def test_gate_bounds_and_monotonic():
    n = np.array([0, 1, 5, 50, 500])
    a = gate_alpha(n, k=10)
    assert a[0] == 0 and np.all((a >= 0) & (a <= 1)) and np.all(np.diff(a) > 0)


def test_split_no_time_leak():
    df = pd.DataFrame({"user_id": ["u"] * 10, "book_id": list("abcdefghij"),
                       "date_updated": pd.date_range("2015-01-01", periods=10)})
    train, val, test = per_user_time_split(df, val_ratio=0.1, test_ratio=0.2)
    assert train.date_updated.max() <= val.date_updated.min() <= test.date_updated.min()
    assert len(train) + len(val) + len(test) == 10