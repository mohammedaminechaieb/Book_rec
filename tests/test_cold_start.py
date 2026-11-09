"""
==============================================================================
File      : tests/test_cold_start.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead, tests Ella's code)  |  Ella (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/data/cold_start.py:
    - each cold book keeps at most `keep` train interactions; other books unchanged
    - each cold user keeps only their `keep` EARLIEST interactions
    - results are reproducible with the same seed
Input:  synthetic train DataFrames
Output: pytest pass / fail
==============================================================================
"""
import pandas as pd

from src.data.cold_start import make_cold_books, make_cold_users


def _toy_train():
    return pd.DataFrame({"user_id": [f"u{i}" for i in range(100)],
                         "book_id": [f"b{i % 10}" for i in range(100)],
                         "date_updated": pd.date_range("2015-01-01", periods=100)})


def _toy_users():
    return pd.DataFrame({"user_id": [f"u{i // 10}" for i in range(100)],
                         "book_id": [f"b{i}" for i in range(100)],
                         "date_updated": pd.date_range("2015-01-01", periods=100)})


def test_cold_books_capped():
    train_cold, cold = make_cold_books(_toy_train(), fraction=0.3, keep=3, seed=1)
    counts = train_cold.groupby("book_id").size()
    assert all(counts.get(b, 0) <= 3 for b in cold)
    assert all(counts[b] == 10 for b in counts.index if b not in cold)


def test_cold_books_reproducible():
    _, c1 = make_cold_books(_toy_train(), 0.3, 3, seed=7)
    _, c2 = make_cold_books(_toy_train(), 0.3, 3, seed=7)
    assert sorted(c1) == sorted(c2)


def test_cold_users_keep_earliest():
    df = _toy_users()
    train_cold, cold = make_cold_users(df, fraction=0.2, keep=3, seed=1)
    for u in cold:
        kept = train_cold[train_cold.user_id == u].sort_values("date_updated")
        first3 = df[df.user_id == u].sort_values("date_updated").head(3)
        assert list(kept.book_id) == list(first3.book_id)