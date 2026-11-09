"""
==============================================================================
File      : tests/test_cold_start.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead, tests Ella's code)  |  Ella (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/data/cold_start.py: each cold book keeps at most
  max_train_interactions in train, non-cold books are unchanged, and the
  result is reproducible with the same seed.
Input:  synthetic train DataFrame
Output: pytest pass / fail
==============================================================================
"""
import pandas as pd

from src.data.cold_start import make_cold_books


def _toy_train():
    return pd.DataFrame({"user_id": [f"u{i}" for i in range(100)],
                         "book_id": [f"b{i % 10}" for i in range(100)]})


def test_cold_books_capped():
    train_cold, cold = make_cold_books(_toy_train(), fraction=0.3, keep=3, seed=1)
    counts = train_cold.groupby("book_id").size()
    assert all(counts.get(b, 0) <= 3 for b in cold)
    assert all(counts[b] == 10 for b in counts.index if b not in cold)


def test_reproducible():
    _, c1 = make_cold_books(_toy_train(), 0.3, 3, seed=7)
    _, c2 = make_cold_books(_toy_train(), 0.3, 3, seed=7)
    assert sorted(c1) == sorted(c2)