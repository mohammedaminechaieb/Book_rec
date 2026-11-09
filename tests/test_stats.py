"""
==============================================================================
File      : tests/test_stats.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead, tests Ella's code)  |  Ella (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/data/stats.py: counts, sparsity and Gini on a toy dataset.
Input:  tiny DataFrame / arrays
Output: pytest pass / fail
==============================================================================
"""
import math

import numpy as np
import pandas as pd

from src.data.stats import dataset_stats, gini, sparsity


def test_sparsity():
    assert math.isclose(sparsity(4, 2, 3), 1 - 4 / 6)


def test_gini_equal_and_concentrated():
    assert math.isclose(gini(np.array([5, 5, 5, 5])), 0.0, abs_tol=1e-9)
    assert gini(np.array([0, 0, 0, 100])) > 0.7


def test_dataset_stats_counts():
    df = pd.DataFrame({"user_id": ["a", "a", "b", "b"], "book_id": [1, 2, 2, 3]})
    s = dataset_stats(df, stage="toy")
    assert (s["n_users"], s["n_items"], s["n_interactions"]) == (2, 3, 4)