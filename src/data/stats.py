"""
==============================================================================
File      : src/data/stats.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Computes the dataset statistics required by the assignment (step 2):
    - number of users, items, interactions
    - density = interactions / (users x items), sparsity = 1 - density
    - interactions per user / per item (mean, median, min, max)
    - popularity concentration: Gini of item counts, share of interactions
      captured by the top 20% most popular books, share of books with <= 3 interactions
  It runs on raw-sampled, cleaned and train data so the effect of cleaning is visible.
Input:
  - interactions DataFrame [user_id, book_id, ...] (any stage)
Output:
  - dict of statistics (one stage)
  - results/tables/dataset_stats.csv (one row per stage: raw / clean / train)
==============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd


def dataset_stats(df: pd.DataFrame, stage: str = "clean") -> dict:
    """Return the statistics listed in the header for one stage."""
    raise NotImplementedError("TODO (Ella)")


def sparsity(n_interactions: int, n_users: int, n_items: int) -> float:
    """1 - n_interactions / (n_users * n_items)."""
    raise NotImplementedError("TODO (Ella)")


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of the item popularity counts (0 = equal, 1 = one item has everything)."""
    raise NotImplementedError("TODO (Ella)")


def stats_table(stages: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    """stages = {"raw": df_raw, "clean": df_clean, "train": train}; writes the CSV."""
    raise NotImplementedError("TODO (Ella)")