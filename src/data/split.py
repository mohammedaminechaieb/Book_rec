"""
==============================================================================
File      : src/data/split.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Per-user time split. Each user's interactions are sorted by date_updated:
  the last test_ratio go to TEST, the val_ratio before them go to VALIDATION,
  and the rest go to TRAIN. Items that never appear in train are dropped from
  val/test (except the simulated cold books, see cold_start.py).
Input:
  - data/interim/interactions_clean.parquet
  - config.yaml -> split.val_ratio, split.test_ratio
Output:
  - data/processed/train.parquet, val.parquet, test.parquet  [user_id, book_id, date_updated]
==============================================================================
"""
import pandas as pd


def per_user_time_split(df: pd.DataFrame, val_ratio: float, test_ratio: float
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test). Every user keeps at least 1 train interaction."""
    raise NotImplementedError("TODO (Amine)")