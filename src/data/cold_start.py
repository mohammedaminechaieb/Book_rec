"""
==============================================================================
File      : src/data/cold_start.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Simulates cold start (Goodreads has no truly new books). It picks a random
  cold_book_fraction of the books and removes all but max_train_interactions
  of their TRAIN interactions. Their val/test interactions are kept, so we can
  measure how well we recommend them.
Input:
  - data/processed/train.parquet
  - config.yaml -> cold_start.cold_book_fraction, cold_start.max_train_interactions, seed
Output:
  - data/processed/train_cold.parquet   (train with the hidden interactions removed)
  - data/processed/cold_books.json      (list of book_ids that were made cold)
==============================================================================
"""
import numpy as np
import pandas as pd


def make_cold_books(train: pd.DataFrame, fraction: float, keep: int, seed: int
                    ) -> tuple[pd.DataFrame, list]:
    """Return (train_cold, cold_book_ids)."""
    raise NotImplementedError("TODO (Ella)")


def book_counts(train: pd.DataFrame) -> pd.Series:
    """n = number of training interactions per book (this is the input of the gate)."""
    raise NotImplementedError("TODO (Ella)")