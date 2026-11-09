"""
==============================================================================
File      : src/data/cold_start.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Simulates cold start (Goodreads data is from 2017, so nothing is truly new).
  1. New ITEMS: picks a random cold_book_fraction of the books and removes all
     but max_train_interactions of their TRAIN interactions.
  2. New USERS: picks a random cold_user_fraction of the users and keeps only
     their max_user_train_interactions EARLIEST train interactions.
  Val/test interactions are kept in both cases, so the evaluation can measure
  how well these books / users are served.
Input:
  - data/processed/train.parquet
  - config.yaml -> cold_start.*, seed
Output:
  - data/processed/train_cold.parquet   (train after both simulations)
  - data/processed/cold_books.json      (book_ids made cold)
  - data/processed/cold_users.json      (user_ids made cold)
==============================================================================
"""
import numpy as np
import pandas as pd


def make_cold_books(train: pd.DataFrame, fraction: float, keep: int, seed: int
                    ) -> tuple[pd.DataFrame, list]:
    """Return (train_cold, cold_book_ids)."""
    raise NotImplementedError("TODO (Ella)")


def make_cold_users(train: pd.DataFrame, fraction: float, keep: int, seed: int
                    ) -> tuple[pd.DataFrame, list]:
    """Return (train_cold, cold_user_ids). Keep the `keep` earliest rows (date_updated) of each cold user."""
    raise NotImplementedError("TODO (Ella)")


def book_counts(train: pd.DataFrame) -> pd.Series:
    """n = number of training interactions per book (this is the input of the gate)."""
    raise NotImplementedError("TODO (Ella)")