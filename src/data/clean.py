"""
==============================================================================
File      : src/data/clean.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  1. Keeps positive interactions (is_read == True OR rating >= 4).
  2. Removes duplicates (user, book).
  3. Applies an iterative k-core filter (>= 5 per user AND per book).
  4. Cleans book text: strips HTML, lowercases, and uses the title when the
     description is missing. Drops books that have no interactions left.
Input:
  - interactions and books DataFrames from src/data/load.py
  - config.yaml -> data.positive_rating, data.min_user_interactions, data.min_book_interactions
Output:
  - data/interim/interactions_clean.parquet  [user_id, book_id, date_updated]
  - data/interim/books_clean.parquet         [book_id, title, text, shelves, n_ratings]
==============================================================================
"""
import pandas as pd


def keep_positive(df: pd.DataFrame, positive_rating: int) -> pd.DataFrame:
    raise NotImplementedError("TODO (Ella)")


def k_core_filter(df: pd.DataFrame, min_user: int, min_book: int) -> pd.DataFrame:
    """Repeat the user and book filters until nothing changes."""
    raise NotImplementedError("TODO (Ella)")


def clean_books(books: pd.DataFrame, keep_ids: set) -> pd.DataFrame:
    raise NotImplementedError("TODO (Ella)")