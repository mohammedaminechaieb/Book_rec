"""
==============================================================================
File      : src/data/load.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Reads the raw line-delimited JSON (.json.gz) files in chunks, keeps only the
  useful columns and samples users so the data fits in memory.
Input:
  - data/raw/goodreads_interactions_<genre>.json.gz
      (user_id, book_id, is_read, rating, date_added, date_updated, ...)
  - data/raw/goodreads_books_<genre>.json.gz
      (book_id, title, description, popular_shelves, authors, ratings_count, ...)
  - data/raw/goodreads_book_genres_initial.json.gz (book_id, genres)
Output:
  - interactions DataFrame [user_id, book_id, is_read, rating, date_updated]
  - books DataFrame        [book_id, title, description, shelves(list[str]), ratings_count]
  - genres DataFrame       [book_id, genre]  (main genre of each book)
==============================================================================
"""
from pathlib import Path

import pandas as pd


def load_interactions(path: Path, n_users_sample: int, seed: int) -> pd.DataFrame:
    """Read in chunks (pd.read_json(lines=True, chunksize=...)), then sample n users."""
    raise NotImplementedError("TODO (Amine)")


def load_books(path: Path, top_shelves: int = 20) -> pd.DataFrame:
    """Keep title, description, the top-N popular shelf names and ratings_count."""
    raise NotImplementedError("TODO (Amine)")


def load_genres(path: Path) -> pd.DataFrame:
    """Return one main genre per book (the genre with the highest count)."""
    raise NotImplementedError("TODO (Amine)")