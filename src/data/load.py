"""
==============================================================================
File      : src/data/load.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Reads the raw line-delimited JSON (.json.gz) files line by line, keeps only
  the useful columns and samples users so the data fits in memory.
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
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Goodreads date format, e.g. "Tue Oct 17 09:40:11 -0700 2017"
DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _read_lines(path: Path):
    """Yield one dict per line of a .json.gz file (one JSON object per line)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _to_int(value, default: int = 0) -> int:
    """Goodreads stores numbers as text and sometimes leaves them empty."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def load_interactions(path: Path, n_users_sample: int, seed: int) -> pd.DataFrame:
    """Read the interactions of a random sample of n_users_sample users.

    Two passes over the file: (1) list the users and sample them,
    (2) keep only the rows of the sampled users.
    """
    # Pass 1: every distinct user id.
    users = set()
    for row in _read_lines(path):
        users.add(row["user_id"])
    users = sorted(users)                      # sorted -> same sample for the same seed

    if n_users_sample < len(users):
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(users, size=n_users_sample, replace=False).tolist())
    else:
        keep = set(users)                      # fewer users than asked: keep all

    # Pass 2: rows of the sampled users, useful columns only.
    rows = []
    for row in _read_lines(path):
        if row["user_id"] in keep:
            rows.append((row["user_id"], str(row["book_id"]), bool(row["is_read"]),
                        _to_int(row["rating"]), row["date_updated"]))

    df = pd.DataFrame(rows, columns=["user_id", "book_id", "is_read", "rating", "date_updated"])
    df["date_updated"] = pd.to_datetime(df["date_updated"], format=DATE_FORMAT,
                                        utc=True, errors="coerce")
    return df

def load_books(path: Path, top_shelves: int = 20, keep_ids: set | None = None) -> pd.DataFrame:
    """Keep title, description, the top-N popular shelf names and ratings_count.

    keep_ids (optional): only load these book ids (the books left after cleaning).
    """
    rows = []
    for book in _read_lines(path):
        book_id = str(book["book_id"])
        if keep_ids is not None and book_id not in keep_ids:
            continue
        # popular_shelves = [{"count": "153", "name": "fantasy"}, ...]
        shelves = sorted(book.get("popular_shelves", []),
                        key=lambda s: _to_int(s.get("count")), reverse=True)
        names = [s["name"] for s in shelves[:top_shelves]]
        rows.append((book_id, book.get("title", ""), book.get("description", ""),
                    names, _to_int(book.get("ratings_count"))))

    return pd.DataFrame(rows, columns=["book_id", "title", "description", "shelves", "ratings_count"])


def load_genres(path: Path, keep_ids: set | None = None) -> pd.DataFrame:
    """Return one main genre per book (the genre with the highest count).

    keep_ids (optional): only load these book ids. The file covers ALL Goodreads
    books (about 2 million lines), so passing keep_ids keeps the table small.
    """
    rows = []
    for record in _read_lines(path):
        book_id = str(record["book_id"])
        if keep_ids is not None and book_id not in keep_ids:
            continue
        genres = record.get("genres") or {}        # e.g. {"fantasy, paranormal": 31, "romance": 4}
        if not genres:
            continue                               # book with no genre information
        rows.append((book_id, max(genres, key=genres.get)))

    return pd.DataFrame(rows, columns=["book_id", "genre"])