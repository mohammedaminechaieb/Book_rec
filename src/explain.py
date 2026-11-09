"""
==============================================================================
File      : src/explain.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Builds a human-readable explanation for each recommended book (assignment
  step 6). For user u and book i it returns:
    - gate weight alpha_i = n_i / (n_i + k) and the CF / content share of the score
    - the liked book most similar to i in TF-IDF space ("because you liked X")
    - the popular shelves shared between i and the user's liked books
    - a popularity label (bestseller / mid / long tail / new book)
    - one sentence that sums it up, e.g.
      "Because you liked The Name of the Wind. Shared tags: epic-fantasy, magic.
       Few readers so far (n=2): ranked 90% on content."
Input:
  - user_idx, book_idx, train CSR, TF-IDF item matrix, n_items, k
  - books table (title, shelves) aligned with book indices
Output:
  - dict {alpha, cf_share, content_share, closest_book, shared_shelves, popularity, sentence}
==============================================================================
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def score_breakdown(alpha: float, cf_score: float, content_score: float) -> dict:
    """Return {"cf_share", "content_share"}: the part of the hybrid score from each branch (sum = 1)."""
    raise NotImplementedError("TODO (Amine)")


def closest_liked_book(book_idx: int, liked_idx: np.ndarray, item_vectors: sp.csr_matrix) -> tuple[int, float]:
    """Liked book with the highest cosine similarity to book_idx -> (index, similarity)."""
    raise NotImplementedError("TODO (Amine)")


def shared_shelves(book_shelves: list[str], liked_shelves: list[list[str]], top: int = 3) -> list[str]:
    """Shelves of the book that appear most often among the liked books' shelves."""
    raise NotImplementedError("TODO (Amine)")


def popularity_label(n: int, n_items: np.ndarray, cold_threshold: int = 3) -> str:
    """'new book' (n <= cold_threshold), 'long tail', 'mid', or 'bestseller' (top 1%)."""
    raise NotImplementedError("TODO (Amine)")


def explain(user_idx: int, book_idx: int, train: sp.csr_matrix, item_vectors: sp.csr_matrix,
            books: pd.DataFrame, n_items: np.ndarray, k: float,
            cf_score: float, content_score: float, top_shelves: int = 3) -> dict:
    """Combine the helpers above into the dict described in the header."""
    raise NotImplementedError("TODO (Amine)")