"""
==============================================================================
File      : src/evaluation/exposure_metrics.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Long-tail / exposure metrics on the full set of top-N lists:
    coverage      = share of the catalogue recommended at least once
    novelty       = mean over lists of mean(-log2(pop_i)), pop_i = n_i / n_users
    genre_variety = mean over lists of (#distinct genres in list / N)
    tail_share    = share of recommended books outside the top-20% most popular
Input:
  - top_n: array (n_users x N) of book indices
  - n_items: train interaction count per book, n_users (int), n_books (int)
  - book_genre: array (n_books,) with the genre id of each book
Output:
  - floats (one per metric)
==============================================================================
"""
import numpy as np


def coverage(top_n: np.ndarray, n_books: int) -> float:
    raise NotImplementedError("TODO (Ella)")


def novelty(top_n: np.ndarray, n_items: np.ndarray, n_users: int) -> float:
    raise NotImplementedError("TODO (Ella): add 1 to n_items to avoid log(0)")


def genre_variety(top_n: np.ndarray, book_genre: np.ndarray) -> float:
    raise NotImplementedError("TODO (Ella)")


def tail_share(top_n: np.ndarray, n_items: np.ndarray, head_fraction: float = 0.2) -> float:
    raise NotImplementedError("TODO (Ella)")