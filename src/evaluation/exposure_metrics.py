
"""
==============================================================================
File      : src/evaluation/exposure_metrics.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Computes long-tail exposure metrics:
  - Catalogue coverage
  - Novelty
  - Genre variety
  - Tail share

Input:
  - top_n: (n_users, N) book index array
  - n_items: training interaction counts per book
  - n_users: number of training users
  - n_books: catalogue size
  - book_genre: genre IDs aligned with book indices

Output:
  - Float values for each exposure metric
==============================================================================
"""

import numpy as np


def coverage(top_n: np.ndarray, n_books: int) -> float:
    """Share of the catalogue recommended at least once."""

    if n_books <= 0:
        return 0.0

    top_n = np.asarray(top_n, dtype=int)

    if top_n.size == 0:
        return 0.0

    # Ignore padding / invalid book indices.
    valid = top_n[(top_n >= 0) & (top_n < n_books)]

    if valid.size == 0:
        return 0.0

    unique_books = np.unique(valid)

    return float(len(unique_books) / n_books)


def novelty(
    top_n: np.ndarray,
    n_items: np.ndarray,
    n_users: int
) -> float:
    """
    Mean self-information of recommended books.

    Lower popularity produces higher novelty.
    Add-one smoothing prevents log2(0).
    """

    counts = np.asarray(n_items, dtype=float)
    top_n = np.asarray(top_n, dtype=int)

    if n_users <= 0 or top_n.size == 0 or counts.size == 0:
        return 0.0

    valid = top_n[
        (top_n >= 0) & (top_n < len(counts))
    ]

    if valid.size == 0:
        return 0.0

    # Smoothed probability of each recommended book.
    popularity = (counts[valid] + 1) / (n_users + 1)

    # Self-information: -log2(popularity).
    information = -np.log2(popularity)

    return float(np.mean(information))


def genre_variety(
    top_n: np.ndarray,
    book_genre: np.ndarray
) -> float:
    """Mean fraction of distinct genres in recommendation lists."""

    top_n = np.asarray(top_n, dtype=int)
    book_genre = np.asarray(book_genre)

    if top_n.size == 0 or book_genre.size == 0:
        return 0.0

    if top_n.ndim == 1:
        top_n = top_n.reshape(1, -1)

    varieties = []

    for recommendations in top_n:
        valid = recommendations[
            (recommendations >= 0)
            & (recommendations < len(book_genre))
        ]

        if len(valid) == 0:
            varieties.append(0.0)
            continue

        genres = book_genre[valid]

        distinct_genres = len(np.unique(genres))

        varieties.append(distinct_genres / len(valid))

    return float(np.mean(varieties))


def tail_share(
    top_n: np.ndarray,
    n_items: np.ndarray,
    head_fraction: float = 0.2
) -> float:
    """
    Share of recommended books outside the most popular head.

    The head consists of the top head_fraction books,
    ranked by training interaction count.
    """

    if not 0 <= head_fraction <= 1:
        raise ValueError("head_fraction must be between 0 and 1")

    counts = np.asarray(n_items)
    top_n = np.asarray(top_n, dtype=int)

    if top_n.size == 0 or counts.size == 0:
        return 0.0

    n_books = len(counts)

    valid = top_n[
        (top_n >= 0) & (top_n < n_books)
    ]

    if valid.size == 0:
        return 0.0

    # Determine the number of popular books.
    n_head = int(np.ceil(n_books * head_fraction))

    if n_head == 0:
        return 1.0

    # Rank books by popularity (descending).
    ranked_books = np.argsort(-counts, kind="stable")

    head_books = ranked_books[:n_head]

    # Recommendations not belonging to the popular head.
    is_tail = ~np.isin(valid, head_books)

    return float(np.mean(is_tail))
