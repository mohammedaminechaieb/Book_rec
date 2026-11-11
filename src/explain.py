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
      (generic shelves such as "to-read" or "owned" are ignored)
    - a popularity label (bestseller / mid / long tail / new book)
    - one sentence that sums it up, e.g.
      "Because you liked Saga, Vol. 1. Shared tags: space-opera, sci-fi.
       Few readers so far (n=2): ranked 90% on content."
Input:
  - user_idx, book_idx, train CSR, TF-IDF item matrix, n_items, k
  - books table (title, shelves) aligned with book indices (row i = book i)
Output:
  - dict {alpha, cf_share, content_share, closest_book, closest_similarity,
          shared_shelves, popularity, n, sentence}
==============================================================================
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.models.hybrid_gate import gate_alpha

# Shelves that say how a reader stores a book, not what the book is about.
GENERIC_SHELVES = {
    "to-read", "currently-reading", "read", "favorites", "favourites", "owned",
    "books-i-own", "owned-books", "i-own", "my-books", "my-library", "library",
    "default", "wish-list", "wishlist", "to-buy", "kindle", "ebook", "ebooks",
    "audiobook", "audiobooks", "re-read", "dnf", "did-not-finish", "abandoned",
    "unfinished", "on-hold", "borrowed", "home-library", "maybe", "books", "series",
}
_GENERIC_PATTERN = re.compile(r"(to-?read|^own|-own|read-?in-|^read-|\d{4}|kindle|wish|"
                              r"favou?rite|currently|e-?book|audio|library|tbr)")


def is_generic_shelf(name: str) -> bool:
    """True for storage / status shelves ("to-read", "read-2016", "owned"...)."""
    name = str(name).lower()
    return name in GENERIC_SHELVES or bool(_GENERIC_PATTERN.search(name))

def score_breakdown(alpha: float, cf_score: float, content_score: float) -> dict:
    """Return {"cf_share", "content_share"}: the part of the hybrid score from each branch (sum = 1).

    cf_score and content_score are the per-user min-max normalised scores (0..1),
    i.e. the values the gate actually mixes.
    """
    cf_part = alpha * cf_score
    content_part = (1.0 - alpha) * content_score
    total = cf_part + content_part
    if total <= 0:                       # both scores are 0: fall back to the weights
        return {"cf_share": float(alpha), "content_share": float(1.0 - alpha)}
    return {"cf_share": float(cf_part / total), "content_share": float(content_part / total)}

def closest_liked_book(book_idx: int, liked_idx: np.ndarray,
                      item_vectors: sp.csr_matrix) -> tuple[int, float]:
    """Liked book with the highest cosine similarity to book_idx -> (index, similarity)."""
    liked_idx = np.asarray(liked_idx)
    if liked_idx.size == 0:
        return -1, 0.0
    target = normalize(item_vectors[book_idx])         # 1 x vocab, length 1
    liked = normalize(item_vectors[liked_idx])          # n_liked x vocab, length 1 each
    sims = np.asarray((liked @ target.T).todense()).ravel()
    best = int(np.argmax(sims))
    return int(liked_idx[best]), float(sims[best])

def shared_shelves(book_shelves: list[str], liked_shelves: list[list[str]], top: int = 3) -> list[str]:
    """Shelves of the book that appear most often among the liked books' shelves."""
    # how many liked books carry each shelf (a shelf counts once per book)
    counts = Counter(s for shelves in liked_shelves for s in set(shelves))
    candidates = [s for s in dict.fromkeys(book_shelves)          # keep order, no duplicates
                  if counts[s] > 0 and not is_generic_shelf(s)]
    # most shared first; ties keep the book's own order (its most popular shelves first)
    return sorted(candidates, key=lambda s: -counts[s])[:top]


def popularity_label(n: int, n_items: np.ndarray, cold_threshold: int = 3) -> str:
    """'new book' (n <= cold_threshold), 'long tail', 'mid', or 'bestseller' (top 1%)."""
    if n <= cold_threshold:
        return "new book"
    if n >= np.quantile(n_items, 0.99):
        return "bestseller"
    if n >= np.quantile(n_items, 0.80):       # top 20% = the "head" (same as tail_share)
        return "mid"
    return "long tail"

def explain(user_idx: int, book_idx: int, train: sp.csr_matrix, item_vectors: sp.csr_matrix,
            books: pd.DataFrame, n_items: np.ndarray, k: float,
            cf_score: float, content_score: float, top_shelves: int = 3,
            alpha: float | None = None, cold_threshold: int = 3) -> dict:
    """Combine the helpers above into the dict described in the header.

    alpha: leave None for the gated hybrid (computed from n and k); pass 1.0 for
    "ALS only", 0.0 for "Content only", 0.5 for the 50/50 baseline.
    """
    liked_idx = train[user_idx].indices                       # books the user read in train
    n = int(n_items[book_idx])
    if alpha is None:
        alpha = float(gate_alpha(np.array([n]), k)[0])
    shares = score_breakdown(alpha, cf_score, content_score)

    closest_idx, similarity = closest_liked_book(book_idx, liked_idx, item_vectors)
    closest_title = books.iloc[closest_idx]["title"] if closest_idx >= 0 else None
    tags = shared_shelves(list(books.iloc[book_idx]["shelves"]),
                          [list(books.iloc[i]["shelves"]) for i in liked_idx], top=top_shelves)
    label = popularity_label(n, n_items, cold_threshold)

    # --- one readable sentence ---
    parts = []
    if closest_title is not None and similarity > 0:
        parts.append(f"Because you liked {closest_title}.")
    if tags:
        parts.append("Shared tags: " + ", ".join(tags) + ".")
    content_pct, cf_pct = round(100 * shares["content_share"]), round(100 * shares["cf_share"])
    if label == "new book":
        parts.append(f"Few readers so far (n={n}): ranked {content_pct}% on content.")
    else:
        parts.append(f"{label.capitalize()} ({n} readers): ranked {cf_pct}% on similar readers "
                    f"and {content_pct}% on content.")

    return {"alpha": alpha, **shares, "closest_book": closest_title,
            "closest_similarity": similarity, "shared_shelves": tags,
            "popularity": label, "n": n, "sentence": " ".join(parts)}