"""
==============================================================================
File      : src/explain.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Builds a human-readable explanation, in French, for each recommended book
  (assignment step 6). For user u and book i it returns:
    - gate weight alpha_i = n_i / (n_i + k) and the CF / content share of the score
    - the liked book most similar to i in TF-IDF space ("Parce que vous avez aimé X")
    - the popular shelves shared between i and the user's liked books
      (storage shelves such as "to-read" and near-universal ones such as
       "comics" are ignored)
    - a popularity class: head (best-seller) / mid (intermédiaire) / tail (pépite),
      with the thresholds of src/labels.py
    - one sentence that sums it up, e.g.
      "Parce que vous avez aimé « Saga, Vol. 1 ». Tags communs : space-opera.
       Livre encore peu lu (2 lecteurs) : score porté à 90 % par le contenu."
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

from src.labels import NBSP, POP_LABEL, popularity_class
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

def frequent_shelves(all_shelves, max_share: float = 0.30) -> set[str]:
    """Shelves carried by more than max_share of the books (e.g. "comics", "graphic-novels").

    In a comics-only dataset such tags describe every book, so showing them as
    "shared tags" explains nothing. Computed once on the whole catalogue.
    """
    all_shelves = list(all_shelves)
    if not all_shelves:
        return set()
    counts = Counter(s for shelves in all_shelves for s in set(list(shelves)))
    limit = max_share * len(all_shelves)
    return {s for s, c in counts.items() if c > limit}


def shared_shelves(book_shelves: list[str], liked_shelves: list[list[str]], top: int = 3,
                   exclude: set | None = None) -> list[str]:
    """Shelves of the book that appear most often among the liked books' shelves.

    exclude: shelves never shown (use frequent_shelves() for near-universal tags).
    """
    exclude = exclude or set()
    # how many liked books carry each shelf (a shelf counts once per book)
    counts = Counter(s for shelves in liked_shelves for s in set(shelves))
    candidates = [s for s in dict.fromkeys(book_shelves)          # keep order, no duplicates
                  if counts[s] > 0 and not is_generic_shelf(s) and s not in exclude]
    # most shared first; ties keep the book's own order (its most popular shelves first)
    return sorted(candidates, key=lambda s: -counts[s])[:top]


def popularity_label(n: int, n_items: np.ndarray) -> str:
    """'head' (best-seller, top 5 %), 'mid' (top 20 %) or 'tail' (pépite, the other 80 %).

    Same thresholds as src/labels.py, so badges, gauges and the tail_share metric agree.
    """
    return popularity_class(n, n_items)


def _readers(n: int) -> str:
    return f"{n}{NBSP}lecteur" + ("s" if n > 1 else "")


def _clean_title(title: str) -> str:
    """Title without trailing dot, so the sentence never ends with '..'."""
    return str(title).strip().rstrip(".").strip()


def explain(user_idx: int, book_idx: int, train: sp.csr_matrix, item_vectors: sp.csr_matrix,
            books: pd.DataFrame, n_items: np.ndarray, k: float,
            cf_score: float, content_score: float, top_shelves: int = 3,
            alpha: float | None = None, cold_threshold: int = 3,
            exclude_shelves: set | None = None, mode: str = "gate") -> dict:
    """Combine the helpers above into the dict described in the header.

    mode: "gate" (alpha from n and k), "fixed" (alpha given, 50/50 baseline),
          "als" (alpha = 1) or "content" (alpha = 0). The sentence is written for the mode.
    """
    liked_idx = train[user_idx].indices                       # books the user read in train
    n = int(n_items[book_idx])
    if mode == "als":
        alpha = 1.0
    elif mode == "content":
        alpha = 0.0
    elif alpha is None:
        alpha = float(gate_alpha(np.array([n]), k)[0])
    shares = score_breakdown(alpha, cf_score, content_score)

    closest_idx, similarity = closest_liked_book(book_idx, liked_idx, item_vectors)
    closest_title = books.iloc[closest_idx]["title"] if closest_idx >= 0 else None
    tags = shared_shelves(list(books.iloc[book_idx]["shelves"]),
                          [list(books.iloc[i]["shelves"]) for i in liked_idx],
                          top=top_shelves, exclude=exclude_shelves)
    label = popularity_label(n, n_items)

    # --- one readable sentence (French, like the app and the report) ---
    parts = []
    if closest_title is not None and similarity > 0:
        parts.append(f"Parce que vous avez aimé « {_clean_title(closest_title)} ».")
    if tags:
        parts.append(f"Tags communs{NBSP}: " + ", ".join(tags) + ".")
    content_pct, cf_pct = round(100 * shares["content_share"]), round(100 * shares["cf_share"])
    if mode == "als":
        parts.append(f"Apprécié par des lecteurs aux goûts proches des vôtres ({_readers(n)}).")
    elif mode == "content":
        parts.append(f"Choisi pour la proximité de son contenu ({_readers(n)}).")
    elif n <= cold_threshold:
        parts.append(f"Livre encore peu lu ({_readers(n)}){NBSP}: score porté à "
                     f"{content_pct}{NBSP}% par le contenu.")
    else:
        parts.append(f"{POP_LABEL[label]} ({_readers(n)}){NBSP}: score porté à {cf_pct}{NBSP}% "
                     f"par les lecteurs similaires et {content_pct}{NBSP}% par le contenu.")

    return {"alpha": alpha, **shares, "closest_book": closest_title,
            "closest_similarity": similarity, "shared_shelves": tags,
            "popularity": label, "n": n, "sentence": " ".join(parts)}
