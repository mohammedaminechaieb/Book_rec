"""
==============================================================================
File      : src/models/hybrid_gate.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  The core of the project: the per-book gate.
    alpha_i = n_i / (n_i + k)
    score(u, i) = alpha_i * CF(u, i) + (1 - alpha_i) * CONTENT(u, i)
  n_i = number of training interactions of book i. A book with n_i = 0 is
  ranked on content alone; a popular book mostly on CF. k = 0 means CF only
  for every book with n_i > 0. Both score matrices are min-max normalised per
  user before they are mixed. It also supports a fixed alpha (50/50 baseline).
  k is the number of interactions at which CF and content weigh the same
  (alpha = 0.5 when n_i = k).
Input:
  - CF scores and CONTENT scores (n_batch x n_books)
  - n_i per book (n_books,), k (float) OR fixed_alpha
Output:
  - hybrid score matrix (n_batch x n_books)
  - gate values alpha_i (n_books,) (plotted in results/figures)
==============================================================================
"""
import numpy as np
import pandas as pd


def gate_alpha(n: np.ndarray, k: float) -> np.ndarray:
    """alpha = n / (n + k); define alpha = 0 where n = 0 and alpha = 1 where n > 0 and k = 0."""
    n = np.asarray(n, dtype=np.float64)
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return (n > 0).astype(np.float64)       # k = 0 -> pure CF for every known book
    return n / (n + k)                          # n = 0 gives 0 -> pure content


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Rescale each row (one user) to [0, 1]. A constant row becomes all zeros."""
    scores = np.asarray(scores, dtype=np.float64)
    lo = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - lo
    span[span == 0] = 1.0                        # avoid division by zero
    return (scores - lo) / span


def combine(cf_scores: np.ndarray, content_scores: np.ndarray, alpha) -> np.ndarray:
    """alpha * CF + (1 - alpha) * content, after per-user min-max normalisation.

    alpha: one value (fixed baseline) or one value per book (gate).
    Kept separate so tune_k.py can reuse the same score matrices for every k.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    return alpha * minmax_rows(cf_scores) + (1.0 - alpha) * minmax_rows(content_scores)


def counts_to_array(counts: pd.Series, book2idx: dict) -> np.ndarray:
    """Turn book_counts() (indexed by book_id) into an array aligned with the matrix columns."""
    n = np.zeros(len(book2idx), dtype=np.float64)
    for book_id, count in counts.items():
        if book_id in book2idx:
            n[book2idx[book_id]] = count
    return n

class HybridGate:
    def __init__(self, cf_model, content_model, n_items: np.ndarray,
                 k: float | None = 20, fixed_alpha: float | None = None):
        if k is None and fixed_alpha is None:
            raise ValueError("Give k (gate) or fixed_alpha (fixed baseline).")
        self.cf, self.content, self.n, self.k, self.fixed_alpha = (
            cf_model, content_model, np.asarray(n_items, dtype=np.float64), k, fixed_alpha)
        self.alpha = self._compute_alpha()

    def _compute_alpha(self) -> np.ndarray:
        if self.fixed_alpha is not None:                        # 50/50 baseline
            return np.full(len(self.n), float(self.fixed_alpha))
        return gate_alpha(self.n, self.k)                       # per-book gate

    def set_k(self, k: float) -> "HybridGate":
        """Change k without rebuilding the models (used for the k sweep)."""
        self.k, self.fixed_alpha = k, None
        self.alpha = self._compute_alpha()
        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """Hybrid scores for a batch of users -> (len(user_idx), n_books)."""
        user_idx = np.asarray(user_idx)
        return combine(self.cf.score(user_idx), self.content.score(user_idx), self.alpha)