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
Input:
  - CF scores and CONTENT scores (n_batch x n_books)
  - n_i per book (n_books,), k (float) OR fixed_alpha
Output:
  - hybrid score matrix (n_batch x n_books)
  - gate values alpha_i (n_books,) (plotted in results/figures)
==============================================================================
"""
import numpy as np


def gate_alpha(n: np.ndarray, k: float) -> np.ndarray:
    """alpha = n / (n + k); define alpha = 0 where n = 0 and alpha = 1 where n > 0 and k = 0."""
    raise NotImplementedError("TODO (Amine)")


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO (Amine)")


class HybridGate:
    def __init__(self, cf_model, content_model, n_items: np.ndarray,
                 k: float | None = 20, fixed_alpha: float | None = None):
        self.cf, self.content, self.n, self.k, self.fixed_alpha = (
            cf_model, content_model, n_items, k, fixed_alpha)

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO (Amine)")