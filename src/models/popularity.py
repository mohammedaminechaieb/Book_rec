"""
==============================================================================
File      : src/models/popularity.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Baseline: recommends the most-read books in train to everyone (excluding the
  books the user already read). It is the worst case for popularity bias.
Input:
  - train CSR matrix (users x books)
Output:
  - score matrix (n_users x n_books) or top-N indices per user
==============================================================================
"""
import numpy as np
import scipy.sparse as sp


class PopularityModel:
    def fit(self, train: sp.csr_matrix) -> "PopularityModel":
        raise NotImplementedError("TODO (Ella): item counts")

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """Return (len(user_idx), n_books) scores: the same popularity row for every user."""
        raise NotImplementedError("TODO (Ella)")