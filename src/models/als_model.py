"""
==============================================================================
File      : src/models/als_model.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Collaborative branch: implicit-feedback ALS (implicit library). It learns a
  vector for every user and every book. score(u, i) = user_u . item_i
Input:
  - train CSR matrix (users x books, binary), confidence = 1 + alpha * r
  - config.yaml -> als.factors, als.regularization, als.iterations, als.alpha
Output:
  - fitted model; user_factors (n_users x f), item_factors (n_books x f)
  - score matrix for a batch of users (n_batch x n_books)
  - optional: data/processed/als_factors.npz
==============================================================================
"""
import numpy as np
import scipy.sparse as sp


class ALSModel:
    def __init__(self, factors: int = 64, regularization: float = 0.01,
                 iterations: int = 20, alpha: float = 40.0, seed: int = 42):
        self.params = dict(factors=factors, regularization=regularization,
                           iterations=iterations, alpha=alpha, seed=seed)
        self.user_factors = None
        self.item_factors = None

    def fit(self, train: sp.csr_matrix) -> "ALSModel":
        """Use implicit.als.AlternatingLeastSquares on (alpha * train)."""
        raise NotImplementedError("TODO (Amine)")

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO (Amine)")