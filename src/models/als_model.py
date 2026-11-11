"""
==============================================================================
File      : src/models/als_model.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Collaborative branch: implicit-feedback ALS (implicit library). It learns a
  vector for every user and every book. score(u, i) = user_u . item_i
  Confidence of an observed interaction = 1 + alpha * r (Hu, Koren & Volinsky 2008);
  unobserved pairs have confidence 1 and preference 0.
Input:
  - train CSR matrix (users x books, binary), confidence = 1 + alpha * r
  - config.yaml -> als.factors, als.regularization, als.iterations, als.alpha
Output:
  - fitted model; user_factors (n_users x f), item_factors (n_books x f)
  - score matrix for a batch of users (n_batch x n_books)
  - optional: data/processed/als_factors.npz
==============================================================================
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

class ALSModel:
    def __init__(self, factors: int = 64, regularization: float = 0.01,
                iterations: int = 20, alpha: float = 40.0, seed: int = 42):
        self.params = dict(factors=factors, regularization=regularization,
                          iterations=iterations, alpha=alpha, seed=seed)
        self.user_factors = None
        self.item_factors = None

    @classmethod
    def from_config(cls, cfg: dict) -> "ALSModel":
        """Build the model from config.yaml -> als.* and seed."""
        a = cfg["als"]
        return cls(factors=a["factors"], regularization=a["regularization"],
                  iterations=a["iterations"], alpha=a["alpha"], seed=cfg["seed"])

    def fit(self, train: sp.csr_matrix) -> "ALSModel":
        """Use implicit.als.AlternatingLeastSquares on the confidence matrix 1 + alpha * train."""
        from implicit.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        p = self.params
        # Confidence matrix: only the observed (non-zero) cells change -> 1 + alpha * 1.
        confidence = train.tocsr().astype(np.float32, copy=True)
        confidence.data = 1.0 + p["alpha"] * confidence.data

        model = AlternatingLeastSquares(factors=p["factors"], regularization=p["regularization"],
                                        iterations=p["iterations"], random_state=p["seed"],
                                        use_gpu=False)
        # implicit runs its own threads; limit BLAS to 1 thread to avoid slowdowns.
        with threadpool_limits(limits=1, user_api="blas"):
            model.fit(confidence)          # implicit >= 0.7 expects users x items

        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self