
"""
==============================================================================
File      : src/models/popularity.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Popularity baseline that recommends the most popular books.

  - fit(): calculates the number of positive training interactions per book.
  - score(): returns popularity scores for requested users.

Input:
  - Binary CSR training matrix (users x books)
  - Array of user indices

Output:
  - Score matrix (n_requested_users x n_books)
==============================================================================
"""

import numpy as np
import scipy.sparse as sp


class PopularityModel:

    def fit(self, train: sp.csr_matrix) -> "PopularityModel":
        """
        Learn book popularity from the training matrix.

        Each nonzero element represents a positive interaction.
        """

        if not sp.issparse(train):
            raise TypeError("train must be a scipy sparse matrix")

        train = train.tocsr(copy=True)

        # Normalize sparse representation.
        train.sum_duplicates()
        train.eliminate_zeros()

        # Count nonzero interactions for each book.
        self.item_counts_ = np.asarray(
            train.getnnz(axis=0),
            dtype=np.float32
        ).ravel()

        # Store matrix dimensions.
        self.n_users_ = train.shape[0]
        self.n_books_ = train.shape[1]

        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """
        Return popularity scores for the requested users.

        Every user receives the same popularity scores.
        Already-read books must be filtered during recommendation.
        """

        if not hasattr(self, "item_counts_"):
            raise RuntimeError(
                "Model must be fitted before calling score()."
            )

        user_idx = np.atleast_1d(user_idx)

        if np.any(user_idx < 0) or np.any(user_idx >= self.n_users_):
            raise ValueError("Invalid user index")

        # Repeat the popularity scores for every requested user.
        scores = np.tile(
            self.item_counts_,
            (len(user_idx), 1)
        )

        return scores

