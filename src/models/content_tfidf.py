"""
==============================================================================
File      : src/models/content_tfidf.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Content branch. Each book gets an L2-normalised TF-IDF vector built from its
  description and its popular shelves (tags such as "dystopian"). A user
  profile is the mean vector of the books the user liked in train.
  score(u, i) = cosine(profile_u, book_i). This works for cold books too.
Input:
  - data/interim/books_clean.parquet (text, shelves), aligned with book2idx
  - train CSR matrix
  - config.yaml -> tfidf.*
Output:
  - item matrix (n_books x vocab, sparse)
  - score matrix for a batch of users (n_batch x n_books)
==============================================================================
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


class ContentTFIDFModel:
    def __init__(self, max_features: int = 50000, min_df: int = 3, ngram_range=(1, 2),
                 use_description: bool = True, use_shelves: bool = True):
        self.params = dict(max_features=max_features, min_df=min_df, ngram_range=tuple(ngram_range),
                           use_description=use_description, use_shelves=use_shelves)
        self.item_vectors = None

    def build_text(self, books: pd.DataFrame) -> list[str]:
        """Join the description and the shelf tags (shelves as single tokens, e.g. historical_fiction)."""
        raise NotImplementedError("TODO (Ella)")

    def fit(self, books: pd.DataFrame, train: sp.csr_matrix) -> "ContentTFIDFModel":
        raise NotImplementedError("TODO (Ella)")

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO (Ella)")