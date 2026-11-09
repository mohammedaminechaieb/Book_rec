"""
==============================================================================
File      : src/evaluation/evaluate.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Evaluation runner. For a model with .score(user_idx), it scores users in
  batches, masks the books seen in train, takes the top-N, and computes:
    - Recall@N, NDCG@N on ALL held-out books
    - Recall@N, NDCG@N on the COLD subset (books with n <= cold_threshold)
    - coverage, novelty, genre variety, tail share
Input:
  - model, train CSR, held-out DataFrame (val or test) mapped to indices
  - n_items, book_genre, cold_threshold, top_n, batch_size
Output:
  - dict of metrics for one model
  - results/tables/<split>_metrics.csv (one row per model, see save_table)
==============================================================================
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def top_n_from_scores(scores: np.ndarray, seen: sp.csr_matrix, n: int) -> np.ndarray:
    """Set seen items to -inf, then argpartition + sort -> (n_batch x n)."""
    raise NotImplementedError("TODO (Amine)")


def evaluate_model(model, train: sp.csr_matrix, heldout: dict, n_items: np.ndarray,
                   book_genre: np.ndarray, cold_threshold: int = 3,
                   top_n: int = 20, batch_size: int = 2000) -> dict:
    """heldout = {user_idx: set(book_idx)}. Returns the metrics dict."""
    raise NotImplementedError("TODO (Amine)")


def save_table(rows: list[dict], path) -> pd.DataFrame:
    raise NotImplementedError("TODO (Amine)")