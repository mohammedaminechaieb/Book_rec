"""
==============================================================================
File      : src/evaluation/evaluate.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Evaluation runner (assignment step 5). For a model with .score(user_idx), it
  scores users in batches, masks the books seen in train and takes the top-N.
  It computes, for every K in eval.ks:
    Precision@K, Recall@K, NDCG@K, HR@K on 3 subsets:
      - all       : every test user, every held-out book
      - cold_items: held-out books restricted to the cold books (users with >= 1 of them)
      - cold_users: only the simulated new users
  and, on the top-N lists of all users:
    coverage, novelty, genre variety, tail share
Input:
  - model, train CSR, heldout {user_idx: set(book_idx)}
  - n_items, book_genre, cold_item_idx (set), cold_user_idx (set)
  - ks, top_n, batch_size
Output:
  - flat dict of metrics, e.g. {"all/recall@20": ..., "cold_items/ndcg@10": ..., "coverage": ...}
  - results/tables/<split>_metrics.csv (one row per model, see save_table)
==============================================================================
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def top_n_from_scores(scores: np.ndarray, seen: sp.csr_matrix, n: int) -> np.ndarray:
    """Set seen items to -inf, then argpartition + sort -> (n_batch x n)."""
    raise NotImplementedError("TODO (Amine)")


def accuracy_on_subset(top_n: dict, heldout: dict, ks: list[int]) -> dict:
    """Average Precision/Recall/NDCG/HR at each K over the users of `heldout`."""
    raise NotImplementedError("TODO (Amine)")


def evaluate_model(model, train: sp.csr_matrix, heldout: dict, n_items: np.ndarray,
                   book_genre: np.ndarray, cold_item_idx: set, cold_user_idx: set,
                   ks=(5, 10, 20), top_n: int = 20, batch_size: int = 2000) -> dict:
    raise NotImplementedError("TODO (Amine)")


def save_table(rows: list[dict], path) -> pd.DataFrame:
    raise NotImplementedError("TODO (Amine)")