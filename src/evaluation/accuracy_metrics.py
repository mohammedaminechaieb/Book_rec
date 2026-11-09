"""
==============================================================================
File      : src/evaluation/accuracy_metrics.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Accuracy metrics on top-N lists:
    Recall@N = |rec_N intersect relevant| / min(|relevant|, N)
    NDCG@N   = DCG@N / IDCG@N, with binary relevance and log2 discount
Input:
  - recommended: list/array of book indices (top-N, ordered)
  - relevant:    set of held-out book indices for that user
Output:
  - float per user (evaluate.py averages them)
==============================================================================
"""
import numpy as np


def recall_at_k(recommended, relevant: set, k: int = 20) -> float:
    raise NotImplementedError("TODO (Amine)")


def ndcg_at_k(recommended, relevant: set, k: int = 20) -> float:
    raise NotImplementedError("TODO (Amine)")