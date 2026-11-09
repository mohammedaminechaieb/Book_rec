"""
==============================================================================
File      : src/evaluation/accuracy_metrics.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Accuracy metrics on one top-K list (binary relevance):
    Precision@K = |rec_K intersect relevant| / K
    Recall@K    = |rec_K intersect relevant| / min(|relevant|, K)
    NDCG@K      = DCG@K / IDCG@K, log2 discount
    HR@K        = 1 if at least one relevant book is in rec_K, else 0
Input:
  - recommended: list/array of book indices (ordered, length >= K)
  - relevant:    set of held-out book indices for that user
Output:
  - float per user (evaluate.py averages them over users)
==============================================================================
"""
import numpy as np


def precision_at_k(recommended, relevant: set, k: int = 20) -> float:
    raise NotImplementedError("TODO (Amine)")


def recall_at_k(recommended, relevant: set, k: int = 20) -> float:
    raise NotImplementedError("TODO (Amine)")


def ndcg_at_k(recommended, relevant: set, k: int = 20) -> float:
    raise NotImplementedError("TODO (Amine)")


def hit_rate_at_k(recommended, relevant: set, k: int = 20) -> float:
    raise NotImplementedError("TODO (Amine)")