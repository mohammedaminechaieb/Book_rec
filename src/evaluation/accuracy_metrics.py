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
  A user with no relevant book gets 0 on every metric (evaluate.py skips such users).
Input:
  - recommended: list/array of book indices (ordered, length >= K)
  - relevant:    set of held-out book indices for that user
Output:
  - float per user (evaluate.py averages them over users)
==============================================================================
"""
import numpy as np


def _hits(recommended, relevant: set, k: int) -> list[bool]:
    """For each of the first k recommendations: is it a relevant book?"""
    return [item in relevant for item in list(recommended)[:k]]


def precision_at_k(recommended, relevant: set, k: int = 20) -> float:
    if not relevant:
        return 0.0
    return sum(_hits(recommended, relevant, k)) / k


def recall_at_k(recommended, relevant: set, k: int = 20) -> float:
    if not relevant:
        return 0.0
    return sum(_hits(recommended, relevant, k)) / min(len(relevant), k)


def ndcg_at_k(recommended, relevant: set, k: int = 20) -> float:
    if not relevant:
        return 0.0
    hits = _hits(recommended, relevant, k)
    # rank r (0-based) is discounted by log2(r + 2): rank 0 -> 1, rank 1 -> 1/log2(3), ...
    dcg = sum(1.0 / np.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return float(dcg / idcg)


def hit_rate_at_k(recommended, relevant: set, k: int = 20) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(_hits(recommended, relevant, k)) else 0.0