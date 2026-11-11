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
  Exposure metrics (Ella's exposure_metrics.py) can be switched off with
  with_exposure=False, so accuracy can be checked before they are written.
Input:
  - model, train CSR, heldout {user_idx: set(book_idx)}
  - n_items, book_genre, cold_item_idx (set), cold_user_idx (set)
  - ks, top_n, batch_size
Output:
  - flat dict of metrics, e.g. {"all/recall@20": ..., "cold_items/ndcg@10": ..., "coverage": ...}
  - results/tables/<split>_metrics.csv (one row per model, see save_table)
==============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation.accuracy_metrics import (hit_rate_at_k, ndcg_at_k,
                                            precision_at_k, recall_at_k)

METRICS = {"precision": precision_at_k, "recall": recall_at_k,
          "ndcg": ndcg_at_k, "hr": hit_rate_at_k}

def heldout_from_df(df: pd.DataFrame, user2idx: dict, book2idx: dict) -> dict:
    """Turn val.parquet / test.parquet into {user_idx: set(book_idx)}.

    Rows whose user or book is not in the train id maps are skipped.
    """
    heldout: dict[int, set] = {}
    for user_id, book_id in zip(df["user_id"], df["book_id"]):
        u, b = user2idx.get(user_id), book2idx.get(book_id)
        if u is not None and b is not None:
            heldout.setdefault(u, set()).add(b)
    return heldout

def top_n_from_scores(scores: np.ndarray, seen: sp.csr_matrix, n: int) -> np.ndarray:
    """Set seen items to -inf, then argpartition + sort -> (n_batch x n)."""
    scores = np.array(scores, dtype=np.float64, copy=True)   # never modify the model's array
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf                              # never recommend a book already read

    n = min(n, scores.shape[1])
    # argpartition finds the n best columns of each row in linear time (unordered) ...
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    # ... then only those n are sorted, best first.
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def accuracy_on_subset(top_n: dict, heldout: dict, ks: list[int]) -> dict:
    """Average Precision/Recall/NDCG/HR at each K over the users of `heldout`."""
    sums = {f"{name}@{k}": 0.0 for name in METRICS for k in ks}
    users = [u for u, rel in heldout.items() if rel and u in top_n]
    for u in users:
        for k in ks:
            for name, fn in METRICS.items():
                sums[f"{name}@{k}"] += fn(top_n[u], heldout[u], k)
    n_users = len(users)
    result = {key: (value / n_users if n_users else 0.0) for key, value in sums.items()}
    result["n_users"] = n_users
    return result


def evaluate_model(model, train: sp.csr_matrix, heldout: dict, n_items: np.ndarray,
                   book_genre: np.ndarray | None, cold_item_idx: set, cold_user_idx: set,
                   ks=(5, 10, 20), top_n: int = 20, batch_size: int = 2000,
                   with_exposure: bool = True) -> dict:
    """Score every held-out user in batches and return the flat metrics dict."""
    ks = list(ks)
    list_len = max(top_n, max(ks))
    users = np.array(sorted(u for u, rel in heldout.items() if rel))

    # 1. Top-N list of every user, in batches (limits memory).
    top: dict[int, np.ndarray] = {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        lists = top_n_from_scores(model.score(batch), train[batch], list_len)
        top.update(zip(batch.tolist(), lists))

    # 2. The three subsets.
    subsets = {
        "all": heldout,
        "cold_items": {u: rel & cold_item_idx for u, rel in heldout.items() if rel & cold_item_idx},
        "cold_users": {u: rel for u, rel in heldout.items() if u in cold_user_idx},
    }
    out = {}
    for name, subset in subsets.items():
        for metric, value in accuracy_on_subset(top, subset, ks).items():
            out[f"{name}/{metric}"] = value

    # 3. Exposure metrics on the top_n lists of all users (Ella's functions).
    if with_exposure:
        from src.evaluation import exposure_metrics as em
        lists = np.stack([top[u][:top_n] for u in users.tolist()])
        out["coverage"] = em.coverage(lists, n_books=train.shape[1])
        out["novelty"] = em.novelty(lists, n_items, n_users=train.shape[0])
        out["tail_share"] = em.tail_share(lists, n_items)
        if book_genre is not None:
            out["genre_variety"] = em.genre_variety(lists, book_genre)
    return out


def evaluate_all(models: dict, **kwargs) -> list[dict]:
    """Evaluate several models with the same settings -> one row per model.

    models = {"Popularity": pop, "ALS": als, "Content": tfidf,
              "Hybrid 50/50": fixed, "Gated hybrid": gate}
    kwargs = the arguments of evaluate_model except `model`.
    """
    rows = []
    for name, model in models.items():
        print(f"evaluating {name} ...")
        rows.append({"model": name, **evaluate_model(model, **kwargs)})
    return rows


def save_table(rows: list[dict], path) -> pd.DataFrame:
    """Write the results table (one row per model) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).set_index("model").round(4)
    table.to_csv(path)
    return table