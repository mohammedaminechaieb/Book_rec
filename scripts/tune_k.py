
"""
==============================================================================
File      : scripts/tune_k.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Tune the gated hybrid parameter k on the VALIDATION set.

  ALS and TF-IDF scores are computed only once per user batch.
  The script evaluates every k without retraining the models.

  The best k is selected by validation NDCG@20.

Input:
  - config/config.yaml
  - data/processed/*
  - data/interim/genres.parquet

Output:
  - results/tables/k_sweep_val.csv
  - results/tables/best_k.json
  - results/figures/tradeoff_curve.png
  - results/figures/gate_curve.png

Run:
  python scripts/tune_k.py
==============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, set_seed
from src.recommend import load_artifacts
from src.models.hybrid_gate import gate_alpha, minmax_rows
from src.evaluation.evaluate import (
    heldout_from_df,
    top_n_from_scores,
    accuracy_on_subset,
)
from src.evaluation import exposure_metrics as em
from src.visualization import plots


def load_validation_data(cfg, art):
    """Load validation interactions and cold-book identifiers."""

    processed = Path(cfg["paths"]["processed_dir"])

    val_path = processed / "val.parquet"
    cold_path = processed / "cold_books.json"

    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing validation file: {val_path}"
        )

    val = pd.read_parquet(val_path)

    # The saved ID maps use string keys.
    val["user_id"] = val["user_id"].astype(str)
    val["book_id"] = val["book_id"].astype(str)

    heldout = heldout_from_df(
        val,
        art["user2idx"],
        art["book2idx"],
    )

    if not heldout:
        raise ValueError(
            "No validation interactions match the saved ID maps."
        )

    cold_book_ids = set()

    if cold_path.exists():
        with cold_path.open("r", encoding="utf-8") as file:
            cold_data = json.load(file)

        cold_book_ids = {
            str(book_id)
            for book_id in cold_data.get("book_ids", [])
        }

    cold_item_idx = {
        art["book2idx"][book_id]
        for book_id in cold_book_ids
        if book_id in art["book2idx"]
    }

    cold_heldout = {
        user_id: relevant & cold_item_idx
        for user_id, relevant in heldout.items()
        if relevant & cold_item_idx
    }

    return heldout, cold_heldout


def load_book_genres(cfg, art):
    """
    Load the dominant genre for every book.

    The returned array must follow the same order
    as the columns of the training matrix.
    """

    path = (
        Path(cfg["paths"]["interim_dir"])
        / "genres.parquet"
    )

    if not path.exists():
        print(
            "Genre file missing: genre variety will not "
            "be calculated."
        )
        return None

    genres = pd.read_parquet(path)

    if not {"book_id", "genre"}.issubset(genres.columns):
        print(
            "Genre columns missing: skipping genre variety."
        )
        return None

    genre_series = (
        genres.assign(
            book_id=genres["book_id"].astype(str)
        )
        .drop_duplicates(subset=["book_id"])
        .set_index("book_id")["genre"]
    )

    aligned = (
        genre_series
        .reindex(art["idx2book"])
        .fillna("unknown")
    )

    genre_codes, _ = pd.factorize(aligned)

    return genre_codes


def evaluate_k_values(cfg, art, heldout, cold_heldout, book_genre):
    """
    Compute ALS and TF-IDF scores once per batch,
    then evaluate all k values.
    """

    k_grid = [
        float(k)
        for k in cfg["gate"]["k_grid"]
    ]

    if not k_grid:
        raise ValueError("gate.k_grid cannot be empty.")

    if any(k < 0 for k in k_grid):
        raise ValueError("All k values must be non-negative.")

    k_grid = list(dict.fromkeys(k_grid))

    train = art["train"]
    n_items = np.asarray(art["n_items"], dtype=float)

    top_n = int(cfg["eval"]["top_n"])
    batch_size = int(cfg["eval"]["batch_size"])

    # Ensure NDCG@20 and Recall@20 can be computed.
    list_length = max(top_n, 20)

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    users = np.array(
        sorted(
            user_id
            for user_id, relevant in heldout.items()
            if relevant
        ),
        dtype=int,
    )

    print(
        f"Validation users: {len(users)}"
    )
    print(
        f"Books: {train.shape[1]}"
    )
    print(
        f"Testing {len(k_grid)} k values: {k_grid}"
    )

    # One top-N dictionary for each value of k.
    recommendations = {
        k: {}
        for k in k_grid
    }

    # Gate weights depend only on book popularity.
    alpha_by_k = {
        k: gate_alpha(n_items, k)
        for k in k_grid
    }

    for start in range(0, len(users), batch_size):

        batch = users[
            start:start + batch_size
        ]

        print(
            f"Scoring users "
            f"{start + 1}-{start + len(batch)} "
            f"/ {len(users)}"
        )

        # Score both branches ONLY ONCE per batch.
        als_scores = art["als"].score(batch)

        content_scores = art["content"].score(batch)

        # HybridGate uses per-user min-max normalization.
        als_normalized = minmax_rows(als_scores)
        content_normalized = minmax_rows(content_scores)

        seen = train[batch]

        for k in k_grid:

            alpha = alpha_by_k[k]

            # Same formula as HybridGate.combine().
            hybrid_scores = (
                alpha * als_normalized
                + (1.0 - alpha) * content_normalized
            )

            # Remove training books and get the best items.
            top_lists = top_n_from_scores(
                hybrid_scores,
                seen,
                list_length,
            )

            for user_id, book_list in zip(
                batch.tolist(),
                top_lists,
            ):
                recommendations[k][user_id] = book_list

    rows = []

    # Calculate metrics for each candidate k.
    for k in k_grid:

        top = recommendations[k]

        all_accuracy = accuracy_on_subset(
            top,
            heldout,
            [20],
        )

        cold_accuracy = accuracy_on_subset(
            top,
            cold_heldout,
            [20],
        )

        lists = np.stack([
            top[user_id][:top_n]
            for user_id in users.tolist()
        ])

        row = {
            "k": k,
            "recall": all_accuracy["recall@20"],
            "ndcg": all_accuracy["ndcg@20"],
            "cold_recall": cold_accuracy["recall@20"],
            "coverage": em.coverage(
                lists,
                n_books=train.shape[1],
            ),
            "novelty": em.novelty(
                lists,
                n_items,
                n_users=train.shape[0],
            ),
            "tail_share": em.tail_share(
                lists,
                n_items,
            ),
        }

        if book_genre is not None:
            row["variety"] = em.genre_variety(
                lists,
                book_genre,
            )
        else:
            row["variety"] = np.nan

        rows.append(row)

    table = pd.DataFrame(rows)

    return table


def select_best_k(table):
    """
    Select k by maximum validation NDCG@20.

    Ties:
      1. Higher coverage
      2. Smaller k
    """

    ordered = table.sort_values(
        by=["ndcg", "coverage", "k"],
        ascending=[False, False, True],
        kind="stable",
    )

    return ordered.iloc[0]


def save_results(cfg, table, best):
    """Save tuning metrics, best k, and report figures."""

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save validation sweep.
    table.to_csv(
        tables_dir / "k_sweep_val.csv",
        index=False,
    )

    # Save selected hyperparameter.
    best_result = {
        "k": float(best["k"]),
        "selection_metric": "validation_ndcg@20",
        "ndcg": float(best["ndcg"]),
        "recall": float(best["recall"]),
        "coverage": float(best["coverage"]),
    }

    best_path = tables_dir / "best_k.json"

    with best_path.open("w", encoding="utf-8") as f:
        json.dump(
            best_result,
            f,
            indent=2,
        )

    # The plotting function accepts all/recall@20.
    plot_table = table.rename(
        columns={
            "recall": "all/recall@20",
        }
    )

    plots.tradeoff_curve(
        plot_table,
        figures_dir / "tradeoff_curve.png",
    )

    # The plotting function handles positive k.
    # For k=0, the ALS weight is 1 on known books.
    if float(best["k"]) > 0:

        plots.gate_curve(
            k=float(best["k"]),
            out=figures_dir / "gate_curve.png",
        )

    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = np.arange(501)

        alpha = gate_alpha(
            n,
            k=0.0,
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(n, alpha, linewidth=2)

        ax.set_title("Hybrid Gate Curve (k=0)")
        ax.set_xlabel("Training interactions per book")
        ax.set_ylabel("ALS weight (alpha)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            figures_dir / "gate_curve.png",
            dpi=200,
        )

        plt.close(fig)

    print("\nSaved:")
    print(tables_dir / "k_sweep_val.csv")
    print(best_path)
    print(figures_dir / "tradeoff_curve.png")
    print(figures_dir / "gate_curve.png")


def main() -> None:

    cfg = load_config()

    set_seed(cfg["seed"])

    print("=" * 65)
    print("BOOKREC - GATE PARAMETER TUNING")
    print("=" * 65)

    # Reuse trained models. Do not fit again.
    art = load_artifacts(cfg)

    heldout, cold_heldout = load_validation_data(
        cfg,
        art,
    )

    book_genre = load_book_genres(
        cfg,
        art,
    )

    table = evaluate_k_values(
        cfg,
        art,
        heldout,
        cold_heldout,
        book_genre,
    )

    best = select_best_k(table)

    save_results(
        cfg,
        table,
        best,
    )

    print("\nVALIDATION RESULTS")
    print(
        table.round(4).to_string(index=False)
    )

    print("\nBEST k:", float(best["k"]))
    print("NDCG@20:", round(float(best["ndcg"]), 4))
    print("Recall@20:", round(float(best["recall"]), 4))
    print("Coverage:", round(float(best["coverage"]), 4))

    print("\nTuning complete!")


if __name__ == "__main__":
    main()
