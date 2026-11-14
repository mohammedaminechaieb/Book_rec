
"""
==============================================================================
File      : src/data/stats.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer + writes the tests)
------------------------------------------------------------------------------
What it does:
  Computes dataset statistics:
  - numbers of users, books, interactions
  - density and sparsity
  - interactions per user/book
  - Gini coefficient of book popularity
  - top 20% popularity share
  - share of books with at most 3 interactions

Input:
  - interactions DataFrame with user_id and book_id columns

Output:
  - dictionary of statistics
  - comparison CSV for raw, clean and train stages
==============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


def sparsity(
    n_interactions: int,
    n_users: int,
    n_items: int
) -> float:
    """Compute sparsity of the user-item interaction matrix."""

    total_possible = n_users * n_items

    if total_possible == 0:
        return 0.0

    return float(1 - n_interactions / total_possible)


def gini(counts: np.ndarray) -> float:
    """Compute the Gini coefficient of book popularity counts."""

    counts = np.asarray(counts, dtype=float)

    if counts.size == 0:
        return 0.0

    if np.any(counts < 0):
        raise ValueError("Popularity counts must be non-negative.")

    total = counts.sum()

    if total == 0:
        return 0.0

    sorted_counts = np.sort(counts)
    n = len(sorted_counts)

    indices = np.arange(1, n + 1)

    coefficient = (
        2 * np.sum(indices * sorted_counts) / (n * total)
        - (n + 1) / n
    )

    return float(coefficient)


def dataset_stats(
    df: pd.DataFrame,
    stage: str = "clean"
) -> dict:
    """Compute descriptive statistics for one dataset stage."""

    required_columns = {"user_id", "book_id"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain user_id and book_id columns."
        )

    n_users = int(df["user_id"].nunique())
    n_items = int(df["book_id"].nunique())
    n_interactions = int(len(df))

    total_possible = n_users * n_items

    density = (
        n_interactions / total_possible
        if total_possible > 0
        else 0.0
    )

    sparse = sparsity(n_interactions, n_users, n_items)

    # Number of interactions for each user and book.
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("book_id").size()

    def describe_counts(counts):
        """Return mean, median, min and max safely."""
        if counts.empty:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0,
                "max": 0
            }

        return {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "min": int(counts.min()),
            "max": int(counts.max())
        }

    user_summary = describe_counts(user_counts)
    item_summary = describe_counts(item_counts)

    popularity_gini = gini(item_counts.to_numpy())

    # Share of interactions captured by the most popular 20% of books.
    if n_items > 0:
        top_count = int(np.ceil(n_items * 0.20))
        top_items = item_counts.nlargest(top_count)

        top_20_share = (
            float(top_items.sum() / n_interactions)
            if n_interactions > 0
            else 0.0
        )

        # Fraction of books with 3 or fewer interactions.
        low_popularity_share = float(
            (item_counts <= 3).mean()
        )
    else:
        top_20_share = 0.0
        low_popularity_share = 0.0

    stats = {
        "stage": stage,

        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,

        "density": float(density),
        "sparsity": sparse,

        "user_interactions_mean": user_summary["mean"],
        "user_interactions_median": user_summary["median"],
        "user_interactions_min": user_summary["min"],
        "user_interactions_max": user_summary["max"],

        "item_interactions_mean": item_summary["mean"],
        "item_interactions_median": item_summary["median"],
        "item_interactions_min": item_summary["min"],
        "item_interactions_max": item_summary["max"],

        "gini": popularity_gini,
        "top_20pct_share": top_20_share,
        "low_popularity_share": low_popularity_share
    }

    return stats


def stats_table(
    stages: dict[str, pd.DataFrame],
    out: Path
) -> pd.DataFrame:
    """Compute statistics for each stage and save a CSV table."""

    rows = []

    for stage_name, df in stages.items():
        rows.append(dataset_stats(df, stage=stage_name))

    result = pd.DataFrame(rows)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(out, index=False)

    return result
