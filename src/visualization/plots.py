
"""
==============================================================================
File      : src/visualization/plots.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Creates all figures used in the BookRec report:

  1. long_tail_distribution
  2. tradeoff_curve
  3. gate_curve
  4. popularity_of_recs
  5. baseline_bars

Input:
  - Evaluation DataFrames
  - Training interaction counts per book
  - Recommendation indices

Output:
  - PNG figures saved at 200 dpi
==============================================================================
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_output(out: Path) -> Path:
    """Create the output directory if necessary."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    return out


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first available column among several possible names."""

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Missing column. Expected one of: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# =============================================================================
# 1. LONG-TAIL DISTRIBUTION
# =============================================================================

def long_tail_distribution(
    n_items: np.ndarray,
    out: Path
) -> None:
    """
    Plot the number of training interactions for each book.

    Books are sorted from most popular to least popular.
    """

    counts = np.asarray(n_items).ravel()

    if counts.size == 0:
        raise ValueError("n_items cannot be empty.")

    if np.any(counts < 0):
        raise ValueError(
            "Interaction counts cannot be negative."
        )

    # Sort books by decreasing popularity.
    counts = np.sort(counts)[::-1]

    ranks = np.arange(
        1,
        len(counts) + 1
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        ranks,
        counts,
        linewidth=1.8,
        color="steelblue"
    )

    # symlog allows us to display books with zero interactions.
    ax.set_yscale(
        "symlog",
        linthresh=1
    )

    ax.set_title(
        "Long-tail distribution of book popularity",
        fontsize=14
    )

    ax.set_xlabel(
        "Book rank (most to least popular)"
    )

    ax.set_ylabel(
        "Number of training interactions"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        _prepare_output(out),
        dpi=200
    )

    plt.close(fig)


# =============================================================================
# 2. TRADEOFF CURVE
# =============================================================================

def tradeoff_curve(
    k_table: pd.DataFrame,
    out: Path
) -> None:
    """
    Plot the accuracy-diversity tradeoff for different k values.

    X-axis: Catalogue coverage
    Y-axis: Recall@20
    """

    data = k_table.copy()

    k_col = _find_column(
        data,
        ["k"]
    )

    recall_col = _find_column(
        data,
        [
            "all/recall@20",
            "recall_at_20",
            "recall@20",
            "recall_20",
            "Recall@20",
        ]
    )

    coverage_col = _find_column(
        data,
        [
            "coverage",
            "catalogue_coverage",
            "coverage_at_20",
        ]
    )

    data = data[
        [k_col, recall_col, coverage_col]
    ].copy()

    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna()

    if data.empty:
        raise ValueError(
            "No valid tradeoff data available."
        )

    # Sort the points by coverage for the visual representation.
    data = data.sort_values(
        coverage_col
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        data[coverage_col],
        data[recall_col],
        marker="o",
        markersize=6,
        linewidth=1.8,
        color="royalblue"
    )

    # Label each point with its k value.
    for _, row in data.iterrows():

        ax.annotate(
            f"k={row[k_col]:g}",
            (
                row[coverage_col],
                row[recall_col]
            ),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_title(
        "Recall@20 vs Catalogue Coverage",
        fontsize=14
    )

    ax.set_xlabel(
        "Catalogue Coverage"
    )

    ax.set_ylabel(
        "Recall@20"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        _prepare_output(out),
        dpi=200
    )

    plt.close(fig)


# =============================================================================
# 3. GATE CURVE
# =============================================================================

def gate_curve(
    k: float,
    out: Path,
    n_max: int = 500
) -> None:
    """
    Show how the hybrid model adjusts the ALS weight.

    Formula:
        alpha = n / (n + k)

    n = number of training interactions for a book.
    k = gate parameter.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if n_max < 0:
        raise ValueError(
            "n_max must be non-negative."
        )

    n = np.arange(
        n_max + 1,
        dtype=float
    )

    # Weight assigned to ALS.
    alpha = n / (n + k)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        n,
        alpha,
        linewidth=2,
        color="darkorange",
        label="ALS weight"
    )

    ax.axhline(
        y=0.5,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label="Equal ALS/content weight"
    )

    # At n=k, alpha=0.5.
    ax.axvline(
        x=k,
        linestyle=":",
        color="gray",
        alpha=0.7,
        label=f"k = {k:g}"
    )

    ax.set_title(
        f"Hybrid Gate Curve (k={k:g})",
        fontsize=14
    )

    ax.set_xlabel(
        "Number of training interactions per book"
    )

    ax.set_ylabel(
        "ALS weight (alpha)"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        _prepare_output(out),
        dpi=200
    )

    plt.close(fig)


# =============================================================================
# 4. POPULARITY OF RECOMMENDATIONS
# =============================================================================

def popularity_of_recs(
    recs_by_model: dict,
    n_items: np.ndarray,
    out: Path
) -> None:
    """
    Compare the popularity distributions of recommended books.

    recs_by_model:
        {model_name: array of recommended book indices}

    n_items:
        number of training interactions per book.
    """

    counts = np.asarray(n_items).ravel()

    if counts.size == 0:
        raise ValueError(
            "n_items cannot be empty."
        )

    if np.any(counts < 0):
        raise ValueError(
            "Interaction counts cannot be negative."
        )

    if not recs_by_model:
        raise ValueError(
            "No model recommendations provided."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0

    for model_name, recommendations in recs_by_model.items():

        indices = np.asarray(
            recommendations,
            dtype=int
        ).ravel()

        # Ignore invalid book indices such as -1.
        valid = indices[
            (indices >= 0)
            & (indices < len(counts))
        ]

        if valid.size == 0:
            continue

        popularities = counts[valid]

        # The log1p transformation handles zero-interaction books.
        log_popularity = np.log1p(popularities)

        ax.hist(
            log_popularity,
            bins=30,
            alpha=0.45,
            density=True,
            label=str(model_name)
        )

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(
            "No valid recommendations found."
        )

    ax.set_title(
        "Popularity Distribution of Recommended Books",
        fontsize=14
    )

    ax.set_xlabel(
        "log(1 + number of training interactions)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        _prepare_output(out),
        dpi=200
    )

    plt.close(fig)


# =============================================================================
# 5. BASELINE MODEL COMPARISON
# =============================================================================

def baseline_bars(
    metrics: pd.DataFrame,
    out: Path
) -> None:
    """
    Compare the evaluation metrics of all recommendation models.

    Supports two DataFrame formats:

    1. Model names stored in a column:
         model | recall_at_20 | coverage | ...

    2. Model names stored in the index:
         index=model, columns=all/recall@20, coverage, ...

    The pipeline uses the second format.
    """

    data = metrics.copy()

    # Detect if the model names are already a column.
    model_candidates = [
        "model",
        "method",
        "baseline"
    ]

    if not any(
        col in data.columns
        for col in model_candidates
    ):

        # Convert DataFrame index into a column.
        data = data.reset_index()

        # Rename the first column as model.
        data = data.rename(
            columns={
                data.columns[0]: "model"
            }
        )

    model_col = _find_column(
        data,
        model_candidates
    )

    # Metrics selected for the final report.
    selected_metrics = [
        "all/recall@20",
        "all/ndcg@20",
        "coverage",
        "novelty",
        "tail_share",
        "genre_variety",
    ]

    available_metrics = [
        col
        for col in selected_metrics
        if col in data.columns
    ]

    # Compatibility with small test DataFrames.
    if not available_metrics:

        available_metrics = [
            col
            for col in data.columns
            if col != model_col
            and pd.api.types.is_numeric_dtype(
                data[col]
            )
        ]

    if not available_metrics:
        raise ValueError(
            "No numeric metrics available for plotting."
        )

    n_metrics = len(available_metrics)

    n_cols = min(
        3,
        n_metrics
    )

    n_rows = int(
        np.ceil(n_metrics / n_cols)
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            5 * n_cols,
            4.5 * n_rows
        ),
        squeeze=False
    )

    model_names = data[
        model_col
    ].astype(str)

    # Create one subplot for each metric.
    for i, metric in enumerate(available_metrics):

        ax = axes.flat[i]

        values = pd.to_numeric(
            data[metric],
            errors="coerce"
        )

        ax.bar(
            model_names,
            values,
            color="cornflowerblue"
        )

        title = (
            metric
            .replace("all/", "")
            .replace("_", " ")
        )

        ax.set_title(
            title,
            fontsize=11
        )

        ax.set_ylabel(
            metric
        )

        ax.tick_params(
            axis="x",
            labelrotation=35,
            labelsize=8
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

    # Hide unused panels.
    for i in range(
        n_metrics,
        axes.size
    ):
        axes.flat[i].set_visible(False)

    fig.suptitle(
        "Recommendation Models Comparison",
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        _prepare_output(out),
        dpi=200
    )

    plt.close(fig)
