
"""
==============================================================================
File      : src/visualization/plots.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Creates five figures for the report:
    1. Long-tail distribution of book interactions
    2. Recall@20 versus catalogue coverage
    3. Hybrid gate alpha versus book interaction count
    4. Popularity of recommended books by model
    5. Baseline metrics comparison

Input:
  - Evaluation DataFrames, item counts, recommendation indices

Output:
  - PNG figures at 200 dpi
==============================================================================
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_output(out: Path) -> Path:
    """Create the output directory when necessary."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _find_column(df, candidates):
    """Find a column using supported aliases."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Missing column. Expected one of: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def long_tail_distribution(
    n_items: np.ndarray,
    out: Path
) -> None:
    """Plot training interaction counts ranked by book popularity."""

    counts = np.asarray(n_items).ravel()

    if counts.size == 0:
        raise ValueError("n_items cannot be empty.")

    if np.any(counts < 0):
        raise ValueError("Interaction counts cannot be negative.")

    counts = np.sort(counts)[::-1]

    ranks = np.arange(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ranks, counts, linewidth=1.8)

    # A symmetric log scale can display zero counts.
    ax.set_yscale("symlog", linthresh=1)

    ax.set_title("Long-tail distribution of book popularity")
    ax.set_xlabel("Book rank (most to least popular)")
    ax.set_ylabel("Number of training interactions")

    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(_prepare_output(out), dpi=200)
    plt.close(fig)


def tradeoff_curve(
    k_table: pd.DataFrame,
    out: Path
) -> None:
    """Plot Recall@20 against coverage for different gate k values."""

    k_col = _find_column(k_table, ["k"])
    recall_col = _find_column(
        k_table,
        ["recall_at_20", "recall@20", "recall_20", "Recall@20"]
    )
    coverage_col = _find_column(
        k_table,
        ["coverage", "catalogue_coverage", "coverage_at_20"]
    )

    data = k_table[
        [k_col, recall_col, coverage_col]
    ].copy()

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna().sort_values(coverage_col)

    if data.empty:
        raise ValueError("No valid tradeoff data available.")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        data[coverage_col],
        data[recall_col],
        marker="o",
        linewidth=1.7
    )

    for _, row in data.iterrows():
        ax.annotate(
            f"k={row[k_col]:g}",
            (row[coverage_col], row[recall_col]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_title("Accuracy vs catalogue coverage")
    ax.set_xlabel("Catalogue coverage")
    ax.set_ylabel("Recall@20")

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_prepare_output(out), dpi=200)
    plt.close(fig)


def gate_curve(
    k: float,
    out: Path,
    n_max: int = 500
) -> None:
    """Plot the ALS weight as a function of book interactions."""

    if k <= 0:
        raise ValueError("k must be positive.")

    if n_max < 0:
        raise ValueError("n_max must be non-negative.")

    n = np.arange(n_max + 1, dtype=float)

    alpha = n / (n + k)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(n, alpha, linewidth=2)

    ax.axhline(
        0.5,
        linestyle="--",
        alpha=0.6,
        label="Equal ALS/content weight"
    )

    ax.set_title(f"Hybrid gate weight (k={k:g})")
    ax.set_xlabel("Number of training interactions per book")
    ax.set_ylabel("ALS weight (alpha)")

    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(_prepare_output(out), dpi=200)
    plt.close(fig)


def popularity_of_recs(
    recs_by_model: dict,
    n_items: np.ndarray,
    out: Path
) -> None:
    """Compare popularity distributions of recommended books."""

    counts = np.asarray(n_items).ravel()

    if counts.size == 0:
        raise ValueError("n_items cannot be empty.")

    if np.any(counts < 0):
        raise ValueError("Interaction counts cannot be negative.")

    if not recs_by_model:
        raise ValueError("No model recommendations provided.")

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0

    for model_name, recommendations in recs_by_model.items():

        indices = np.asarray(recommendations, dtype=int).ravel()

        valid = indices[
            (indices >= 0) & (indices < len(counts))
        ]

        if valid.size == 0:
            continue

        popularities = counts[valid]

        # log1p handles both popular and zero-interaction books.
        ax.hist(
            np.log1p(popularities),
            bins=30,
            alpha=0.45,
            density=True,
            label=str(model_name)
        )

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("No valid recommendations found.")

    ax.set_title("Popularity distribution of recommendations")
    ax.set_xlabel("log(1 + training interactions per book)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(_prepare_output(out), dpi=200)
    plt.close(fig)


def baseline_bars(
    metrics: pd.DataFrame,
    out: Path
) -> None:
    """Compare model metrics using separate bar panels."""

    model_col = _find_column(
        metrics,
        ["model", "method", "baseline"]
    )

    numeric_columns = [
        column
        for column in metrics.columns
        if column != model_col
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]

    if not numeric_columns:
        raise ValueError("No numeric metric columns available.")

    n_metrics = len(numeric_columns)

    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    model_names = metrics[model_col].astype(str)

    for i, metric in enumerate(numeric_columns):

        ax = axes.flat[i]

        values = metrics[metric]

        ax.bar(model_names, values)

        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel(metric)

        ax.tick_params(
            axis="x",
            labelrotation=40
        )

        ax.grid(axis="y", alpha=0.2)

    # Hide unused panels.
    for i in range(n_metrics, axes.size):
        axes.flat[i].set_visible(False)

    fig.suptitle(
        "Recommendation models comparison",
        fontsize=15
    )

    fig.tight_layout()
    fig.savefig(_prepare_output(out), dpi=200)
    plt.close(fig)
