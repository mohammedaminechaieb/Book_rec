
"""
==============================================================================
File      : src/data/clean.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  1. Keeps positive interactions (is_read == True OR rating >= threshold).
  2. Removes duplicate user-book pairs.
  3. Applies an iterative k-core filter.
  4. Cleans book text and keeps only relevant books.

Input:
  - interactions and books DataFrames from src/data/load.py
  - filtering thresholds supplied by config.yaml through the pipeline

Output:
  - cleaned interactions: user_id, book_id, date_updated
  - cleaned books: book_id, title, text, shelves, n_ratings
==============================================================================
"""

import html
import re

import pandas as pd


def keep_positive(df: pd.DataFrame, positive_rating: int) -> pd.DataFrame:
    """Keep positive interactions and remove duplicate user-book pairs."""

    required = {"user_id", "book_id", "is_read", "rating"}

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    result = df.copy()

    # Remove interactions without valid identifiers.
    result = result.dropna(subset=["user_id", "book_id"])

    # Convert rating to numeric safely.
    ratings = pd.to_numeric(result["rating"], errors="coerce")

    # A positive interaction is a read book OR a high rating.
    is_read = result["is_read"].fillna(False).eq(True)

    positive_mask = is_read | (ratings >= positive_rating)

    result = result.loc[positive_mask].copy()

    # Keep the most recent interaction for each user-book pair.
    if "date_updated" in result.columns:
        result = result.sort_values(
            "date_updated",
            kind="stable",
            na_position="first"
        )

    result = result.drop_duplicates(
        subset=["user_id", "book_id"],
        keep="last"
    )

    # Keep columns required by the downstream pipeline.
    output_columns = ["user_id", "book_id"]

    if "date_updated" in result.columns:
        output_columns.append("date_updated")

    return result[output_columns].reset_index(drop=True)


def k_core_filter(
    df: pd.DataFrame,
    min_user: int,
    min_book: int
) -> pd.DataFrame:
    """Repeatedly filter users and books until no rows are removed."""

    if min_user < 1 or min_book < 1:
        raise ValueError("Minimum interaction counts must be >= 1.")

    result = df.copy()

    while True:
        previous_size = len(result)

        # Count positive interactions per user.
        user_counts = result["user_id"].value_counts()

        valid_users = user_counts[
            user_counts >= min_user
        ].index

        result = result[
            result["user_id"].isin(valid_users)
        ].copy()

        # Recalculate book counts after removing users.
        book_counts = result["book_id"].value_counts()

        valid_books = book_counts[
            book_counts >= min_book
        ].index

        result = result[
            result["book_id"].isin(valid_books)
        ].copy()

        # Stop when the number of rows no longer changes.
        if len(result) == previous_size:
            break

    return result.reset_index(drop=True)


def clean_books(books: pd.DataFrame, keep_ids: set) -> pd.DataFrame:
    """Clean metadata and keep only books with remaining interactions."""

    required = {"book_id", "title", "description", "shelves"}

    if not required.issubset(books.columns):
        raise ValueError(f"Missing columns: {required - set(books.columns)}")

    result = books.copy()

    # Keep only books present in cleaned interactions.
    keep_ids = {str(book_id) for book_id in keep_ids}

    result = result[
        result["book_id"].astype(str).isin(keep_ids)
    ].copy()

    result = result.drop_duplicates(
        subset=["book_id"],
        keep="first"
    )

    # Text cleaning helper.
    def normalize_text(value):
        if not isinstance(value, str):
            return ""

        text = html.unescape(value)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    # Preserve original titles for display in the app.
    result["title"] = result["title"].fillna("").astype(str)

    # Clean descriptions.
    result["text"] = result["description"].apply(normalize_text)

    # If description is missing, use the title.
    missing_text = result["text"].eq("")

    result.loc[missing_text, "text"] = (
        result.loc[missing_text, "title"].apply(normalize_text)
    )

    # Normalize shelf names while preserving them as a list.
    def normalize_shelves(value):
        if not isinstance(value, (list, tuple)):
            return []

        return [
            cleaned
            for shelf in value
            if (cleaned := normalize_text(shelf))
        ]

    result["shelves"] = result["shelves"].apply(normalize_shelves)

    # Match the column name expected by the cleaned dataset.
    if "ratings_count" in result.columns:
        result["n_ratings"] = pd.to_numeric(
            result["ratings_count"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        result["n_ratings"] = 0

    columns = ["book_id", "title", "text", "shelves", "n_ratings"]

    # Keep the Goodreads work id when present: it groups the editions of a book.
    if "work_id" in result.columns:
        result["work_id"] = result["work_id"].fillna("").astype(str)
        columns.append("work_id")

    return result[columns].reset_index(drop=True)
