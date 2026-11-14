
import numpy as np
import pandas as pd


def make_cold_books(
    train: pd.DataFrame,
    fraction: float,
    keep: int,
    seed: int
) -> tuple[pd.DataFrame, list]:
    """
    Simulate cold-start books.

    Select a fraction of training books randomly and retain
    at most 'keep' training interactions for each selected book.

    Returns:
        train_cold: reduced training DataFrame
        cold_book_ids: selected book identifiers
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    if keep < 0:
        raise ValueError("keep must be non-negative")

    rng = np.random.default_rng(seed)

    book_ids = train["book_id"].dropna().unique()

    n_cold = int(len(book_ids) * fraction)

    if n_cold == 0:
        return train.copy().reset_index(drop=True), []

    cold_books = rng.choice(
        book_ids,
        size=n_cold,
        replace=False
    ).tolist()

    # All interactions belonging to non-cold books.
    normal_mask = ~train["book_id"].isin(cold_books)
    normal_rows = train.loc[normal_mask].copy()

    # Interactions belonging to selected cold books.
    cold_rows = train.loc[~normal_mask].copy()

    # Randomize and retain at most 'keep' rows per cold book.
    if keep == 0:
        cold_rows = cold_rows.iloc[0:0].copy()
    else:
        random_state = int(
            rng.integers(0, np.iinfo(np.int32).max)
        )

        cold_rows = cold_rows.sample(
            frac=1,
            random_state=random_state
        )

        cold_rows = (
            cold_rows.groupby("book_id", sort=False)
            .head(keep)
        )

    train_cold = pd.concat(
        [normal_rows, cold_rows]
    ).sort_index(kind="stable").reset_index(drop=True)

    return train_cold, cold_books


def make_cold_users(
    train: pd.DataFrame,
    fraction: float,
    keep: int,
    seed: int
) -> tuple[pd.DataFrame, list]:
    """
    Simulate users with limited histories.

    For each selected cold user, keep only the earliest
    'keep' training interactions based on date_updated.

    Returns:
        train_cold: reduced training DataFrame
        cold_user_ids: selected user identifiers
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    if keep < 0:
        raise ValueError("keep must be non-negative")

    if "date_updated" not in train.columns:
        raise ValueError("date_updated column is required")

    rng = np.random.default_rng(seed)

    user_ids = train["user_id"].dropna().unique()

    n_cold = int(len(user_ids) * fraction)

    if n_cold == 0:
        return train.copy().reset_index(drop=True), []

    cold_users = rng.choice(
        user_ids,
        size=n_cold,
        replace=False
    ).tolist()

    # Keep all interactions of users not selected as cold.
    normal_mask = ~train["user_id"].isin(cold_users)
    normal_rows = train.loc[normal_mask].copy()

    # Sort cold-user interactions chronologically.
    cold_rows = train.loc[~normal_mask].copy()

    cold_rows["date_updated"] = pd.to_datetime(
        cold_rows["date_updated"],
        errors="coerce"
    )

    cold_rows = cold_rows.sort_values(
        "date_updated",
        kind="stable",
        na_position="last"
    )

    # Retain the earliest interactions for each cold user.
    cold_rows = (
        cold_rows.groupby("user_id", sort=False)
        .head(keep)
    )

    train_cold = pd.concat(
        [normal_rows, cold_rows]
    ).sort_index(kind="stable").reset_index(drop=True)

    return train_cold, cold_users


def book_counts(train: pd.DataFrame) -> pd.Series:
    """
    Count the number of training interactions per book.

    The counts are used by the hybrid gate to control
    the weight of ALS versus the content-based model.

    Returns:
        pandas Series indexed by book_id
    """
    counts = train.groupby("book_id").size()
    counts.name = "n_interactions"

    return counts
