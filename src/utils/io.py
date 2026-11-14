
"""
==============================================================================
File      : src/utils/io.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Helpers for saving and loading DataFrames (parquet), sparse matrices
  (npz) and JSON. Builds user/book index mappings and a binary CSR matrix.
Input:
  - pandas DataFrames, scipy sparse matrices, dictionaries
Output:
  - Saved files and loaded objects; user/book mappings; CSR matrices
==============================================================================
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp


def save_df(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_df(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a Parquet file."""
    return pd.read_parquet(Path(path))


def save_sparse(mat: sp.csr_matrix, path: Path) -> None:
    """Save a sparse matrix in NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, mat.tocsr())


def load_sparse(path: Path) -> sp.csr_matrix:
    """Load a sparse matrix as CSR."""
    return sp.load_npz(Path(path)).tocsr()


def save_json(obj: dict, path: Path) -> None:
    """Save a dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_id_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build user and book mappings using the TRAIN interactions.

    Returns:
        user2idx: {user_id: row_index}
        book2idx: {book_id: column_index}
    """
    user_ids = pd.unique(df["user_id"])
    book_ids = pd.unique(df["book_id"])

    user2idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    book2idx = {
        book_id: idx
        for idx, book_id in enumerate(book_ids)
    }

    return user2idx, book2idx


def to_csr(
    df: pd.DataFrame,
    user2idx: dict,
    book2idx: dict
) -> sp.csr_matrix:
    """
    Build a binary user x book interaction matrix.

    A value of 1 indicates a positive interaction.
    Unknown users/books are ignored.
    Duplicate interactions are collapsed to 1.
    """
    user_indices = df["user_id"].map(user2idx)
    book_indices = df["book_id"].map(book2idx)

    valid = user_indices.notna() & book_indices.notna()

    rows = user_indices[valid].to_numpy(dtype=np.int64)
    cols = book_indices[valid].to_numpy(dtype=np.int64)

    values = np.ones(len(rows), dtype=np.float32)

    mat = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user2idx), len(book2idx)),
        dtype=np.float32
    )

    # Multiple interactions for the same pair count as one.
    mat.sum_duplicates()
    mat.data[:] = 1.0

    return mat
