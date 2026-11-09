"""
==============================================================================
File      : src/utils/io.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Small helpers for saving and loading the intermediate files: DataFrames
  (parquet), sparse matrices (npz), dense arrays (npy), JSON. It also builds
  the id <-> index maps used to turn user_id/book_id into matrix rows/cols.
Input:
  - pandas DataFrames, scipy sparse matrices, numpy arrays, dicts
Output:
  - files in data/interim, data/processed, results/ (and the reverse: loaded objects)
==============================================================================
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp


def save_df(df: pd.DataFrame, path: Path) -> None:
    raise NotImplementedError("TODO (Ella): parquet write, create parent dir")


def load_df(path: Path) -> pd.DataFrame:
    raise NotImplementedError("TODO (Ella)")


def save_sparse(mat: sp.csr_matrix, path: Path) -> None:
    raise NotImplementedError("TODO (Ella): sp.save_npz")


def load_sparse(path: Path) -> sp.csr_matrix:
    raise NotImplementedError("TODO (Ella)")


def save_json(obj: dict, path: Path) -> None:
    raise NotImplementedError("TODO (Ella)")


def build_id_maps(df: pd.DataFrame) -> tuple[dict, dict]:
    """Return (user2idx, book2idx) from the user_id / book_id columns of the TRAIN set."""
    raise NotImplementedError("TODO (Ella)")


def to_csr(df: pd.DataFrame, user2idx: dict, book2idx: dict) -> sp.csr_matrix:
    """Build the binary user x book interaction matrix (1 = positive)."""
    raise NotImplementedError("TODO (Ella)")