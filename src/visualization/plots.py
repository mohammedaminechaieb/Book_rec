"""
==============================================================================
File      : src/visualization/plots.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  All report figures (matplotlib, saved as PNG at 200 dpi):
    1. long_tail_distribution  - interactions per book (log scale), the data's long tail
    2. tradeoff_curve          - Recall@20 vs coverage, one point per k (main result)
    3. gate_curve              - alpha against n for the chosen k
    4. popularity_of_recs      - histogram of the popularity of recommended books per model
    5. baseline_bars           - all metrics per model (popularity / ALS / content / 50-50 / gate)
Input:
  - DataFrames from results/tables/*.csv, n_items array
Output:
  - results/figures/*.png
==============================================================================
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def long_tail_distribution(n_items: np.ndarray, out: Path) -> None:
    raise NotImplementedError("TODO (Ella)")


def tradeoff_curve(k_table: pd.DataFrame, out: Path) -> None:
    raise NotImplementedError("TODO (Ella)")


def gate_curve(k: float, out: Path, n_max: int = 500) -> None:
    raise NotImplementedError("TODO (Ella)")


def popularity_of_recs(recs_by_model: dict, n_items: np.ndarray, out: Path) -> None:
    raise NotImplementedError("TODO (Ella)")


def baseline_bars(metrics: pd.DataFrame, out: Path) -> None:
    raise NotImplementedError("TODO (Ella)")