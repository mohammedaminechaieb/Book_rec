"""
==============================================================================
File      : tests/test_exposure_metrics.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead, tests Ella's code)  |  Ella (co-owner)
------------------------------------------------------------------------------
What it does:
  Unit tests for src/evaluation/exposure_metrics.py (coverage, novelty,
  genre variety, tail share).
Input:  small hand-made top-N arrays
Output: pytest pass / fail   (run: pytest tests/test_exposure_metrics.py)
==============================================================================
"""
import numpy as np

from src.evaluation.exposure_metrics import coverage, genre_variety, novelty


def test_coverage_half_catalogue():
    top_n = np.array([[0, 1], [1, 0]])
    assert coverage(top_n, n_books=4) == 0.5


def test_genre_variety_all_same():
    top_n = np.array([[0, 1]])
    assert genre_variety(top_n, np.array([3, 3])) == 0.5


def test_novelty_rare_books_higher():
    n_items = np.array([1000, 1])
    assert novelty(np.array([[1]]), n_items, 1000) > novelty(np.array([[0]]), n_items, 1000)