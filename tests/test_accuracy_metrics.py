"""
==============================================================================
File      : tests/test_accuracy_metrics.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead, tests Amine's code)  |  Amine (co-owner)
------------------------------------------------------------------------------
What it does:
  Unit tests for src/evaluation/accuracy_metrics.py, on hand-computed cases.
Input:  small hand-made lists
Output: pytest pass / fail   (run: pytest tests/test_accuracy_metrics.py)
==============================================================================
"""
import math

from src.evaluation.accuracy_metrics import recall_at_k, ndcg_at_k


def test_recall_perfect():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_recall_half():
    assert recall_at_k([1, 9, 8, 7], {1, 2}, k=4) == 0.5


def test_ndcg_first_hit():
    assert math.isclose(ndcg_at_k([5, 1, 2], {5}, k=3), 1.0)


def test_ndcg_second_position():
    assert math.isclose(ndcg_at_k([9, 5], {5}, k=2), 1 / math.log2(3))