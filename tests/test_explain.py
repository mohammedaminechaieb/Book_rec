"""
==============================================================================
File      : tests/test_explain.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead, tests Amine's code)  |  Amine (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/explain.py: score shares sum to 1, alpha = 0 gives a 100% content
  share, the closest liked book is found, and shared shelves are ranked.
Input:  tiny hand-made vectors and shelf lists
Output: pytest pass / fail
==============================================================================
"""
import math

import numpy as np
import scipy.sparse as sp

from src.explain import closest_liked_book, score_breakdown, shared_shelves


def test_breakdown_sums_to_one():
    b = score_breakdown(alpha=0.3, cf_score=0.8, content_score=0.5)
    assert math.isclose(b["cf_share"] + b["content_share"], 1.0)


def test_breakdown_new_book_is_content_only():
    assert score_breakdown(alpha=0.0, cf_score=0.9, content_score=0.4)["content_share"] == 1.0


def test_closest_liked_book():
    vecs = sp.csr_matrix(np.array([[1, 0], [0, 1], [0.9, 0.1]]))
    idx, sim = closest_liked_book(2, np.array([0, 1]), vecs)
    assert idx == 0 and sim > 0.9


def test_shared_shelves_order():
    out = shared_shelves(["magic", "dragons", "romance"],
                         [["magic", "dragons"], ["magic"], ["horror"]], top=2)
    assert out == ["magic", "dragons"]