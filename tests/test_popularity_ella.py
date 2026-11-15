"""
==============================================================================
File      : tests/test_popularity_ella.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner)
------------------------------------------------------------------------------
What it does:
  Unit tests for src/models/popularity.py: same popularity scores for every user, errors before fit.
  Note: these tests were written by the author of the code; the cross-testing
  rule (each person tests the other's code) applies to the other test files.
Input:  small CSR matrices
Output: pytest pass / fail
==============================================================================
"""
import numpy as np
import scipy.sparse as sp
import pytest

from src.models.popularity import PopularityModel


def test_popularity_scores():

    train = sp.csr_matrix([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0]
    ])

    model = PopularityModel()

    model.fit(train)

    scores = model.score(np.array([0, 1]))

    # 2 users and 4 books.
    assert scores.shape == (2, 4)

    # Popularity: A=3, B=2, C=1, D=1.
    np.testing.assert_array_equal(
        scores[0],
        np.array([3, 2, 1, 1])
    )

    # Every user receives the same scores.
    np.testing.assert_array_equal(
        scores[0],
        scores[1]
    )


def test_fit_returns_model():

    train = sp.csr_matrix([
        [1, 0],
        [0, 1]
    ])

    model = PopularityModel()

    assert model.fit(train) is model


def test_score_before_fit():

    model = PopularityModel()

    with pytest.raises(RuntimeError):
        model.score(np.array([0]))


def test_invalid_user():

    train = sp.csr_matrix([
        [1, 0],
        [0, 1]
    ])

    model = PopularityModel().fit(train)

    with pytest.raises(ValueError):
        model.score(np.array([5]))
