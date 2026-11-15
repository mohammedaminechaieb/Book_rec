"""
==============================================================================
File      : tests/test_tfidf_ella.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner)
------------------------------------------------------------------------------
What it does:
  Unit tests for src/models/content_tfidf.py: text building, shapes, similar books score higher, cold books and users.
  Note: these tests were written by the author of the code; the cross-testing
  rule (each person tests the other's code) applies to the other test files.
Input:  3 toy books and a small CSR matrix
Output: pytest pass / fail
==============================================================================
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

from src.models.content_tfidf import ContentTFIDFModel


def sample_data():
    books = pd.DataFrame({
        "book_id": ["b1", "b2", "b3"],
        "text": [
            "magic dragons fantasy adventure",
            "magic wizard fantasy castle",
            "romance love relationships"
        ],
        "shelves": [
            ["fantasy", "magic"],
            ["fantasy", "wizard"],
            ["romance"]
        ]
    })

    train = sp.csr_matrix([
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    return books, train


def test_build_text():
    books, _ = sample_data()

    model = ContentTFIDFModel(min_df=1)

    documents = model.build_text(books)

    assert len(documents) == 3
    assert "magic" in documents[0]
    assert "fantasy" in documents[0]


def test_fit():
    books, train = sample_data()

    model = ContentTFIDFModel(min_df=1)
    result = model.fit(books, train)

    assert result is model
    assert model.item_vectors.shape[0] == 3
    assert sp.issparse(model.item_vectors)


def test_scores_shape():
    books, train = sample_data()

    model = ContentTFIDFModel(min_df=1).fit(
        books, train
    )

    scores = model.score(np.array([0, 1]))

    assert scores.shape == (2, 3)
    assert np.isfinite(scores).all()


def test_similar_fantasy_books():
    books, train = sample_data()

    model = ContentTFIDFModel(min_df=1).fit(
        books, train
    )

    scores = model.score(np.array([0]))

    # User 0 likes fantasy book b1.
    # Fantasy book b2 should be closer than romance b3.
    assert scores[0, 1] > scores[0, 2]


def test_cold_book():
    books, train = sample_data()

    model = ContentTFIDFModel(min_df=1).fit(
        books, train
    )

    # Book b2 has no training interactions.
    assert train[:, 1].nnz == 0

    # But TF-IDF can still score it.
    scores = model.score(np.array([0]))

    assert scores[0, 1] > 0


def test_new_user_without_history():
    books, train = sample_data()

    model = ContentTFIDFModel(min_df=1).fit(
        books, train
    )

    scores = model.score(np.array([2]))

    # No liked books -> no preference profile.
    assert np.allclose(scores, 0)


def test_score_before_fit():
    model = ContentTFIDFModel(min_df=1)

    with pytest.raises(RuntimeError):
        model.score(np.array([0]))
