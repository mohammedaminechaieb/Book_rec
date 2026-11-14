
import pandas as pd

from src.data.clean import (
    keep_positive,
    k_core_filter,
    clean_books,
)


def test_keep_positive():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3"],
        "book_id": ["b1", "b2", "b3", "b4"],
        "is_read": [True, False, False, True],
        "rating": [2, 5, 2, 0],
    })

    result = keep_positive(df, positive_rating=4)

    assert len(result) == 3
    assert "b3" not in result["book_id"].values


def test_remove_duplicates():
    df = pd.DataFrame({
        "user_id": ["u1", "u1"],
        "book_id": ["b1", "b1"],
        "is_read": [True, True],
        "rating": [4, 5],
    })

    result = keep_positive(df, positive_rating=4)

    assert len(result) == 1


def test_k_core():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "book_id": ["b1", "b2", "b1", "b2", "b3"],
    })

    result = k_core_filter(df, min_user=2, min_book=2)

    assert len(result) == 4
    assert set(result["user_id"]) == {"u1", "u2"}
    assert set(result["book_id"]) == {"b1", "b2"}


def test_clean_books():
    books = pd.DataFrame({
        "book_id": ["b1", "b2"],
        "title": ["Fantasy Book", "Other Book"],
        "description": ["<p>MAGIC Adventure</p>", ""],
        "shelves": [["fantasy"], ["fiction"]],
        "ratings_count": [100, 10],
    })

    result = clean_books(books, keep_ids={"b1"})

    assert len(result) == 1
    assert result.iloc[0]["text"] == "magic adventure"
    assert result.iloc[0]["n_ratings"] == 100
