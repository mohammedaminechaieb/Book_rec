"""
==============================================================================
File      : tests/test_labels.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead, tests Amine's code)  |  Amine (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/labels.py and the labels used by src/explain.py:
    - every popularity class returned for a book is one the app gauge knows
      (this bug once hid all best-sellers from the gauge)
    - pépites = the books outside the top 20 % (same as the tail_share metric)
    - French number formatting and model names
    - explanation sentences are in French
Input:  small arrays
Output: pytest pass / fail
==============================================================================
"""
import numpy as np
import scipy.sparse as sp
import pandas as pd

from src.explain import explain, frequent_shelves, popularity_label
from src.labels import POP_CLASSES, POP_LABEL, fr_number, fr_percent, model_k, model_label


def test_every_label_is_a_gauge_class():
    n_items = np.arange(1, 101)
    labels = {popularity_label(int(n), n_items) for n in n_items}
    assert labels <= set(POP_CLASSES) and labels == set(POP_CLASSES)
    assert all(c in POP_LABEL for c in labels)


def test_tail_is_outside_top_20_percent():
    n_items = np.arange(1, 101)
    tail = [n for n in n_items if popularity_label(int(n), n_items) == "tail"]
    assert len(tail) == 80


def test_french_numbers():
    assert fr_number(0.2836, 3) == "0,284"
    assert fr_number(2845, 0) == "2 845"
    assert fr_percent(0.816, 1) == "81,6 %"


def test_model_names():
    assert model_label("Gated hybrid (k=20)").startswith("Hybride adaptatif")
    assert model_label("Hybrid 50/50") == "Hybride 50/50"
    assert model_k("Hybride adaptatif (k=5)") == 5.0


def test_frequent_shelves():
    shelves = [["comics", "manga"], ["comics", "horror"], ["comics", "memoir"]]
    assert frequent_shelves(shelves, max_share=0.5) == {"comics"}


def test_explanation_is_french():
    books = pd.DataFrame({"title": ["Watchmen.", "Saga", "Tiny"],
                          "shelves": [["superheroes"], ["space-opera"], ["superheroes", "indie"]]})
    train = sp.csr_matrix(np.array([[1, 1, 0]]))
    vecs = sp.csr_matrix(np.array([[1.0, 0.2], [0.0, 1.0], [0.9, 0.1]]))
    out = explain(0, 2, train, vecs, books, np.array([500, 300, 2]), k=20, cf_score=0.7, content_score=0.9)
    assert out["sentence"].startswith("Parce que vous avez aimé « Watchmen »")
    assert ".." not in out["sentence"] and "Because" not in out["sentence"]
