"""
==============================================================================
File      : tests/test_editions.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead, tests Ella's code)  |  Ella (co-owner)
------------------------------------------------------------------------------
What it does:
  Tests src/editions.py: other editions of a picked / read book are not
  recommended again ("Maus I ..." vs "Maus: ... (Maus, #1)", "Vol." vs
  "Volume", "The Complete Maus"), but other volumes of a series are kept.
Input:  real-looking Goodreads titles
Output: pytest pass / fail
==============================================================================
"""
from src.editions import dedupe_candidates, series_key, title_key


def test_keys():
    assert series_key("Maus I: A Survivor's Tale (Maus, #1)") == "maus#1"
    assert title_key("Batman, Vol. 1: The Court of Owls") == title_key("Batman, Volume 1: The Court of Owls")


def test_dedupe_removes_editions_keeps_other_volumes():
    picked = ["Maus I: A Survivor's Tale: My Father Bleeds History (Maus, #1)",
              "Batman, Volume 1: The Court of Owls"]
    candidates = ["Maus: A Survivor's Tale : My Father Bleeds History (Maus, #1)",   # same work
                  "Batman, Vol. 1: The Court of Owls",                               # same work
                  "The Complete Maus",                                               # collected edition
                  "Maus II: A Survivor's Tale: And Here My Troubles Began (Maus, #2)",  # next volume: keep
                  "Saga, Vol. 1 (Saga, #1)",
                  "Saga, Volume 1 (Saga, #1)"]                                       # duplicate of the previous
    kept = [candidates[i] for i in dedupe_candidates(candidates, picked, limit=10)]
    assert kept == ["Maus II: A Survivor's Tale: And Here My Troubles Began (Maus, #2)",
                    "Saga, Vol. 1 (Saga, #1)"]


def test_limit():
    assert dedupe_candidates([f"Book {i}" for i in range(10)], [], limit=3) == [0, 1, 2]


def test_accents_volumes_and_collections():
    picked = ["Persepolis: The Story of a Childhood (Persepolis, #1)"]
    candidates = ["Persepolis: The Story of a Childhood and the Story of a Return",   # contains the picked book
                  "Persépolis (Persepolis #1-4)",                                    # collected, accent
                  "Persepolis, Volume 2",                                            # next volume: keep
                  "Persepolis 2: The Story of a Return",                             # same volume 2: drop
                  "Blacksad (Blacksad, #1-3)",                                       # collected: keep once
                  "Arctic-Nation (Blacksad, #2)",                                    # inside the collection: drop
                  "Saga, Vol. 1 (Saga, #1)",
                  "Saga, Vol. 2 (Saga, #2)"]                                         # other volume: keep
    kept = [candidates[i] for i in dedupe_candidates(candidates, picked, limit=10)]
    assert kept == ["Persepolis, Volume 2", "Blacksad (Blacksad, #1-3)",
                    "Saga, Vol. 1 (Saga, #1)", "Saga, Vol. 2 (Saga, #2)"]


def test_work_id_wins():
    kept = dedupe_candidates(["Totally Different Title"], ["Original Title"], limit=5,
                             candidate_work_ids=["42"], reference_work_ids=["42"])
    assert kept == []
