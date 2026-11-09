"""
==============================================================================
File      : src/recommend.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Shared recommendation logic used by the Streamlit app (app/streamlit_app.py)
  and by the command line:
    - load_artifacts(): loads the saved models and tables from data/processed
    - recommend_for_user(): top-N for an existing reader + one explanation per book
    - recommend_for_new_reader(): top-N from a few books picked by a new reader
      (content only, since there is no CF vector: the cold-user case)
Input:
  - data/processed/* written by scripts/run_pipeline.py
    (id maps, n_items.npy, als_factors.npz, tfidf_items.npz, books_clean.parquet, best_k.json)
  - --user <user_id> on the command line
Output:
  - DataFrame [rank, title, score, alpha, popularity, explanation]
  - console print + results/tables/demo_<user_id>.csv (CLI)
Run:
  python -m src.recommend --user <user_id>
==============================================================================
"""
import argparse

import pandas as pd


def load_artifacts(cfg: dict) -> dict:
    raise NotImplementedError("TODO (Ella)")


def recommend_for_user(user_id: str, art: dict, model_name: str = "gate", top_n: int = 10) -> pd.DataFrame:
    """model_name in {"gate", "als", "content", "popularity"}. Uses src.explain.explain for each row."""
    raise NotImplementedError("TODO (Ella)")


def recommend_for_new_reader(book_ids: list[str], art: dict, top_n: int = 10) -> pd.DataFrame:
    raise NotImplementedError("TODO (Ella)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--model", default="gate")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    raise NotImplementedError("TODO (Ella)")


if __name__ == "__main__":
    main()