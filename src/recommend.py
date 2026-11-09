"""
==============================================================================
File      : src/recommend.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Demo for the presentation. For one user it prints the books the user liked
  and the top-20 from ALS, next to the top-20 from the gated hybrid, with
  title, n (popularity) and alpha, so the long-tail effect can be seen.
Input:
  - trained models (or saved scores), book2idx, books_clean.parquet
  - --user <user_id> (CLI argument)
Output:
  - table printed to the console
  - results/tables/demo_<user_id>.csv
Run:
  python -m src.recommend --user <user_id>
==============================================================================
"""
import argparse


def recommend_for_user(user_id: str, model, k: int = 20):
    raise NotImplementedError("TODO (Ella)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    args = parser.parse_args()
    raise NotImplementedError("TODO (Ella)")


if __name__ == "__main__":
    main()