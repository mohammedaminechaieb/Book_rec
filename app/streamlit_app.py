"""
==============================================================================
File      : app/streamlit_app.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Demo interface (assignment step 6). Two modes:
    1. Existing reader: choose a reader and a model (gated hybrid / ALS /
       content / popularity). It shows the books they liked, then the top-N,
       with an explanation for each book (closest liked book, shared shelves,
       CF vs content share, popularity label). A side-by-side view compares
       ALS and the gate, so the long-tail effect is visible.
    2. New reader (cold start): pick a few books you like and get content-based
       recommendations, with the same explanations.
Input:
  - artifacts in data/processed (run scripts/run_pipeline.py first)
  - config.yaml -> app.*
Output:
  - web page (local)
Run:
  streamlit run app/streamlit_app.py
==============================================================================
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src import recommend  # noqa: E402

MODELS = {"Gated hybrid": "gate", "ALS only": "als", "Content only": "content", "Popularity": "popularity"}


@st.cache_resource
def get_artifacts():
    cfg = load_config()
    return cfg, recommend.load_artifacts(cfg)


def show_existing_reader(cfg: dict, art: dict) -> None:
    # TODO (Ella): selectbox of art["demo_users"]; show liked books;
    #   call recommend.recommend_for_user(); show the table with the explanation column;
    #   checkbox "compare with ALS" -> two columns side by side
    st.info("TODO (Ella): existing reader view")


def show_new_reader(cfg: dict, art: dict) -> None:
    # TODO (Ella): multiselect of book titles (>= 3); call recommend.recommend_for_new_reader()
    st.info("TODO (Ella): new reader view")


def main() -> None:
    st.set_page_config(page_title="book_rec demo", layout="wide")
    st.title("Book recommendations with less popularity bias")
    st.caption("Goodreads - ALS + TF-IDF with a per-book gate alpha = n / (n + k)")
    try:
        cfg, art = get_artifacts()
    except (NotImplementedError, FileNotFoundError) as err:
        st.warning(f"Artifacts not ready: {err}. Run scripts/run_pipeline.py first.")
        return
    mode = st.sidebar.radio("Mode", ["Existing reader", "New reader (cold start)"])
    if mode == "Existing reader":
        show_existing_reader(cfg, art)
    else:
        show_new_reader(cfg, art)


if __name__ == "__main__":
    main()