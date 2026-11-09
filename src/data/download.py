"""
==============================================================================
File      : src/data/download.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Downloads the Goodreads genre files (interactions, books, genres) from the
  UCSD Book Graph repository into data/raw. Skips files that already exist.
  Check the URLs on https://mengtingwan.github.io/data/goodreads.html
Input:
  - config.yaml -> data.base_url, data.*_file
Output:
  - data/raw/goodreads_interactions_<genre>.json.gz
  - data/raw/goodreads_books_<genre>.json.gz
  - data/raw/goodreads_book_genres_initial.json.gz
Run:
  python -m src.data.download
==============================================================================
"""
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils.config import load_config


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a file to disk with a progress bar. Skip it if it already exists."""
    raise NotImplementedError("TODO (Amine)")


def main() -> None:
    cfg = load_config()
    # TODO (Amine): build the 3 URLs (byGenre/ for interactions + books, root for genres) and call download_file
    raise NotImplementedError


if __name__ == "__main__":
    main()