"""
==============================================================================
File      : src/data/download.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Downloads the Goodreads genre files (interactions, books, genres) from the
  UCSD Book Graph repository into data/raw. Skips files that already exist.
  Each file is first written as <name>.part and renamed at the end, so a
  download that was cut is never mistaken for a complete file.
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
    if dest.exists():
        print(f"skip  {dest.name} (already downloaded)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")          # temporary name while downloading

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()                    # stop with an error on 404, 500, ...
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc=dest.name) as bar:
            for block in response.iter_content(chunk_size=chunk):
                f.write(block)
                bar.update(len(block))

    tmp.replace(dest)


def main() -> None:
    cfg = load_config()
    data = cfg["data"]
    raw_dir = cfg["paths"]["raw_dir"]
    base = data["base_url"].rstrip("/")

    # The interactions and books files are per genre (byGenre/ folder).
    # The genres file covers all books and sits at the root.
    urls = {
        data["interactions_file"]: f"{base}/byGenre/{data['interactions_file']}",
        data["books_file"]: f"{base}/byGenre/{data['books_file']}",
        data["genres_file"]: f"{base}/{data['genres_file']}",
    }
    for name, url in urls.items():
        download_file(url, raw_dir / name)
    print(f"Done. Files are in {raw_dir}")


if __name__ == "__main__":
    main()