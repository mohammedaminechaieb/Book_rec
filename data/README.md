# data/
**Maintainer:** Amine (raw, processed) | Ella (interim). Nothing in here is committed to git.

| Folder | Content | Produced by |
|---|---|---|
| `raw/` | Goodreads `*.json.gz` files, never edited | `src/data/download.py` |
| `interim/` | `interactions_clean.parquet`, `books_clean.parquet` | `src/data/clean.py` |
| `processed/` | `train/val/test.parquet`, `train_cold.parquet`, `cold_books.json`, id maps, CSR `.npz` | `split.py`, `cold_start.py`, `utils/io.py` |

Source: UCSD Book Graph (https://mengtingwan.github.io/data/goodreads.html). Academic use only. Do not redistribute the files.