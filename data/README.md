# data/
**Maintainer:** Amine (raw, processed) | Ella (interim). Nothing in here is committed to git.
Every file below is written by `scripts/run_pipeline.py` (calling the modules listed).

| Folder | Content | Written by (module) |
|---|---|---|
| `raw/` | Goodreads `goodreads_interactions_comics_graphic.json.gz`, `goodreads_books_comics_graphic.json.gz`, `goodreads_book_genres_initial.json.gz` — never edited | `src/data/download.py` |
| `interim/` | `interactions_raw.parquet` (user sample), `interactions_clean.parquet`, `books_clean.parquet`, `genres.parquet` | `load.py`, `clean.py` via the pipeline |
| `processed/` | `train/val/test.parquet`, `train_cold.parquet`, `cold_books.json`, `cold_users.json` | `split.py`, `cold_start.py` |
| `processed/` | `id_maps.json`, `train_csr.npz`, `n_items.npy`, `books_aligned.parquet` (row i = book i) | `run_pipeline.py` (`step_matrices`) |
| `processed/` | `als_factors.npz`, `tfidf_items.npz` (read by the app) | `als_model.py`, `content_tfidf.py` |

Source: UCSD Book Graph (https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html). Academic use only. Do not redistribute the files.
