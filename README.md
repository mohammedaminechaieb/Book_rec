# book_rec - Reducing popularity bias in book recommendation (Goodreads)

| | |
|---|---|
| **Team** | Amine, Ella (equal split; see `docs/team_split.md`) |
| **Course** | Big Data Project - Proposal 3 |
| **Dataset** | UCSD Book Graph (Goodreads, Wan & McAuley), one genre subset |
| **Output** | Top-20 book list per reader, evaluated on accuracy AND long-tail exposure |

## 1. Problem
Collaborative filtering mostly recommends bestsellers, because they have the most data.
Niche books (translations, small publishers, older titles) stay invisible. This is **popularity bias**.
**Question:** how much more of the long tail can we recommend without losing accuracy?

## 2. Method

    Interactions + books -> Cleaning and split -> Implicit ALS + TF-IDF vectors -> Gate -> Top-20 -> Accuracy + exposure metrics

- **Collaborative branch:** implicit-feedback ALS (`implicit` library).
- **Content branch:** TF-IDF over the book description and its popular shelves. A book's score is its
  cosine similarity to the mean vector of the books the user liked.
- **Per-book gate:** `score = a * CF + (1 - a) * content`, with `a = n / (n + k)`.
  `n` = number of training interactions of the book, `k` tuned on validation.
- **Positive interaction:** the book was read, or rated 4 or more.
- **Cold start (simulated):** most interactions of a random set of books are hidden from training.
  The data was collected in 2017, so there are no truly new books (we state this in the report).

## 3. Evaluation
- Accuracy: Recall@20, NDCG@20 (on all books, and on the cold subset).
- Exposure: catalogue coverage, novelty (mean self-information), genre variety per list.
- Baselines: Popularity, ALS only, Content only, fixed 50/50 hybrid, gated hybrid.
- Main figure: accuracy vs coverage trade-off curve for different values of k.

## 4. Repository layout
| Folder | Content |
|---|---|
| `config/` | `config.yaml`: every path and hyper-parameter |
| `data/` | `raw/` (downloaded), `interim/` (cleaned), `processed/` (splits, matrices) - not committed |
| `src/` | Python package: data, models, evaluation, visualization, utils |
| `scripts/` | Entry points: `run_pipeline.py`, `tune_k.py` |
| `notebooks/` | EDA and results analysis |
| `tests/` | pytest unit tests (each person tests the other's code) |
| `results/` | Figures, tables, run logs |
| `reports/` | Final report outline, slides |
| `docs/` | Team split, git workflow |

## 5. Quick start (PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m src.data.download
    python scripts/run_pipeline.py
    python scripts/tune_k.py
    pytest -q

## 6. Ownership summary
| Area | Lead | Co-owner |
|---|---|---|
| Config, download, loading, split | Amine | Ella |
| Cleaning, cold-start simulation, I/O | Ella | Amine |
| ALS model, hybrid gate | Amine | Ella |
| Popularity baseline, TF-IDF content model | Ella | Amine |
| Accuracy metrics, evaluation runner, pipeline | Amine | Ella |
| Exposure metrics, plots, recommend demo, k tuning | Ella | Amine |
| Report and slides | Both (see `reports/report_outline.md`) | |

## 7. Limits
- Goodreads data is from 2017, so cold start is simulated.
- We use one genre and sample users, so the results are for that subset only.
- The dataset is for academic use. Do not redistribute the raw files.