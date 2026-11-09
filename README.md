# book_rec - Reducing popularity bias in book recommendation (Goodreads)

| | |
|---|---|
| **Team** | Amine, Ella (equal split; see `docs/team_split.md`) |
| **Course** | Big Data Project - Hybrid recommender system |
| **Dataset** | UCSD Book Graph (Goodreads, Wan & McAuley), one genre subset |
| **Output** | Top-N book list per reader with explanations, evaluated on accuracy AND long-tail exposure |

## 1. Problem
Collaborative filtering mostly recommends bestsellers, because they have the most data.
Niche books stay invisible. This is **popularity bias**.
**Question:** how much more of the long tail can we recommend without losing accuracy?

## 2. Method

    Interactions + books -> stats + cleaning + split -> cold-start simulation
      -> ALS (CF) + TF-IDF (content) [+ MiniLM embeddings, optional]
      -> per-book gate -> top-N + explanation -> accuracy + exposure metrics

- **Collaborative:** implicit-feedback ALS.
- **Content:** TF-IDF of the description + popular shelves; the user profile is the mean of the books they liked.
- **Optional recent approach:** frozen sentence embeddings (all-MiniLM-L6-v2).
- **Hybrid gate:** `score = a * CF + (1 - a) * content`, `a = n / (n + k)`, with k tuned on validation.
- **Explanations:** closest liked book, shared shelves, CF vs content share, popularity label.

## 3. Evaluation (assignment step 5)
- Accuracy: Precision@K, Recall@K, NDCG@K, HR@K for K = 5, 10, 20.
- Beyond accuracy: coverage, novelty, genre variety, tail share.
- Compared: Popularity, ALS only, Content only, fixed 50/50, gated hybrid.
- Subsets: all users; **cold items** (simulated new books); **cold users** (simulated new readers).

## 4. How the project covers the assignment
| Assignment step | Where |
|---|---|
| 1. Domain and dataset | `src/data/download.py`, `src/data/load.py` |
| 2. Statistics, sparsity, popularity, cleaning, split | `src/data/stats.py`, `clean.py`, `split.py`, `notebooks/01_eda.ipynb` |
| 3. At least two base approaches | `src/models/als_model.py`, `content_tfidf.py`, `popularity.py`, optional `content_embeddings.py` |
| 4. Justified hybrid | `src/models/hybrid_gate.py`, `scripts/tune_k.py` |
| 5. Evaluation incl. beyond-accuracy and cold start | `src/evaluation/*`, `src/data/cold_start.py` |
| 6. Demo with explanations | `app/streamlit_app.py`, `src/recommend.py`, `src/explain.py` |
| Deliverable: notebook + commented code | `notebooks/00_main.ipynb`, `src/` |
| Deliverable: report (10 pages max) | `reports/report_outline.md` |

## 5. Repository layout
| Folder | Content |
|---|---|
| `config/` | `config.yaml`: every path and hyper-parameter |
| `data/` | raw / interim / processed (not committed) |
| `src/` | Python package: data, models, evaluation, visualization, utils, recommend, explain |
| `app/` | Streamlit demo |
| `scripts/` | `run_pipeline.py`, `tune_k.py` |
| `notebooks/` | `00_main` (handed in), `01_eda`, `02_results_analysis` |
| `tests/` | pytest (each person tests the other's code) |
| `results/` | figures, tables, runs |
| `reports/` | report outline, slides |
| `docs/` | team split, workflow |

## 6. Quick start (PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m src.data.download
    python scripts/run_pipeline.py
    python scripts/tune_k.py
    streamlit run app/streamlit_app.py
    pytest -q

## 7. Limits
- Goodreads data is from 2017, so new books and new users are simulated.
- One genre and a user sample, so the results hold for that subset only.
- Academic-use dataset. Do not redistribute the raw files.