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
      -> ALS (CF) + TF-IDF (content)
      -> per-book gate -> top-N + explanation -> accuracy + exposure metrics

- **Collaborative:** implicit-feedback ALS.
- **Content:** TF-IDF of the description + popular shelves; the user profile is the mean of the books they liked.
- **Future work:** frozen sentence embeddings (e.g. all-MiniLM-L6-v2) instead of TF-IDF.
- **Hybrid gate (weighted hybrid, adaptive weight per book):** `score = a * CF + (1 - a) * content`,
  `a = n / (n + k)`. Before mixing, each branch's scores are turned into **per-user percentile ranks**
  (`gate.normalization: rank`). With min-max scaling, TF-IDF scores stay near 0 while ALS scores do not,
  so the books the gate sends to content (rare books) were pushed down and the gate favoured best-sellers. k is chosen on validation with a stated trade-off: the **largest k whose
  NDCG@20 stays within 5 %** of the best (`gate.ndcg_tolerance`), because a larger k sends more
  weight to content, i.e. to long-tail and new books.
- **Explanations (in French):** closest liked book, shared informative tags (storage shelves such as
  `to-read` and near-universal ones such as `comics` are ignored), share of the score from ALS and
  from content, popularity class.
- **Popularity classes** (one vocabulary everywhere, `src/labels.py`): *best-seller* = top 5 % most-read
  books, *intermédiaire* = next 15 %, *pépite* = the other 80 % (the long tail, same definition as
  the `tail_share` metric).
- **Editions:** Goodreads lists each edition separately; other editions of a book already read or
  picked, collected editions ("The Complete Maus", "(Persepolis #1-4)") and duplicates inside a list
  are never recommended (`src/editions.py`, using the Goodreads `work_id` when present).
- **Demo readers:** not the simulated cold users, and at least 8 training books (`app.min_demo_books`):
  after the 5-per-user filter and the time split many users keep only 3 books in training.

## 3. Evaluation (assignment step 5)
- Accuracy: Precision@K, Recall@K, NDCG@K, HR@K for K = 5, 10, 20.
- Beyond accuracy: coverage, novelty, tail share (genre variety is computed but not used:
  almost every book's main genre is "comics" in this subset, so it does not separate the models).
- Compared: Popularity, ALS only, Content only, fixed 50/50, gated hybrid.
- Subsets: all users; **cold items** (simulated new books); **cold users** (simulated new readers).

## 4. How the project covers the assignment
| Assignment step | Where |
|---|---|
| 1. Domain and dataset | `src/data/download.py`, `src/data/load.py` |
| 2. Statistics, sparsity, popularity, cleaning, split | `src/data/stats.py`, `clean.py`, `split.py`, `notebooks/00_main.ipynb` |
| 3. At least two base approaches | `src/models/als_model.py`, `content_tfidf.py`, `popularity.py` |
| 4. Justified hybrid | `src/models/hybrid_gate.py`, `scripts/tune_k.py` |
| 5. Evaluation incl. beyond-accuracy and cold start | `src/evaluation/*`, `src/data/cold_start.py` |
| 6. Demo with explanations | `app/streamlit_app.py`, `src/recommend.py`, `src/explain.py`, `src/labels.py`, `src/editions.py` |
| Deliverable: notebook + commented code | `notebooks/00_main.ipynb`, `src/` |
| Deliverable: report (10 pages max) | `reports/Rapport_BookRec.pdf` (plan: `reports/report_outline.md`) |

## 5. Repository layout
| Folder | Content |
|---|---|
| `config/` | `config.yaml`: every path and hyper-parameter |
| `data/` | raw / interim / processed (not committed) |
| `src/` | Python package: data, models, evaluation, visualization, utils, recommend, explain, labels, editions |
| `app/`, `.streamlit/` | Streamlit demo and its theme / toolbar settings |
| `scripts/` | `run_pipeline.py`, `tune_k.py` |
| `notebooks/` | `00_main.ipynb` (the notebook handed in) |
| `tests/` | pytest (each person tests the other's code) |
| `results/` | figures, tables, runs |
| `reports/` | report outline, slides |
| `docs/` | team split, workflow |

## 6. Quick start (PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m src.data.download
    python scripts/run_pipeline.py --force      # 50 000 sampled users (data.n_users_sample)
    python scripts/tune_k.py                    # chooses k on VALIDATION -> best_k.json
    python scripts/run_pipeline.py              # again, WITHOUT --force: re-evaluates TEST with that k
    pytest -q
    streamlit run app/streamlit_app.py

The **second** `run_pipeline.py` is required: otherwise `test_metrics.csv` still holds the
results of the default k (the app warns on the Résultats page if the two k differ).
For a quick trial, add `--users 5000` to the first command (and `--force` whenever you change it).
The results quoted in the report must come from the full run.

## 7. Limits
- Goodreads data is from 2017, so new books and new users are simulated.
- One genre and a user sample, so the results hold for that subset only.
- Goodreads lists editions separately: classics such as *Persepolis* can look "rarely read" in the
  sample because their readers are split across editions.
- In our first runs the adaptive hybrid did not beat the fixed 50/50 hybrid on overall diversity;
  its advantage is on new books (cold start). The report states this honestly.
- Academic-use dataset. Do not redistribute the raw files.