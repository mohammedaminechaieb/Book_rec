<!--
File    : docs/team_split.md
Owner   : shared (Amine + Ella)
What    : who leads / reviews each file, plus the plan
Input   : -
Output  : reference for the team and the teacher
-->
# Team split - Amine and Ella

Rule: every file has a **lead** (writes most of it) and a **co-owner** (pair-programs, reviews the PR).
Each person writes the **tests for the other's code**. Split: 15 lead files each.

| File | Lead | Co-owner |
|---|---|---|
| src/utils/config.py | Amine | Ella |
| src/utils/io.py | Ella | Amine |
| src/data/download.py | Amine | Ella |
| src/data/load.py | Amine | Ella |
| src/data/stats.py | Ella | Amine |
| src/data/clean.py | Ella | Amine |
| src/data/split.py | Amine | Ella |
| src/data/cold_start.py (cold books + cold users) | Ella | Amine |
| src/models/popularity.py | Ella | Amine |
| src/models/als_model.py | Amine | Ella |
| src/models/content_tfidf.py | Ella | Amine |
| src/models/content_embeddings.py (optional) | Amine | Ella |
| src/models/hybrid_gate.py | Amine | Ella |
| src/evaluation/accuracy_metrics.py | Amine | Ella |
| src/evaluation/exposure_metrics.py | Ella | Amine |
| src/evaluation/evaluate.py | Amine | Ella |
| src/visualization/plots.py | Ella | Amine |
| src/recommend.py | Ella | Amine |
| src/explain.py | Amine | Ella |
| app/streamlit_app.py | Ella | Amine |
| scripts/run_pipeline.py | Amine | Ella |
| scripts/tune_k.py | Ella | Amine |
| tests/test_accuracy_metrics.py | Ella | Amine |
| tests/test_split_and_gate.py | Ella | Amine |
| tests/test_explain.py | Ella | Amine |
| tests/test_exposure_metrics.py | Amine | Ella |
| tests/test_cold_start.py | Amine | Ella |
| tests/test_stats.py | Amine | Ella |
| notebooks/01_eda.ipynb | Ella | Amine |
| notebooks/02_results_analysis.ipynb | Amine | Ella |
| notebooks/00_main.ipynb, README, config, report, slides | Both | Both |

## Plan
| Phase | Amine | Ella | Together |
|---|---|---|---|
| 1 | config, download, load | io, stats, clean, EDA notebook | agree on the data schema |
| 2 | split, ALS | cold books + cold users, popularity, TF-IDF | first run on a small sample |
| 3 | gate, accuracy metrics, evaluate | exposure metrics, plots | cross-tests |
| 4 | explain, run_pipeline, (embeddings) | recommend, Streamlit app, tune_k | k sweep, results notebook |
| 5 | report sections (see outline) | report sections | 00_main notebook, results + discussion |
| 6 | slides (half) | slides (half) | rehearsal: each explains the other's code |