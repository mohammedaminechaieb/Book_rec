<!--
File    : docs/team_split.md
Owner   : shared (Amine + Ella)
What    : who leads / reviews each file, plus the two-week plan
Input   : -
Output  : reference for the team and the teacher
-->
# Team split - Amine and Ella

Rule: every file has a **lead** (writes most of it) and a **co-owner** (pair-programs, reviews the PR).
Each person writes the **tests for the other's code**. Split: 12 lead files each.

| File | Lead | Co-owner |
|---|---|---|
| src/utils/config.py | Amine | Ella |
| src/utils/io.py | Ella | Amine |
| src/data/download.py | Amine | Ella |
| src/data/load.py | Amine | Ella |
| src/data/clean.py | Ella | Amine |
| src/data/split.py | Amine | Ella |
| src/data/cold_start.py | Ella | Amine |
| src/models/popularity.py | Ella | Amine |
| src/models/als_model.py | Amine | Ella |
| src/models/content_tfidf.py | Ella | Amine |
| src/models/hybrid_gate.py | Amine | Ella |
| src/evaluation/accuracy_metrics.py | Amine | Ella |
| src/evaluation/exposure_metrics.py | Ella | Amine |
| src/evaluation/evaluate.py | Amine | Ella |
| src/visualization/plots.py | Ella | Amine |
| src/recommend.py | Ella | Amine |
| scripts/run_pipeline.py | Amine | Ella |
| scripts/tune_k.py | Ella | Amine |
| tests/test_accuracy_metrics.py | Ella | Amine |
| tests/test_split_and_gate.py | Ella | Amine |
| tests/test_exposure_metrics.py | Amine | Ella |
| tests/test_cold_start.py | Amine | Ella |
| notebooks/01_eda.ipynb | Ella | Amine |
| notebooks/02_results_analysis.ipynb | Amine | Ella |
| README.md, config.yaml, requirements.txt, report, slides | Both | Both |

## Two-week plan
| Days | Amine | Ella | Together |
|---|---|---|---|
| 1-2 | config, download, load | io, clean, EDA notebook | agree on the data schema |
| 3-4 | split, ALS | cold_start, popularity, TF-IDF | first full run on a small sample |
| 5-7 | gate, accuracy metrics, evaluate | exposure metrics, plots | cross-tests |
| 8-9 | run_pipeline, results notebook | tune_k, recommend demo | k sweep on full sample |
| 10-12 | report sections (see outline) | report sections | results + discussion |
| 13-14 | slides (half) | slides (half) | rehearsal |