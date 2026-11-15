# src/
**Maintainer:** shared. Python package; import it as `from src.<sub>.<module> import ...` from the project root.

| Part | Role | Leads |
|---|---|---|
| `utils/` | config loading, file I/O, id maps | Amine, Ella |
| `data/` | download -> load -> stats -> clean -> split -> cold start | Amine, Ella |
| `models/` | popularity, ALS, TF-IDF, hybrid gate (`content_embeddings.py`: future work, not implemented) | Amine, Ella |
| `evaluation/` | accuracy and exposure metrics, evaluation runner | Amine, Ella |
| `visualization/` | report figures (French) | Ella |
| `recommend.py` | recommendation logic shared by the app and the CLI | Ella |
| `explain.py` | explanation (in French) of each recommended book | Amine |
| `labels.py` | one vocabulary: popularity classes and thresholds, French model / metric names, number format | Amine |
| `editions.py` | recognises other editions of the same work, so they are not recommended again | Ella |

Every model has the same interface: `fit(...)` and `score(user_idx) -> (n_batch x n_books)` array.
