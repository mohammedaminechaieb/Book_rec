# src/
**Maintainer:** shared. Python package; import it as `from src.<sub>.<module> import ...` from the project root.

| Part | Role | Leads |
|---|---|---|
| `utils/` | config loading, file I/O, id maps | Amine, Ella |
| `data/` | download -> load -> stats -> clean -> split -> cold start | Amine, Ella |
| `models/` | popularity, ALS, TF-IDF, optional embeddings, hybrid gate | Amine, Ella |
| `evaluation/` | accuracy and exposure metrics, evaluation runner | Amine, Ella |
| `visualization/` | report figures | Ella |
| `recommend.py` | recommendation logic shared by the app and the CLI | Ella |
| `explain.py` | explanation of each recommended book | Amine |

Every model has the same interface: `fit(...)` and `score(user_idx) -> (n_batch x n_books)` array.