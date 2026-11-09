# src/
**Maintainer:** shared. Python package; import it as `from src.<sub>.<module> import ...` from the project root.

| Subpackage | Role | Leads |
|---|---|---|
| `utils/` | config loading, file I/O, id maps | Amine, Ella |
| `data/` | download -> load -> clean -> split -> cold start | Amine, Ella |
| `models/` | popularity, ALS, TF-IDF content, hybrid gate | Amine, Ella |
| `evaluation/` | accuracy and exposure metrics, evaluation runner | Amine, Ella |
| `visualization/` | report figures | Ella |
| `recommend.py` | single-user demo | Ella |

Every model has the same interface: `fit(...)` and `score(user_idx) -> (n_batch x n_books)` array.