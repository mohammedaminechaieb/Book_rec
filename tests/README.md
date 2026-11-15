# tests/
Run `pytest -q` from the project root. **Cross-testing:** each person tests the other's code.

| Test file | Written by | Tests code of |
|---|---|---|
| `test_accuracy_metrics.py` | Ella | Amine |
| `test_split_and_gate.py` | Ella | Amine |
| `test_explain.py` | Ella | Amine |
| `test_labels.py` | Ella | Amine (labels, French explanations, popularity classes) |
| `test_exposure_metrics.py` | Amine | Ella |
| `test_cold_start.py` | Amine | Ella |
| `test_stats.py` | Amine | Ella |
| `test_editions.py` | Amine | Ella (edition filtering) |
| `test_clean_ella.py`, `test_plots_ella.py`, `test_popularity_ella.py`, `test_tfidf_ella.py` | Ella | Ella (own code, extra checks) |

All tests pass on the current code (`50 passed`).
