# src/models/
All models have `fit(...)` and `score(user_idx)`.

| File | Lead | Role |
|---|---|---|
| `popularity.py` | Ella | baseline: most-read books |
| `als_model.py` | Amine | collaborative branch (implicit ALS) |
| `content_tfidf.py` | Ella | content branch (TF-IDF description + shelves) |
| `content_embeddings.py` | Amine | optional recent approach (MiniLM sentence embeddings) |
| `hybrid_gate.py` | Amine | `a = n/(n+k)` gate + fixed 50/50 baseline |