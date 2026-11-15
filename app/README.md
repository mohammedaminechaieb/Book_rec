# app/
**Lead:** Ella | **Co-owner:** Amine

`streamlit_app.py` is the demo interface for the defense (assignment step 6), in French.

    streamlit run app/streamlit_app.py

Run the pipeline first (see `scripts/README.md`): the app reads the saved models from `data/processed/`
and the results from `results/`. `.streamlit/config.toml` sets the dark theme and hides the
"Deploy" toolbar. Set the environment variable `BOOKREC_DEBUG=1` to see full error traces.

| Page | Shows |
|---|---|
| Vue d'ensemble | key figures, the two models and the gate α = n/(n+k) (log-scale curve), how to read the gauge |
| Lecteur existant | 3 named demo readers (grand public, pépites, éclectique) + others; 5 models; duel ALS vs hybride adaptatif with an honest verdict |
| Nouveau lecteur | search and add 3+ books; content-only recommendations (α = 0); other editions excluded |
| Résultats & analyses | relevance vs long-tail chart, French metric table (incl. cold start), how k was chosen, figures |

Every list has a gauge Best‑seller / Intermédiaire / Pépite (thresholds shown under it). Each card shows
the class, the number of readers, the share of the score from ALS and from content (hybrids only) and
one French sentence. The logic lives in `src/recommend.py`, `src/explain.py`, `src/labels.py`,
`src/editions.py`; the app only displays it.

Before the defense: run it once **offline** (the fonts come from Google Fonts; without internet the
system fonts are used, which is fine but looks different).
