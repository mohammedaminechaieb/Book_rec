# app/
**Lead:** Ella | **Co-owner:** Amine

`streamlit_app.py` is the demo interface for the defense (assignment step 6).

    streamlit run app/streamlit_app.py

Run `python scripts/run_pipeline.py` first: the app reads the saved models from `data/processed/`.

| Mode | Shows |
|---|---|
| Existing reader | liked books, top-N for the chosen model, an explanation per book, ALS vs gate side by side |
| New reader (cold start) | pick a few books, get content-based top-N with explanations |

Each explanation gives: the closest liked book, the shared shelves, the CF vs content share (gate alpha) and the popularity label.
The logic lives in `src/recommend.py` and `src/explain.py`. The app only displays it.