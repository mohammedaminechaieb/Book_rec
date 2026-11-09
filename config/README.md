# config/
**Maintainer:** shared (Amine + Ella)

`config.yaml` holds every path and hyper-parameter (data sampling, split ratios, cold-start settings, ALS, TF-IDF, gate k grid, top-N).
It is read by `src/utils/config.py -> load_config()`. Change values here, never in the code.