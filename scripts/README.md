# scripts/
Entry points; run them from the project root.

| Script | Lead | Command | Produces |
|---|---|---|---|
| `run_pipeline.py` | Amine | `python scripts/run_pipeline.py [--force] [--k 20]` | processed data, `test_metrics.csv`, figures |
| `tune_k.py` | Ella | `python scripts/tune_k.py` | `k_sweep_val.csv`, `best_k.json`, `tradeoff_curve.png` |