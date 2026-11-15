# scripts/
Entry points; run them from the project root, **in this order**:

| Step | Script | Lead | Command | Produces |
|---|---|---|---|---|
| 1 | `run_pipeline.py` | Amine | `python scripts/run_pipeline.py --force [--users 5000] [--k 20] [--no-exposure]` | data/interim, data/processed, `dataset_stats.csv`, `test_metrics.csv`, `long_tail.png`, `baseline_bars.png`, `popularity_of_recs.png`, `results/runs/<date>.json` |
| 2 | `tune_k.py` | Ella | `python scripts/tune_k.py` | `k_sweep_val.csv`, `best_k.json`, `tradeoff_curve.png`, `gate_curve.png` |
| 3 | `run_pipeline.py` | Amine | `python scripts/run_pipeline.py` (no `--force`) | `test_metrics.csv` recomputed with the k of `best_k.json` |

- `--force` reruns every step (needed when the data or `--users` change).
- `--users N` overrides `data.n_users_sample` for a quick trial; final results use the default (50 000).
- `--k` forces a k; otherwise `best_k.json`, otherwise `gate.k` in `config.yaml`.
- `--no-exposure` skips coverage / novelty / tail share.
