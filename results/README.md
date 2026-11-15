# results/
**Maintainer:** Ella (figures) | Amine (tables, runs)

| Folder | Content |
|---|---|
| `figures/` | `long_tail.png`, `baseline_bars.png`, `popularity_of_recs.png` (pipeline), `tradeoff_curve.png`, `gate_curve.png` (tune_k) — French titles, used in the report and the app |
| `tables/` | `dataset_stats.csv`, `test_metrics.csv` (one row per model), `k_sweep_val.csv`, `best_k.json`, `demo_<user>.csv` |
| `runs/` | one JSON per pipeline run (settings + metrics), not committed |

`best_k.json` = `{k, selection_metric, selection_rule, ndcg_tolerance, best_ndcg, ndcg, recall, cold_recall, coverage, tail_share}`.

After the **final** full run (see `scripts/README.md`), commit `tables/` and `figures/`: the report
and the demo must show the same numbers.
