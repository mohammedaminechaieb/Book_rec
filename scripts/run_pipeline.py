"""
==============================================================================
File      : scripts/run_pipeline.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  End-to-end pipeline:
    load -> stats (raw) -> clean -> stats (clean) -> split -> stats (train)
    -> cold-start simulation (new books + new users) -> id maps / CSR
    -> fit Popularity, ALS, TF-IDF content [+ embeddings if enabled]
    -> build the 50/50 and gated hybrids
    -> evaluate every model on TEST (all / cold items / cold users; P, R, NDCG, HR @K;
       coverage, novelty, genre variety, tail share)
    -> save tables, figures and the artifacts used by the demo app
  Each step skips its work if its output file already exists (--force reruns it).
Input:
  - config/config.yaml, data/raw/*
Output:
  - data/interim/*, data/processed/* (splits, cold lists, id maps, n_items.npy,
    als_factors.npz, tfidf_items.npz, [book_embeddings.npy])
  - results/tables/dataset_stats.csv, test_metrics.csv
  - results/figures/long_tail.png, baseline_bars.png, popularity_of_recs.png
  - results/runs/<timestamp>.json (config + metrics of the run)
Run:
  python scripts/run_pipeline.py [--force] [--k 20]
==============================================================================
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--k", type=float, default=None, help="override gate.k")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    # TODO (Amine): chain the steps listed in the header
    raise NotImplementedError


if __name__ == "__main__":
    main()