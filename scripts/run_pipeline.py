"""
==============================================================================
File      : scripts/run_pipeline.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  End-to-end pipeline:
    load -> clean -> split -> cold-start simulation -> id maps / CSR
    -> fit Popularity, ALS, Content -> build 50/50 and gated hybrids
    -> evaluate all 5 models on TEST -> save the table and figures
  Each step skips its work if its output file already exists (--force reruns it).
Input:
  - config/config.yaml, data/raw/*
Output:
  - data/interim/*, data/processed/*
  - results/tables/test_metrics.csv
  - results/figures/baseline_bars.png, popularity_of_recs.png, long_tail.png
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