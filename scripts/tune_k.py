"""
==============================================================================
File      : scripts/tune_k.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Sweeps the gate parameter k over gate.k_grid on the VALIDATION set. ALS and
  content scores are computed once and reused for every k (only the gate
  changes). It picks the best k by NDCG@20 on validation and draws the
  accuracy vs coverage trade-off curve (the main figure of the report).
Input:
  - data/processed/* (train_cold, val, id maps), trained ALS + content models
  - config.yaml -> gate.k_grid
Output:
  - results/tables/k_sweep_val.csv  (k, recall, ndcg, cold_recall, coverage, novelty, variety)
  - results/tables/best_k.json
  - results/figures/tradeoff_curve.png, gate_curve.png
Run:
  python scripts/tune_k.py
==============================================================================
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    # TODO (Ella): loop over cfg["gate"]["k_grid"], evaluate, save the table, plot
    raise NotImplementedError


if __name__ == "__main__":
    main()