
import numpy as np
import pandas as pd

from src.visualization.plots import (
    long_tail_distribution,
    tradeoff_curve,
    gate_curve,
    popularity_of_recs,
    baseline_bars,
)


def test_long_tail(tmp_path):
    counts = np.array([100, 50, 20, 5, 1, 0])

    out = tmp_path / "long_tail.png"
    long_tail_distribution(counts, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_tradeoff(tmp_path):
    data = pd.DataFrame({
        "k": [5, 10, 20, 50],
        "recall_at_20": [0.31, 0.30, 0.28, 0.25],
        "coverage": [0.15, 0.22, 0.35, 0.45],
    })

    out = tmp_path / "tradeoff.png"
    tradeoff_curve(data, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_gate(tmp_path):
    out = tmp_path / "gate.png"

    gate_curve(k=20, out=out, n_max=500)

    assert out.exists()
    assert out.stat().st_size > 0


def test_popularity_of_recs(tmp_path):
    counts = np.array([1000, 500, 100, 20, 5, 1])

    recs = {
        "Popularity": np.array([[0, 1], [0, 2]]),
        "TF-IDF": np.array([[3, 4], [4, 5]]),
        "Hybrid": np.array([[1, 4], [2, 5]]),
    }

    out = tmp_path / "popularity.png"

    popularity_of_recs(recs, counts, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_baseline_bars(tmp_path):
    metrics = pd.DataFrame({
        "model": ["Popularity", "ALS", "TF-IDF", "Hybrid"],
        "recall_at_20": [0.15, 0.30, 0.25, 0.29],
        "ndcg_at_20": [0.12, 0.27, 0.21, 0.26],
        "coverage": [0.05, 0.15, 0.50, 0.40],
        "novelty": [2.5, 3.0, 5.5, 4.8],
    })

    out = tmp_path / "baselines.png"

    baseline_bars(metrics, out)

    assert out.exists()
    assert out.stat().st_size > 0
