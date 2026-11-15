"""
==============================================================================
File      : src/labels.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  One shared vocabulary for everything shown to a user (app, figures, report):
    - popularity classes of a book, with their thresholds
        best-seller  : top 5 % most-read books
        intermédiaire: next 15 % (so best-seller + intermédiaire = the "head",
                       the top 20 % used by the tail_share metric)
        pépite       : the remaining 80 % = the long tail (same definition as tail_share)
    - French display names of the models and the metrics
    - French number formatting (decimal comma, non-breaking space before %)
  Keeping these in one place avoids the same idea having five names.
Input:
  - n_items (training interactions per book), model / metric keys, numbers
Output:
  - class keys "head" / "mid" / "tail", French labels, formatted strings
==============================================================================
"""
import re

import numpy as np

NBSP = " "            # non-breaking space (before %, :, ;)

# ---------------------------------------------------------------- popularity
HEAD_QUANTILE = 0.95       # best-sellers: top 5 %
MID_QUANTILE = 0.80        # head = top 20 % (same as exposure_metrics.tail_share)

POP_CLASSES = ("head", "mid", "tail")
POP_LABEL = {"head": "Best‑seller", "mid": "Intermédiaire", "tail": "Pépite"}
POP_LABEL_PLURAL = {"head": "Best‑sellers", "mid": "Intermédiaires", "tail": "Pépites"}


def popularity_thresholds(n_items: np.ndarray) -> tuple[float, float]:
    """(mid_min, head_min): number of readers needed to be 'intermédiaire' / 'best-seller'."""
    n_items = np.asarray(n_items, dtype=float)
    return float(np.quantile(n_items, MID_QUANTILE)), float(np.quantile(n_items, HEAD_QUANTILE))


def popularity_class(n: float, n_items: np.ndarray | None = None,
                     thresholds: tuple[float, float] | None = None) -> str:
    """'head', 'mid' or 'tail' for a book read n times in train."""
    if thresholds is None:
        thresholds = popularity_thresholds(n_items)
    mid_min, head_min = thresholds
    if n >= head_min:
        return "head"
    if n >= mid_min:
        return "mid"
    return "tail"


# ---------------------------------------------------------------- models / metrics
MODEL_LABEL = {
    "gate": "Hybride adaptatif",
    "fixed": "Hybride 50/50",
    "als": "ALS (collaboratif)",
    "content": "TF‑IDF (contenu)",
    "embeddings": "Embeddings (contenu)",
    "popularity": "Popularité",
}

METRIC_LABEL = {
    "all/recall@20": "Rappel@20",
    "all/ndcg@20": "NDCG@20",
    "all/precision@20": "Précision@20",
    "all/hr@20": "Taux de succès@20",
    "cold_items/recall@20": "Rappel@20 · livres nouveaux",
    "cold_users/recall@20": "Rappel@20 · lecteurs nouveaux",
    "coverage": "Couverture du catalogue",
    "novelty": "Nouveauté",
    "tail_share": "Part de pépites",
    "genre_variety": "Variété des genres",
}

METRIC_HELP = {
    "all/recall@20": "Part des livres réellement lus ensuite retrouvés dans le top 20 (plus haut = mieux).",
    "all/ndcg@20": "Comme le rappel, mais un bon livre compte plus s'il est en haut de la liste (plus haut = mieux).",
    "cold_items/recall@20": "Rappel limité aux livres nouveaux (cold start simulé).",
    "cold_users/recall@20": "Rappel pour les lecteurs nouveaux (3 livres en entraînement).",
    "coverage": "Part du catalogue recommandée au moins une fois (plus haut = plus de diversité).",
    "novelty": "Rareté moyenne des livres recommandés, en bits (plus haut = moins connus).",
    "tail_share": "Part des recommandations hors des 20 % de livres les plus lus.",
    "genre_variety": "Nombre de genres distincts par liste ; peu informatif ici (presque tout est « comics »).",
}

PERCENT_METRICS = {"coverage", "tail_share"}


def model_label(name: str) -> str:
    """French display name for a model key or a pipeline row name such as 'Gated hybrid (k=20)'."""
    raw = str(name)
    key = raw.lower()
    k = re.search(r"k\s*=\s*([0-9.]+)", raw)
    suffix = f" (k{NBSP}={NBSP}{fr_number(float(k.group(1)), 0 if float(k.group(1)).is_integer() else 1)})" if k else ""
    if "gate" in key or "adaptatif" in key:
        return MODEL_LABEL["gate"] + suffix
    if "50/50" in key or "fixed" in key:
        return MODEL_LABEL["fixed"]
    if "embedding" in key:
        return MODEL_LABEL["embeddings"]
    if "tf-idf" in key or "content" in key or "contenu" in key:
        return MODEL_LABEL["content"]
    if "als" in key:
        return MODEL_LABEL["als"]
    if "popular" in key:
        return MODEL_LABEL["popularity"]
    return raw


def model_k(name: str) -> float | None:
    """k written in a row name such as 'Hybride adaptatif (k=20)', else None."""
    m = re.search(r"k\s*=\s*([0-9.]+)", str(name))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- numbers
def fr_number(value, decimals: int = 2) -> str:
    """1234.5 -> '1 234,50' (French decimal comma and thin grouping)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    text = f"{value:,.{decimals}f}"
    return text.replace(",", NBSP).replace(".", ",")


def fr_percent(value, decimals: int = 0) -> str:
    """0.816 -> '81,6 %' style (value is a share between 0 and 1)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{fr_number(value * 100, decimals)}{NBSP}%"


def fr_metric(metric: str, value) -> str:
    """Format a metric value the French way (percent for shares, 3 decimals otherwise)."""
    if metric in PERCENT_METRICS:
        return fr_percent(value, 1)
    if metric == "novelty":
        return fr_number(value, 2)
    return fr_number(value, 3)
