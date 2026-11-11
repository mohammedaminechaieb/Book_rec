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
  Steps that call code not written yet (NotImplementedError) in stats / plots
  are skipped with a message instead of stopping the run.
Input:
  - config/config.yaml, data/raw/*
Output:
  - data/interim/interactions_raw.parquet, interactions_clean.parquet, books_clean.parquet, genres.parquet
  - data/processed/train|val|test.parquet, train_cold.parquet, cold_books.json, cold_users.json,
    id_maps.json, train_csr.npz, n_items.npy, books_aligned.parquet,
    als_factors.npz, tfidf_items.npz, [book_embeddings.npy]
  - results/tables/dataset_stats.csv, test_metrics.csv
  - results/figures/long_tail.png, baseline_bars.png, popularity_of_recs.png
  - results/runs/<timestamp>.json (settings + metrics of the run)
Run:
  python scripts/run_pipeline.py [--force] [--k 20] [--users 5000] [--no-exposure]
==============================================================================
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # must be set BEFORE numpy is imported

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import load_config, set_seed  # noqa: E402
from src.utils import io  # noqa: E402
from src.data import clean, cold_start, split, stats  # noqa: E402
from src.data import load as loader  # noqa: E402
from src.models.popularity import PopularityModel  # noqa: E402
from src.models.als_model import ALSModel  # noqa: E402
from src.models.content_tfidf import ContentTFIDFModel  # noqa: E402
from src.models.hybrid_gate import HybridGate, counts_to_array  # noqa: E402
from src.evaluation.evaluate import (evaluate_all, heldout_from_df,  # noqa: E402
                                    save_table, top_n_from_scores)

def need(path: Path, force: bool) -> bool:
    """True if this step must run: its output is missing, or --force was given."""
    if force or not Path(path).exists():
        return True
    print(f"    skip (exists): {Path(path).name}")
    return False


def optional(fn, label: str):
    """Run fn(); if the code behind it is not written yet, warn and go on."""
    try:
        return fn()
    except NotImplementedError as err:
        print(f"    skipped {label}: not implemented yet ({err})")
        return None


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path) -> None:
    """JSON writer that also accepts Paths and numpy numbers (default=str)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        
def step_data(cfg: dict, n_users: int, force: bool):
    """Load the raw sample, clean it, load books and genres for the kept books."""
    d, P = cfg["data"], cfg["paths"]
    raw_path = P["interim_dir"] / "interactions_raw.parquet"
    clean_path = P["interim_dir"] / "interactions_clean.parquet"
    books_path = P["interim_dir"] / "books_clean.parquet"
    genres_path = P["interim_dir"] / "genres.parquet"

    print("[1] load interactions")
    if need(raw_path, force):
        raw = loader.load_interactions(P["raw_dir"] / d["interactions_file"], n_users, cfg["seed"])
        io.save_df(raw, raw_path)
    raw = io.load_df(raw_path)

    print("[2] clean")
    if need(clean_path, force):
        df = clean.keep_positive(raw, d["positive_rating"])
        df = clean.k_core_filter(df, d["min_user_interactions"], d["min_book_interactions"])
        io.save_df(df, clean_path)
    df = io.load_df(clean_path)
    keep_ids = set(df["book_id"])

    if need(books_path, force):
        books = loader.load_books(P["raw_dir"] / d["books_file"],
                                  cfg["tfidf"]["top_shelves_per_book"], keep_ids=keep_ids)
        io.save_df(clean.clean_books(books, keep_ids), books_path)
    if need(genres_path, force):
        io.save_df(loader.load_genres(P["raw_dir"] / d["genres_file"], keep_ids=keep_ids), genres_path)
    return raw, df, io.load_df(books_path), io.load_df(genres_path)


def step_split(cfg: dict, df: pd.DataFrame, force: bool):
    """Time split, then simulate new books and new users on the train part."""
    P, c = cfg["paths"]["processed_dir"], cfg["cold_start"]

    print("[3] split")
    if need(P / "train.parquet", force):
        train, val, test = split.per_user_time_split(df, cfg["split"]["val_ratio"],
                                                    cfg["split"]["test_ratio"])
        val, test = split.drop_unknown_books(train, val), split.drop_unknown_books(train, test)
        for name, part in (("train", train), ("val", val), ("test", test)):
            io.save_df(part, P / f"{name}.parquet")
    train, val, test = (io.load_df(P / f"{n}.parquet") for n in ("train", "val", "test"))

    print("[4] cold start")
    if need(P / "train_cold.parquet", force):
        train_cold, cold_books = cold_start.make_cold_books(
            train, c["cold_book_fraction"], c["max_train_interactions"], cfg["seed"])
        train_cold, cold_users = cold_start.make_cold_users(
            train_cold, c["cold_user_fraction"], c["max_user_train_interactions"], cfg["seed"])
        io.save_df(train_cold, P / "train_cold.parquet")
        write_json({"book_ids": [str(b) for b in cold_books]}, P / "cold_books.json")
        write_json({"user_ids": [str(u) for u in cold_users]}, P / "cold_users.json")
    train_cold = io.load_df(P / "train_cold.parquet")
    cold_books = set(read_json(P / "cold_books.json")["book_ids"])
    cold_users = set(read_json(P / "cold_users.json")["user_ids"])
    return train, val, test, train_cold, cold_books, cold_users

def step_matrices(cfg: dict, train, train_cold, books, genres) -> dict:
    """Id maps, the train matrix, n per book, and tables aligned with the matrix columns."""
    P = cfg["paths"]["processed_dir"]
    print("[5] matrices")
    # Id maps from the train set BEFORE the cold-start simulation: every known book keeps a
    # column, even if the simulation removed all its interactions (it then has n = 0).
    user2idx, book2idx = io.build_id_maps(train)
    user2idx = {str(k): int(v) for k, v in user2idx.items()}
    book2idx = {str(k): int(v) for k, v in book2idx.items()}
    X = io.to_csr(train_cold, user2idx, book2idx)
    n_items = counts_to_array(cold_start.book_counts(train_cold), book2idx)

    # books / genres in matrix-column order (row i = book i), as explain.py expects
    idx2book = sorted(book2idx, key=book2idx.get)
    books_aligned = books.assign(book_id=books["book_id"].astype(str)).set_index("book_id").reindex(idx2book)
    books_aligned["title"] = books_aligned["title"].fillna("")
    books_aligned["text"] = books_aligned["text"].fillna("")
    books_aligned["shelves"] = [list(s) if isinstance(s, (list, np.ndarray)) else []
                                for s in books_aligned["shelves"]]
    books_aligned = books_aligned.reset_index()
    genre = (genres.assign(book_id=genres["book_id"].astype(str)).set_index("book_id")["genre"]
             .reindex(idx2book).fillna("unknown"))
    book_genre, _ = pd.factorize(genre)

    io.save_sparse(X, P / "train_csr.npz")
    write_json({"user2idx": user2idx, "book2idx": book2idx}, P / "id_maps.json")
    np.save(P / "n_items.npy", n_items)
    io.save_df(books_aligned, P / "books_aligned.parquet")
    print(f"    {X.shape[0]:,} users x {X.shape[1]:,} books, {X.nnz:,} train interactions")
    return {"X": X, "user2idx": user2idx, "book2idx": book2idx, "n_items": n_items,
            "books": books_aligned, "book_genre": book_genre}


def choose_k(cfg: dict, cli_k: float | None) -> float:
    """--k on the command line > best k from tune_k.py > gate.k in config.yaml."""
    best_path = cfg["paths"]["tables_dir"] / "best_k.json"
    if cli_k is not None:
        return cli_k
    if best_path.exists():
        return float(read_json(best_path)["k"])
    return float(cfg["gate"]["k"])


def step_models(cfg: dict, m: dict, k: float) -> dict:
    """Fit the base models and build the two hybrids (all share score(user_idx))."""
    P, t = cfg["paths"]["processed_dir"], cfg["tfidf"]
    print("[6] models")
    X, n_items = m["X"], m["n_items"]

    pop = PopularityModel().fit(X)
    als = ALSModel.from_config(cfg).fit(X)
    als.save(P / "als_factors.npz")
    tfidf = ContentTFIDFModel(max_features=t["max_features"], min_df=t["min_df"],
                              ngram_range=t["ngram_range"], use_description=t["use_description"],
                              use_shelves=t["use_shelves"]).fit(m["books"], X)
    io.save_sparse(tfidf.item_vectors, P / "tfidf_items.npz")

    models = {
        "Popularity": pop,
        "ALS": als,
        "Content (TF-IDF)": tfidf,
        "Hybrid 50/50": HybridGate(als, tfidf, n_items, k=None,
                                  fixed_alpha=cfg["gate"]["fixed_alpha"]),
        f"Gated hybrid (k={k:g})": HybridGate(als, tfidf, n_items, k=k),
    }
    if cfg["embeddings"]["enabled"]:                     # optional "recent approach"
        from src.models.content_embeddings import ContentEmbeddingModel
        e = cfg["embeddings"]
        emb = ContentEmbeddingModel(e["model_name"], e["batch_size"]).fit(
            m["books"], X, cache_path=P / "book_embeddings.npy")
        models["Content (embeddings)"] = emb
        models[f"Gated hybrid + embeddings (k={k:g})"] = HybridGate(als, emb, n_items, k=k)
    return models
  
def step_evaluate(cfg: dict, m: dict, models: dict, test, cold_books, cold_users,
                  with_exposure: bool) -> pd.DataFrame:
    """Evaluate every model on TEST and write test_metrics.csv."""
    e, T = cfg["eval"], cfg["paths"]["tables_dir"]
    print("[7] evaluate on test")
    b2i, u2i = m["book2idx"], m["user2idx"]
    rows = evaluate_all(
        models, train=m["X"], heldout=heldout_from_df(test, u2i, b2i), n_items=m["n_items"],
        book_genre=m["book_genre"],
        cold_item_idx={b2i[b] for b in cold_books if b in b2i},
        cold_user_idx={u2i[u] for u in cold_users if u in u2i},
        ks=e["ks"], top_n=e["top_n"], batch_size=e["batch_size"], with_exposure=with_exposure)
    table = save_table(rows, T / "test_metrics.csv")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table)
    return table


def step_figures(cfg: dict, m: dict, models: dict, table: pd.DataFrame, test) -> None:
    """Report figures (Ella's plots.py); skipped if not written yet."""
    from src.visualization import plots
    F, e = cfg["paths"]["figures_dir"], cfg["eval"]
    print("[8] figures")
    optional(lambda: plots.long_tail_distribution(m["n_items"], F / "long_tail.png"), "long_tail")
    optional(lambda: plots.baseline_bars(table, F / "baseline_bars.png"), "baseline_bars")

    def recs_sample():
        users = np.array(sorted(heldout_from_df(test, m["user2idx"], m["book2idx"])))
        sample = np.random.default_rng(cfg["seed"]).choice(users, min(2000, len(users)), replace=False)
        recs = {name: top_n_from_scores(model.score(sample), m["X"][sample], e["top_n"])
                for name, model in models.items()}
        plots.popularity_of_recs(recs, m["n_items"], F / "popularity_of_recs.png")
    optional(recs_sample, "popularity_of_recs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun every step")
    parser.add_argument("--k", type=float, default=None, help="override gate.k")
    parser.add_argument("--users", type=int, default=None,
                        help="override data.n_users_sample (use with --force), e.g. 5000 for a quick run")
    parser.add_argument("--no-exposure", action="store_true",
                        help="skip coverage / novelty / ... (if exposure_metrics.py is not ready)")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    n_users = args.users or cfg["data"]["n_users_sample"]
    start = time.time()

    raw, df, books, genres = step_data(cfg, n_users, args.force)
    train, val, test, train_cold, cold_books, cold_users = step_split(cfg, df, args.force)
    optional(lambda: stats.stats_table({"raw": raw, "clean": df, "train": train_cold},
                                      cfg["paths"]["tables_dir"] / "dataset_stats.csv"), "stats")
    m = step_matrices(cfg, train, train_cold, books, genres)
    k = choose_k(cfg, args.k)
    models = step_models(cfg, m, k)
    table = step_evaluate(cfg, m, models, test, cold_books, cold_users, not args.no_exposure)
    step_figures(cfg, m, models, table, test)

    run = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "k": k, "n_users_sample": n_users,
          "matrix": {"users": m["X"].shape[0], "books": m["X"].shape[1], "train_nnz": int(m["X"].nnz)},
          "settings": {key: cfg[key] for key in ("data", "split", "cold_start", "als", "tfidf", "gate", "eval")},
          "metrics": table.reset_index().to_dict(orient="records")}
    write_json(run, cfg["paths"]["runs_dir"] / f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
