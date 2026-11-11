"""
==============================================================================
File      : src/data/split.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Per-user time split. Each user's interactions are sorted by date_updated:
  the last test_ratio go to TEST, the val_ratio before them go to VALIDATION,
  and the rest go to TRAIN. Items that never appear in train are dropped from
  val/test (except the simulated cold books, see cold_start.py).
  Rules per user with n interactions:
    n_test = max(1, floor(n * test_ratio))   if n >= 2, else 0
    n_val  = max(1, floor(n * val_ratio))    if n >= 3, else 0
    train  = the rest (always >= 1)
Input:
  - data/interim/interactions_clean.parquet
  - config.yaml -> split.val_ratio, split.test_ratio
Output:
  - data/processed/train.parquet, val.parquet, test.parquet  [user_id, book_id, date_updated]
Run:
  python -m src.data.split
==============================================================================
"""
import numpy as np
import pandas as pd

COLUMNS = ["user_id", "book_id", "date_updated"]

def _holdout_sizes(n: np.ndarray, val_ratio: float, test_ratio: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Number of test and validation rows for users with n interactions each."""
    n_test = np.where(n >= 2, np.maximum(1, np.floor(n * test_ratio)), 0).astype(int)
    n_val = np.where(n >= 3, np.maximum(1, np.floor(n * val_ratio)), 0).astype(int)
    # never leave a user without a train interaction
    n_val = np.minimum(n_val, n - n_test - 1).clip(min=0)
    return n_test, n_val

def per_user_time_split(df: pd.DataFrame, val_ratio: float, test_ratio: float
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test). Every user keeps at least 1 train interaction."""
    # 1. Sort each user's history in time; book_id breaks ties so the result is always the same.
    df = df[COLUMNS].sort_values(["user_id", "date_updated", "book_id"]).reset_index(drop=True)

    # 2. Position of each row counted from the END of the user's history (0 = most recent).
    n_per_row = df.groupby("user_id")["book_id"].transform("size").to_numpy()
    pos_from_end = n_per_row - 1 - df.groupby("user_id").cumcount().to_numpy()

    # 3. Hold-out sizes for each row's user.
    n_test, n_val = _holdout_sizes(n_per_row, val_ratio, test_ratio)

    is_test = pos_from_end < n_test
    is_val = (~is_test) & (pos_from_end < n_test + n_val)
    is_train = ~(is_test | is_val)

    train, val, test = df[is_train], df[is_val], df[is_test]

    # 4. Remove val/test books never seen in train: no model can learn them.
    known_books = set(train["book_id"])
    val = val[val["book_id"].isin(known_books)]
    test = test[test["book_id"].isin(known_books)]

    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))
    
def main() -> None:
    from src.utils.config import load_config
    from src.utils.io import load_df, save_df          # Ella's io.py

    cfg = load_config()
    interim, processed = cfg["paths"]["interim_dir"], cfg["paths"]["processed_dir"]

    df = load_df(interim / "interactions_clean.parquet")
    train, val, test = per_user_time_split(df, cfg["split"]["val_ratio"], cfg["split"]["test_ratio"])

    save_df(train, processed / "train.parquet")
    save_df(val, processed / "val.parquet")
    save_df(test, processed / "test.parquet")
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} rows, "
          f"{train['user_id'].nunique():,} users")


if __name__ == "__main__":
    main()