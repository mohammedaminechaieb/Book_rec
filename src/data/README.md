# src/data/
Pipeline order: `download -> load -> clean -> split -> cold_start`

| File | Lead | Input | Output |
|---|---|---|---|
| `download.py` | Amine | URLs in config | `data/raw/*.json.gz` |
| `load.py` | Amine | raw files | interactions, books, genres DataFrames |
| `clean.py` | Ella | DataFrames | `data/interim/*_clean.parquet` |
| `split.py` | Amine | clean interactions | `train/val/test.parquet` |
| `cold_start.py` | Ella | train | `train_cold.parquet`, `cold_books.json` |