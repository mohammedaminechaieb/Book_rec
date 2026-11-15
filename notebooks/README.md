# notebooks/
The real logic lives in `src/`; notebooks call it.

| Notebook | Lead | Purpose |
|---|---|---|
| `00_main.ipynb` | Both | **The notebook handed in**: assignment steps 1 to 6, reading the artifacts of the final run |
| `01_eda.ipynb` | Ella | dataset statistics, long tail, missing text (exploration) |
| `02_results_analysis.ipynb` | Amine | model comparison, cold items / cold users, k trade-off |

Run the pipeline first (see `scripts/README.md`), then run `00_main.ipynb` top to bottom before
handing it in (Kernel > Restart & Run All). Clear the outputs of 01 / 02 before committing.
