<!--
File    : reports/report_outline.md
Owner   : shared - section leads below
What    : skeleton of the final report, with a lead and a reviewer for each section
Input   : results/tables/*.csv, results/figures/*.png, notebooks
Output  : final report (PDF) + slides in reports/slides/
-->
# Reducing popularity bias in book recommendation on Goodreads

| # | Section | Lead | Reviewer |
|---|---|---|---|
| 1 | Introduction and problem (popularity bias, long tail) | Ella | Amine |
| 2 | Related work (CF, content, hybrids, accuracy-only evaluation) | Amine | Ella |
| 3 | Data (UCSD Book Graph, genre subset, filtering, simulated cold start) | Amine | Ella |
| 4.1 | Method: ALS collaborative branch | Amine | Ella |
| 4.2 | Method: TF-IDF content branch | Ella | Amine |
| 4.3 | Method: per-book gate alpha = n / (n + k) | Amine | Ella |
| 5 | Evaluation protocol (split, Recall/NDCG, coverage, novelty, variety) | Ella | Amine |
| 6 | Results: baselines table, cold subset, trade-off curve | Both | Both |
| 7 | Discussion and limitations (2017 data, one genre, simulated cold start) | Ella | Amine |
| 8 | Conclusion and future work (learned gate, embeddings) | Amine | Ella |

Slides: 50/50. Amine presents sections 2-4.1, 4.3 and 8; Ella presents 1, 4.2, 5 and 7. You present Results (6) together.