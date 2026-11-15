<!--
File    : reports/report_outline.md
Owner   : shared - section leads below
What    : skeleton of the final report (10 pages MAX, in French), with a lead and a reviewer for each section
Input   : results/tables/*.csv, results/figures/*.png, notebooks
Output  : final report (PDF) + slides in reports/slides/
-->
# Système de recommandation hybride : réduire le biais de popularité (Goodreads)

**Limit: 10 pages maximum** (excluding the appendix with the AI conversations). Page budget per section is given in brackets.

| # | Section | Assignment step | Pages | Lead | Reviewer |
|---|---|---|---|---|---|
| 1 | Introduction: problem, popularity bias, long tail | - | 0.75 | Ella | Amine |
| 2 | Related work (CF, content, hybrids, accuracy-only evaluation) | - | 0.5 | Amine | Ella |
| 3 | Domain and dataset (UCSD Book Graph, genre subset) | 1 | 0.5 | Amine | Ella |
| 4 | Data analysis and preparation: stats table, sparsity, popularity distribution, cleaning, split, cold-start simulation | 2 | 1.25 | Ella | Amine |
| 5.1 | Base approaches: popularity, TF-IDF content | 3 | 0.75 | Ella | Amine |
| 5.2 | Base approaches: ALS (embeddings = future work) | 3 | 0.75 | Amine | Ella |
| 6 | Hybrid: weighted hybrid with an adaptive weight per book a = n/(n+k); why; how k is chosen (largest k within 5 % of the best validation NDCG@20, stated trade-off); rank normalisation and why min-max made the gate favour best-sellers (AI error found by the audit) | 4 | 1 | Amine | Ella |
| 7 | Evaluation protocol: P/R/NDCG/HR@K, coverage, novelty, tail share (genre variety computed but not informative here); all / cold items / cold users; popularity classes (top 5 % / next 15 % / 80 %) | 5 | 0.75 | Ella | Amine |
| 8 | Results: comparison table, cold-start results, trade-off curve, k sweep. Say honestly where the adaptive hybrid wins (new books) and where the fixed 50/50 hybrid is better (overall diversity), and why: the gate trusts ALS for books with many readers, and ALS favours popular books | 5 | 1.5 | Both | Both |
| 9 | Demonstration: interface and explanations (screenshots) | 6 | 0.75 | Ella | Amine |
| 10 | Use of generative AI: generated vs hand-written/modified parts, AI errors found and fixed, our own technical choices | 7 | 1 | Both | Both |
| 11 | Limits and conclusion: simulated cold start (2017 data), one genre and a user sample, editions listed separately on Goodreads (classics look rare; we filter editions in recommendations), genre variety not informative | - | 0.5 | Amine | Ella |

Slides: 50/50. Amine presents sections 2, 3, 5.2, 6 and 11; Ella presents 1, 4, 5.1, 7 and 9. You present 8 and 10 together, then the live demo.