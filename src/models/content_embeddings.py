"""
==============================================================================
File      : src/models/content_embeddings.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  OPTIONAL "recent approach" (assignment step 3). Same role as the TF-IDF
  content branch, but each book is encoded with a pretrained sentence
  transformer (all-MiniLM-L6-v2, 384 dims, frozen, CPU is enough).
  score(u, i) = cosine(mean of liked book embeddings, embedding_i).
  It can be dropped into the gate in place of TF-IDF and compared in the
  results table. Enabled with config.yaml -> embeddings.enabled: true.
Input:
  - books_clean.parquet (text), train CSR
  - config.yaml -> embeddings.model_name, embeddings.batch_size
Output:
  - data/processed/book_embeddings.npy (n_books x 384, L2-normalised)
  - score matrix for a batch of users (n_batch x n_books)
==============================================================================
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


class ContentEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256):
        self.model_name, self.batch_size = model_name, batch_size
        self.item_vectors = None

    def fit(self, books: pd.DataFrame, train: sp.csr_matrix, cache_path=None) -> "ContentEmbeddingModel":
        """Encode the book texts (load from cache_path if it exists) and keep train for the user profiles."""
        raise NotImplementedError("TODO (Amine): from sentence_transformers import SentenceTransformer")

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO (Amine)")