
"""
==============================================================================
File      : src/models/content_tfidf.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Content-based recommendation with TF-IDF.
  - Builds book documents from descriptions and shelf tags.
  - Learns L2-normalized TF-IDF item representations.
  - Builds user profiles from positively interacted books.
  - Computes cosine similarity between users and books.

Input:
  - books DataFrame aligned with book2idx
  - binary user-item CSR training matrix
  - TF-IDF parameters provided through config.yaml

Output:
  - sparse TF-IDF item matrix
  - user-book score matrix
==============================================================================
"""

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.explain import is_generic_shelf


class ContentTFIDFModel:

    def __init__(
        self,
        max_features: int = 50000,
        min_df: int = 3,
        ngram_range=(1, 2),
        use_description: bool = True,
        use_shelves: bool = True
    ):
        self.params = dict(
            max_features=max_features,
            min_df=min_df,
            ngram_range=tuple(ngram_range),
            use_description=use_description,
            use_shelves=use_shelves
        )

        self.item_vectors = None
        self.vectorizer = None
        self.train = None

    def build_text(self, books: pd.DataFrame) -> list[str]:
        """
        Build one text document per book.

        Shelf tags such as 'historical fiction' become
        'historical_fiction' so they are treated as one token.
        Storage / status shelves ('to-read', 'owned', 'read-2016'...)
        say nothing about the content and are left out.
        """

        documents = []

        for _, book in books.iterrows():

            parts = []

            # Add description if enabled.
            if self.params["use_description"]:
                description = book.get("text", "")

                if isinstance(description, str):
                    parts.append(description.lower())

            # Add shelves if enabled.
            if self.params["use_shelves"]:
                shelves = book.get("shelves", [])

                if isinstance(shelves, (list, tuple, np.ndarray)):
                    tags = []

                    for shelf in shelves:
                        if not isinstance(shelf, str) or is_generic_shelf(shelf):
                            continue

                        tag = shelf.strip().lower()
                        tag = re.sub(r"[\s-]+", "_", tag)

                        if tag:
                            tags.append(tag)

                    parts.extend(tags)

            documents.append(" ".join(parts).strip())

        return documents

    def fit(
        self,
        books: pd.DataFrame,
        train: sp.csr_matrix
    ) -> "ContentTFIDFModel":
        """
        Fit TF-IDF on book metadata and store user interactions.

        books row i must correspond to train column i.
        """

        if not sp.issparse(train):
            raise TypeError("train must be a sparse matrix")

        train = train.tocsr(copy=True)

        if len(books) != train.shape[1]:
            raise ValueError(
                "Number of book rows must match train columns."
            )

        documents = self.build_text(books)

        # Replace empty documents with a placeholder.
        # It avoids an entirely empty document, although
        # an all-placeholder corpus is not informative.
        documents = [
            doc if doc.strip() else "unknownbook"
            for doc in documents
        ]

        self.vectorizer = TfidfVectorizer(
            max_features=self.params["max_features"],
            min_df=self.params["min_df"],
            ngram_range=self.params["ngram_range"],
            norm="l2",
            dtype=np.float32
        )

        # Sparse matrix: n_books x vocabulary_size.
        self.item_vectors = self.vectorizer.fit_transform(
            documents
        ).tocsr()

        # Make interactions binary even if duplicates
        # were accidentally summed in the input.
        train.sum_duplicates()
        train.eliminate_zeros()
        train.data[:] = 1.0

        self.train = train

        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """
        Compute user-book cosine similarities.

        Returns:
            dense array (n_selected_users, n_books)
        """

        if self.item_vectors is None or self.train is None:
            raise RuntimeError(
                "The TF-IDF model must be fitted first."
            )

        user_idx = np.atleast_1d(
            np.asarray(user_idx, dtype=int)
        )

        if np.any(user_idx < 0) or np.any(
            user_idx >= self.train.shape[0]
        ):
            raise ValueError("Invalid user index.")

        # Select the requested users.
        interactions = self.train[user_idx]

        # Compute user profiles from liked books.
        # Matrix shape: n_users_batch x vocabulary_size.
        user_profiles = interactions @ self.item_vectors

        # Mean and sum have the same direction before
        # L2 normalization for a given user.
        user_profiles = normalize(
            user_profiles,
            norm="l2",
            axis=1
        )

        # Item vectors are already L2-normalized.
        # Dot product therefore equals cosine similarity.
        similarities = user_profiles @ self.item_vectors.T

        return similarities.toarray().astype(
            np.float32,
            copy=False
        )
