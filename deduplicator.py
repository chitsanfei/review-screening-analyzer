"""
Deduplicator - Removes duplicate citations using TF-IDF and cosine similarity.
Optimized with numpy vectorization for better performance on large datasets.
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
REQUIRED_COLUMNS = ('Title', 'Authors', 'Abstract', 'DOI')
CLUSTER_COLUMNS = ('Cluster_ID', 'Index', 'Title', 'Authors', 'DOI', 'Abstract')


class UnionFind:
    """Optimized Union-Find data structure with path compression and union by rank."""

    __slots__ = ('parent', 'rank')

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if union was performed."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True

    def get_clusters(self) -> Dict[int, List[int]]:
        """Get all clusters as a dictionary."""
        clusters: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


class Deduplicator:
    """Removes duplicate entries using TF-IDF similarity matching."""

    __slots__ = ()

    def process_dataframes(
        self,
        dataframes: List[pd.DataFrame],
        threshold: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process multiple DataFrames and remove duplicates.

        Args:
            dataframes: List of DataFrames to process
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            Tuple of (unique entries DataFrame, duplicate clusters DataFrame)
        """
        # Validate input
        for df in dataframes:
            self._validate_dataframe(df)

        # Combine all DataFrames
        combined = pd.concat(dataframes, ignore_index=True)

        # Create combined text for similarity comparison
        combined['_text'] = (
            combined['Title'].fillna('') + ' ' +
            combined['Authors'].fillna('')
        )

        # Find duplicates
        unique_df, clusters_df = self._find_duplicates(combined, threshold)

        # Clean up and standardize output
        unique_df = self._standardize_output(unique_df)
        clusters_df = self._standardize_clusters(clusters_df)

        return unique_df, clusters_df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def _find_duplicates(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find duplicate clusters using TF-IDF and cosine similarity."""
        n = len(df)

        if n == 0:
            return df.copy(), pd.DataFrame(columns=list(CLUSTER_COLUMNS))

        if n == 1:
            return df.copy(), pd.DataFrame(columns=list(CLUSTER_COLUMNS))

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            max_features=10000  # Limit features for performance
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(df['_text'])
        except ValueError:
            # Empty vocabulary - no valid text to compare
            return df.copy(), pd.DataFrame(columns=list(CLUSTER_COLUMNS))

        # Compute similarity matrix efficiently
        # For large datasets, process in batches
        uf = UnionFind(n)

        if n > 1000:
            # Batch processing for large datasets
            self._batch_similarity(tfidf_matrix, threshold, uf)
        else:
            # Direct computation for smaller datasets
            similarity = cosine_similarity(tfidf_matrix)
            # Use numpy to find pairs above threshold
            rows, cols = np.where(similarity > threshold)
            for i, j in zip(rows, cols):
                if i < j:  # Only process upper triangle
                    uf.union(i, j)

        # Build output DataFrames
        clusters = uf.get_clusters()
        return self._build_output(df, clusters)

    def _batch_similarity(
        self,
        tfidf_matrix,
        threshold: float,
        uf: UnionFind,
        batch_size: int = 500
    ) -> None:
        """Compute similarity in batches for memory efficiency."""
        n = tfidf_matrix.shape[0]

        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch = tfidf_matrix[i:end_i]

            # Compare batch with all remaining rows
            for j in range(i, n, batch_size):
                end_j = min(j + batch_size, n)
                other = tfidf_matrix[j:end_j]

                sim = cosine_similarity(batch, other)
                rows, cols = np.where(sim > threshold)

                for r, c in zip(rows, cols):
                    abs_r, abs_c = i + r, j + c
                    if abs_r < abs_c:
                        uf.union(abs_r, abs_c)

    def _build_output(
        self,
        df: pd.DataFrame,
        clusters: Dict[int, List[int]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build output DataFrames from clusters."""
        cluster_data = []
        unique_indices = []

        for cluster_id, indices in clusters.items():
            if len(indices) > 1:
                # Multiple items in cluster - these are duplicates
                for idx in indices:
                    row = df.iloc[idx]
                    cluster_data.append({
                        'Cluster_ID': cluster_id,
                        'Index': idx,
                        'Title': row.get('Title', ''),
                        'Authors': row.get('Authors', ''),
                        'DOI': row.get('DOI', ''),
                        'Abstract': row.get('Abstract', '')
                    })
                # Keep first occurrence as unique
                unique_indices.append(indices[0])
            else:
                # Single item - unique entry
                unique_indices.extend(indices)

        # Build DataFrames
        clusters_df = pd.DataFrame(cluster_data) if cluster_data else pd.DataFrame(
            columns=list(CLUSTER_COLUMNS)
        )

        unique_df = df.iloc[unique_indices].copy()
        unique_df = unique_df.reset_index(drop=True)
        unique_df.index.name = 'Index'

        # Remove temporary column
        if '_text' in unique_df.columns:
            unique_df = unique_df.drop(columns=['_text'])

        return unique_df, clusters_df

    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize output DataFrame format."""
        if df.index.name != 'Index':
            df = df.reset_index(drop=True)
            df.index.name = 'Index'

        # Ensure required columns exist
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        return df[list(REQUIRED_COLUMNS)]

    def _standardize_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize clusters DataFrame format."""
        for col in CLUSTER_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df[list(CLUSTER_COLUMNS)]
