import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class Deduplicator:
    def __init__(self):
        self.required_columns = ['Title', 'Authors', 'Abstract', 'DOI']

    def validate_dataframe(self, df):
        """Validate if dataframe has required columns"""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        return True

    def process_dataframes(self, dataframes, threshold=0.8):
        """Process multiple dataframes and remove duplicates"""
        try:
            # Validate and combine dataframes
            for df in dataframes:
                self.validate_dataframe(df)
            
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Create Title_Author column for similarity comparison
            combined_df['Title_Author'] = combined_df['Title'].fillna('') + ' ' + combined_df['Authors'].fillna('')
            
            # Find duplicate clusters
            clusters_df, unique_df = self.find_duplicate_clusters(combined_df, threshold)
            
            # Ensure output format consistency
            unique_df = self.standardize_output(unique_df)
            clusters_df = self.standardize_clusters(clusters_df)
            
            return unique_df, clusters_df
            
        except Exception as e:
            logging.error(f"Error in deduplication process: {str(e)}")
            raise

    def find_duplicate_clusters(self, df, threshold):
        """Find duplicate clusters using TF-IDF and cosine similarity"""
        vectorizer = TfidfVectorizer().fit_transform(df['Title_Author'])
        cosine_sim = cosine_similarity(vectorizer)
        
        n = cosine_sim.shape[0]
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        # Build clusters
        for i in range(n):
            for j in range(i + 1, n):
                if cosine_sim[i, j] > threshold:
                    union(i, j)
        
        # Collect clusters
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        # Prepare output dataframes
        cluster_data = []
        unique_indices = []
        
        for cluster_id, indices in clusters.items():
            if len(indices) > 1:
                for index in indices:
                    cluster_data.append({
                        "Cluster_ID": cluster_id,
                        "Index": index,
                        "Title": df.iloc[index]["Title"],
                        "Authors": df.iloc[index]["Authors"],
                        "DOI": df.iloc[index]["DOI"],
                        "Abstract": df.iloc[index]["Abstract"]
                    })
                unique_indices.append(indices[0])  # Keep first occurrence
            else:
                unique_indices.extend(indices)
        
        clusters_df = pd.DataFrame(cluster_data) if cluster_data else pd.DataFrame(columns=["Cluster_ID", "Index", "Title", "Authors", "DOI", "Abstract"])
        unique_df = df.iloc[unique_indices].copy()
        
        # Reset index to ensure it starts from 0
        unique_df = unique_df.reset_index(drop=True)
        # Add Index column that matches NBIB/RIS format
        unique_df.index.name = 'Index'
        
        return clusters_df, unique_df

    def standardize_output(self, df):
        """Ensure output dataframe has consistent format"""
        # Make sure Index is properly set
        if 'Index' not in df.index.name:
            df = df.reset_index(drop=True)
            df.index.name = 'Index'
        
        # Ensure all required columns exist
        required_columns = ['Title', 'Authors', 'Abstract', 'DOI']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Select and order columns while preserving the index
        df = df[required_columns]
        return df

    def standardize_clusters(self, df):
        """Ensure clusters dataframe has consistent format"""
        required_columns = ['Cluster_ID', 'Index', 'Title', 'Authors', 'DOI', 'Abstract']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        return df[required_columns] 