import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from src.preprocessor import IndexMapper, build_interaction_matrix
from src import config

class ItemCollaborativeFiltering:
    """
    Item-based Collaborative Filtering Recommender.
    Computes item-item cosine similarity and predicts ratings using
    weighted averages of user's ratings on similar items.
    """
    def __init__(self, mapper: IndexMapper, k_neighbors: int = 40):
        self.mapper = mapper
        self.k_neighbors = k_neighbors
        self.interaction_matrix = None
        self.item_similarity = None
        self.item_means = None
        self.user_means = None
        self.global_mean = 0.0
        
    def fit(self, train_df: pd.DataFrame) -> 'ItemCollaborativeFiltering':
        """Fits the model by building the sparse matrix and computing item similarities."""
        print("Fitting Item-Based CF model...")
        # 1. Build interaction matrix (users x movies)
        self.interaction_matrix = build_interaction_matrix(train_df, self.mapper)
        
        # Operate directly on the sparse matrix to avoid densification
        # Calculate item means (over rated users) using getcol()
        self.item_means = np.zeros(self.mapper.num_movies)
        for m in range(self.mapper.num_movies):
            col = self.interaction_matrix.getcol(m)
            rated = col.data  # only non-zero values, no densification
            self.item_means[m] = np.mean(rated) if len(rated) > 0 else 3.0
                
        # Calculate user means (over rated items) using getrow()
        self.user_means = np.zeros(self.mapper.num_users)
        for u in range(self.mapper.num_users):
            row = self.interaction_matrix.getrow(u)
            rated = row.data  # only non-zero values, no densification
            self.user_means[u] = np.mean(rated) if len(rated) > 0 else 3.0
                
        all_ratings = train_df["rating"].values
        self.global_mean = float(np.mean(all_ratings)) if len(all_ratings) > 0 else 3.5
        
        # 2. Compute item-item similarity matrix
        # Columns in interaction_matrix represent items (movies)
        # Cosine similarity between columns (transpose the matrix to compute similarity between items)
        print("Computing item-item cosine similarity matrix...")
        self.item_similarity = cosine_similarity(self.interaction_matrix.T)
        
        # Set self-similarity to 0 to prevent predicting based on the same item
        np.fill_diagonal(self.item_similarity, 0.0)
        
        print("Item-Based CF fitting complete.")
        return self
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts rating for a given user_id and movie_id using the formula:
        P_u,i = sum(sim(i, j) * R_u,j) / sum(|sim(i, j)|)
        for movies j rated by user u.
        """
        # Cold start checks
        if user_id not in self.mapper.user_to_idx:
            # New user: return global mean
            return self.global_mean
        if movie_id not in self.mapper.movie_to_idx:
            # New movie: return user's mean rating if user is known
            u_idx = self.mapper.user_to_idx[user_id]
            return self.user_means[u_idx]
            
        u_idx = self.mapper.user_to_idx[user_id]
        m_idx = self.mapper.movie_to_idx[movie_id]
        
        # Get all movies rated by user u_idx
        user_ratings = self.interaction_matrix[u_idx].toarray().flatten()
        rated_movie_indices = np.where(user_ratings > 0)[0]
        
        if len(rated_movie_indices) == 0:
            return self.item_means[m_idx]
            
        # Similarities between target movie m_idx and movies rated by user
        similarities = self.item_similarity[m_idx, rated_movie_indices]
        
        # Sort and select top K neighbors
        if len(similarities) > self.k_neighbors:
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:]
            similarities = similarities[top_k_indices]
            rated_movie_indices = rated_movie_indices[top_k_indices]
            user_ratings = user_ratings[rated_movie_indices]
            
        # Calculate sum of absolute similarities
        sim_sum = np.sum(np.abs(similarities))
        if sim_sum == 0:
            # Fallback: return the item's mean rating or user mean rating
            return 0.5 * (self.item_means[m_idx] + self.user_means[u_idx])
            
        # Calculate weighted rating
        predicted_rating = np.sum(similarities * user_ratings) / sim_sum
        
        # Bound rating to [1.0, 5.0]
        return float(np.clip(predicted_rating, 1.0, 5.0))
        
    def get_top_k_recommendations(
        self, 
        user_id: int, 
        k: int = 10,
        train_ratings_df: pd.DataFrame = None
    ) -> List[Tuple[int, float]]:
        """
        Generates Top-K recommended movie IDs for a target user.
        Excludes movies that the user has already rated in the training set.
        """
        if user_id not in self.mapper.user_to_idx:
            # Cold start: handled by popularity fallback
            return []
            
        u_idx = self.mapper.user_to_idx[user_id]
        
        # Get movies already rated by the user in train
        if train_ratings_df is not None:
            watched_movies = set(train_ratings_df[train_ratings_df["user_id"] == user_id]["movie_id"])
        else:
            watched_indices = np.where(self.interaction_matrix[u_idx].toarray().flatten() > 0)[0]
            watched_movies = {self.mapper.idx_to_movie[m] for m in watched_indices}
            
        # Score all movies that the user has NOT seen
        scores = []
        for mid, m_idx in self.mapper.movie_to_idx.items():
            if mid in watched_movies:
                continue
            pred = self.predict_rating(user_id, mid)
            scores.append((mid, pred))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def save_item_cf_model(model: ItemCollaborativeFiltering, filepath: str) -> None:
    """Saves Item-CF model using pickle."""
    print(f"Saving Item-CF model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print("Item-CF model saved successfully.")


def load_item_cf_model(filepath: str) -> ItemCollaborativeFiltering:
    """Loads Item-CF model from pickle."""
    print(f"Loading Item-CF model from {filepath}...")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
