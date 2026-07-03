import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from src.preprocessor import IndexMapper, build_interaction_matrix
from src import config

class UserCollaborativeFiltering:
    """
    User-based Collaborative Filtering Recommender.
    Computes user-user cosine similarity and predicts ratings using
    mean-centered weighted averages of similar users' ratings.
    """
    def __init__(self, mapper: IndexMapper, k_neighbors: int = 40):
        self.mapper = mapper
        self.k_neighbors = k_neighbors
        self.interaction_matrix = None
        self.user_similarity = None
        self.user_means = None
        self.global_mean = 0.0
        
    def fit(self, train_df: pd.DataFrame) -> 'UserCollaborativeFiltering':
        """Fits the model by building the sparse matrix and computing user similarities."""
        print("Fitting User-Based CF model...")
        # 1. Build interaction matrix (users x movies)
        self.interaction_matrix = build_interaction_matrix(train_df, self.mapper)
        # Convert to CSC for fast column slicing
        self.interaction_matrix_csc = self.interaction_matrix.tocsc()
        
        # 2. Compute global mean and user average ratings
        # Operate directly on the sparse matrix to avoid densification
        self.user_means = np.zeros(self.mapper.num_users)
        for u in range(self.mapper.num_users):
            row = self.interaction_matrix.getrow(u)
            rated = row.data  # only non-zero values, no densification
            self.user_means[u] = np.mean(rated) if len(rated) > 0 else 3.0
                
        all_ratings = train_df["rating"].values
        self.global_mean = float(np.mean(all_ratings)) if len(all_ratings) > 0 else 3.5
        
        # 3. Compute user-user similarity matrix
        # Cosine similarity is computed directly on the user-item interaction matrix
        # Note: Cosine similarity treats unrated items as 0
        print("Computing user-user cosine similarity matrix...")
        self.user_similarity = cosine_similarity(self.interaction_matrix)
        
        # Set self-similarity to 0 to prevent recommending based on oneself
        np.fill_diagonal(self.user_similarity, 0.0)
        
        print("User-Based CF fitting complete.")
        return self
        
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts rating for a given user_id and movie_id using the formula:
        P_u,i = mean_u + sum(sim(u, v) * (R_v,i - mean_v)) / sum(|sim(u, v)|)
        """
        # Cold start checks
        if user_id not in self.mapper.user_to_idx:
            return self.global_mean
        if movie_id not in self.mapper.movie_to_idx:
            # Fall back to user mean if user is known
            u_idx = self.mapper.user_to_idx[user_id]
            return self.user_means[u_idx]
            
        u_idx = self.mapper.user_to_idx[user_id]
        m_idx = self.mapper.movie_to_idx[movie_id]
        
        # Find all users who rated movie_id
        # In sparse matrix, get the column corresponding to movie_id (using fast CSC)
        movie_ratings = self.interaction_matrix_csc[:, m_idx].toarray().flatten()
        users_who_rated = np.where(movie_ratings > 0)[0]
        
        if len(users_who_rated) == 0:
            return self.user_means[u_idx]
            
        # Similarities between target user and users who rated the movie
        similarities = self.user_similarity[u_idx, users_who_rated]
        
        # Filter to top K neighbors
        if len(similarities) > self.k_neighbors:
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:]
            similarities = similarities[top_k_indices]
            users_who_rated = users_who_rated[top_k_indices]
            movie_ratings = movie_ratings[users_who_rated]
            
        # Calculate sum of absolute similarities
        sim_sum = np.sum(np.abs(similarities))
        if sim_sum == 0:
            return self.user_means[u_idx]
            
        # Get rating deviations: R_v,i - mean_v
        neighbor_means = self.user_means[users_who_rated]
        deviations = movie_ratings - neighbor_means
        
        # Calculate predicted rating
        predicted_rating = self.user_means[u_idx] + np.sum(similarities * deviations) / sim_sum
        
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
            # Cold start: user not in training. Handled by global popularity in recommender.py.
            return []
            
        u_idx = self.mapper.user_to_idx[user_id]
        
        # Get movies already rated by the user in train
        if train_ratings_df is not None:
            watched_movies = set(train_ratings_df[train_ratings_df["user_id"] == user_id]["movie_id"])
        else:
            # Fall back to checking the interaction matrix
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


def save_user_cf_model(model: UserCollaborativeFiltering, filepath: str) -> None:
    """Saves User-CF model using pickle."""
    print(f"Saving User-CF model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print("User-CF model saved successfully.")


def load_user_cf_model(filepath: str) -> UserCollaborativeFiltering:
    """Loads User-CF model from pickle."""
    print(f"Loading User-CF model from {filepath}...")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
