import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from src import config
from src.models import svd_model, item_cf, user_cf
from src.preprocessor import IndexMapper

class Recommender:
    """
    Production-quality Recommender interface.
    Combines SVD and Item-CF models into a Hybrid Recommender,
    handles cold-start users (0 ratings or 1-5 ratings),
    and generates explainable recommendations.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.mapper: IndexMapper = None
        self.svd_model = None
        self.item_cf_model: item_cf.ItemCollaborativeFiltering = None
        self.user_cf_model: user_cf.UserCollaborativeFiltering = None
        self.ratings_df = None
        self.movies_df = None
        self.movie_id_to_title = {}
        self.movie_title_to_id = {}
        self.movie_id_to_year = {}
        
    def load_models_and_data(self) -> None:
        """Loads all serialized models and preprocessed data files."""
        print("Loading models and data for Recommender...")
        
        # Load datasets
        if not os.path.exists(config.PROCESSED_RATINGS_PARQUET):
            raise FileNotFoundError("Processed ratings parquet file not found. Run preprocessing notebook first.")
        self.ratings_df = pd.read_parquet(config.PROCESSED_RATINGS_PARQUET)
        self.movies_df = pd.read_parquet(config.PROCESSED_MOVIES_PARQUET)
        
        # Map movie ID metadata
        for _, row in self.movies_df.iterrows():
            mid = int(row["movie_id"])
            title = str(row["movie_title"])
            year = row["year"]
            self.movie_id_to_title[mid] = title
            self.movie_title_to_id[title.lower()] = mid
            self.movie_id_to_year[mid] = int(year) if pd.notna(year) else 0
            
        # Load models
        if os.path.exists(config.SVD_MODEL_PATH):
            self.svd_model = svd_model.load_svd_model(config.SVD_MODEL_PATH)
        else:
            print("SVD model pkl not found. SVD recommendations will be unavailable.")
            
        if os.path.exists(config.ITEM_CF_PATH):
            self.item_cf_model = item_cf.load_item_cf_model(config.ITEM_CF_PATH)
            self.mapper = self.item_cf_model.mapper # retrieve the shared IndexMapper
        else:
            print("Item-CF model pkl not found. Item-CF recommendations will be unavailable.")
            
        if os.path.exists(config.USER_CF_PATH):
            self.user_cf_model = user_cf.load_user_cf_model(config.USER_CF_PATH)
        else:
            print("User-CF model pkl not found. User-CF recommendations will be unavailable.")
            
    def get_global_popular_movies(self, k: int = 10) -> List[Tuple[str, float, int]]:
        """
        Cold Start Strategy for 0 ratings.
        Returns the top-K globally popular movies weighted by rating count and average rating.
        Formula: Bayesian Weighted Rating or simple count * mean.
        """
        # Count and Mean ratings
        movie_stats = self.ratings_df.groupby("movie_id").agg(
            count=("rating", "count"),
            mean=("rating", "mean")
        )
        # Global rating mean
        C = movie_stats["count"].mean()
        m = self.ratings_df["rating"].mean()
        
        # IMDB Formula: (v / (v+m_val)) * R + (m_val / (v+m_val)) * C_val
        movie_stats["weighted_rating"] = (
            (movie_stats["count"] / (movie_stats["count"] + C)) * movie_stats["mean"] +
            (C / (movie_stats["count"] + C)) * m
        )
        
        top_mids = movie_stats.sort_values(by="weighted_rating", ascending=False).head(k).index.tolist()
        
        recs = []
        for mid in top_mids:
            title = self.movie_id_to_title.get(mid, f"Movie {mid}")
            year = self.movie_id_to_year.get(mid, 0)
            score = movie_stats.loc[mid, "weighted_rating"]
            recs.append((title, float(score), year))
            
        return recs

    def get_cold_start_recommendations_from_ratings(
        self, 
        user_ratings: Dict[int, float], 
        k: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Cold Start Strategy for 1-5 ratings.
        Uses the item-item cosine similarity matrix to recommend movies
        similar to the few movies the user has rated.
        user_ratings: Dict of {movie_id: rating}
        """
        if not self.item_cf_model:
            return self.get_global_popular_movies(k)
            
        # Get item similarity matrix from Item-CF model
        item_sim = self.item_cf_model.item_similarity
        
        # Array to aggregate similarities
        agg_scores = np.zeros(self.mapper.num_movies)
        sim_sums = np.zeros(self.mapper.num_movies)
        
        rated_mids = list(user_ratings.keys())
        rated_idxs = []
        
        for mid, rating in user_ratings.items():
            if mid in self.mapper.movie_to_idx:
                m_idx = self.mapper.movie_to_idx[mid]
                rated_idxs.append(m_idx)
                
                # Weight similarity by rating deviation (rating - 3.0) to favor positive feedback
                weight = rating - 3.0
                similarities = item_sim[m_idx]
                
                agg_scores += similarities * weight
                sim_sums += np.abs(similarities)
                
        # Exclude already rated movies
        for m_idx in rated_idxs:
            agg_scores[m_idx] = -np.inf
            
        # Normalize scores
        non_zero_sim = sim_sums > 0
        agg_scores[non_zero_sim] /= sim_sums[non_zero_sim]
        
        # Sort and return top-K
        top_indices = np.argsort(agg_scores)[::-1][:k]
        
        recs = []
        for idx in top_indices:
            if agg_scores[idx] == -np.inf:
                continue
            mid = self.mapper.idx_to_movie[idx]
            title = self.movie_id_to_title.get(mid, f"Movie {mid}")
            year = self.movie_id_to_year.get(mid, 0)
            score = float(agg_scores[idx] + 3.0) # Shift back to 1-5 scale
            recs.append((title, score, year))
            
        return recs

    def predict_hybrid_rating(self, user_id: int, movie_id: int) -> float:
        """
        Combines SVD and Item-CF scores:
        Score = alpha * SVD_score + (1 - alpha) * ItemCF_score
        """
        svd_pred = 3.5
        item_pred = 3.5
        
        if self.svd_model:
            svd_pred = self.svd_model.predict(uid=user_id, iid=movie_id).est
            
        if self.item_cf_model:
            item_pred = self.item_cf_model.predict_rating(user_id, movie_id)
            
        # If one model fails/returns fallback, we lean on the other
        # If both are available, do the weighted average
        if self.svd_model and self.item_cf_model:
            hybrid_score = self.alpha * svd_pred + (1 - self.alpha) * item_pred
        elif self.svd_model:
            hybrid_score = svd_pred
        else:
            hybrid_score = item_pred
            
        return float(np.clip(hybrid_score, 1.0, 5.0))

    def get_top_k_recommendations(
        self, 
        user_id: int, 
        k: int = 10,
        model_type: str = "hybrid"
    ) -> List[Tuple[str, float, int]]:
        """
        Returns list of (movie_title, predicted_rating, year)
        Supports: 'svd', 'user_cf', 'item_cf', 'hybrid'
        Also handles cold-start users.
        """
        # Check if user exists in train
        is_known_user = self.mapper and user_id in self.mapper.user_to_idx
        
        if not is_known_user:
            # Cold Start: 0 ratings -> Recommend globally popular
            print(f"Cold Start: User {user_id} is unknown. Recommending popular items.")
            return self.get_global_popular_movies(k)
            
        # Get movies seen by user in train
        user_ratings_in_train = self.ratings_df[self.ratings_df["user_id"] == user_id]
        
        # Cold Start: 1-5 ratings -> Use item-based CF on these ratings
        if len(user_ratings_in_train) <= 5:
            print(f"Cold Start: User {user_id} has only {len(user_ratings_in_train)} ratings. Running Item-CF on ratings.")
            user_ratings_dict = dict(zip(user_ratings_in_train["movie_id"], user_ratings_in_train["rating"]))
            return self.get_cold_start_recommendations_from_ratings(user_ratings_dict, k)
            
        watched_movies = set(user_ratings_in_train["movie_id"])
        
        # Build index-to-movie and movie-to-index mappings for fast array construction
        all_movie_ids = list(self.movie_id_to_title.keys())
        n_movies = len(all_movie_ids)
        
        if model_type == "hybrid" or model_type == "svd":
            # --- Vectorized SVD scoring ---
            svd_scores = np.full(n_movies, 3.5)
            if self.svd_model:
                trainset = self.svd_model.trainset
                try:
                    u_inner = trainset.to_inner_uid(user_id)
                    pu = self.svd_model.pu[u_inner]          # user latent vector
                    qi = self.svd_model.qi                    # all item latent vectors
                    bu = self.svd_model.bu[u_inner]           # user bias
                    bi = self.svd_model.bi                    # all item biases
                    global_mean = trainset.global_mean
                    # Map each movie_id to its inner SVD item index
                    for i, mid in enumerate(all_movie_ids):
                        try:
                            i_inner = trainset.to_inner_iid(mid)
                            svd_scores[i] = global_mean + bu + bi[i_inner] + qi[i_inner].dot(pu)
                        except Exception:
                            svd_scores[i] = global_mean + bu
                except Exception:
                    # User not in SVD trainset; fall back to global mean
                    pass
        
        if model_type == "hybrid" or model_type == "item_cf":
            # --- Vectorized ItemCF scoring ---
            item_cf_scores = np.full(n_movies, 3.0)
            if self.item_cf_model:
                u_idx = self.mapper.user_to_idx[user_id]
                user_rating_vec = self.item_cf_model.interaction_matrix[u_idx].toarray().flatten()
                all_item_scores = self.item_cf_model.item_similarity.dot(user_rating_vec)
                sim_sums = np.abs(self.item_cf_model.item_similarity).dot(
                    (user_rating_vec > 0).astype(float)
                )
                all_item_scores = np.divide(
                    all_item_scores, sim_sums,
                    out=np.full_like(sim_sums, 3.0),
                    where=sim_sums > 0
                )
                # Map internal item indices back to the all_movie_ids order
                for i, mid in enumerate(all_movie_ids):
                    if mid in self.mapper.movie_to_idx:
                        m_idx = self.mapper.movie_to_idx[mid]
                        item_cf_scores[i] = float(np.clip(all_item_scores[m_idx], 1.0, 5.0))
        
        if model_type == "user_cf":
            # UserCF: compute scores per movie (no matrix shortcut available)
            user_cf_scores = np.full(n_movies, 3.0)
            if self.user_cf_model:
                for i, mid in enumerate(all_movie_ids):
                    user_cf_scores[i] = self.user_cf_model.predict_rating(user_id, mid)
        
        # Combine scores based on model_type
        alpha = self.alpha
        if model_type == "hybrid" and self.svd_model and self.item_cf_model:
            hybrid_scores = alpha * svd_scores + (1 - alpha) * item_cf_scores
        elif model_type == "hybrid" and self.svd_model:
            hybrid_scores = svd_scores
        elif model_type == "hybrid":
            hybrid_scores = item_cf_scores
        elif model_type == "svd":
            hybrid_scores = svd_scores
        elif model_type == "item_cf":
            hybrid_scores = item_cf_scores
        elif model_type == "user_cf":
            hybrid_scores = user_cf_scores
        else:
            hybrid_scores = np.full(n_movies, 3.0)
        
        # Exclude watched movies by setting their scores to -inf
        for i, mid in enumerate(all_movie_ids):
            if mid in watched_movies:
                hybrid_scores[i] = -np.inf
        
        # Return top-k from a single argsort
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        
        result = []
        for i in top_indices:
            if hybrid_scores[i] == -np.inf:
                break
            mid = all_movie_ids[i]
            title = self.movie_id_to_title.get(mid, f"Movie {mid}")
            year = self.movie_id_to_year.get(mid, 0)
            result.append((title, float(hybrid_scores[i]), year))
        
        return result

    def get_similar_movies(self, movie_title: str, k: int = 10) -> List[Tuple[str, float, int]]:
        """
        Returns list of (movie_title, similarity_score, year) similar to the target movie_title.
        """
        target_mid = self.movie_title_to_id.get(movie_title.lower())
        if not target_mid or not self.item_cf_model:
            print(f"Movie '{movie_title}' or Item-CF similarity matrix not found.")
            return []
            
        m_idx = self.mapper.movie_to_idx[target_mid]
        similarities = self.item_cf_model.item_similarity[m_idx]
        
        # Sort indices by similarity descending
        similar_indices = np.argsort(similarities)[::-1]
        
        recs = []
        count = 0
        for idx in similar_indices:
            mid = self.mapper.idx_to_movie[idx]
            sim = float(similarities[idx])
            
            # Skip if similarity is very low or 0
            if sim <= 0.0:
                continue
                
            title = self.movie_id_to_title.get(mid, f"Movie {mid}")
            year = self.movie_id_to_year.get(mid, 0)
            recs.append((title, sim, year))
            count += 1
            if count >= k:
                break
                
        return recs

    def explain_recommendation(self, user_id: int, movie_id: int) -> str:
        """
        Explains a recommendation: "Because you liked Movie X and Movie Y..."
        Finds the movies rated highly by the user that are most similar to the recommended movie.
        """
        title = self.movie_id_to_title.get(movie_id, f"Movie {movie_id}")
        
        # If user is not in mapping, we can't explain
        if not self.mapper or user_id not in self.mapper.user_to_idx:
            return f"We recommended '{title}' because it is one of our top-rated popular movies."
            
        # Get user's ratings in train
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        
        if len(user_ratings) == 0:
            return f"We recommended '{title}' because it is highly popular among all Netflix viewers."
            
        # Filter to user's highly rated movies (rating >= 4.0)
        highly_rated = user_ratings[user_ratings["rating"] >= 4.0]
        if len(highly_rated) == 0:
            highly_rated = user_ratings # fall back to any movie rated if none rated >= 4.0
            
        if movie_id not in self.mapper.movie_to_idx or not self.item_cf_model:
            return f"We recommended '{title}' based on your general viewing history."
            
        m_idx = self.mapper.movie_to_idx[movie_id]
        item_sim = self.item_cf_model.item_similarity
        
        similarities = []
        for _, row in highly_rated.iterrows():
            rated_mid = int(row["movie_id"])
            if rated_mid in self.mapper.movie_to_idx:
                rated_idx = self.mapper.movie_to_idx[rated_mid]
                sim = item_sim[m_idx, rated_idx]
                similarities.append((rated_mid, sim, row["rating"]))
                
        # Sort by similarity score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Pick top 2 matches
        valid_matches = [x for x in similarities if x[1] > 0.05][:2]
        
        if len(valid_matches) == 2:
            m1_title = self.movie_id_to_title.get(valid_matches[0][0])
            m2_title = self.movie_id_to_title.get(valid_matches[1][0])
            r1 = valid_matches[0][2]
            r2 = valid_matches[1][2]
            return (f"We recommended '{title}' because you gave highly positive ratings to "
                    f"'{m1_title}' (rated {r1}★) and '{m2_title}' (rated {r2}★), which are "
                    f"highly similar to this title.")
        elif len(valid_matches) == 1:
            m1_title = self.movie_id_to_title.get(valid_matches[0][0])
            r1 = valid_matches[0][2]
            return (f"We recommended '{title}' because you watched and liked "
                    f"'{m1_title}' (rated {r1}★), which has a strong similarity to this title.")
        else:
            # Fall back to user's favorite genre/movies
            top_movie_id = user_ratings.sort_values(by="rating", ascending=False).iloc[0]["movie_id"]
            top_movie_title = self.movie_id_to_title.get(top_movie_id)
            return (f"We recommended '{title}' based on its popularity and because your top rated "
                    f"movie was '{top_movie_title}'.")
