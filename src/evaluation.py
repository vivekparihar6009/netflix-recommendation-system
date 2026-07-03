import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from surprise import SVD
from src import config

def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Computes the Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))

def compute_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Computes the Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(predictions - actuals)))

def get_recommendations_for_user(
    model: Any, 
    user_id: int, 
    train_data: pd.DataFrame, 
    all_movie_ids: List[int],
    k: int = 10
) -> List[int]:
    """
    Generates Top-K recommended movie IDs for a user, excluding movies
    already seen in the training data.
    Supports Surprise SVD, UserCF, and ItemCF models.
    """
    # Get movies seen in training
    user_train_movies = set(train_data[train_data["user_id"] == user_id]["movie_id"])
    
    # Check if the model is Surprise SVD
    if isinstance(model, SVD):
        # Predict ratings for all movies the user hasn't seen
        unseen_movies = [m for m in all_movie_ids if m not in user_train_movies]
        preds = []
        for m in unseen_movies:
            # SVD predict returns a Prediction object, .est holds the predicted rating
            pred = model.predict(uid=user_id, iid=m).est
            preds.append((m, pred))
        # Sort by predicted rating descending
        preds.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in preds[:k]]
        
    elif hasattr(model, "get_top_k_recommendations"):
        # Custom CF models (UserCF/ItemCF) or hybrid
        recs = model.get_top_k_recommendations(user_id=user_id, k=k, train_ratings_df=train_data)
        return [m for m, _ in recs]
        
    else:
        # Fallback to popularity if model is not recognized
        movie_popularity = train_data["movie_id"].value_counts()
        unseen_movies = [m for m in all_movie_ids if m not in user_train_movies]
        popular_unseen = [m for m in movie_popularity.index if m in unseen_movies]
        return popular_unseen[:k]


def evaluate_ranking_metrics(
    model: Any, 
    test_data: pd.DataFrame, 
    train_data: pd.DataFrame, 
    k: int = 10, 
    relevance_threshold: float = 3.5,
    max_users: int = 200,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Computes MAP@K, Precision@K, Recall@K, and NDCG@K for the model on the test data.
    
    For each user in the test set:
    1. Generate Top-K recommended movies (excluding train set movies).
    2. Get actual test ratings for these recommended movies.
    3. A movie is RELEVANT if actual rating >= relevance_threshold.
    4. Compute precision, recall, Average Precision (AP), and DCG.
    """
    all_movie_ids = list(train_data["movie_id"].unique())
    test_users = test_data["user_id"].unique()
    if len(test_users) > max_users:
        rng = np.random.default_rng(random_state)
        test_users = rng.choice(test_users, size=max_users, replace=False)
    
    precisions = []
    recalls = []
    aps = []
    ndcgs = []
    
    # Store test ratings in a dictionary for fast lookup
    # user_id -> movie_id -> rating
    test_ratings_dict = {}
    for user_id, group in test_data.groupby("user_id"):
        test_ratings_dict[user_id] = dict(zip(group["movie_id"], group["rating"]))
        
    for user_id in tqdm(test_users, desc="Evaluating ranking metrics"):
        # Get actual test ratings and relevance for this user
        user_test_ratings = test_ratings_dict.get(user_id, {})
        user_relevant_movies = {m for m, r in user_test_ratings.items() if r >= relevance_threshold}
        
        # If user has no relevant items in test set, skip or count as zero
        if len(user_relevant_movies) == 0:
            # Standard practice: we can skip users with no relevant items in test set
            # or record 0. Let's record 0.0 for precision/recall/map/ndcg or skip them
            # Let's skip to avoid skewing recall (recall denominator would be 0)
            continue
            
        # 1. Generate top-K recommendations
        recs = get_recommendations_for_user(model, user_id, train_data, all_movie_ids, k=k)
        
        # 2. Compute Precision@K and Recall@K
        hits = 0
        dcg = 0.0
        ap_sum = 0.0
        
        for p, rec_movie in enumerate(recs):
            rank = p + 1
            if rec_movie in user_relevant_movies:
                hits += 1
                ap_sum += hits / rank
                dcg += 1.0 / np.log2(rank + 1)
                
        # Precision@K
        precision = hits / k
        precisions.append(precision)
        
        # Recall@K
        recall = hits / len(user_relevant_movies)
        recalls.append(recall)
        
        # Average Precision (AP) for MAP@K
        ap = ap_sum / min(k, len(user_relevant_movies))
        aps.append(ap)
        
        # Ideal DCG (IDCG@K)
        idcg = 0.0
        for p in range(min(k, len(user_relevant_movies))):
            idcg += 1.0 / np.log2(p + 2)
            
        # NDCG@K
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
        
    return {
        f"map_at_{k}": float(np.mean(aps)) if aps else 0.0,
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0
    }
