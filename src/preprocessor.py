import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple, Dict
from src import config

class IndexMapper:
    """
    Manages bidirectional mapping between original IDs (user_id, movie_id)
    and contiguous integer indices (0 to N-1).
    """
    def __init__(self):
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        
    def fit(self, ratings_df: pd.DataFrame) -> None:
        """Creates mappings for all unique users and movies in the DataFrame."""
        unique_users = sorted(ratings_df["user_id"].unique())
        unique_movies = sorted(ratings_df["movie_id"].unique())
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}
        
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie = {idx: mid for idx, mid in enumerate(unique_movies)}
        
    def map_users(self, user_series: pd.Series) -> pd.Series:
        """Maps original user_ids to mapped integer indices."""
        return user_series.map(self.user_to_idx)
        
    def map_movies(self, movie_series: pd.Series) -> pd.Series:
        """Maps original movie_ids to mapped integer indices."""
        return movie_series.map(self.movie_to_idx)
        
    @property
    def num_users(self) -> int:
        return len(self.user_to_idx)
        
    @property
    def num_movies(self) -> int:
        return len(self.movie_to_idx)


def split_data_stratified(
    ratings_df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs an 80/20 train-test split stratified by user.
    Guarantees that every user in the test set also appears in the train set.
    For each user:
      - If user has 1 rating, it goes to the train set.
      - If user has N > 1 ratings, round(N * test_size) ratings go to the test set,
        and the rest (at least 1) go to the train set.
    """
    print(f"Performing stratified train-test split (test_size={test_size})...")
    np.random.seed(random_state)
    
    train_indices = []
    test_indices = []
    
    # Group by user_id and split indices
    for _, group in ratings_df.groupby("user_id"):
        indices = group.index.values.copy()
        np.random.shuffle(indices)
        
        n_ratings = len(indices)
        if n_ratings <= 1:
            # If only 1 rating, it must be in the train set to avoid cold-start/unseen user in train
            train_indices.extend(indices)
        else:
            n_test = int(np.round(n_ratings * test_size))
            n_test = max(0, min(n_test, n_ratings - 1)) # ensure at least 1 in train
            
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])
            
    train_df = ratings_df.loc[train_indices].copy()
    test_df = ratings_df.loc[test_indices].copy()
    
    # Reset index for clean dataframes
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Assert check to make sure every user in test is also in train
    train_users = set(train_df["user_id"])
    test_users = set(test_df["user_id"])
    assert test_users.issubset(train_users), "Error: Found users in test set that are not in train set!"
    
    print(f"Split complete. Train set: {len(train_df):,} ratings. Test set: {len(test_df):,} ratings.")
    return train_df, test_df


def build_interaction_matrix(
    ratings_df: pd.DataFrame, 
    mapper: IndexMapper
) -> csr_matrix:
    """
    Builds a sparse user-item interaction matrix from a ratings DataFrame.
    The values in the matrix are the ratings.
    """
    # Map IDs to continuous indices
    user_indices = mapper.map_users(ratings_df["user_id"])
    movie_indices = mapper.map_movies(ratings_df["movie_id"])
    
    # Drop any ratings that didn't map (i.e. movies/users not in mapper)
    valid_mask = user_indices.notna() & movie_indices.notna()
    
    row = user_indices[valid_mask].astype(int).values
    col = movie_indices[valid_mask].astype(int).values
    data = ratings_df.loc[valid_mask, "rating"].values
    
    # Create the CSR matrix
    interaction_matrix = csr_matrix(
        (data, (row, col)), 
        shape=(mapper.num_users, mapper.num_movies),
        dtype=float
    )
    
    return interaction_matrix
