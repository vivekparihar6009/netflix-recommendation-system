import os
import pickle
import pandas as pd
from typing import Dict, Any, Tuple
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from src import config

def run_svd_grid_search(
    train_df: pd.DataFrame, 
    param_grid: Dict[str, list] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Runs Grid Search Cross-Validation to find the best hyperparameters for SVD.
    Returns:
        tuple: (best_params, best_rmse_score)
    """
    print("Running SVD Grid Search CV...")
    if param_grid is None:
        param_grid = config.SVD_GRID_SEARCH
        
    # Representative sample of 100k for speed
    grid_df = train_df
    if len(train_df) > 100000:
        grid_df = train_df.sample(100000, random_state=config.RANDOM_STATE)
        
    reader = Reader(rating_scale=(1.0, 5.0))
    # Surprise expects exactly [userID, itemID, rating]
    data = Dataset.load_from_df(grid_df[["user_id", "movie_id", "rating"]], reader)
    
    gs = GridSearchCV(
        SVD, 
        param_grid, 
        measures=["rmse", "mae"], 
        cv=3, 
        n_jobs=1, 
        joblib_verbose=2
    )
    gs.fit(data)
    
    best_params = gs.best_params["rmse"]
    best_score = gs.best_score["rmse"]
    
    print(f"Grid Search complete. Best RMSE: {best_score:.4f}")
    print("Best params:", best_params)
    return best_params, best_score


def train_svd(
    train_df: pd.DataFrame, 
    params: Dict[str, Any] = None
) -> SVD:
    """
    Trains the SVD model on the entire training set with the given parameters.
    """
    if params is None:
        params = config.SVD_BEST_PARAMS
        
    print(f"Training SVD model with parameters: {params}...")
    reader = Reader(rating_scale=(1.0, 5.0))
    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    
    # Trainset is built on the entire data loaded
    trainset = data.build_full_trainset()
    
    # Instantiate and fit
    # SVD parameters mapping:
    # n_factors, n_epochs, lr_all, reg_all
    model = SVD(
        n_factors=params.get("n_factors", 100),
        n_epochs=params.get("n_epochs", 20),
        lr_all=params.get("lr_all", 0.005),
        reg_all=params.get("reg_all", 0.02),
        random_state=config.RANDOM_STATE
    )
    model.fit(trainset)
    print("SVD model training complete.")
    return model


def save_svd_model(model: SVD, filepath: str) -> None:
    """
    Saves the SVD model using pickle.
    """
    print(f"Saving SVD model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print("SVD model saved successfully.")


def load_svd_model(filepath: str) -> SVD:
    """
    Loads the SVD model from the given path.
    """
    print(f"Loading SVD model from {filepath}...")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model
