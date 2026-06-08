import os

# Base paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
REPORT_FIGS_DIR = os.path.join(PROJECT_DIR, "report_figures")
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, "notebooks")

# Create directories if they do not exist
for d in [DATA_DIR, RESULTS_DIR, REPORT_FIGS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(d, exist_ok=True)

# Dataset file paths
RAW_MOVIE_TITLES = os.path.join(DATA_DIR, "movie_titles.csv")
RAW_DATA_FILES = [
    os.path.join(DATA_DIR, "combined_data_1.txt"),
    os.path.join(DATA_DIR, "combined_data_2.txt"),
    os.path.join(DATA_DIR, "combined_data_3.txt"),
    os.path.join(DATA_DIR, "combined_data_4.txt"),
]

# Processed parquet file paths
PROCESSED_RATINGS_PARQUET = os.path.join(DATA_DIR, "processed_ratings.parquet")
PROCESSED_MOVIES_PARQUET = os.path.join(DATA_DIR, "processed_movies.parquet")

# Subsetting parameters
TOP_USERS_COUNT = 5000
TOP_MOVIES_COUNT = 2000

# Evaluation parameters
TEST_SIZE = 0.20
RELEVANCE_THRESHOLD = 3.5

# Model hyperparameters
SVD_GRID_SEARCH = {
    "n_factors": [50, 100, 150],
    "n_epochs": [15, 20, 25],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1]
}

# Best default parameters (from description)
SVD_BEST_PARAMS = {
    "n_factors": 100,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02
}

# Model pickle paths
SVD_MODEL_PATH = os.path.join(RESULTS_DIR, "svd_model.pkl")
USER_CF_PATH = os.path.join(RESULTS_DIR, "user_cf.pkl")
ITEM_CF_PATH = os.path.join(RESULTS_DIR, "item_cf.pkl")

# Evaluation metrics results file
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")
SAMPLE_RECOMMENDATIONS_CSV = os.path.join(RESULTS_DIR, "sample_recommendations.csv")

# Reproducibility
RANDOM_STATE = 42
