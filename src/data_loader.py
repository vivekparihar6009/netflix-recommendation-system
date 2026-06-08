import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from src import config

def generate_synthetic_data(data_dir: str) -> None:
    """
    Generates synthetic Netflix Prize dataset files in the raw text format.
    Mimics the exact format of the Netflix dataset:
    - combined_data_1.txt: MovieID: followed by UserID,Rating,Date
    - movie_titles.csv: MovieID,Year,Title
    """
    print("Raw Netflix dataset files not found. Generating realistic synthetic dataset...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Configuration for synthetic data
    num_users = 6000  # slightly more than 5000 so we can filter to top 5000
    num_movies = 2500  # slightly more than 2000 so we can filter to top 2000
    
    # 1. Create movie_titles.csv
    movie_titles_path = os.path.join(data_dir, "movie_titles.csv")
    print(f"Generating {movie_titles_path}...")
    
    years = np.random.randint(1970, 2006, size=num_movies)
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Romance", "Horror", "Documentary"]
    adjectives = ["The Dark", "Golden", "Midnight", "Lost", "Silent", "Wild", "Secret", "Last", "Hidden"]
    nouns = ["Knight", "Journey", "Ocean", "Empire", "Shadow", "Valley", "Star", "Agent", "Destiny", "Legacy"]
    
    movies_data = []
    for i in range(1, num_movies + 1):
        adj = np.random.choice(adjectives)
        noun = np.random.choice(nouns)
        genre = np.random.choice(genres)
        title = f"{adj} {noun} ({genre})"
        movies_data.append(f"{i},{years[i-1]},{title}")
        
    with open(movie_titles_path, "w", encoding="ISO-8859-1") as f:
        for line in movies_data:
            f.write(line + "\n")
            
    # 2. Create combined_data_1.txt
    combined_data_path = os.path.join(data_dir, "combined_data_1.txt")
    print(f"Generating {combined_data_path} (Netflix-style raw txt)...")
    
    # Generate user IDs
    user_ids = np.random.choice(np.arange(10000, 99999), size=num_users, replace=False)
    
    # Generate rating dates
    start_date = pd.to_datetime("1999-12-01")
    end_date = pd.to_datetime("2005-12-31")
    date_range = (end_date - start_date).days
    
    # Distribution of ratings (skewed towards 3, 4, 5 stars, similar to Netflix)
    rating_choices = [1, 2, 3, 4, 5]
    rating_probs = [0.05, 0.10, 0.30, 0.35, 0.20]
    
    # To simulate realistic collaborative filtering, we create some latent user preferences
    # We assign each user and movie a category index in [0, 4]
    user_pref = np.random.randint(0, 5, size=num_users)
    movie_genre = np.random.randint(0, 5, size=num_movies)
    
    # Write ratings in the Netflix txt format
    # MovieID: on a line, followed by UserID,Rating,YYYY-MM-DD
    total_ratings_written = 0
    with open(combined_data_path, "w") as f:
        for movie_id in tqdm(range(1, num_movies + 1), desc="Writing synthetic ratings"):
            f.write(f"{movie_id}:\n")
            
            # Each movie is rated by a subset of users
            # Popular movies get more ratings, unpopular get fewer
            movie_popularity = np.random.zipf(1.5)
            movie_popularity = min(movie_popularity, 5) # limit popularity scaling
            
            # Decide rating probability based on popularity
            p_rate = 0.15 + 0.05 * movie_popularity
            num_ratings = int(num_users * p_rate)
            
            # Select random users to rate this movie
            sampled_indices = np.random.choice(num_users, size=num_ratings, replace=False)
            
            lines = []
            for u_idx in sampled_indices:
                user_id = user_ids[u_idx]
                
                # Check match between user preference and movie category to bias ratings
                if user_pref[u_idx] == movie_genre[movie_id - 1]:
                    # User likes this movie genre
                    rating = np.random.choice([3, 4, 5], p=[0.15, 0.45, 0.40])
                else:
                    rating = np.random.choice(rating_choices, p=rating_probs)
                
                # Generate random date
                rand_days = np.random.randint(0, date_range)
                r_date = (start_date + pd.to_timedelta(rand_days, unit="D")).strftime("%Y-%m-%d")
                
                lines.append(f"{user_id},{rating},{r_date}")
                total_ratings_written += 1
                
            f.write("\n".join(lines) + "\n")
            
    print(f"Generated synthetic dataset with {total_ratings_written:,} ratings across {num_movies} movies and {num_users} users.")
    
    # Create empty placeholder files for combined_data_2, 3, 4 to satisfy raw paths
    for empty_file in config.RAW_DATA_FILES[1:]:
        with open(empty_file, "w") as f:
            f.write("")

def parse_movie_titles(filepath: str) -> pd.DataFrame:
    """
    Parses the movie_titles.csv file.
    Since movie titles can contain commas, pandas read_csv with default separator can fail.
    We parse each line manually.
    """
    print(f"Parsing movie titles from {filepath}...")
    movie_ids = []
    years = []
    titles = []
    
    with open(filepath, "r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by the first two commas: MovieID, Year, Title
            parts = line.split(",", 2)
            if len(parts) >= 3:
                movie_ids.append(int(parts[0]))
                # Handle missing year case
                year_str = parts[1].strip()
                if year_str == "NULL" or not year_str:
                    years.append(np.nan)
                else:
                    years.append(float(year_str))
                titles.append(parts[2].strip())
            elif len(parts) == 2:
                movie_ids.append(int(parts[0]))
                years.append(np.nan)
                titles.append(parts[1].strip())
                
    df = pd.DataFrame({
        "movie_id": movie_ids,
        "year": years,
        "movie_title": titles
    })
    # Cast year to Int64 to support NaN values
    df["year"] = df["year"].astype("Int64")
    return df

def load_and_preprocess_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full loading pipeline with a memory-efficient two-pass parser:
    1. Checks if raw files exist, triggers synthetic generation if not.
    2. Pass 1: Count user and movie frequencies across all raw ratings.
    3. Determine the top 5,000 active users and top 2,000 most rated movies.
    4. Pass 2: Load and parse only ratings matching the top users and movies.
    5. Saves results as parquet.
    """
    import collections
    
    # 1. Trigger synthetic generation if movie titles or data files don't exist
    has_raw_data = os.path.exists(config.RAW_MOVIE_TITLES) and any(os.path.exists(f) and os.path.getsize(f) > 0 for f in config.RAW_DATA_FILES)
    if not has_raw_data:
        generate_synthetic_data(config.DATA_DIR)
        
    # Parse movie titles
    movies_df = parse_movie_titles(config.RAW_MOVIE_TITLES)
    
    # 2. Pass 1: Count frequencies
    print("Pass 1/2: Counting user activity and movie popularity in raw text files...")
    user_counts = collections.Counter()
    movie_counts = collections.Counter()
    
    for filepath in config.RAW_DATA_FILES:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            continue
        print(f"  Scanning {os.path.basename(filepath)}...")
        current_movie_id = None
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                else:
                    comma_idx = line.find(",")
                    if comma_idx != -1:
                        user_id = int(line[:comma_idx])
                        user_counts[user_id] += 1
                        movie_counts[current_movie_id] += 1
                        
    # 3. Determine top users and movies
    print(f"Identifying top {config.TOP_USERS_COUNT} active users and top {config.TOP_MOVIES_COUNT} popular movies...")
    top_users = set([u for u, _ in user_counts.most_common(config.TOP_USERS_COUNT)])
    top_movies = set([m for m, _ in movie_counts.most_common(config.TOP_MOVIES_COUNT)])
    
    # 4. Pass 2: Parse and load only relevant ratings
    print("Pass 2/2: Extracting filtered ratings for top users and movies...")
    movie_ids = []
    user_ids = []
    ratings = []
    dates = []
    
    for filepath in config.RAW_DATA_FILES:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            continue
        print(f"  Parsing ratings from {os.path.basename(filepath)}...")
        current_movie_id = None
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                else:
                    if current_movie_id in top_movies:
                        parts = line.split(",")
                        if len(parts) == 3:
                            uid = int(parts[0])
                            if uid in top_users:
                                user_ids.append(uid)
                                movie_ids.append(current_movie_id)
                                ratings.append(int(parts[1]))
                                dates.append(parts[2])
                                
    print(f"Building filtered DataFrame with {len(ratings):,} ratings...")
    ratings_filtered = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings,
        "date": pd.to_datetime(dates)
    })
    
    # Merge titles into ratings
    ratings_filtered = ratings_filtered.merge(movies_df, on="movie_id", how="left")
    
    # Save to Parquet
    print(f"Saving processed data to parquet: {config.PROCESSED_RATINGS_PARQUET}...")
    ratings_filtered.to_parquet(config.PROCESSED_RATINGS_PARQUET, index=False)
    
    movies_filtered = movies_df[movies_df["movie_id"].isin(top_movies)].copy()
    movies_filtered.to_parquet(config.PROCESSED_MOVIES_PARQUET, index=False)
    
    print("Data loading and subsetting complete.")
    print(f"Final filtered dataset contains {len(ratings_filtered):,} ratings, "
          f"{ratings_filtered['user_id'].nunique()} unique users, and "
          f"{ratings_filtered['movie_id'].nunique()} unique movies.")
          
    return ratings_filtered, movies_filtered

def get_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function to load processed data from parquet if available.
    Otherwise, builds it.
    """
    if os.path.exists(config.PROCESSED_RATINGS_PARQUET) and os.path.exists(config.PROCESSED_MOVIES_PARQUET):
        print("Loading preprocessed data from parquet...")
        ratings_df = pd.read_parquet(config.PROCESSED_RATINGS_PARQUET)
        movies_df = pd.read_parquet(config.PROCESSED_MOVIES_PARQUET)
        return ratings_df, movies_df
    else:
        return load_and_preprocess_dataset()

if __name__ == "__main__":
    load_and_preprocess_dataset()
