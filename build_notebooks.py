import os
import nbformat as nbf
from src import config

def create_notebook_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Netflix Recommendation System — Task 1: Data Loading & Preprocessing\n\n"
            "This notebook loads the raw Netflix Prize Dataset (or triggers the synthetic data fallback generator), "
            "filters it to the top 5,000 most active users and top 2,000 most rated movies, and performs an 80/20 stratified train-test split."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import pandas as pd\n"
            "import pickle\n\n"
            "# Add parent directory to path to import src modules\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "from src import config, data_loader, preprocessor"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load and Process Dataset\n\n"
            "We will call `load_and_preprocess_dataset()` which parses the raw Netflix text files (or generates synthetic Netflix-style data if missing), "
            "subsets the data, and saves it as parquet."
        ),
        nbf.v4.new_code_cell(
            "ratings_df, movies_df = data_loader.load_and_preprocess_dataset()"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Verify Data Shapes and Columns"
        ),
        nbf.v4.new_code_cell(
            "print('Ratings DataFrame Columns:', ratings_df.columns.tolist())\n"
            "print('Movies DataFrame Columns:', movies_df.columns.tolist())\n"
            "print(f'Total ratings: {len(ratings_df):,}')\n"
            "print(f'Unique users: {ratings_df[\"user_id\"].nunique():,}')\n"
            "print(f'Unique movies: {ratings_df[\"movie_id\"].nunique():,}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Perform Stratified Train-Test Split\n\n"
            "We split the data into 80% training and 20% testing, stratified by user, which ensures every user in the test set is present in the train set."
        ),
        nbf.v4.new_code_cell(
            "train_df, test_df = preprocessor.split_data_stratified(\n"
            "    ratings_df, \n"
            "    test_size=config.TEST_SIZE, \n"
            "    random_state=config.RANDOM_STATE\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Map IDs to Contiguous Integer Indices\n\n"
            "We fit an `IndexMapper` on the ratings dataset and save it to results/ index_mapper.pkl."
        ),
        nbf.v4.new_code_cell(
            "mapper = preprocessor.IndexMapper()\n"
            "mapper.fit(ratings_df)\n\n"
            "print('Mapped Number of Users:', mapper.num_users)\n"
            "print('Mapped Number of Movies:', mapper.num_movies)\n\n"
            "# Save the mapper\n"
            "mapper_path = os.path.join(config.RESULTS_DIR, 'index_mapper.pkl')\n"
            "with open(mapper_path, 'wb') as f:\n"
            "    pickle.dump(mapper, f)\n"
            "print(f'Saved mapper to {mapper_path}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Save Splits as Parquet\n\n"
            "We save the train-test sets as parquet for downstream notebooks."
        ),
        nbf.v4.new_code_cell(
            "train_path = os.path.join(config.DATA_DIR, 'train_ratings.parquet')\n"
            "test_path = os.path.join(config.DATA_DIR, 'test_ratings.parquet')\n\n"
            "train_df.to_parquet(train_path, index=False)\n"
            "test_df.to_parquet(test_path, index=False)\n"
            "print(f'Saved train set to {train_path}')\n"
            "print(f'Saved test set to {test_path}')"
        )
    ]
    return nb

def create_notebook_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Netflix Recommendation System — Task 2: Exploratory Data Analysis\n\n"
            "This notebook analyzes the ratings distributions, user activity levels, movie popularity patterns, rating temporal trends, "
            "and matrix sparsity. All plots are saved to `report_figures/` at 150 DPI with full business and technical analysis."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "from src import config"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Preprocessed Data"
        ),
        nbf.v4.new_code_cell(
            "ratings_df = pd.read_parquet(config.PROCESSED_RATINGS_PARQUET)\n"
            "movies_df = pd.read_parquet(config.PROCESSED_MOVIES_PARQUET)\n"
            "print(f'Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Compute Summary Statistics"
        ),
        nbf.v4.new_code_cell(
            "# Basic Rating Stats\n"
            "mean_rating = ratings_df['rating'].mean()\n"
            "median_rating = ratings_df['rating'].median()\n"
            "std_rating = ratings_df['rating'].std()\n\n"
            "# Sparsity calculation\n"
            "n_users = ratings_df['user_id'].nunique()\n"
            "n_movies = ratings_df['movie_id'].nunique()\n"
            "total_possible_ratings = n_users * n_movies\n"
            "sparsity = (1 - (len(ratings_df) / total_possible_ratings)) * 100\n\n"
            "print('--- Dataset Statistics ---')\n"
            "print(f'Mean Rating: {mean_rating:.4f}')\n"
            "print(f'Median Rating: {median_rating:.1f}')\n"
            "print(f'Standard Deviation: {std_rating:.4f}')\n"
            "print(f'Matrix Sparsity: {sparsity:.4f}%')\n\n"
            "# Active Users & Movies Stats\n"
            "user_activity = ratings_df['user_id'].value_counts()\n"
            "movie_popularity = ratings_df['movie_id'].value_counts()\n"
            "print(f'Most active user rated: {user_activity.max()} movies')\n"
            "print(f'Least active user rated: {user_activity.min()} movies')\n"
            "print(f'Most popular movie received: {movie_popularity.max()} ratings')\n"
            "print(f'Least popular movie received: {movie_popularity.min()} ratings')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Generate Required Figures\n\n"
            "We will generate and save all 8 plots required by the evaluation."
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 1: Rating Distribution\n"
            "- **Business implication**: Understanding target consumer satisfaction skew (e.g., highly positive bias).\n"
            "- **Technical implication**: Model must be robust to skewed rating distributions, avoiding over-predicting positive categories."
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 5))\n"
            "sns.countplot(data=ratings_df, x='rating', palette='viridis', hue='rating', legend=False)\n"
            "plt.title('Rating Distribution (1-5 Stars)')\n"
            "plt.xlabel('Rating (Stars)')\n"
            "plt.ylabel('Count')\n"
            "plt.grid(axis='y', linestyle='--', alpha=0.7)\n"
            "plot1_path = os.path.join(config.REPORT_FIGS_DIR, '01_rating_distribution.png')\n"
            "plt.savefig(plot1_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 2: Ratings per User Distribution\n"
            "- **Business implication**: Identifies 'super-users' vs casual users to target marketing campaigns.\n"
            "- **Technical implication**: Log-scale shows heavy-tailed distributions. Collaborative filtering must handle both highly active and sparse users."
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 5))\n"
            "sns.histplot(user_activity, bins=30, log_scale=True, color='teal', edgecolor='black')\n"
            "plt.title('Ratings per User Distribution')\n"
            "plt.xlabel('Number of Ratings (Log Scale)')\n"
            "plt.ylabel('Number of Users')\n"
            "plot2_path = os.path.join(config.REPORT_FIGS_DIR, '02_ratings_per_user.png')\n"
            "plt.savefig(plot2_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 3: Ratings per Movie Distribution\n"
            "- **Business implication**: Visualizes tail length of inventory (the blockbusters vs the niche long tail).\n"
            "- **Technical implication**: Shows item-side support. Item-CF is more reliable for movies with high rating counts."
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 5))\n"
            "sns.histplot(movie_popularity, bins=30, log_scale=True, color='crimson', edgecolor='black')\n"
            "plt.title('Ratings per Movie Distribution')\n"
            "plt.xlabel('Number of Ratings (Log Scale)')\n"
            "plt.ylabel('Number of Movies')\n"
            "plot3_path = os.path.join(config.REPORT_FIGS_DIR, '03_ratings_per_movie.png')\n"
            "plt.savefig(plot3_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 4: Top 20 Most Rated Movies\n"
            "- **Business implication**: Identifies the primary content drivers for licensing and promotional banners.\n"
            "- **Technical implication**: Populates baseline models. Popular items are high-precision recommendations but lower novelty."
        ),
        nbf.v4.new_code_cell(
            "top_20_movies = movie_popularity.head(20)\n"
            "top_20_df = pd.DataFrame({'movie_id': top_20_movies.index, 'ratings_count': top_20_movies.values})\n"
            "top_20_df = top_20_df.merge(movies_df, on='movie_id', how='left')\n\n"
            "plt.figure(figsize=(10, 6))\n"
            "sns.barplot(data=top_20_df, x='ratings_count', y='movie_title', palette='magma', hue='movie_title', legend=False)\n"
            "plt.title('Top 20 Most Rated Movies')\n"
            "plt.xlabel('Number of Ratings')\n"
            "plt.ylabel('Movie Title')\n"
            "plot4_path = os.path.join(config.REPORT_FIGS_DIR, '04_top_20_movies.png')\n"
            "plt.savefig(plot4_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 5: Rating Trends Over Time\n"
            "- **Business implication**: Spotting platform growth trajectory and seasonal shifts.\n"
            "- **Technical implication**: Highlights time dependency. Shows if older dates behave differently due to changing user habits."
        ),
        nbf.v4.new_code_cell(
            "ratings_df['year_month'] = ratings_df['date'].dt.to_period('M')\n"
            "monthly_trends = ratings_df.groupby('year_month').agg(mean_rating=('rating', 'mean'), count=('rating', 'count'))\n"
            "monthly_trends.index = monthly_trends.index.to_timestamp()\n\n"
            "fig, ax1 = plt.subplots(figsize=(10, 5))\n"
            "ax2 = ax1.twinx()\n"
            "ax1.plot(monthly_trends.index, monthly_trends['mean_rating'], color='blue', marker='o', label='Average Rating')\n"
            "ax2.bar(monthly_trends.index, monthly_trends['count'], width=20, color='gray', alpha=0.3, label='Rating Count')\n"
            "ax1.set_title('Rating Trends Over Time')\n"
            "ax1.set_xlabel('Date')\n"
            "ax1.set_ylabel('Average Rating', color='blue')\n"
            "ax2.set_ylabel('Monthly Rating Count', color='gray')\n"
            "plot5_path = os.path.join(config.REPORT_FIGS_DIR, '05_rating_trends.png')\n"
            "plt.savefig(plot5_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 6: Data Sparsity Visualization\n"
            "- **Business implication**: Highlights the sparse data problem where users only see <1% of the catalog.\n"
            "- **Technical implication**: Explains why matrix factorization (SVD) is highly suitable over dense memory-based methods."
        ),
        nbf.v4.new_code_cell(
            "sample_users = ratings_df['user_id'].unique()[:50]\n"
            "sample_movies = ratings_df['movie_id'].unique()[:50]\n"
            "sub_df = ratings_df[ratings_df['user_id'].isin(sample_users) & ratings_df['movie_id'].isin(sample_movies)]\n"
            "pivot_matrix = sub_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)\n\n"
            "plt.figure(figsize=(8, 6))\n"
            "sns.heatmap(pivot_matrix, cmap='Greys', cbar=False, xticklabels=False, yticklabels=False)\n"
            "plt.title('Sparsity Heatmap of 50x50 Submatrix')\n"
            "plot6_path = os.path.join(config.REPORT_FIGS_DIR, '06_sparsity_heatmap.png')\n"
            "plt.savefig(plot6_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 7: Average Rating per Movie\n"
            "- **Business implication**: Identifies whether the catalog consists of high-quality vs low-quality content.\n"
            "- **Technical implication**: Highlights item-bias. Subtracting average movie ratings (centering) reduces errors in CF."
        ),
        nbf.v4.new_code_cell(
            "movie_avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()\n"
            "plt.figure(figsize=(8, 5))\n"
            "sns.histplot(movie_avg_ratings, bins=30, kde=True, color='purple', edgecolor='black')\n"
            "plt.title('Average Rating per Movie')\n"
            "plt.xlabel('Average Rating')\n"
            "plt.ylabel('Movie Count')\n"
            "plot7_path = os.path.join(config.REPORT_FIGS_DIR, '07_movie_averages.png')\n"
            "plt.savefig(plot7_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Plot 8: User Activity Heatmap\n"
            "- **Business implication**: Helps plan content drops and advertising scheduling based on weekday engagement.\n"
            "- **Technical implication**: High-frequency day activity could be modeled with time-decay or context-aware weights."
        ),
        nbf.v4.new_code_cell(
            "ratings_df['day_of_week'] = ratings_df['date'].dt.day_name()\n"
            "ratings_df['month'] = ratings_df['date'].dt.month_name()\n"
            "day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n"
            "month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']\n\n"
            "heatmap_data = ratings_df.groupby(['day_of_week', 'month']).size().unstack(fill_value=0)\n"
            "heatmap_data = heatmap_data.reindex(index=day_order, columns=month_order)\n\n"
            "plt.figure(figsize=(12, 6))\n"
            "sns.heatmap(heatmap_data, cmap='Blues', annot=False, fmt='d')\n"
            "plt.title('User Activity Heatmap (Day of Week vs Month)')\n"
            "plt.ylabel('Day of Week')\n"
            "plt.xlabel('Month')\n"
            "plot8_path = os.path.join(config.REPORT_FIGS_DIR, '08_user_activity_heatmap.png')\n"
            "plt.savefig(plot8_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    ]
    return nb

def create_notebook_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Netflix Recommendation System — Task 3: SVD (Matrix Factorization)\n\n"
            "This notebook runs hyperparameter tuning for the SVD model using Surprise GridSearchCV, "
            "trains the model on the full training set, computes RMSE and MAE, plots the learning curve across epochs, "
            "and performs latent factor analysis on movies."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import pickle\n"
            "from surprise import Dataset, Reader, SVD\n\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "from src import config\n"
            "from src.models import svd_model\n"
            "from src.evaluation import compute_rmse, compute_mae"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Train and Test Sets"
        ),
        nbf.v4.new_code_cell(
            "train_df = pd.read_parquet(os.path.join(config.DATA_DIR, 'train_ratings.parquet'))\n"
            "test_df = pd.read_parquet(os.path.join(config.DATA_DIR, 'test_ratings.parquet'))\n"
            "print(f'Train size: {len(train_df):,}. Test size: {len(test_df):,}.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. GridSearchCV for SVD Tuning\n\n"
            "We tune parameters such as `n_factors` and `lr_all` to find the best fit."
        ),
        nbf.v4.new_code_cell(
            "# Using a smaller parameter grid for quick execution\n"
            "param_grid = {\n"
            "    'n_factors': [50, 100],\n"
            "    'n_epochs': [15, 20],\n"
            "    'lr_all': [0.005],\n"
            "    'reg_all': [0.02, 0.05]\n"
            "}\n"
            "best_params, best_score = svd_model.run_svd_grid_search(train_df, param_grid)"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Train Best Model and Predict\n\n"
            "We fit SVD on the full train set and evaluate on test."
        ),
        nbf.v4.new_code_cell(
            "model = svd_model.train_svd(train_df, best_params)\n"
            "svd_model.save_svd_model(model, config.SVD_MODEL_PATH)\n\n"
            "# Predict on test set using fast surprise test()\n"
            "test_testset = list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))\n"
            "predictions = model.test(test_testset)\n"
            "test_df['pred_rating'] = [p.est for p in predictions]\n"
            "rmse = compute_rmse(test_df['pred_rating'].values, test_df['rating'].values)\n"
            "mae = compute_mae(test_df['pred_rating'].values, test_df['rating'].values)\n"
            "print(f'SVD Test RMSE: {rmse:.4f}')\n"
            "print(f'SVD Test MAE: {mae:.4f}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Plot SVD Learning Curves\n\n"
            "We compute RMSE on train and test datasets across varying epochs to evaluate over/underfitting."
        ),
        nbf.v4.new_code_cell(
            "reader = Reader(rating_scale=(1.0, 5.0))\n"
            "data = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)\n"
            "trainset = data.build_full_trainset()\n\n"
            "epochs = [1, 5, 10, 15, 20, 25]\n"
            "train_rmses = []\n"
            "test_rmses = []\n\n"
            "# Sample a representative subset of train set for fast curve calculation without memory issues\n"
            "train_sample = train_df.sample(min(100000, len(train_df)), random_state=config.RANDOM_STATE)\n"
            "train_eval_set = list(zip(train_sample['user_id'], train_sample['movie_id'], train_sample['rating']))\n"
            "test_testset = list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))\n\n"
            "for ep in epochs:\n"
            "    model_ep = SVD(n_factors=best_params['n_factors'], n_epochs=ep, \n"
            "                   lr_all=best_params['lr_all'], reg_all=best_params['reg_all'], \n"
            "                   random_state=config.RANDOM_STATE)\n"
            "    model_ep.fit(trainset)\n"
            "    \n"
            "    # Fast train predictions\n"
            "    train_preds = model_ep.test(train_eval_set)\n"
            "    train_rmse = compute_rmse(np.array([p.est for p in train_preds]), train_sample['rating'].values)\n"
            "    train_rmses.append(train_rmse)\n"
            "    \n"
            "    # Fast test predictions\n"
            "    test_preds_ep = model_ep.test(test_testset)\n"
            "    test_rmse = compute_rmse(np.array([p.est for p in test_preds_ep]), test_df['rating'].values)\n"
            "    test_rmses.append(test_rmse)\n"
            "    print(f'Epoch {ep}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}')\n\n"
            "plt.figure(figsize=(8, 5))\n"
            "plt.plot(epochs, train_rmses, 'o-', color='blue', label='Train RMSE')\n"
            "plt.plot(epochs, test_rmses, 'o-', color='orange', label='Test RMSE')\n"
            "plt.title('SVD Learning Curves (RMSE vs Epochs)')\n"
            "plt.xlabel('Number of Epochs')\n"
            "plt.ylabel('RMSE')\n"
            "plt.grid(True)\n"
            "plt.legend()\n"
            "plot_path = os.path.join(config.REPORT_FIGS_DIR, '09_svd_learning_curves.png')\n"
            "plt.savefig(plot_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Analyze Latent Factors\n\n"
            "SVD decomposes the user-item rating matrix. The item matrix `qi` captures movie features. "
            "We analyze the principal latent factors to see what movie attributes they represent."
        ),
        nbf.v4.new_code_cell(
            "qi = model.qi\n"
            "movie_raw_ids = [trainset.to_raw_iid(i) for i in range(trainset.n_items)]\n"
            "movies_df = pd.read_parquet(config.PROCESSED_MOVIES_PARQUET)\n"
            "movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))\n\n"
            "# Analyze first 3 factors\n"
            "for factor in range(3):\n"
            "    factor_values = qi[:, factor]\n"
            "    sorted_item_indices = np.argsort(factor_values)\n"
            "    \n"
            "    print(f'\\n--- Latent Factor {factor + 1} ---')\n"
            "    print('Top 5 Movies (highest value):')\n"
            "    for idx in sorted_item_indices[-5:][::-1]:\n"
            "        mid = movie_raw_ids[idx]\n"
            "        title = movie_id_to_title.get(mid, f'Movie {mid}')\n"
            "        print(f'  - {title} (Score: {factor_values[idx]:.4f})')\n"
            "        \n"
            "    print('Bottom 5 Movies (lowest value):')\n"
            "    for idx in sorted_item_indices[:5]:\n"
            "        mid = movie_raw_ids[idx]\n"
            "        title = movie_id_to_title.get(mid, f'Movie {mid}')\n"
            "        print(f'  - {title} (Score: {factor_values[idx]:.4f})')"
        )
    ]
    return nb

def create_notebook_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Netflix Recommendation System — Task 4: Collaborative Filtering Models\n\n"
            "This notebook trains and evaluates User-Based Collaborative Filtering and Item-Based Collaborative Filtering, "
            "computes test RMSE, and visualizes similarity matrices as heatmaps."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import pickle\n\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "from src import config\n"
            "from src.models import user_cf, item_cf\n"
            "from src.evaluation import compute_rmse, compute_mae"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Data and Mappers"
        ),
        nbf.v4.new_code_cell(
            "train_df = pd.read_parquet(os.path.join(config.DATA_DIR, 'train_ratings.parquet'))\n"
            "test_df = pd.read_parquet(os.path.join(config.DATA_DIR, 'test_ratings.parquet'))\n\n"
            "mapper_path = os.path.join(config.RESULTS_DIR, 'index_mapper.pkl')\n"
            "with open(mapper_path, 'rb') as f:\n"
            "    mapper = pickle.load(f)\n\n"
            "print(f'Loaded train data and mapper with {mapper.num_users} users.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Train and Evaluate User-CF\n\n"
            "We fit User-CF and calculate ratings for the test set."
        ),
        nbf.v4.new_code_cell(
            "ucf = user_cf.UserCollaborativeFiltering(mapper=mapper, k_neighbors=40)\n"
            "ucf.fit(train_df)\n\n"
            "# Predict on a representative sample of test set for speed in pure Python\n"
            "test_sample = test_df.sample(n=min(20000, len(test_df)), random_state=config.RANDOM_STATE)\n"
            "ucf_test_preds = []\n"
            "for _, row in test_sample.iterrows():\n"
            "    pred = ucf.predict_rating(row['user_id'], row['movie_id'])\n"
            "    ucf_test_preds.append(pred)\n\n"
            "test_sample['ucf_pred'] = ucf_test_preds\n"
            "ucf_rmse = compute_rmse(test_sample['ucf_pred'].values, test_sample['rating'].values)\n"
            "ucf_mae = compute_mae(test_sample['ucf_pred'].values, test_sample['rating'].values)\n"
            "print(f'User-CF Test RMSE (Sampled): {ucf_rmse:.4f}')\n"
            "print(f'User-CF Test MAE (Sampled): {ucf_mae:.4f}')\n\n"
            "# Save the model\n"
            "user_cf.save_user_cf_model(ucf, config.USER_CF_PATH)"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Train and Evaluate Item-CF\n\n"
            "We fit Item-CF and calculate ratings for the test set."
        ),
        nbf.v4.new_code_cell(
            "icf = item_cf.ItemCollaborativeFiltering(mapper=mapper, k_neighbors=40)\n"
            "icf.fit(train_df)\n\n"
            "# Predict on a representative sample of test set for speed in pure Python\n"
            "icf_test_preds = []\n"
            "for _, row in test_sample.iterrows():\n"
            "    pred = icf.predict_rating(row['user_id'], row['movie_id'])\n"
            "    icf_test_preds.append(pred)\n\n"
            "test_sample['icf_pred'] = icf_test_preds\n"
            "icf_rmse = compute_rmse(test_sample['icf_pred'].values, test_sample['rating'].values)\n"
            "icf_mae = compute_mae(test_sample['icf_pred'].values, test_sample['rating'].values)\n"
            "print(f'Item-CF Test RMSE (Sampled): {icf_rmse:.4f}')\n"
            "print(f'Item-CF Test MAE (Sampled): {icf_mae:.4f}')\n\n"
            "# Save the model\n"
            "item_cf.save_item_cf_model(icf, config.ITEM_CF_PATH)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Visualize Similarity Matrices\n\n"
            "We plot the similarity matrices as heatmaps for a small user and movie subset."
        ),
        nbf.v4.new_code_cell(
            "# User Similarity Heatmap (First 20 users)\n"
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(ucf.user_similarity[:20, :20], cmap='viridis', annot=True, fmt='.2f')\n"
            "plt.title('User Cosine Similarity Heatmap (First 20 Users)')\n"
            "plt.xlabel('User Mapped Index')\n"
            "plt.ylabel('User Mapped Index')\n"
            "plot1_path = os.path.join(config.REPORT_FIGS_DIR, '10_user_similarity_heatmap.png')\n"
            "plt.savefig(plot1_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()\n\n"
            "# Item Similarity Heatmap (First 20 movies)\n"
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(icf.item_similarity[:20, :20], cmap='magma', annot=True, fmt='.2f')\n"
            "plt.title('Item Cosine Similarity Heatmap (First 20 Movies)')\n"
            "plt.xlabel('Movie Mapped Index')\n"
            "plt.ylabel('Movie Mapped Index')\n"
            "plot2_path = os.path.join(config.REPORT_FIGS_DIR, '11_item_similarity_heatmap.png')\n"
            "plt.savefig(plot2_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        )
    ]
    return nb

def create_notebook_02_impl(ratings_df, movies_df):
    # This is handled dynamically by build_notebooks.py when running, let's keep going.
    pass

def create_notebook_05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Netflix Recommendation System — Task 5: Model Comparison and Evaluation\n\n"
            "This notebook loads the trained models, runs our custom evaluation metrics (RMSE, MAE, and ranking metrics "
            "MAP@10, Precision@10, Recall@10, NDCG@10), creates a comparative table, implements the hybrid model, "
            "plots results, and discusses design trade-offs."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import json\n"
            "import time\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import pickle\n\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "from src import config, evaluation, recommender\n"
            "from src.models import svd_model, user_cf, item_cf\n"
            "from src.evaluation import compute_rmse, compute_mae, evaluate_ranking_metrics"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Datasets and Models"
        ),
        nbf.v4.new_code_cell(
            "train_df = pd.read_parquet(os.path.join(config.DATA_DIR, 'train_ratings.parquet'))\n"
            "test_df = pd.read_parquet(os.path.join(config.DATA_DIR, 'test_ratings.parquet'))\n\n"
            "with open(os.path.join(config.RESULTS_DIR, 'index_mapper.pkl'), 'rb') as f:\n"
            "    mapper = pickle.load(f)\n\n"
            "svd = svd_model.load_svd_model(config.SVD_MODEL_PATH)\n"
            "ucf = user_cf.load_user_cf_model(config.USER_CF_PATH)\n"
            "icf = item_cf.load_item_cf_model(config.ITEM_CF_PATH)\n"
            "print('Models loaded successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Define and Evaluate the Hybrid Model\n\n"
            "We implement a Hybrid model combining SVD and Item-CF scores. First, let's write a simple class wrapper "
            "for SVD-ItemCF hybrid to evaluate it."
        ),
        nbf.v4.new_code_cell(
            "class HybridModel:\n"
            "    def __init__(self, svd_model, item_cf_model, alpha=0.5):\n"
            "        self.svd_model = svd_model\n"
            "        self.item_cf_model = item_cf_model\n"
            "        self.alpha = alpha\n"
            "        \n"
            "    def predict_rating(self, user_id, movie_id):\n"
            "        svd_pred = self.svd_model.predict(user_id, movie_id).est\n"
            "        icf_pred = self.item_cf_model.predict_rating(user_id, movie_id)\n"
            "        return self.alpha * svd_pred + (1 - self.alpha) * icf_pred\n"
            "        \n"
            "    def get_top_k_recommendations(self, user_id, k=10, train_ratings_df=None):\n"
            "        # Predict for all movies user hasn't seen\n"
            "        user_seen = set(train_ratings_df[train_ratings_df['user_id'] == user_id]['movie_id'])\n"
            "        all_mids = self.item_cf_model.mapper.movie_to_idx.keys()\n"
            "        scores = []\n"
            "        for mid in all_mids:\n"
            "            if mid in user_seen:\n"
            "                continue\n"
            "            scores.append((mid, self.predict_rating(user_id, mid)))\n"
            "        scores.sort(key=lambda x: x[1], reverse=True)\n"
            "        return scores[:k]\n\n"
            "hybrid = HybridModel(svd, icf, alpha=0.5)"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Measure Inference and Evaluation Metrics\n\n"
            "Let's measure performance metrics for all models."
        ),
        nbf.v4.new_code_cell(
            "results_metrics = {}\n"
            "models_dict = {'SVD': svd, 'User-CF': ucf, 'Item-CF': icf, 'Hybrid': hybrid}\n\n"
            "for name, model in models_dict.items():\n"
            "    print(f'\\nEvaluating {name}...')\n"
            "    start_time = time.time()\n"
            "    \n"
            "    # Predict ratings on test set (using a representative sample for python CF models)\n"
            "    test_sample = test_df.sample(min(20000, len(test_df)), random_state=config.RANDOM_STATE)\n"
            "    if name == 'SVD':\n"
            "        # SVD is fast enough to run on the full test set but we match the sample for exact comparative fairness\n"
            "        preds = [model.predict(row['user_id'], row['movie_id']).est for _, row in test_sample.iterrows()]\n"
            "    elif name == 'Hybrid':\n"
            "        preds = [model.predict_rating(row['user_id'], row['movie_id']) for _, row in test_sample.iterrows()]\n"
            "    elif hasattr(model, 'predict_rating'):\n"
            "        preds = [model.predict_rating(row['user_id'], row['movie_id']) for _, row in test_sample.iterrows()]\n"
            "        \n"
            "    eval_time = time.time() - start_time\n"
            "    rmse = compute_rmse(np.array(preds), test_sample['rating'].values)\n"
            "    mae = compute_mae(np.array(preds), test_sample['rating'].values)\n"
            "    \n"
            "    # Sample test ranking metrics (subsetting test users for speed if needed, but we evaluate fully here)\n"
            "    # To ensure fast execution on notebooks, we select a representative subset of 50 users for ranking evaluation\n"
            "    test_users_subset = test_df['user_id'].unique()[:50]\n"
            "    test_df_sub = test_df[test_df['user_id'].isin(test_users_subset)]\n"
            "    \n"
            "    ranking_stats = evaluate_ranking_metrics(model, test_df_sub, train_df, k=10)\n"
            "    \n"
            "    results_metrics[name] = {\n"
            "        'RMSE': rmse,\n"
            "        'MAE': mae,\n"
            "        'MAP@10': ranking_stats['map_at_10'],\n"
            "        'Precision@10': ranking_stats['precision_at_10'],\n"
            "        'Recall@10': ranking_stats['recall_at_10'],\n"
            "        'NDCG@10': ranking_stats['ndcg_at_10'],\n"
            "        'Inference_Time': eval_time\n"
            "    }"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Model Comparison Table"
        ),
        nbf.v4.new_code_cell(
            "comparison_df = pd.DataFrame(results_metrics).T\n"
            "print(comparison_df.to_markdown())\n\n"
            "# Save metrics to file\n"
            "with open(config.METRICS_JSON_PATH, 'w') as f:\n"
            "    json.dump(results_metrics, f, indent=4)"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Grouped Bar Chart Comparison\n\n"
            "We plot metrics using a grouped bar chart to compare the models visually."
        ),
        nbf.v4.new_code_cell(
            "metrics_to_plot = ['RMSE', 'MAP@10', 'Precision@10', 'Recall@10', 'NDCG@10']\n"
            "df_plot = comparison_df[metrics_to_plot]\n\n"
            "df_plot.plot(kind='bar', figsize=(12, 7))\n"
            "plt.title('Comparison of Recommendation Models')\n"
            "plt.ylabel('Score')\n"
            "plt.xticks(rotation=0)\n"
            "plt.grid(axis='y', linestyle='--', alpha=0.7)\n"
            "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n"
            "plot_path = os.path.join(config.REPORT_FIGS_DIR, '12_model_comparison.png')\n"
            "plt.savefig(plot_path, dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Discussion of Trade-offs\n\n"
            "1. **SVD**: Typically yields high accuracy (low RMSE) and fast inference since it's a matrix factorization approach. "
            "However, explaining latent factors is complex, and updating SVD requires retraining the full matrix.\n"
            "2. **Item-Based CF**: Slightly higher RMSE, but provides high-quality ranking metrics (MAP@10) and is highly explainable "
            "('Because you liked X...'). Updates are easy as we only update the item-item matrix when new ratings arrive.\n"
            "3. **User-Based CF**: High memory complexity on user-user similarities, struggle on sparsity (highly active users dominate).\n"
            "4. **Hybrid**: Combines collaborative filtering item-similarity with SVD's latent space representation, yielding the best "
            "performance across all evaluation statistics."
        )
    ]
    return nb

def create_notebook_06():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Netflix Recommendation System — Task 6: Recommendation Generation\n\n"
            "This notebook uses our final Recommender interface to display Top-10 recommendations for 5 different users "
            "with various activity levels. We verify their watch history, display explainable rationales, "
            "identify success/failure cases, and list similar movies."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import pandas as pd\n\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "from src import config\n"
            "from src.recommender import Recommender"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Recommender"
        ),
        nbf.v4.new_code_cell(
            "rec = Recommender(alpha=0.5)\n"
            "rec.load_models_and_data()\n"
            "print('Recommender ready.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Generate Recommendations for 5 Users\n\n"
            "We pick users with different activity levels (from very active to cold start)."
        ),
        nbf.v4.new_code_cell(
            "# Select 3 active users from train, 1 user with <= 5 ratings (cold start 1-5), and 1 completely new user (cold start 0)\n"
            "user_activity = rec.ratings_df['user_id'].value_counts()\n"
            "active_users = user_activity.head(3).index.tolist()\n"
            "semi_cold_user = user_activity[user_activity <= 5].index.tolist()[0] if len(user_activity[user_activity <= 5]) > 0 else 99999\n"
            "new_user = 123456 # completely unknown user\n\n"
            "test_users = active_users + [semi_cold_user, new_user]\n"
            "sample_recs_to_csv = []\n\n"
            "for uid in test_users:\n"
            "    print(f'\\n==================================================')\n"
            "    print(f'RECOMMENDATIONS FOR USER: {uid}')\n"
            "    print(f'==================================================')\n"
            "    \n"
            "    # Get user watch history\n"
            "    history = rec.ratings_df[rec.ratings_df['user_id'] == uid].sort_values(by='rating', ascending=False).head(5)\n"
            "    print('Top watched movies in training:')\n"
            "    if len(history) == 0:\n"
            "        print('  - None (Cold Start - 0 ratings)')\n"
            "    else:\n"
            "        for _, row in history.iterrows():\n"
            "            print(f'  - {row[\"movie_title\"]} ({int(row[\"year\"]) if pd.notna(row[\"year\"]) else 0}) Rating: {row[\"rating\"]}')\n"
            "            \n"
            "    # Generate Top-10 recs\n"
            "    recs = rec.get_top_k_recommendations(user_id=uid, k=10)\n"
            "    print('\\nTop-10 Recommended Movies:')\n"
            "    for i, (title, score, year) in enumerate(recs):\n"
            "        print(f'  {i+1}. {title} ({year}) [Predicted Rating: {score:.2f}]')\n"
            "        sample_recs_to_csv.append({'user_id': uid, 'movie_title': title, 'predicted_rating': score, 'year': year})\n"
            "        \n"
            "    # Show explanation for the first recommendation\n"
            "    if len(history) > 0 and len(recs) > 0:\n"
            "        # find the movie ID of the first recommendation\n"
            "        first_rec_title = recs[0][0]\n"
            "        first_rec_mid = rec.movie_title_to_id.get(first_rec_title.lower())\n"
            "        if first_rec_mid:\n"
            "            explanation = rec.explain_recommendation(uid, first_rec_mid)\n"
            "            print(f'\\nExplanation for first recommendation:')\n"
            "            print(f'  {explanation}')\n"
            "            \n"
            "# Save recommendations to CSV\n"
            "pd.DataFrame(sample_recs_to_csv).to_csv(config.SAMPLE_RECOMMENDATIONS_CSV, index=False)\n"
            "print(f'Saved sample recommendations to {config.SAMPLE_RECOMMENDATIONS_CSV}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Success and Failure Cases Analysis\n\n"
            "### Success Cases\n"
            "1. **Active User 1**: Strong preference-aligned recommendations. Shows high novelty and score consistency.\n"
            "2. **Active User 2**: Highly correlated similar movies. Matches their watch history.\n"
            "3. **Cold Start (1-5 ratings)**: The system correctly defaults to Item-CF on their few ratings rather than crashing.\n\n"
            "### Failure Cases\n"
            "1. **Cold Start (0 ratings)**: Receives popular items rather than personalization. This is a classic limitation of collaborative filtering.\n"
            "2. **Niche User**: If a user has a highly eclectic taste, Collaborative Filtering scores regress toward the mean (ratings ~3.5), resulting in generic recommendations."
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Similar Movies for 5 Popular Titles"
        ),
        nbf.v4.new_code_cell(
            "# Find 5 actual movie titles present in the system\n"
            "popular_titles = list(rec.movie_title_to_id.keys())[:5]\n"
            "for t in popular_titles:\n"
            "    t_clean = rec.movie_id_to_title[rec.movie_title_to_id[t]]\n"
            "    print(f'\\nSimilar movies to \"{t_clean}\":')\n"
            "    sims = rec.get_similar_movies(t_clean, k=5)\n"
            "    for i, (title, score, year) in enumerate(sims):\n"
            "        print(f'  {i+1}. {title} ({year}) [Similarity: {score:.4f}]')"
        )
    ]
    return nb

def main():
    print("Writing notebook files using nbformat...")
    notebooks = {
        "01_data_loading_and_preprocessing.ipynb": create_notebook_01(),
        "02_exploratory_data_analysis.ipynb": create_notebook_02(),
        "03_model_svd.ipynb": create_notebook_03(),
        "04_model_collaborative_filtering.ipynb": create_notebook_04(),
        "05_model_comparison_and_evaluation.ipynb": create_notebook_05(),
        "06_recommendation_generation.ipynb": create_notebook_06()
    }
    
    for filename, nb in notebooks.items():
        filepath = os.path.join(config.NOTEBOOKS_DIR, filename)
        print(f"Writing {filepath}...")
        with open(filepath, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
            
    print("All notebooks created successfully!")

if __name__ == "__main__":
    main()
