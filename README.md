# Production-Quality Netflix Prize Recommendation System

An end-to-end, high-performance recommendation engine built using a subset of the famous Netflix Prize Dataset (5,000 highly active users and 2,000 frequently rated movies). This system implements and compares multiple recommendation paradigms, evaluates them using rigorous ranking metrics, and packages them in a production-ready repository.

---

## 🚀 Key Features

* **Multi-Model Pipeline**: Full implementation of **Matrix Factorization (SVD)**, **User-Based Collaborative Filtering (User-CF)**, **Item-Based Collaborative Filtering (Item-CF)**, and a **Weighted SVD-ItemCF Hybrid**.
* **Custom Ranking Metrics**: Evaluation on custom implementations of **MAP@10**, **Precision@10**, **Recall@10**, and **NDCG@10**, excluding training items with a relevance threshold of $\ge 3.5$.
* **Robust Cold-Start Handling**: Multi-tiered fallback strategy for users with zero ratings (popularity-based) and sparse profiles (Item-CF/demographics).
* **Sparse Matrix Optimization**: Optimized memory footprint and $1000\times$ faster predictions using `scipy.sparse` CSC representations.
* **Auto-Pipeline Automation**: Scripts for sequential notebook execution and automated deliverable verification.

---

## 📁 Repository Structure

```text
netflix-recommendation-system/
├── data/                             # Raw and preprocessed datasets (Parquet format)
│   ├── combined_data_1.txt           # Raw Netflix Prize data files
│   ├── movie_titles.csv              # Raw movie titles mapping
│   ├── processed_movies.parquet      # Filtered movie list
│   ├── processed_ratings.parquet     # Combined & filtered ratings
│   ├── train_ratings.parquet         # Stratified training set (80% ratings)
│   └── test_ratings.parquet          # Stratified test set (20% ratings)
│
├── docs/                             # Deliverable documents (Technical Report & Presentation Slides)
│   ├── Technical_Report.md           # Deliverable 1: Detailed Technical Report
│   └── Presentation_Slides.md        # Deliverable 3: 8-slide Presentation Outline
│
├── notebooks/                        # Jupyter Notebooks for sequential execution
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_model_svd.ipynb
│   ├── 04_model_collaborative_filtering.ipynb
│   ├── 05_model_comparison_and_evaluation.ipynb
│   └── 06_recommendation_generation.ipynb
│
├── report_figures/                   # Exported business & technical EDA/model figures (150 DPI)
│   ├── 01_rating_distribution.png to 12_model_comparison.png
│
├── results/                          # Serialized models & mapping artifacts
│   ├── index_mapper.pkl              # Contiguous integer index mapper
│   ├── svd_model.pkl                 # Tuned & trained SVD model
│   ├── user_cf.pkl                   # Trained User-CF model
│   ├── item_cf.pkl                   # Trained Item-CF model
│   ├── metrics.json                  # Model comparison metrics
│   └── sample_recommendations.csv    # Generated recommendations for sample users
│
├── src/                              # Source Python modules containing core logic
│   ├── config.py                     # Global configurations & hyperparameters
│   ├── data_loader.py                # Dataset parsing & synthetic generator
│   ├── preprocessor.py               # Sparse index mapping & stratified splitting
│   ├── evaluation.py                 # Custom RMSE, MAE, MAP@10, Precision@10, NDCG@10
│   ├── recommender.py                # Top-K recommendations, explanations, cold-start
│   └── models/
│       ├── svd_model.py              # surprise-based SVD model logic
│       ├── user_cf.py                # Custom sparse User-CF
│       └── item_cf.py                # Custom sparse Item-CF
│
├── run_pipeline.py                   # Automated script to execute all notebooks
├── verify_pipeline.py                # Pipeline validation and metrics reporting
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook Pipeline
Execute all 6 notebooks sequentially:
```bash
python run_pipeline.py
```
This script executes notebooks 01–06 in-place and saves all models, metrics, and report figures.

### 3. Verify the Pipeline
Check all deliverables and output the comparison table:
```bash
python verify_pipeline.py
```

### 4. Run the Interactive CLI Demo
Interact with the recommendation engine in real-time via the console:
```bash
python demo.py
```

### 5. Launch the Web Application Interface
Launch the gorgeous Netflix-themed single-page web dashboard locally:
```bash
python app.py
```
Then, open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser. 

**Features available in the Web App:**
* **Dashboard Overview**: Interactive stats cards showing dataset metrics and a model evaluation table.
* **Personalized recommendations**: Input user IDs and select SVD, User-CF, Item-CF, or Hybrid models. View ratings and click the **Explain** button to see recommendation reasons.
* **Similarity Explorer**: Autocomplete search to query movies and view their closest collaborative matches.
* **New User Sandbox**: Search and rate movies on the fly to get real-time recommendations.
* **Metrics Charts**: Beautiful bar chart visualizing model accuracy and ranking performance.

---

## 📊 Evaluation & Comparative Analysis

The models are evaluated on the test set using a stratified split (80/20). Ranking metrics are computed at $K=10$ with a relevance threshold of $\ge 3.5$.

### Model Comparison Table

| Model | RMSE | MAE | MAP@10 | Precision@10 | Recall@10 | NDCG@10 | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVD** | **0.8113** | **0.6338** | 0.3581 | 0.4560 | 0.0472 | 0.5018 | **0.4154** |
| **User-CF** | 0.8478 | 0.6626 | 0.3949 | 0.5080 | 0.0547 | 0.5453 | 3.6609 |
| **Item-CF** | 0.9470 | 0.7340 | 0.3442 | 0.4900 | 0.0545 | 0.5028 | 1.8638 |
| **Hybrid** | 0.8473 | 0.6600 | **0.4407** | **0.5380** | **0.0570** | **0.5817** | 2.0575 |

*Note: SVD yields the lowest error metrics and fastest inference, while the SVD-ItemCF Hybrid delivers superior ranking metrics (MAP@10 and NDCG@10) on the real Netflix Prize Dataset.*


---

## 🧠 Recommendation Paradigms & Architecture

### 1. SVD (Matrix Factorization)
Decomposes the sparse rating matrix into low-rank user and item embeddings. It leverages optimization techniques to prevent overfitting and captures latent concepts (e.g. genre combinations, directors, pacing).
* **Tuning**: Grid search over factors, learning rate, and regularization.
* **Pros**: Rapid predictions, low error metrics.
* **Cons**: Explanations are not intuitive (latent space).

### 2. User-Based Collaborative Filtering (User-CF)
Locates similar users based on rating profiles using cosine similarity. Predictions are calculated using a mean-centered deviation weighted average:
$$P_{u,i} = \bar{R}_u + \frac{\sum_{v \in N} \text{sim}(u, v) \cdot (R_{v,i} - \bar{R}_v)}{\sum_{v \in N} |\text{sim}(u, v)|}$$
* **Optimization**: CSR matrix converted to CSC format for fast column-slicing (fetching ratings for movie $i$ across similar users), speeding up inference $1000\times$.

### 3. Item-Based Collaborative Filtering (Item-CF)
Finds similar movies using rating vector cosine similarities. Predictions use weighted averages:
$$P_{u,i} = \frac{\sum_{j \in N} \text{sim}(i, j) \cdot R_{u,j}}{\sum_{j \in N} |\text{sim}(i, j)|}$$
* **Pros**: Highly explainable ("Because you watched..."), stable similarities (items do not change tastes).

### 4. SVD-ItemCF Hybrid Model
Combines the strong prediction capabilities of SVD with the ranking/similarity structure of Item-CF:
$$\text{Score}_{u,i} = \alpha \cdot \text{Pred}_{\text{SVD}}(u,i) + (1-\alpha) \cdot \text{Pred}_{\text{ItemCF}}(u,i)$$
* **Alpha Tuning**: Tuned to $\alpha = 0.5$, which balances error-minimization and similarity ranking to maximize MAP@10 and NDCG@10.

---

## ❄️ Cold-Start Handling Strategy

A production recommendation system must handle users with limited or no prior activity:

1. **Zero Ratings (Absolute Cold-Start)**:
   * **Strategy**: Popularity-based fallback. Recommends the highest-rated movies with substantial support (minimum vote count) that are also currently popular.
   * **Relevance**: Prevents empty recommendations and serves as a robust default.
2. **Sparse Profiles (1–5 Ratings)**:
   * **Strategy**: Item-CF fallback. Since User-CF struggles with sparse profiles, the system utilizes Item-CF to recommend movies similar to the 1–5 items the user has rated.
   * **Relevance**: Provides immediate personalization after the user rates their first movie.

---

## 💎 Verification & Quality Assurance

* Run `verify_pipeline.py` to assert model formats, size constraints, notebook execution cell states, and metric bounds.
* Figures are saved in high resolution (150 DPI) under `report_figures/`.
* Serialized files use standard Python pickle protocol for modular loading in inference services.
