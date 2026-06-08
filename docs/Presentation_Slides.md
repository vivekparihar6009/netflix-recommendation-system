# Presentation: Personalized Content Discovery Engine
**Dataset**: Netflix Prize Dataset (Subset)  
**Authors**: Machine Learning Engineering Team  
**Date**: June 8, 2026  

---

## Slide 1: Problem Overview & Streaming Platform Motivation
### The Business Challenge
* **Objective**: Connect users with relevant content to maximize session length, viewer retention, and user engagement.
* **The Opportunity**: Recommendation engines directly power modern digital streaming services (e.g., Netflix, YouTube).
* **The Netflix Prize Dataset Benchmark**: 
  * Originally released to improve movie rating prediction accuracy.
  * Contains 100M+ ratings across 480k users and 17.7k movies.
* **Sub-Matrix Scope**: We built a high-performance recommendation pipeline on a sub-matrix of **5,000 top active users** and **2,000 popular movies** (representing **5.35M interactions**).

---

## Slide 2: Data Understanding & EDA Insights
### Understanding the Data Dimensions
* **Positive Rating Skew**: Average rating is **3.60 stars**, with **4.0 stars** being the most popular choice. Baseline predictors must account for user rating biases.
* **Log-Scale Power Law Distribution**: Movie popularity exhibits a heavy-tailed distribution, where top blockbusters receive the vast majority of interactions.
* **Interaction Matrix Density**: Our sub-matrix is **53.53% dense** (sparsity is **46.47%**), providing a robust neighborhood matrix for collaborative filtering models.

---

## Slide 3: Recommendation Methodology & Models
### Three Core Paradigms
1. **Matrix Factorization (SVD)**:
   * Maps users and movies to a shared latent space (e.g., 100 dimensions).
   * Models rating magnitude accurately: $\hat{R}_{u,i} = \mu + b_u + b_i + p_u^T q_i$.
2. **User-Based Collaborative Filtering (User-CF)**:
   * Locates similar users using cosine similarity and deviation vectors.
3. **Item-Based Collaborative Filtering (Item-CF)**:
   * Recommends movies similar to user's highly rated titles using movie-movie cosine similarity.

---

## Slide 4: Hybrid Recommender Design
### Blending Strengths for Content Discovery
* **The Trade-Off**: SVD excels at prediction accuracy (RMSE), while Item-CF excels at ranking relevance.
* **Weighted Ensemble Formula**:
  $$\text{Score}_{u,i} = \alpha \cdot \hat{R}_{\text{SVD}}(u,i) + (1-\alpha) \cdot \hat{R}_{\text{ItemCF}}(u,i)$$
* **Hyperparameter Tuning**: Blending coefficient set to **$\alpha = 0.5$** to optimize precision and mean average precision.
* **Explainable Recommendations**: Neighborhood models provide natural language explanations (e.g., *"We recommended Movie X because you rated similar Movie Y highly"*).

---

## Slide 5: Experimental Evaluation Results
### Comparison of Recommendation Algorithms on Test Set
* Stratified 80/20 train-test split (ensuring test users are in training).
* Relevance threshold for ranking: rating **$\ge 3.5$ stars**.

| Model | RMSE (↓) | MAE (↓) | MAP@10 (↑) | Precision@10 (↑) | NDCG@10 (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVD** | **0.8113** | **0.6338** | 0.3581 | 0.4560 | 0.5018 |
| **User-CF** | 0.8478 | 0.6626 | 0.3949 | 0.5080 | 0.5453 |
| **Item-CF** | 0.9470 | 0.7340 | 0.3442 | 0.4900 | 0.5028 |
| **Hybrid** | 0.8473 | 0.6600 | **0.4407** | **0.5380** | **0.5817** |

---

## Slide 6: Key Insights: RMSE vs. MAP@10
### The Prediction-Ranking Paradox
* **SVD minimizes prediction error** (RMSE = 0.8113) but underperforms in ranking relevance (MAP@10 = 0.3581).
* **Hybrid Model delivers the best ranking metrics** (MAP@10 = 0.4407, NDCG@10 = 0.5817).
* **Relative Gain**: Ensembling SVD and Item-CF similarities achieves a **23% relative improvement** in recommendation ranking performance.
* **Takeaway**: Minimizing RMSE is crucial for rating prediction, but ranking optimization (MAP@10/NDCG@10) is the metric that governs actual content discovery quality.

---

## Slide 7: Recommendation Case Studies
### Success Case: Coherent Interests
* *User Profile*: User rated action/sci-fi titles highly.
* *Recommendations*: Top titles included *Lord of the Rings*, *Blade Runner*, and *Indiana Jones*.
* *Outcome*: Accurate alignment between latent factor representations and item neighborhood similarities.

### Failure Case: Sparse/Conflicting Profile
* *User Profile*: User rated romantic comedy high and action adventure low.
* *Recommendations*: System recommended general blockbusters like *Gladiator*.
* *Outcome*: Variance in sparse profiles causes neighborhood projection overlap, forcing the system to fall back on global popularity biases.

---

## Slide 8: Future Directions & Deployment
### Interactive Web Dashboard & Real-Time APIs
* **Production Sandbox**: Enables new users to rate movies on the fly and view real-time recommendations.
* **Explainability Modals**: Shows the underlying item similarity matches.
* **Cold-Start Fallbacks**: Robust Bayesian popular default when zero ratings exist, transitioning to Item-CF filtering when sparse ratings are collected.

### Next Steps for Scalability
1. **Temporal Decay**: Discount older ratings to track user taste drift.
2. **Implicit Signals**: Blend ratings with implicit features (clicks, watch duration).
3. **Deep Latent Models**: Transition to Neural Collaborative Filtering.
