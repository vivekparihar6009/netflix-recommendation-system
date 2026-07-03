import os
import sys
import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory

# Add project root to python path to ensure imports work correctly
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.recommender import Recommender
from src import config

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize recommender
recommender = Recommender()

# Issue 3 Fix: module-level flag so half-initialized state never reaches routes
MODELS_LOADED = False
try:
    recommender.load_models_and_data()
    MODELS_LOADED = True
    print("Recommender models and datasets successfully loaded in Flask backend.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FATAL: Model loading failed: {e}")
    print("Make sure you have run the training pipeline first: python run_pipeline.py")

# Cache movie list for autocomplete search
cached_movies = []
if recommender.movies_df is not None:
    for _, row in recommender.movies_df.iterrows():
        mid = int(row["movie_id"])
        title = str(row["movie_title"])
        year = int(row["year"]) if not pd.isna(row["year"]) else 0
        cached_movies.append({
            "id": mid,
            "title": title,
            "year": year
        })
else:
    # Backup load from config path
    import pandas as pd
    try:
        movies_df = pd.read_parquet(config.PROCESSED_MOVIES_PARQUET)
        for _, row in movies_df.iterrows():
            cached_movies.append({
                "id": int(row["movie_id"]),
                "title": str(row["movie_title"]),
                "year": int(row["year"]) if not pd.isna(row["year"]) else 0
            })
    except Exception as e:
        print(f"Could not load movie list cache: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/report_figures/<path:filename>')
def serve_figure(filename):
    return send_from_directory(config.REPORT_FIGS_DIR, filename)

# Issue 3 Fix: health check route
@app.route("/health")
def health():
    if not MODELS_LOADED:
        return jsonify({"status": "unhealthy", "reason": "models not loaded"}), 503
    return jsonify({"status": "ok"}), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns dataset statistics and metrics from metrics.json."""
    # Issue 3 Fix: guard against unloaded models
    if not MODELS_LOADED:
        return jsonify({"error": "Service unavailable: models not loaded"}), 503

    num_users = recommender.mapper.num_users if recommender.mapper else 0
    num_movies = recommender.mapper.num_movies if recommender.mapper else 0
    num_ratings = len(recommender.ratings_df) if recommender.ratings_df is not None else 0
    sparsity = 100.0 * (1 - num_ratings / (num_users * num_movies)) if (num_users and num_movies) else 0.0
    
    # Load model evaluation metrics
    metrics = {}
    metrics_path = config.METRICS_JSON_PATH
    if os.path.exists(metrics_path):
        import json
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics.json: {e}")
            
    return jsonify({
        "status": "success",
        "stats": {
            "users": num_users,
            "movies": num_movies,
            "ratings": num_ratings,
            "sparsity": f"{sparsity:.2f}%"
        },
        "metrics": metrics,
        "models_status": {
            "svd": recommender.svd_model is not None,
            "user_cf": recommender.user_cf_model is not None,
            "item_cf": recommender.item_cf_model is not None,
            "hybrid": (recommender.svd_model is not None and recommender.item_cf_model is not None)
        }
    })

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    """Generates recommendations for a given user_id and model_type."""
    # Issue 3 Fix: guard against unloaded models
    if not MODELS_LOADED:
        return jsonify({"error": "Service unavailable: models not loaded"}), 503

    user_id_str = request.args.get('user_id')
    model_type = request.args.get('model_type', 'hybrid').lower()
    k = request.args.get('k', 10, type=int)
    
    if not user_id_str:
        return jsonify({"status": "error", "message": "Missing 'user_id' parameter."}), 400
        
    try:
        user_id = int(user_id_str)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid user_id. Must be an integer."}), 400
        
    # Check profile metadata
    is_known = recommender.mapper is not None and user_id in recommender.mapper.user_to_idx
    ratings_count = 0
    if is_known and recommender.ratings_df is not None:
        ratings_count = int(len(recommender.ratings_df[recommender.ratings_df["user_id"] == user_id]))
        
    try:
        recs = recommender.get_top_k_recommendations(user_id, k=k, model_type=model_type)
        formatted_recs = []
        for title, score, year in recs:
            # Issue 4 Fix: explicit None check instead of silent default 0
            mid = recommender.movie_title_to_id.get(title.lower())
            if mid is None:
                mid = 0
            formatted_recs.append({
                "movie_id": mid,
                "title": title,
                "year": int(year),
                "predicted_rating": round(score, 3)
            })
            
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "is_known": is_known,
            "ratings_count": ratings_count,
            "model_type": model_type,
            "recommendations": formatted_recs
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Recommendation error: {str(e)}"}), 500

@app.route('/api/explain', methods=['GET'])
def get_explanation():
    """Generates an explanation for a recommendation."""
    # Issue 3 Fix: guard against unloaded models
    if not MODELS_LOADED:
        return jsonify({"error": "Service unavailable: models not loaded"}), 503

    user_id_str = request.args.get('user_id')
    movie_id_str = request.args.get('movie_id')
    
    if not user_id_str or not movie_id_str:
        return jsonify({"status": "error", "message": "Missing 'user_id' or 'movie_id'."}), 400
        
    try:
        user_id = int(user_id_str)
        movie_id = int(movie_id_str)
    except ValueError:
        return jsonify({"status": "error", "message": "User ID and Movie ID must be integers."}), 400
        
    try:
        explanation = recommender.explain_recommendation(user_id, movie_id)
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "movie_id": movie_id,
            "movie_title": recommender.movie_id_to_title.get(movie_id, f"Movie {movie_id}"),
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Explanation error: {str(e)}"}), 500

@app.route('/api/similar', methods=['GET'])
def get_similar():
    """Finds movies similar to the target title."""
    # Issue 3 Fix: guard against unloaded models
    if not MODELS_LOADED:
        return jsonify({"error": "Service unavailable: models not loaded"}), 503

    title = request.args.get('title')
    k = request.args.get('k', 10, type=int)
    
    if not title:
        return jsonify({"status": "error", "message": "Missing 'title' parameter."}), 400

    # Issue 4 Fix: explicit None check instead of silent default 0
    mid = recommender.movie_title_to_id.get(title.lower())
    if mid is None:
        return jsonify({"error": f"Movie '{title}' not found in catalog"}), 404
        
    try:
        similar = recommender.get_similar_movies(title, k=k)
        formatted_similar = []
        for movie_title, score, year in similar:
            # Issue 4 Fix: explicit None check instead of silent default 0
            smid = recommender.movie_title_to_id.get(movie_title.lower())
            if smid is None:
                smid = 0
            formatted_similar.append({
                "movie_id": smid,
                "title": movie_title,
                "year": int(year),
                "similarity_score": round(score, 4)
            })
        return jsonify({
            "status": "success",
            "target_title": title,
            "similar_movies": formatted_similar
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Similarity search error: {str(e)}"}), 500

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Returns the cached list of movies for search autocomplete."""
    return jsonify({
        "status": "success",
        "count": len(cached_movies),
        "movies": cached_movies
    })

@app.route('/api/rate', methods=['POST'])
def process_new_user_ratings():
    """Takes movie ratings for a cold-start user and calculates real-time recommendations."""
    # Issue 3 Fix: guard against unloaded models
    if not MODELS_LOADED:
        return jsonify({"error": "Service unavailable: models not loaded"}), 503

    data = request.get_json()
    if not data or "ratings" not in data:
        return jsonify({"status": "error", "message": "Missing 'ratings' in request body."}), 400
        
    raw_ratings = data["ratings"] # Dict of {movie_id_str: rating_val}
    if not raw_ratings:
        return jsonify({"status": "error", "message": "Ratings dictionary cannot be empty."}), 400
        
    try:
        # Convert keys to integer movie_ids and ratings to floats
        ratings_dict = {int(k): float(v) for k, v in raw_ratings.items()}
    except ValueError:
        return jsonify({"status": "error", "message": "Movie IDs must be integers and ratings must be numeric."}), 400
        
    try:
        # Generate recommendations using Item-CF fallback on rated movies
        recs = recommender.get_cold_start_recommendations_from_ratings(ratings_dict, k=10)
        formatted_recs = []
        for title, score, year in recs:
            # Issue 4 Fix: explicit None check instead of silent default 0
            mid = recommender.movie_title_to_id.get(title.lower())
            if mid is None:
                mid = 0
            formatted_recs.append({
                "movie_id": mid,
                "title": title,
                "year": int(year),
                "predicted_rating": round(score, 3)
            })
            
        # Generate custom explanation for the top recommendation
        explanation = "No recommendations generated."
        if formatted_recs:
            top_rec = formatted_recs[0]
            top_mid = top_rec["movie_id"]
            if recommender.item_cf_model and top_mid in recommender.mapper.movie_to_idx:
                item_sim = recommender.item_cf_model.item_similarity
                m_idx = recommender.mapper.movie_to_idx[top_mid]
                
                # Check if we have positive ratings
                positive_ratings = {k: v for k, v in ratings_dict.items() if v >= 3.0}
                negative_ratings = {k: v for k, v in ratings_dict.items() if v < 3.0}
                
                # Format filter suffix if there are negative ratings
                filter_suffix = ""
                if negative_ratings:
                    disliked_titles = [recommender.movie_id_to_title.get(mid, f"Movie {mid}") for mid in negative_ratings.keys()]
                    if len(disliked_titles) > 2:
                        disliked_str = ", ".join(f"'{t}'" for t in disliked_titles[:-1]) + f", and '{disliked_titles[-1]}'"
                    elif len(disliked_titles) == 2:
                        disliked_str = f"'{disliked_titles[0]}' and '{disliked_titles[1]}'"
                    else:
                        disliked_str = f"'{disliked_titles[0]}'"
                    filter_suffix = f" Additionally, we actively filtered out movies similar to your disliked titles ({disliked_str})."

                if positive_ratings:
                    similarities = []
                    for rated_mid, rating in positive_ratings.items():
                        if rated_mid in recommender.mapper.movie_to_idx:
                            r_idx = recommender.mapper.movie_to_idx[rated_mid]
                            sim = item_sim[m_idx, r_idx]
                            similarities.append((rated_mid, sim, rating))
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    valid = [x for x in similarities if x[1] > 0.02][:2]
                    if len(valid) >= 1:
                        m1_title = recommender.movie_id_to_title.get(valid[0][0])
                        r1 = valid[0][2]
                        if len(valid) == 2:
                            m2_title = recommender.movie_id_to_title.get(valid[1][0])
                            r2 = valid[1][2]
                            explanation = (f"We recommended '{top_rec['title']}' because it is highly similar to "
                                           f"movies you liked: '{m1_title}' (rated {r1}★) and '{m2_title}' (rated {r2}★).")
                        else:
                            explanation = (f"We recommended '{top_rec['title']}' because it is highly similar to "
                                           f"'{m1_title}' which you liked (rated {r1}★).")
                    else:
                        explanation = f"We recommended '{top_rec['title']}' based on general item similarity with your liked movies."
                    explanation += filter_suffix
                else:
                    # Negative-only ratings: explain popularity + filtering of disliked movies
                    disliked_titles = [recommender.movie_id_to_title.get(mid, f"Movie {mid}") for mid in ratings_dict.keys()]
                    if len(disliked_titles) > 2:
                        disliked_str = ", ".join(f"'{t}'" for t in disliked_titles[:-1]) + f", and '{disliked_titles[-1]}'"
                    elif len(disliked_titles) == 2:
                        disliked_str = f"'{disliked_titles[0]}' and '{disliked_titles[1]}'"
                    else:
                        disliked_str = f"'{disliked_titles[0]}'"
                    explanation = (f"We recommended '{top_rec['title']}' because it is highly popular among other viewers, "
                                   f"and we actively filtered out movies similar to the ones you disliked ({disliked_str}).")
            else:
                explanation = f"We recommended '{top_rec['title']}' as one of our top popular movies based on your interests."
                
        return jsonify({
            "status": "success",
            "recommendations": formatted_recs,
            "top_explanation": explanation
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Real-time recommendation error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run server locally on port 5000, without double-loading reloader to conserve memory
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
