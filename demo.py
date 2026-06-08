import os
import sys
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add project root to python path to ensure imports work correctly
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.recommender import Recommender
from src import config

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()} ".center(60, "="))
    print("=" * 60 + "\n")

def search_movies(recommender, query):
    query = query.lower()
    matches = []
    for title, mid in recommender.movie_title_to_id.items():
        if query in title:
            exact_title = recommender.movie_id_to_title.get(mid, title)
            year = recommender.movie_id_to_year.get(mid, 0)
            matches.append((exact_title, mid, year))
    return matches

def display_recs_table(recs, title="Recommendations"):
    table_data = []
    for idx, (movie_title, score, year) in enumerate(recs, 1):
        year_str = f"({year})" if year > 0 else ""
        table_data.append([idx, movie_title, year_str, f"{score:.3f}"])
    
    print(tabulate(table_data, headers=["#", "Movie Title", "Year", "Predicted Rating / Score"], tablefmt="fancy_grid"))

def handle_get_recommendations(recommender):
    print_header("Get Personalized Recommendations")
    print("Sample User IDs from dataset:")
    print("  - 387418 (Highly active user)")
    print("  - 305344 (Highly active user)")
    print("  - 2439493 (Active user)")
    print("  - 99999 (Unknown user / Popularity cold-start fallback)")
    print()
    
    try:
        user_id_input = input("Enter User ID to query: ").strip()
        if not user_id_input:
            print("Operation cancelled.")
            return
        user_id = int(user_id_input)
    except ValueError:
        print("Invalid User ID. Please enter an integer.")
        return
        
    print("\nChoose Model:")
    print("  1. Hybrid SVD-ItemCF (Recommended - Best ranking performance)")
    print("  2. Matrix Factorization (SVD) (Best rating accuracy)")
    print("  3. User-Based Collaborative Filtering")
    print("  4. Item-Based Collaborative Filtering")
    print("  5. Compare All Models side-by-side")
    
    choice = input("\nSelect choice (1-5): ").strip()
    
    model_map = {
        "1": "hybrid",
        "2": "svd",
        "3": "user_cf",
        "4": "item_cf"
    }
    
    # Check if user is known
    is_known = recommender.mapper and user_id in recommender.mapper.user_to_idx
    num_ratings_train = 0
    if is_known:
        num_ratings_train = len(recommender.ratings_df[recommender.ratings_df["user_id"] == user_id])
    
    if choice == "5":
        # Compare all models side-by-side
        print(f"\nGenerating top-10 recommendations for User {user_id} across all models...")
        recs_dict = {}
        for m_key, m_name in model_map.items():
            recs_dict[m_name] = recommender.get_top_k_recommendations(user_id, k=10, model_type=m_name)
            
        # Build comparison table
        table_data = []
        for i in range(10):
            row = [i + 1]
            for m_name in ["hybrid", "svd", "user_cf", "item_cf"]:
                if i < len(recs_dict[m_name]):
                    title, val, yr = recs_dict[m_name][i]
                    row.append(f"{title} ({yr}) [{val:.2f}]")
                else:
                    row.append("-")
            table_data.append(row)
            
        print_header(f"Model Comparison for User {user_id}")
        if is_known:
            print(f"User Profile: Known User ({num_ratings_train} ratings in training set)")
        else:
            print("User Profile: Cold-Start User (0 ratings in training set - popularity fallback)")
            
        print(tabulate(table_data, headers=["#", "Hybrid SVD-ItemCF", "SVD", "User-CF", "Item-CF"], tablefmt="fancy_grid"))
        
        # If user is known, offer explanation for the Hybrid top recommendation
        if is_known and len(recs_dict["hybrid"]) > 0:
            top_hybrid_title = recs_dict["hybrid"][0][0]
            top_hybrid_mid = recommender.movie_title_to_id.get(top_hybrid_title.lower())
            if top_hybrid_mid:
                print("\n💡 Recommendation Explanation:")
                explanation = recommender.explain_recommendation(user_id, top_hybrid_mid)
                print(explanation)
                
    elif choice in model_map:
        m_name = model_map[choice]
        print(f"\nGenerating top-10 recommendations for User {user_id} using model: {m_name.upper()}...")
        recs = recommender.get_top_k_recommendations(user_id, k=10, model_type=m_name)
        
        print_header(f"{m_name.upper()} Recommendations for User {user_id}")
        if is_known:
            print(f"User Profile: Known User ({num_ratings_train} ratings in training set)")
        else:
            if num_ratings_train > 0:
                print(f"User Profile: Sparse Profile ({num_ratings_train} ratings - Item-CF fallback)")
            else:
                print("User Profile: Cold-Start User (0 ratings - Popularity fallback)")
                
        display_recs_table(recs)
        
        # Explain recommendations option
        if recs and (is_known or num_ratings_train > 0):
            print("\nWould you like an explanation for one of these recommendations?")
            exp_choice = input("Enter recommendation number (1-10) to explain, or press Enter to skip: ").strip()
            if exp_choice.isdigit():
                idx = int(exp_choice) - 1
                if 0 <= idx < len(recs):
                    movie_title = recs[idx][0]
                    mid = recommender.movie_title_to_id.get(movie_title.lower())
                    if mid:
                        print("\n💡 Explanation:")
                        explanation = recommender.explain_recommendation(user_id, mid)
                        print(explanation)
                    else:
                        print("Could not resolve movie ID.")
                else:
                    print("Index out of range.")
    else:
        print("Invalid model choice.")

def handle_find_similar_movies(recommender):
    print_header("Find Similar Movies")
    query = input("Enter movie title to search: ").strip()
    if not query:
        return
        
    matches = search_movies(recommender, query)
    if not matches:
        print("No matching movies found in the database. Please try a different query.")
        return
        
    print("\nMatching movies found:")
    for idx, (title, mid, year) in enumerate(matches, 1):
        year_str = f"({year})" if year > 0 else ""
        print(f"  {idx}. {title} {year_str}")
        
    choice = input("\nSelect movie number to find similarities for: ").strip()
    try:
        choice_idx = int(choice) - 1
        if not (0 <= choice_idx < len(matches)):
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input.")
        return
        
    target_title, target_mid, target_year = matches[choice_idx]
    print(f"\nFinding top-10 movies similar to '{target_title}' ({target_year})...")
    
    similar_movies = recommender.get_similar_movies(target_title, k=10)
    
    print_header(f"Movies Similar to: {target_title}")
    if similar_movies:
        table_data = []
        for idx, (title, sim, year) in enumerate(similar_movies, 1):
            year_str = f"({year})" if year > 0 else ""
            table_data.append([idx, title, year_str, f"{sim:.4f}"])
        print(tabulate(table_data, headers=["#", "Movie Title", "Year", "Cosine Similarity"], tablefmt="fancy_grid"))
    else:
        print("No similar movies found. (Check if Item-CF model is loaded).")

def handle_new_user_personalization(recommender):
    print_header("Real-Time Personalization for New User")
    print("This mode simulates a new user registering and rating a few movies.")
    print("Our Item-Based Collaborative Filtering model will compute recommendations")
    print("in real-time using similarity vectors weighted by your ratings.")
    print("-" * 60)
    
    rated_movies = {} # Dict of {movie_id: rating}
    
    while True:
        print(f"\nYou have rated {len(rated_movies)} movies.")
        query = input("Search for a movie you've watched (or press Enter to finish and get recs): ").strip()
        if not query:
            if not rated_movies:
                print("No ratings provided. Personalization cancelled.")
                return
            break
            
        matches = search_movies(recommender, query)
        if not matches:
            print("No matching movies found in the database.")
            continue
            
        print("\nMatches found:")
        for idx, (title, mid, year) in enumerate(matches[:10], 1):
            year_str = f"({year})" if year > 0 else ""
            print(f"  {idx}. {title} {year_str}")
            
        choice = input("\nSelect movie number (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            continue
            
        try:
            choice_idx = int(choice) - 1
            if not (0 <= choice_idx < min(len(matches), 10)):
                print("Invalid choice.")
                continue
        except ValueError:
            print("Invalid input.")
            continue
            
        target_title, target_mid, target_year = matches[choice_idx]
        
        if target_mid in rated_movies:
            print(f"You have already rated '{target_title}'.")
            continue
            
        rating_input = input(f"Rate '{target_title}' (1.0 to 5.0 stars): ").strip()
        try:
            rating = float(rating_input)
            if not (1.0 <= rating <= 5.0):
                print("Rating must be between 1.0 and 5.0.")
                continue
        except ValueError:
            print("Invalid rating format.")
            continue
            
        rated_movies[target_mid] = rating
        print(f"Added rating: {target_title} -> {rating}★")
        
        if len(rated_movies) >= 5:
            finish = input("\nYou have rated 5 movies. Generate recommendations now? (y/n): ").strip().lower()
            if finish == 'y':
                break
                
    print("\nGenerating personalized recommendations using real-time Item-CF profile updates...")
    # Generate recommendations using recommender's cold-start method for sparse profiles
    recs = recommender.get_cold_start_recommendations_from_ratings(rated_movies, k=10)
    
    print_header("Your Personalized Recommendations")
    display_recs_table(recs)
    
    # Explanation
    if recs:
        # Show explanation for the top recommended movie based on rated items
        top_rec_title = recs[0][0]
        top_rec_mid = recommender.movie_title_to_id.get(top_rec_title.lower())
        if top_rec_mid:
            print("\n💡 Why did I get these recommendations?")
            # We can mock/compute explanation since this user isn't in train
            # Let's find similarities to the rated movies
            item_sim = recommender.item_cf_model.item_similarity
            m_idx = recommender.mapper.movie_to_idx[top_rec_mid]
            similarities = []
            for rated_mid, rating in rated_movies.items():
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
                    print(f"We recommended '{top_rec_title}' because it has high cosine similarity with "
                          f"movies you rated highly: '{m1_title}' (rated {r1}★) and '{m2_title}' (rated {r2}★).")
                else:
                    print(f"We recommended '{top_rec_title}' because it has high cosine similarity with "
                          f"'{m1_title}' which you rated {r1}★.")
            else:
                print(f"We recommended '{top_rec_title}' based on similarity with your rated movies.")

def handle_show_metrics():
    print_header("Model Evaluation Metrics")
    metrics_path = os.path.join(project_dir, 'results', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            import json
            metrics = json.load(f)
        df_metrics = pd.DataFrame(metrics).T
        print(tabulate(df_metrics, headers="keys", tablefmt="fancy_grid"))
        print("\nNotes:")
        print("  - Relevance Threshold: Ratings >= 3.5 are considered relevant.")
        print("  - Metrics are evaluated on the 20% held-out stratified test set.")
        print("  - SVD performs best in rating prediction (RMSE/MAE).")
        print("  - Hybrid SVD-ItemCF performs best in ranking quality (MAP/NDCG).")
    else:
        print("Metrics file not found. Please run verify_pipeline.py or python run_pipeline.py first.")

def main():
    clear_screen()
    print_header("Netflix Prize Recommendation System Demo")
    
    # Initialize recommender
    recommender = Recommender()
    try:
        recommender.load_models_and_data()
    except Exception as e:
        print(f"\n❌ Error loading models/data: {e}")
        print("Please make sure you have executed the pipeline first:")
        print("  python run_pipeline.py")
        sys.exit(1)
        
    num_users = recommender.mapper.num_users
    num_movies = recommender.mapper.num_movies
    num_ratings = len(recommender.ratings_df)
    
    print("\nDataset Statistics:")
    print(f"  - Active Users: {num_users:,}")
    print(f"  - Popular Movies: {num_movies:,}")
    print(f"  - Total Ratings: {num_ratings:,}")
    print(f"  - Interaction Sparsity: {100.0 * (1 - num_ratings / (num_users * num_movies)):.2f}%")
    print("\nModels Successfully Loaded:")
    print(f"  - SVD (Matrix Factorization)  : {'[OK]' if recommender.svd_model else '[Missing]'}")
    print(f"  - User-CF (User-Based CF)     : {'[OK]' if recommender.user_cf_model else '[Missing]'}")
    print(f"  - Item-CF (Item-Based CF)     : {'[OK]' if recommender.item_cf_model else '[Missing]'}")
    print(f"  - Hybrid SVD-ItemCF Recommender: {'[OK]' if (recommender.svd_model and recommender.item_cf_model) else '[Missing]'}")
    
    input("\nPress Enter to continue to the main menu...")
    
    while True:
        clear_screen()
        print_header("Netflix Prize Recommendation System")
        print("1. Get Personalized Recommendations for User")
        print("2. Find Similar Movies (Cosine Similarity)")
        print("3. Simulated New User Personalization (Real-Time)")
        print("4. View Model Evaluation Metrics & Comparison")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            handle_get_recommendations(recommender)
        elif choice == "2":
            handle_find_similar_movies(recommender)
        elif choice == "3":
            handle_new_user_personalization(recommender)
        elif choice == "4":
            handle_show_metrics()
        elif choice == "5":
            print("\nThank you for using the Netflix Prize Recommendation System! Goodbye.\n")
            break
        else:
            print("Invalid selection. Press Enter to retry...")
            
        input("\nPress Enter to return to main menu...")

if __name__ == '__main__':
    main()
