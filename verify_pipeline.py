import os
import json
import pandas as pd
import nbformat

def check_file_exists(path):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    return exists, size

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"==================================================")
    print(f"VERIFYING NETFLIX RECOMMENDATION SYSTEM PIPELINE")
    print(f"==================================================")
    print(f"Project Directory: {project_dir}\n")

    # 1. Check Preprocessed Data Files
    data_files = [
        "processed_ratings.parquet",
        "processed_movies.parquet",
        "train_ratings.parquet",
        "test_ratings.parquet"
    ]
    print("[1] Checking preprocessed data files...")
    data_ok = True
    for f in data_files:
        path = os.path.join(project_dir, 'data', f)
        exists, size = check_file_exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            data_ok = False
        print(f"  - {f}: {status} ({size:,} bytes if OK)")
    print(f"Data files verification: {'PASSED' if data_ok else 'FAILED'}\n")

    # 2. Check Report Figures
    figures = [
        "01_rating_distribution.png",
        "02_ratings_per_user.png",
        "03_ratings_per_movie.png",
        "04_top_20_movies.png",
        "05_rating_trends.png",
        "06_sparsity_heatmap.png",
        "07_movie_averages.png",
        "08_user_activity_heatmap.png",
        "09_svd_learning_curves.png",
        "10_user_similarity_heatmap.png",
        "11_item_similarity_heatmap.png",
        "12_model_comparison.png"
    ]
    print("[2] Checking report figures in report_figures/...")
    figs_ok = True
    for fig in figures:
        path = os.path.join(project_dir, 'report_figures', fig)
        exists, size = check_file_exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            figs_ok = False
        print(f"  - {fig}: {status} ({size:,} bytes if OK)")
    print(f"Figures verification: {'PASSED' if figs_ok else 'FAILED'}\n")

    # 3. Check Models
    models = [
        "index_mapper.pkl",
        "svd_model.pkl",
        "user_cf.pkl",
        "item_cf.pkl"
    ]
    print("[3] Checking serialized model files in results/...")
    models_ok = True
    for m in models:
        path = os.path.join(project_dir, 'results', m)
        exists, size = check_file_exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            models_ok = False
        print(f"  - {m}: {status} ({size:,} bytes if OK)")
    print(f"Models verification: {'PASSED' if models_ok else 'FAILED'}\n")

    # 4. Check Output Files
    outputs = [
        "metrics.json",
        "sample_recommendations.csv"
    ]
    print("[4] Checking output files in results/...")
    outputs_ok = True
    for out in outputs:
        path = os.path.join(project_dir, 'results', out)
        exists, size = check_file_exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            outputs_ok = False
        print(f"  - {out}: {status} ({size:,} bytes if OK)")
    print(f"Output files verification: {'PASSED' if outputs_ok else 'FAILED'}\n")

    # 5. Check Notebook Execution
    notebooks = [
        "01_data_loading_and_preprocessing.ipynb",
        "02_exploratory_data_analysis.ipynb",
        "03_model_svd.ipynb",
        "04_model_collaborative_filtering.ipynb",
        "05_model_comparison_and_evaluation.ipynb",
        "06_recommendation_generation.ipynb"
    ]
    print("[5] Checking notebooks execution state...")
    notebooks_ok = True
    for nb_name in notebooks:
        path = os.path.join(project_dir, 'notebooks', nb_name)
        exists, _ = check_file_exists(path)
        if not exists:
            print(f"  - {nb_name}: MISSING")
            notebooks_ok = False
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        # Check if notebook has code cells that have run (meaning execution count is not None)
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        executed_cells = [cell for cell in code_cells if cell.get('execution_count') is not None]
        
        if len(code_cells) == 0:
            print(f"  - {nb_name}: WARNING (No code cells)")
        else:
            pct_run = (len(executed_cells) / len(code_cells)) * 100
            print(f"  - {nb_name}: OK (Executed {len(executed_cells)}/{len(code_cells)} code cells, {pct_run:.1f}%)")
            if len(executed_cells) == 0:
                notebooks_ok = False
    print(f"Notebooks verification: {'PASSED' if notebooks_ok else 'FAILED'}\n")

    # 6. Read and Print Evaluation Metrics
    metrics_path = os.path.join(project_dir, 'results', 'metrics.json')
    if os.path.exists(metrics_path):
        print("[6] Reading final evaluation metrics...")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        df_metrics = pd.DataFrame(metrics).T
        print(df_metrics.to_markdown())
        print()
    else:
        print("[6] WARNING: metrics.json is missing, cannot show final comparison table.\n")

    # 7. Read and Print Sample Recommendations
    recs_path = os.path.join(project_dir, 'results', 'sample_recommendations.csv')
    if os.path.exists(recs_path):
        print("[7] Reading sample recommendations...")
        recs_df = pd.read_csv(recs_path)
        print(f"Loaded {len(recs_df)} sample recommendations.")
        print(f"Unique users recommended to: {recs_df['user_id'].nunique()}")
        print(recs_df.head(15).to_markdown(index=False))
        print()
    else:
        print("[7] WARNING: sample_recommendations.csv is missing.\n")

    # Overall Summary
    all_passed = data_ok and figs_ok and models_ok and outputs_ok and notebooks_ok
    print(f"==================================================")
    print(f"OVERALL PIPELINE STATUS: {'PASSED' if all_passed else 'FAILED'}")
    print(f"==================================================")

if __name__ == '__main__':
    main()
