import os
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

def run_notebook(notebook_path):
    print(f"\n==================================================")
    print(f"Executing: {os.path.basename(notebook_path)}")
    print(f"==================================================")
    start_time = time.time()
    
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        # Configure execution preprocessor
        # Set timeout to 1200 seconds (20 mins) per cell to avoid timeout errors
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
        
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        # Save the executed notebook in-place
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        elapsed = time.time() - start_time
        print(f"Success: {os.path.basename(notebook_path)} executed in {elapsed:.2f}s")
        return True, None
        
    except CellExecutionError as e:
        elapsed = time.time() - start_time
        print(f"ERROR executing {os.path.basename(notebook_path)} after {elapsed:.2f}s:")
        print(e)
        return False, e
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Unexpected error executing {os.path.basename(notebook_path)} after {elapsed:.2f}s:")
        print(e)
        return False, e

def main():
    notebooks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
    notebooks = [
        "01_data_loading_and_preprocessing.ipynb",
        "02_exploratory_data_analysis.ipynb",
        "03_model_svd.ipynb",
        "04_model_collaborative_filtering.ipynb",
        "05_model_comparison_and_evaluation.ipynb",
        "06_recommendation_generation.ipynb"
    ]
    
    results = []
    for nb_name in notebooks:
        nb_path = os.path.join(notebooks_dir, nb_name)
        success, error = run_notebook(nb_path)
        results.append((nb_name, success, error))
        if not success:
            print("\nPipeline failed. Stopping execution.")
            sys.exit(1)
            
    print("\n==================================================")
    print("ALL NOTEBOOKS EXECUTED SUCCESSFULLY!")
    print("==================================================")
    for nb_name, success, _ in results:
        print(f" - {nb_name}: {'SUCCESS' if success else 'FAILED'}")

if __name__ == '__main__':
    main()
