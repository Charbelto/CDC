import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from src.config import CONFIG, OUTPUT_DIR
from src.data_preprocessing import find_dataset, load_and_validate_data, preprocess_features
from src.clustering import prepare_cluster_matrix, run_kmeans_sweep, analyze_kmeans_clusters, run_dbscan
from src.classification import run_classification_pipeline

def run_eda(df):
    print("Generating basic exploratory data analysis (EDA) plots...")
    
    # Save basic summaries
    summary = df.describe().T
    summary.to_csv(f"{OUTPUT_DIR}/summary_stats.csv")
    
    # Target distribution plot
    dist = df["Diabetes_binary"].value_counts().sort_index()
    plt.figure()
    plt.bar(["non-diabetic", "diabetic"], dist.values, color=["#3498db", "#e74c3c"])
    plt.title("Target Variable Distribution (Binary)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=200)
    plt.close()
    
    # BMI clipped histogram
    plt.figure()
    plt.hist(df["BMI_clipped"], bins=40, color="#2ecc71", edgecolor="black")
    plt.title("Clipped BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bmi_hist.png", dpi=200)
    plt.close()
    
    # Diabetes rate by age
    age_rate = df.groupby("Age")["Diabetes_binary"].mean().reset_index()
    age_rate.to_csv(f"{OUTPUT_DIR}/diabetes_rate_by_age.csv", index=False)
    plt.figure()
    plt.plot(age_rate["Age"], age_rate["Diabetes_binary"], marker="o", color="#9b59b6")
    plt.title("Diabetes Rate by Age Category")
    plt.xlabel("Age Category (1-13)")
    plt.ylabel("Diabetes Rate")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/diabetes_rate_by_age.png", dpi=200)
    plt.close()

def main():
    print("=== Starting CDC Diabetes Analysis Pipeline ===")
    
    try:
        dataset_path = find_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo run the pipeline:")
        print("1. Download the CDC Diabetes Health Indicators dataset (e.g. from Kaggle).")
        print("2. Create a 'data/' directory inside the CDC folder.")
        print("3. Place the CSV file in 'CDC/data/diabetes_binary_health_indicators_BRFSS2015.csv'.")
        sys.exit(1)
        
    # 1. Load and validate data
    df_raw = load_and_validate_data(dataset_path)
    
    # 2. Preprocess data
    df_processed = preprocess_features(df_raw)
    
    # 3. Exploratory Data Analysis
    run_eda(df_processed)
    
    # 4. Unsupervised Clustering
    X_scaled, scaler = prepare_cluster_matrix(df_processed)
    best_k, k_table = run_kmeans_sweep(X_scaled)
    print(f"Optimal cluster count found: k={best_k}")
    
    df_clustered = analyze_kmeans_clusters(df_processed, X_scaled, best_k)
    dbscan_results = run_dbscan(X_scaled)
    
    # 5. Supervised Classification
    run_classification_pipeline(df_clustered)
    
    print("\n=== Pipeline Execution Completed Successfully ===")
    print(f"Results and visual charts saved to: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
