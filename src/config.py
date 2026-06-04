import os

SEED = 42

# ========= RUN CONFIG =========
CONFIG = {
    # If True: Diabetes_012 in {1,2} -> diabetic (1); 0 -> non-diabetic (0)
    # If False: only Diabetes_012 == 2 -> diabetic (1); 0 or 1 -> non-diabetic (0)
    "include_prediabetes_as_diabetic": True,

    # Train/val/test split
    "TEST_SIZE": 0.20,
    "VAL_SIZE": 0.20,  # applied to remaining train after test split

    # Speed switch: use a stratified sample for tuning/cluster-selection
    "FAST_MODE": True,
    "FAST_SAMPLE_N": 60000,
    "CLUSTER_SAMPLE_N": 60000,
    "DBSCAN_SAMPLE_N": 25000,

    # Clustering
    "K_RANGE": list(range(2, 11)),
    "KMEANS_RESTARTS_FOR_STABILITY": 5,  # stability runs on a sample

    # DBSCAN comparison
    "DBSCAN_PCA_COMPONENTS": 5,
    "DBSCAN_MIN_SAMPLES_GRID": [5, 10, 20],
    "DBSCAN_EPS_GRID": None,  # if None, auto-generate from k-distance percentiles

    # Classification scoring focus (imbalanced-friendly)
    "TUNING_SCORING": "average_precision",  # PR-AUC

    # Imbalance handling
    "RESAMPLING": "none",   # "none", "under", "over"
    "UNDER_SAMPLING_STRATEGY": 0.7,
    "OVER_SAMPLING_STRATEGY": "auto",

    # FCEVR (optional PCA component selection)
    "USE_FCEVR_FOR_DBSCAN_PCA": True,
    "FCEVR_THRESHOLD": 0.95,
    "FCEVR_MAX_COMPONENTS": 20,
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
