import os
import pandas as pd
import numpy as np
from src.config import CONFIG, DATA_DIR

EXPECTED_COLS = [
    "Diabetes_012", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]

def find_dataset():
    """Look for any CSV file inside the data directory."""
    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        # Fallback to check parent directory
        parent_csv = [f for f in os.listdir(os.path.dirname(DATA_DIR)) if f.lower().endswith(".csv")]
        if parent_csv:
            return os.path.join(os.path.dirname(DATA_DIR), parent_csv[0])
        raise FileNotFoundError(
            f"No CSV dataset found in '{DATA_DIR}'. Please download the 'CDC Diabetes Health Indicators' dataset CSV and place it there."
        )
    return os.path.join(DATA_DIR, csv_files[0])

def load_and_validate_data(file_path):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Check columns
    missing_cols = [col for col in EXPECTED_COLS if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Missing expected columns: {missing_cols}")
    
    # Convert types to int
    for col in df.columns:
        if col in EXPECTED_COLS:
            df[col] = df[col].astype(int)
            
    # Reporting duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    # Reporting missing values
    na_counts = df.isna().sum()
    print(f"Missing values found:\n{na_counts[na_counts > 0]}")
    
    return df

def preprocess_features(df):
    df = df.copy()
    
    # Clip BMI at 1st and 99th percentiles for outliers
    bmi_lo, bmi_hi = df["BMI"].quantile([0.01, 0.99]).astype(int)
    print(f"Clipped BMI bounds: 1st percentile={bmi_lo}, 99th percentile={bmi_hi}")
    df["BMI_clipped"] = df["BMI"].clip(lower=bmi_lo, upper=bmi_hi)
    
    # Target binarization
    include_prediabetes = CONFIG["include_prediabetes_as_diabetic"]
    if include_prediabetes:
        df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)
    else:
        df["Diabetes_binary"] = (df["Diabetes_012"] == 2).astype(int)
        
    return df
