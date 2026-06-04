# CDC Diabetes Risk Classification & Clustering Pipeline

This repository implements a complete end-to-end data science pipeline to analyze public health indicators from the CDC's **Behavioral Risk Factor Surveillance System (BRFSS)**. The project combines **unsupervised clustering** (K-Means and DBSCAN) with **supervised machine learning models** (Logistic Regression and HistGradientBoosting) to segment patient risk and classify diabetes status.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Data Description & Features](#-data-description--features)
3. [System Architecture](#-system-architecture)
4. [Methodology & Models](#-methodology--models)
5. [Key Performance & Results](#-key-performance--results)
6. [Visualizations & Graphs](#-visualizations--graphs)
7. [Installation & Usage](#-installation--usage)

---

## 🎯 Project Overview
Diabetes is one of the leading chronic diseases globally, carrying heavy physical and economic burdens. Early risk detection is vital for preventive clinical interventions. 

This project explores the relationships between diabetes indicators (like blood pressure, cholesterol, BMI, activity, and age) and builds:
1. An **Unsupervised Risk Segmentation**: Identifies patient subgroups with varying diabetes rates and characteristics.
2. A **Supervised Predictive Classifier**: Predicts whether an individual is diabetic or pre-diabetic based on behavioral and health indicators, with a strong focus on addressing class imbalance and threshold optimization.
3. A **Subgroup Fairness and Calibration Assessment**: Verifies model performance consistency across demographics (age, sex) and calibrates output probabilities for reliable risk estimation.

---

## 📊 Data Description & Features

This project utilizes the **CDC Diabetes Health Indicators Dataset** (BRFSS 2015). Key variables include:

*   **Target**: `Diabetes_binary` (0 = No Diabetes, 1 = Diabetic/Pre-diabetic).
*   **Demographics**: `Age` (13-level category), `Sex`, `Education` (1–6 scale), `Income` (1–8 scale).
*   **Clinical Indicators**: `HighBP` (High blood pressure), `HighChol` (High cholesterol), `CholCheck` (Cholesterol check frequency), `BMI_clipped` (Body Mass Index capped at 1st/99th percentile to exclude extreme outliers).
*   **Behavioral Risk Factors**: `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits` & `Veggies` intake, `HvyAlcoholConsump`.
*   **Self-Reported Health**: `GenHlth` (1–5 scale), `MentHlth` (bad mental health days), `PhysHlth` (bad physical health days), `DiffWalk` (difficulty walking).

---

## 🛠 System Architecture

The pipeline is refactored into a modular, production-grade Python package:

```text
CDC/
├── data/                    # Put raw CSV dataset files here (ignored by git)
├── outputs/                 # Auto-generated prediction CSVs, reports, and plots
├── src/                     # Core codebase
│   ├── __init__.py
│   ├── config.py            # Global run configuration parameters and thresholds
│   ├── data_preprocessing.py# Dataset ingestion, clinical cleaning, and target mapping
│   ├── clustering.py        # K-Means/DBSCAN training, evaluation, and stability sweeps
│   └── classification.py    # Supervised training, threshold sweeps, calibration, & fairness
├── main.py                  # End-to-end orchestration runner
├── CDC_Dataset.ipynb        # Demonstration Jupyter Notebook
├── requirements.txt         # Package dependencies
└── README.md                # This documentation
```

---

## 🧪 Methodology & Models

### 1. Data Processing
*   **Outlier Treatment**: Extreme BMI measurements are clipped at the 1st and 99th percentiles (typically 18 to 43) to prevent skewing clustering distances.
*   **Imbalance Handling**: Standard training uses sample weights (`class_weight="balanced"`) to handle the heavy majority class skew. The pipeline also supports Random Under-sampling or Over-sampling.

### 2. Unsupervised Clustering
*   **K-Means**: Sweeps $k$ from 2 to 10. Optimal $k$ is automatically selected using the **Silhouette Coefficient**, **Inertia**, and **Davies-Bouldin index**. Stability is verified via **Adjusted Rand Index (ARI)** over multiple random seeds.
*   **DBSCAN**: Implemented in PCA-reduced space. The number of components is selected via **Fraction of Cumulative Explained Variance Ratio (FCEVR)** reaching $\ge 95\%$ variance. Parameters `eps` and `min_samples` are tuned via grid-search.

### 3. Supervised Classification
*   **Baseline**: A `DummyClassifier` predicting the most frequent class.
*   **Logistic Regression**: Tuned with Grid Search (l1, l2, elasticnet penalties).
*   **HistGradientBoosting**: Randomized search over learning rate, tree depth, and leaf configurations with early stopping.
*   **Threshold Selection**: Decision thresholds are tuned on a dedicated **Validation Set** to maximize the F1-Score rather than using default 0.5 rules.
*   **Calibration**: Post-hoc probability calibration (Sigmoid curve via `CalibratedClassifierCV`) aligns predicted scores with empirical outcomes, verified by Brier Score drops.

---

## 📈 Key Performance & Results

When executed, the pipeline generates comprehensive performance reports in the `outputs/` directory:

1.  **Test Set Metrics Comparison** (`outputs/metrics_test_model_comparison.csv`):
    *   Compares Accuracy, Precision, Recall, F1-Score, and PR-AUC.
    *   *Result*: Gradient boosting outperforms the linear baseline, achieving a higher PR-AUC and recall for minority cohorts.
2.  **Cross-Validation Consistency** (`outputs/metrics_cross_validated_trainval.csv`):
    *   Reports mean and standard deviation over 5 stratified folds to verify model stability.
3.  **Brier Score Loss** (`outputs/calibration_brier_comparison.csv`):
    *   *Result*: Calibrating the Gradient Boosting classifier significantly reduces the Brier loss, yielding reliable probabilities for health app integrations.

---

## 🖼 Visualizations & Graphs

The pipeline automatically generates and exports the following figures to the `outputs/` folder:

*   **`target_distribution.png`**: Visualizes class ratios between diabetic and healthy cohorts.
*   **`corr_heatmap.png`**: Heatmap showing physical indicators and target correlations.
*   **`kmeans_silhouette_plot.png` & `kmeans_inertia_plot.png`**: Elbow/silhouette sweeps for cluster selection.
*   **`diabetes_rate_by_cluster.png`**: Showcases risk rates in different K-Means patient clusters.
*   **`perm_importance_hgb_top15.png`**: Directional permutation importance of the top predictors.
*   **`calibration_curve_hgb_calibrated.png`**: Alignment curve showing predicted vs. true probabilities.

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.9+
*   The raw BRFSS dataset (e.g. from Kaggle)

### 1. Setup Environment
Clone the repository, set up a virtual environment, and install dependencies:
```bash
# Navigate to project
cd Projects/CDC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Dataset
1. Download the `diabetes_binary_health_indicators_BRFSS2015.csv` file.
2. Create a `data/` directory and save it there:
```bash
mkdir data
# Copy your CSV file into CDC/data/
```

### 3. Run Pipeline
To run the full preprocessing, clustering, training, validation, and chart generation:
```bash
python main.py
```
Outputs will be written directly to `outputs/`.
 