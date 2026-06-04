import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.neighbors import NearestNeighbors
from src.config import CONFIG, SEED, OUTPUT_DIR

CLUSTER_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI_clipped", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]

def prepare_cluster_matrix(df):
    X = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def run_kmeans_sweep(X_scaled):
    print("Running K-Means inertia and silhouette sweep...")
    n_sample = min(CONFIG["CLUSTER_SAMPLE_N"], len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), size=n_sample, replace=False)
    X_sample = X_scaled[sample_idx]
    
    rows = []
    for k in CONFIG["K_RANGE"]:
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=4096, n_init="auto")
        labels = km.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)
        dbi = davies_bouldin_score(X_sample, labels)
        chi = calinski_harabasz_score(X_sample, labels)
        
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(sil),
            "davies_bouldin": float(dbi),
            "calinski_harabasz": float(chi)
        })
        
    k_table = pd.DataFrame(rows).sort_values("k")
    k_table.to_csv(f"{OUTPUT_DIR}/kmeans_k_selection_metrics.csv", index=False)
    
    # Plot curves
    plt.figure()
    plt.plot(k_table["k"], k_table["inertia"], marker="o")
    plt.title("K-Means Inertia vs k")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kmeans_inertia_plot.png", dpi=200)
    plt.close()
    
    plt.figure()
    plt.plot(k_table["k"], k_table["silhouette"], marker="o")
    plt.title("K-Means Silhouette Score vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kmeans_silhouette_plot.png", dpi=200)
    plt.close()
    
    best_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
    return best_k, k_table

def analyze_kmeans_clusters(df, X_scaled, k):
    print(f"Fitting final K-Means with k={k}...")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=SEED, batch_size=4096, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["Cluster"] = clusters
    
    # Profile clusters
    profile = df.groupby("Cluster")[CLUSTER_FEATURES + ["Diabetes_binary"]].mean()
    profile.to_csv(f"{OUTPUT_DIR}/cluster_profile_means.csv")
    
    # Sizes and rates
    cluster_sizes = df["Cluster"].value_counts().sort_index().rename("count").reset_index().rename(columns={"index": "Cluster"})
    cluster_diab = df.groupby("Cluster")["Diabetes_binary"].mean().reset_index()
    cluster_summary = cluster_sizes.merge(cluster_diab, on="Cluster").rename(columns={"Diabetes_binary": "diabetes_rate"})
    cluster_summary.to_csv(f"{OUTPUT_DIR}/cluster_summary_sizes_diabetes.csv", index=False)
    
    # Plot diabetes rate
    plt.figure()
    plt.bar(cluster_summary["Cluster"].astype(str), cluster_summary["diabetes_rate"])
    plt.title("Diabetes Prevalence by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Diabetes Rate")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/diabetes_rate_by_cluster.png", dpi=200)
    plt.close()
    
    # Differentiators
    overall_mean = df[CLUSTER_FEATURES].mean()
    overall_std = df[CLUSTER_FEATURES].std().replace(0, 1)
    
    diff_rows = []
    for c in sorted(df["Cluster"].unique()):
        c_mean = df.loc[df["Cluster"] == c, CLUSTER_FEATURES].mean()
        z = (c_mean - overall_mean) / overall_std
        top_pos = z.sort_values(ascending=False).head(5)
        top_neg = z.sort_values(ascending=True).head(5)
        diff_rows.append({
            "Cluster": c,
            "TopHigh(+z)": "; ".join([f"{i}({v:+.2f})" for i, v in top_pos.items()]),
            "TopLow(-z)": "; ".join([f"{i}({v:+.2f})" for i, v in top_neg.items()])
        })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(f"{OUTPUT_DIR}/cluster_top_differentiators.csv", index=False)
    
    # Stability Rand-Index checking
    seeds = [SEED + i for i in range(CONFIG["KMEANS_RESTARTS_FOR_STABILITY"])]
    n_sample = min(CONFIG["CLUSTER_SAMPLE_N"], len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), size=n_sample, replace=False)
    X_sample = X_scaled[sample_idx]
    
    label_runs = []
    for s in seeds:
        km = MiniBatchKMeans(n_clusters=k, random_state=s, batch_size=4096, n_init="auto")
        label_runs.append(km.fit_predict(X_sample))
        
    ari_mat = np.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            ari_mat[i, j] = adjusted_rand_score(label_runs[i], label_runs[j])
            
    ari_df = pd.DataFrame(ari_mat, index=[f"seed_{s}" for s in seeds], columns=[f"seed_{s}" for s in seeds])
    ari_df.to_csv(f"{OUTPUT_DIR}/kmeans_stability_ari_matrix.csv")
    
    plt.figure()
    plt.imshow(ari_mat, aspect="auto")
    plt.title("K-Means Stability (ARI) Heatmap")
    plt.colorbar()
    plt.xticks(range(len(seeds)), [str(s) for s in seeds])
    plt.yticks(range(len(seeds)), [str(s) for s in seeds])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kmeans_stability_ari_heatmap.png", dpi=200)
    plt.close()
    
    return df

def fcevr_choose_n_components(X, threshold=0.95, max_components=20):
    n_max = int(min(max_components, X.shape[1]))
    pca = PCA(n_components=n_max, random_state=SEED)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n = int(np.searchsorted(cum, threshold) + 1)
    return n, evr, cum

def run_dbscan(X_scaled):
    print("Running DBSCAN analysis...")
    n_db = min(CONFIG["DBSCAN_SAMPLE_N"], len(X_scaled))
    idx_db = np.random.choice(len(X_scaled), size=n_db, replace=False)
    X_db = X_scaled[idx_db]
    
    if CONFIG["USE_FCEVR_FOR_DBSCAN_PCA"]:
        n_comp, evr, cum = fcevr_choose_n_components(
            X_db,
            threshold=CONFIG["FCEVR_THRESHOLD"],
            max_components=CONFIG["FCEVR_MAX_COMPONENTS"]
        )
        n_comp = max(2, int(n_comp))
        
        plt.figure()
        plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
        plt.axhline(CONFIG["FCEVR_THRESHOLD"], linestyle="--")
        plt.title("PCA Cumulative Explained Variance Ratio (FCEVR)")
        plt.xlabel("# Components")
        plt.ylabel("Cumulative EVR")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fcevr_cumulative_evr_dbscan.png", dpi=200)
        plt.close()
    else:
        n_comp = CONFIG["DBSCAN_PCA_COMPONENTS"]
        
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_db_pca = pca.fit_transform(X_db)
    
    # Plot k-distance
    nbrs = NearestNeighbors(n_neighbors=10).fit(X_db_pca)
    distances, _ = nbrs.kneighbors(X_db_pca)
    k_dist = np.sort(distances[:, -1])
    
    plt.figure()
    plt.plot(k_dist)
    plt.title("DBSCAN k-Distance Curve")
    plt.xlabel("Points Sorted")
    plt.ylabel("10-th Nearest Neighbor Distance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dbscan_k_distance_pca.png", dpi=200)
    plt.close()
    
    # Parameter sweep
    eps_grid = np.unique(np.round(np.percentile(k_dist, [85, 90, 92, 94, 95, 96, 97, 98]), 3))
    min_samples_grid = CONFIG["DBSCAN_MIN_SAMPLES_GRID"]
    
    rows = []
    for eps in eps_grid:
        for ms in min_samples_grid:
            db = DBSCAN(eps=float(eps), min_samples=int(ms))
            labels = db.fit_predict(X_db_pca)
            unique = set(labels)
            n_clusters = len([u for u in unique if u != -1])
            noise_rate = float(np.mean(labels == -1))
            
            sil = np.nan
            if n_clusters >= 2:
                try:
                    sil = float(silhouette_score(X_db_pca, labels))
                except Exception:
                    pass
            rows.append({
                "eps": float(eps),
                "min_samples": int(ms),
                "n_clusters_ex_noise": n_clusters,
                "noise_rate": noise_rate,
                "silhouette": sil
            })
            
    dbscan_table = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    dbscan_table.to_csv(f"{OUTPUT_DIR}/dbscan_grid_results.csv", index=False)
    
    # Plot heatmap
    eps_list = sorted(dbscan_table["eps"].unique())
    ms_list = sorted(dbscan_table["min_samples"].unique())
    mat = np.full((len(ms_list), len(eps_list)), np.nan)
    for i, ms in enumerate(ms_list):
        for j, eps in enumerate(eps_list):
            v = dbscan_table[(dbscan_table["min_samples"] == ms) & (dbscan_table["eps"] == eps)]["silhouette"]
            if len(v) > 0:
                mat[i, j] = v.iloc[0]
                
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title("DBSCAN Grid Search Silhouette Score Heatmap")
    plt.colorbar()
    plt.xticks(range(len(eps_list)), [str(e) for e in eps_list], rotation=45)
    plt.yticks(range(len(ms_list)), [str(m) for m in ms_list])
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dbscan_silhouette_heatmap.png", dpi=200)
    plt.close()
    
    return dbscan_table
