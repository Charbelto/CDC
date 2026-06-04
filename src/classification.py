import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    make_scorer
)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from src.config import CONFIG, SEED, OUTPUT_DIR
from src.clustering import CLUSTER_FEATURES

def make_resampler(kind: str):
    kind = (kind or "none").lower()
    if kind == "under":
        return RandomUnderSampler(
            sampling_strategy=CONFIG["UNDER_SAMPLING_STRATEGY"],
            random_state=SEED
        )
    if kind == "over":
        return RandomOverSampler(
            sampling_strategy=CONFIG["OVER_SAMPLING_STRATEGY"],
            random_state=SEED
        )
    return "passthrough"

def maybe_sample_train(X_tr, y_tr):
    if not CONFIG["FAST_MODE"]:
        return X_tr, y_tr
    n = min(CONFIG["FAST_SAMPLE_N"], len(X_tr))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=SEED)
    idx_small, _ = next(splitter.split(X_tr, y_tr))
    return X_tr.iloc[idx_small].copy(), y_tr.iloc[idx_small].copy()

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

def plot_confusion(y_true, y_prob, threshold=0.5, name="Model"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title(f"{name} Confusion Matrix (thr={threshold:.2f})")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Text annotations
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center", color="black")
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close()

def plot_roc_pr(y_true, y_prob, name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve — {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_roc_curve.png", dpi=200)
    plt.close()
    
    plt.figure()
    plt.plot(rec, prec, label=f"PR (AUC = {average_precision_score(y_true, y_prob):.3f})")
    plt.title(f"Precision-Recall Curve — {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_pr_curve.png", dpi=200)
    plt.close()

def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = [compute_metrics(y_true, y_prob, float(t)) for t in thresholds]
    return pd.DataFrame(rows).sort_values("f1", ascending=False)

def group_metrics(df_X, y_true, y_prob, group_col, threshold):
    out = []
    tmp = df_X.copy()
    tmp["y_true"] = y_true.values
    tmp["y_prob"] = y_prob
    for g, sub in tmp.groupby(group_col):
        if len(sub) < 200:
            continue
        out.append({
            "group_col": group_col,
            "group": int(g),
            "n": len(sub),
            **compute_metrics(sub["y_true"].values, sub["y_prob"].values, threshold=threshold)
        })
    return pd.DataFrame(out).sort_values("pr_auc", ascending=False)

def error_profile(X_part, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    res = X_part.copy()
    res["y_true"] = y_true.values
    res["y_pred"] = y_pred
    res["y_prob"] = y_prob
    
    fn = res[(res["y_true"] == 1) & (res["y_pred"] == 0)]
    fp = res[(res["y_true"] == 0) & (res["y_pred"] == 1)]
    
    fn_stats = fn[CLUSTER_FEATURES].mean().to_frame("FN_mean")
    fp_stats = fp[CLUSTER_FEATURES].mean().to_frame("FP_mean")
    all_stats = res[CLUSTER_FEATURES].mean().to_frame("Overall_mean")
    
    merged = all_stats.join(fn_stats, how="left").join(fp_stats, how="left")
    return fn, fp, merged

def run_classification_pipeline(df):
    X = df[CLUSTER_FEATURES].copy()
    y = df["Diabetes_binary"].copy()
    
    # Train / Val / Test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=CONFIG["TEST_SIZE"],
        random_state=SEED,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=CONFIG["VAL_SIZE"],
        random_state=SEED,
        stratify=y_trainval
    )
    
    # Class weights for training
    sw_train = compute_sample_weight(class_weight="balanced", y=y_train)
    sw_trainval = compute_sample_weight(class_weight="balanced", y=y_trainval)
    
    # ------------------ Model 1: Dummy Baseline ------------------
    from sklearn.dummy import DummyClassifier
    print("Fitting Dummy baseline classifier...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(X_train, y_train)
    dummy_test_prob = dummy.predict_proba(X_test)[:, 1]
    m_dummy = compute_metrics(y_test, dummy_test_prob, threshold=0.5)
    
    # ------------------ Model 2: Logistic Regression ------------------
    print("Running Logistic Regression grid search...")
    preprocess = ColumnTransformer(
        transformers=[("num", StandardScaler(), CLUSTER_FEATURES)],
        remainder="drop"
    )
    USE_RESAMPLING = CONFIG["RESAMPLING"] in ["under", "over"]
    
    logreg = ImbPipeline(steps=[
        ("prep", preprocess),
        ("sampler", make_resampler(CONFIG["RESAMPLING"])),
        ("clf", LogisticRegression(
            solver="saga",
            class_weight=None if USE_RESAMPLING else "balanced",
            max_iter=400,
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    
    C_grid = [0.01, 0.1, 1, 10]
    param_grid = [
        {"clf__penalty": ["l2"], "clf__C": C_grid},
        {"clf__penalty": ["l1"], "clf__C": C_grid},
        {"clf__penalty": ["elasticnet"], "clf__C": C_grid, "clf__l1_ratio": [0.15, 0.5, 0.85]},
    ]
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X_train_small, y_train_small = maybe_sample_train(X_train, y_train)
    
    search_logreg = GridSearchCV(
        logreg,
        param_grid=param_grid,
        scoring=CONFIG["TUNING_SCORING"],
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    search_logreg.fit(X_train_small, y_train_small)
    best_logreg_params = search_logreg.best_params_
    print(f"Best LogReg params: {best_logreg_params}")
    
    # Fit LogReg on training data
    logreg_best = logreg.set_params(**best_logreg_params)
    logreg_best.fit(X_train, y_train)
    
    # Select threshold on validation set
    logreg_val_prob = logreg_best.predict_proba(X_val)[:, 1]
    thr_table_lr = threshold_sweep(y_val, logreg_val_prob)
    thr_table_lr.to_csv(f"{OUTPUT_DIR}/logreg_threshold_sweep_on_val.csv", index=False)
    logreg_thr = float(thr_table_lr.iloc[0]["threshold"])
    print(f"Selected LogReg threshold (best F1 on validation): {logreg_thr}")
    
    # Refit on train+val
    logreg_final = logreg.set_params(**best_logreg_params)
    logreg_final.fit(X_trainval, y_trainval)
    logreg_test_prob = logreg_final.predict_proba(X_test)[:, 1]
    m_logreg = compute_metrics(y_test, logreg_test_prob, threshold=logreg_thr)
    plot_confusion(y_test, logreg_test_prob, threshold=logreg_thr, name="Logistic Regression")
    plot_roc_pr(y_test, logreg_test_prob, name="Logistic Regression")
    
    with open(f"{OUTPUT_DIR}/logreg_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, (logreg_test_prob >= logreg_thr).astype(int), digits=3))
        
    # ------------------ Model 3: HistGradientBoosting Classifier ------------------
    print("Running HistGradientBoosting classifier search...")
    hgb = HistGradientBoostingClassifier(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    param_dist_hgb = {
        "learning_rate": uniform(0.03, 0.17),
        "max_leaf_nodes": randint(16, 64),
        "max_depth": randint(2, 8),
        "min_samples_leaf": randint(20, 200),
        "l2_regularization": uniform(0.0, 1.0),
        "max_iter": randint(80, 220),
    }
    
    sw_small = compute_sample_weight(class_weight="balanced", y=y_train_small)
    search_hgb = RandomizedSearchCV(
        hgb,
        param_distributions=param_dist_hgb,
        n_iter=12,
        scoring=CONFIG["TUNING_SCORING"],
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        verbose=0
    )
    search_hgb.fit(X_train_small, y_train_small, sample_weight=sw_small)
    best_hgb_params = search_hgb.best_params_
    print(f"Best HGB params: {best_hgb_params}")
    
    # Fit HGB on training data
    hgb_best = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, validation_fraction=0.1, **best_hgb_params)
    hgb_best.fit(X_train, y_train, sample_weight=sw_train)
    
    # Select threshold on validation set
    hgb_val_prob = hgb_best.predict_proba(X_val)[:, 1]
    thr_table_hgb = threshold_sweep(y_val, hgb_val_prob)
    thr_table_hgb.to_csv(f"{OUTPUT_DIR}/hgb_threshold_sweep_on_val.csv", index=False)
    hgb_thr = float(thr_table_hgb.iloc[0]["threshold"])
    print(f"Selected HGB threshold (best F1 on validation): {hgb_thr}")
    
    # Refit HGB on train+val
    hgb_final = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, validation_fraction=0.1, **best_hgb_params)
    hgb_final.fit(X_trainval, y_trainval, sample_weight=sw_trainval)
    hgb_test_prob = hgb_final.predict_proba(X_test)[:, 1]
    m_hgb = compute_metrics(y_test, hgb_test_prob, threshold=hgb_thr)
    plot_confusion(y_test, hgb_test_prob, threshold=hgb_thr, name="HistGradientBoosting")
    plot_roc_pr(y_test, hgb_test_prob, name="HistGradientBoosting")
    
    with open(f"{OUTPUT_DIR}/hgb_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, (hgb_test_prob >= hgb_thr).astype(int), digits=3))
        
    # Compare models
    comparison = pd.DataFrame([
        {"model": "Dummy", **m_dummy},
        {"model": "LogReg", **m_logreg},
        {"model": "HGB", **m_hgb},
    ]).sort_values("pr_auc", ascending=False)
    comparison.to_csv(f"{OUTPUT_DIR}/metrics_test_model_comparison.csv", index=False)
    
    # 5-fold cross validation on trainval
    print("Running 5-fold cross-validation comparisons...")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": make_scorer(f1_score),
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
    }
    
    logreg_for_cv = logreg.set_params(**best_logreg_params)
    cv_logreg = cross_validate(logreg_for_cv, X_trainval, y_trainval, scoring=scoring, cv=cv5, n_jobs=-1)
    cv_hgb = cross_validate(hgb_final, X_trainval, y_trainval, scoring=scoring, cv=cv5, n_jobs=-1)
    
    def cv_summary(cvres, name):
        return {
            "model": name,
            "roc_auc_mean": float(np.mean(cvres["test_roc_auc"])),
            "pr_auc_mean": float(np.mean(cvres["test_pr_auc"])),
            "f1_mean": float(np.mean(cvres["test_f1"])),
            "recall_mean": float(np.mean(cvres["test_recall"])),
            "precision_mean": float(np.mean(cvres["test_precision"])),
        }
        
    cv_table = pd.DataFrame([cv_summary(cv_logreg, "LogReg"), cv_summary(cv_hgb, "HGB")])
    cv_table.to_csv(f"{OUTPUT_DIR}/metrics_cross_validated_trainval.csv", index=False)
    
    # Calibration
    print("Running probability calibration curve analysis...")
    calib = CalibratedClassifierCV(estimator=hgb_final, method="sigmoid", cv=3)
    calib.fit(X_trainval, y_trainval, sample_weight=sw_trainval)
    hgb_cal_test_prob = calib.predict_proba(X_test)[:, 1]
    
    brier_uncal = brier_score_loss(y_test, hgb_test_prob)
    brier_cal = brier_score_loss(y_test, hgb_cal_test_prob)
    
    pd.DataFrame({
        "model": ["HGB_uncal", "HGB_calibrated"],
        "brier": [brier_uncal, brier_cal]
    }).to_csv(f"{OUTPUT_DIR}/calibration_brier_comparison.csv", index=False)
    
    # Plot calibration curves
    for probs, name in [(hgb_test_prob, "HGB (uncal)"), (hgb_cal_test_prob, "HGB (calibrated)")]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        plt.figure()
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"Probability Calibration Curve — {name}")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/calibration_curve_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=200)
        plt.close()
        
    # Group metrics
    print("Running subgroup analyses by demographic factors...")
    sex_logreg = group_metrics(X_test, y_test, logreg_test_prob, group_col="Sex", threshold=logreg_thr)
    sex_hgb = group_metrics(X_test, y_test, hgb_test_prob, group_col="Sex", threshold=hgb_thr)
    sex_logreg.to_csv(f"{OUTPUT_DIR}/group_metrics_sex_logreg.csv", index=False)
    sex_hgb.to_csv(f"{OUTPUT_DIR}/group_metrics_sex_hgb.csv", index=False)
    
    age_logreg = group_metrics(X_test, y_test, logreg_test_prob, group_col="Age", threshold=logreg_thr)
    age_hgb = group_metrics(X_test, y_test, hgb_test_prob, group_col="Age", threshold=hgb_thr)
    age_logreg.to_csv(f"{OUTPUT_DIR}/group_metrics_age_logreg.csv", index=False)
    age_hgb.to_csv(f"{OUTPUT_DIR}/group_metrics_age_hgb.csv", index=False)
    
    # Feature coefficients & importance
    print("Generating feature importances and coefficients...")
    coef = logreg_final.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": CLUSTER_FEATURES, "coef": coef}).sort_values("coef", ascending=False)
    coef_df.to_csv(f"{OUTPUT_DIR}/logreg_coefficients.csv", index=False)
    
    n_pi = min(30000, len(X_test))
    idx_pi = np.random.choice(len(X_test), size=n_pi, replace=False)
    X_pi = X_test.iloc[idx_pi]
    y_pi = y_test.iloc[idx_pi]
    
    pi = permutation_importance(hgb_final, X_pi, y_pi, n_repeats=3, random_state=SEED, n_jobs=-1, scoring="average_precision")
    pi_df = pd.DataFrame({
        "feature": CLUSTER_FEATURES,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)
    pi_df.to_csv(f"{OUTPUT_DIR}/permutation_importance_hgb.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    plt.barh(pi_df["feature"].head(15)[::-1], pi_df["importance_mean"].head(15)[::-1])
    plt.title("Top 15 Permutation Importances (HGB)")
    plt.xlabel("Importance Mean (PR-AUC score drop)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/perm_importance_hgb_top15.png", dpi=200)
    plt.close()
    
    # Save final predictions
    preds_out = X_test.copy()
    preds_out["y_true"] = y_test.values
    preds_out["prob_logreg"] = logreg_test_prob
    preds_out["thr_logreg"] = logreg_thr
    preds_out["pred_logreg"] = (logreg_test_prob >= logreg_thr).astype(int)
    preds_out["prob_hgb"] = hgb_test_prob
    preds_out["thr_hgb"] = hgb_thr
    preds_out["pred_hgb"] = (hgb_test_prob >= hgb_thr).astype(int)
    preds_out["prob_hgb_calibrated"] = hgb_cal_test_prob
    
    preds_out.to_csv(f"{OUTPUT_DIR}/test_predictions_with_probs_and_thresholds.csv", index=False)
    print("Classification evaluation complete!")
