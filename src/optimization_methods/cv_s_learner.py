import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from evaluation_metrics.calculate_qini import calculate_qini

def cv_s_learner(model, X, y, treatment, r_state = 42, n_splits = 5):
    """
    Cross-validation for S-Learner models.
    Trains single model with treatment as a feature.
    
    :param model: Sklearn-compatible classifier.
    :param X: Feature matrix (numpy array).
    :param y: Target array.
    :param treatment: Treatment indicator array.
    :param n_splits: Number of CV folds.

    Returns: Array of Qini scores for each fold.
    """
    strat_key = treatment.astype(str) + "_" + y.astype(str)
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = r_state)
    
    qini_scores = []
    
    for train_idx, val_idx in skf.split(X, strat_key):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        treatment_train_fold, treatment_val_fold = treatment[train_idx], treatment[val_idx]
        
        # Add treatment as feature
        X_train_with_t = np.column_stack([X_train_fold, treatment_train_fold])
        
        model_clone = clone(model)
        model_clone.fit(X_train_with_t, y_train_fold)
        
        # Predict under both treatment scenarios
        X_val_treated = np.column_stack([X_val_fold, np.ones(len(X_val_fold))])
        X_val_control = np.column_stack([X_val_fold, np.zeros(len(X_val_fold))])
        
        prob_treated = model_clone.predict_proba(X_val_treated)[:, 1]
        prob_control = model_clone.predict_proba(X_val_control)[:, 1]
        
        # Uplift score
        scores = prob_treated - prob_control
        
        qini = calculate_qini(y_val_fold, treatment_val_fold, scores)
        qini_scores.append(qini)
    
    return np.array(qini_scores)
