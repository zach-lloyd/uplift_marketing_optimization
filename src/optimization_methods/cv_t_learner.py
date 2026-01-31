import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from evaluation_metrics.calculate_qini import calculate_qini

def cv_t_learner(model, X, y, treatment, r_state = 42, n_splits = 5):
    """
    Cross-validation for T-Learner models.
    Trains separate models for treatment and control groups.
    
    :param model: Sklearn-compatible classifier (will be cloned for T and C models).
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
        
        # Split training data by treatment
        treat_mask = treatment_train_fold == 1
        ctrl_mask = treatment_train_fold == 0
        
        # Train separate models
        model_t = clone(model)
        model_c = clone(model)
        
        model_t.fit(X_train_fold[treat_mask], y_train_fold[treat_mask])
        model_c.fit(X_train_fold[ctrl_mask], y_train_fold[ctrl_mask])
        
        # Uplift = P(Y|T=1) - P(Y|T=0)
        prob_treated = model_t.predict_proba(X_val_fold)[:, 1]
        prob_control = model_c.predict_proba(X_val_fold)[:, 1]
        scores = prob_treated - prob_control
        
        qini = calculate_qini(y_val_fold, treatment_val_fold, scores)
        qini_scores.append(qini)
    
    return np.array(qini_scores)
