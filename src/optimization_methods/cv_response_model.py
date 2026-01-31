import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from evaluation_metrics.calculate_qini import calculate_qini

def cv_response_model(model, X, y, treatment, r_state = 42, n_splits = 5):
    """
    Cross-validation for response models.
    Trains on treatment group only, evaluates uplift on full validation set.
    
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
        
        # Train on treatment group only
        treat_mask = treatment_train_fold == 1
        model_clone = clone(model)
        model_clone.fit(X_train_fold[treat_mask], y_train_fold[treat_mask])
        
        # Score: P(conversion | features)
        scores = model_clone.predict_proba(X_val_fold)[:, 1]
        
        qini = calculate_qini(y_val_fold, treatment_val_fold, scores)
        qini_scores.append(qini)
    
    return np.array(qini_scores)
