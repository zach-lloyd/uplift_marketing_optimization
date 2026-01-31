import numpy as np
import pandas as pd
from itertools import product

def hyperparameter_search(model_class, param_grid, X, y, treatment, cv_func, 
                          r_state = 42, n_iter = None, n_splits = 5):
    """
    Custom hyperparameter search using uplift-based CV.
    
    :param model_class: Model class (e.g., LogisticRegression, LGBMClassifier).
    :param param_grid: Dictionary of hyperparameters to search.
    :param X: Feature matrix.
    :param y: Target array.
    :param treatment: Treatment indicator array.
    :param cv_func: Cross-validation function (cv_response_model, cv_s_learner, or cv_t_learner).
    :param n_iter: Number of random combinations to try (None = exhaustive grid search).
    :param n_splits: Number of CV folds.

    Returns: Dictionary with best_params, best_score, and results_df.
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    # Create list of all possible parameter combinations
    all_combinations = list(product(*param_values))
    
    # If n_iter specified, randomly sample and test n_iter combinations of hyperparameters
    if n_iter is not None and n_iter < len(all_combinations):
        np.random.seed(r_state)
        indices = np.random.choice(len(all_combinations), size = n_iter, replace = False)
        combinations_to_try = [all_combinations[i] for i in indices]
    else:
        combinations_to_try = all_combinations
    
    print(f"Testing {len(combinations_to_try)} parameter combinations...")
    
    results = []
    best_score = -np.inf
    best_params = None
    
    # Iterate over the parameter combinations and perform cross-fold validation
    # using each combination
    for i, combo in enumerate(combinations_to_try):
        params = dict(zip(param_names, combo))
        
        # Use try/except statements to handle situation where a parameter
        # combination is not valid for a given model
        try:
            model = model_class(**params)
            qini_scores = cv_func(model, X, y, treatment, n_splits = n_splits)
            # Evaluate using the Qini coefficient, which is the preferred metric
            # for evaluating uplift models
            mean_qini = qini_scores.mean()
            std_qini = qini_scores.std()
            
            results.append({
                **params,
                "mean_qini": mean_qini,
                "std_qini": std_qini
            })
            
            if mean_qini > best_score:
                best_score = mean_qini
                best_params = params
            
            # Progress update every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(combinations_to_try)}, best Qini so far: {best_score:.4f}")
                
        except Exception as e:
            print(f"  Error with params {params}: {e}")
            continue
    
    results_df = pd.DataFrame(results).sort_values("mean_qini", ascending=False)
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best mean Qini: {best_score:.4f}")
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "results_df": results_df
    }
