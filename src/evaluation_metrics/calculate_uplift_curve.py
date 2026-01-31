import pandas as pd
import numpy as np

def calculate_uplift_curve(y_true, treatment, scores, n_bins = 100):
    """
    Calculates the uplift curve data.
    
    :param y_true: array of the actual conversion outcomes (0 = no conversion, 1 = conversion).
    :param treatment: array of the treatment indicators (0 = no email, 1 = email).
    :param scores: array of the predicted uplift scores (higher = more likely to be persuaded).
    :param n_bins: int representing the number of points on the curve.

    Returns: a dictionary with percentile, uplift, and random_uplift arrays.
    """
    # Create dataframe for easier manipulation
    data = pd.DataFrame({
        "y": y_true,
        "treatment": treatment,
        "score": scores
    })
    
    # Sort by score (descending - highest uplift first)
    data = data.sort_values("score", ascending = False).reset_index(drop = True)
    
    percentiles = []
    uplifts = []
    
    for i in range(1, n_bins + 1):
        # Top i% of customers
        cutoff = int(len(data) * i / n_bins)
        subset = data.iloc[:cutoff]
        
        # Count conversions in treatment and control within this subset
        subset_treatment = subset[subset["treatment"] == 1]
        subset_control = subset[subset["treatment"] == 0]
        
        n_treat_subset = len(subset_treatment)
        n_ctrl_subset = len(subset_control)
        
        # Uplift = (treatment conversions) - (control conversions scaled to treatment size)
        # This gives the incremental conversions attributable to treatment
        uplift = subset_treatment["y"].sum() - subset_control["y"].sum() * \
                 (n_treat_subset / n_ctrl_subset if n_ctrl_subset > 0 else 0)
        
        percentiles.append(i / n_bins)
        uplifts.append(uplift)
    
    # Random baseline: uplift grows linearly
    total_uplift = uplifts[-1]  # Total uplift when targeting everyone
    zero_qini = [total_uplift * p for p in percentiles]
    
    return {
        "percentile": np.array(percentiles),
        "uplift": np.array(uplifts),
        "zero_qini": np.array(zero_qini)
    }
