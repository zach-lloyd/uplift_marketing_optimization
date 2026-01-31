from evaluation_metrics.calculate_uplift_curve import calculate_uplift_curve
import numpy as np

def calculate_auuc(y_true, treatment, scores):
    """
    Calculates the area under the uplift curve (AUUC) (higher is better).
    
    :param y_true: array of the actual conversion outcomes (0 = no conversion, 1 = conversion).
    :param treatment: array of the treatment indicators (0 = no email, 1 = email).
    :param scores: array of the predicted uplift scores (higher = more likely to be persuaded).
    """
    curve = calculate_uplift_curve(y_true, treatment, scores)
    
    # Calculate the area under the curve using the trapezoidal rule
    auuc = np.trapezoid(curve["uplift"], curve["percentile"])
    
    return auuc
