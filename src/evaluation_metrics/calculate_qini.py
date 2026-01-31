from evaluation_metrics.calculate_uplift_curve import calculate_uplift_curve
import numpy as np

def calculate_qini(y_true, treatment, scores):
    """
    Calculates the Qini coefficient 
    (area under the model curve - area under the random baseline curve).
    
    :param y_true: array of the actual conversion outcomes (0 = no conversion, 1 = conversion).
    :param treatment: array of the treatment indicators (0 = no email, 1 = email).
    :param scores: array of the predicted uplift scores (higher = more likely to be persuaded). 
    """
    curve = calculate_uplift_curve(y_true, treatment, scores)
    
    # Area between model curve and random baseline using trapezoidal method
    model_area = np.trapezoid(curve["uplift"], curve["percentile"])
    zero_qini_area = np.trapezoid(curve["zero_qini"], curve["percentile"])
    
    qini = model_area - zero_qini_area
    
    return qini
