def print_response_model_stats(scores, predictions):
    """
    Prints some basic statistics of the response models.
    
    :param scores: Response model scores.
    :param predictions: Response model predictions.
    """
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Mean predicted probability: {scores.mean():.4f}")
    print(f"Predicted conversion rate: {predictions.mean():.4f}")
