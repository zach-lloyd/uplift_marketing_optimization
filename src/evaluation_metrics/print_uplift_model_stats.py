def print_uplift_model_stats(uplift):
    """
    Prints some basic statistics of the uplift models.
    
    :param uplift: The model's uplift scores.
    """
    print(f"Uplift score range: [{uplift.min():.4f}, {uplift.max():.4f}]")
    print(f"Mean estimated uplift: {uplift.mean():.4f}")
    print(f"Customers with positive uplift: {(uplift > 0).mean():.1%}")
