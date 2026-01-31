import matplotlib.pyplot as plt
from evaluation_metrics.calculate_uplift_curve import calculate_uplift_curve
from evaluation_metrics.calculate_auuc import calculate_auuc
from evaluation_metrics.calculate_qini import calculate_qini


def plot_uplift_curves(results_dict, y_true, treatment, title = "Uplift Curves"):
    """
    Plots uplift curves for multiple models.
    
    :param results_dict: Dictionary with model names as keys and score arrays as values.
    :param y_true: Array of the actual conversion outcomes (0 = no conversion, 1 = conversion).
    :param treatment: Array of the treatment indicators (0 = no email, 1 = email).
    :param title: String that serves as the title of the plot. Defaults to 'Uplift Curves'.
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    
    colors = plt.cm.tab10.colors
    
    for idx, (name, scores) in enumerate(results_dict.items()):
        curve = calculate_uplift_curve(y_true, treatment, scores)
        auuc = calculate_auuc(y_true, treatment, scores)
        qini = calculate_qini(y_true, treatment, scores)
        
        ax.plot(curve["percentile"] * 100, curve["uplift"], 
                label = f"{name} (AUUC = {auuc:.1f}, Qini = {qini:.1f})",
                color = colors[idx], linewidth = 2)
    
    # Plot the Zero Qini diagonal for comparison
    curve = calculate_uplift_curve(y_true, treatment, list(results_dict.values())[0])
    ax.plot(curve["percentile"] * 100, curve["zero_qini"], 
            "k--", label = "Zero Qini (AUUC = 15, Qini = 0.0)", linewidth = 1.5, alpha = 0.7)
    
    ax.set_xlabel("Percentage of Population Targeted", fontsize = 12)
    ax.set_ylabel("Cumulative Incremental Conversions", fontsize = 12)
    ax.set_title(title, fontsize = 14)
    ax.legend(loc = "lower right")
    ax.grid(True, alpha = 0.3)
    ax.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.show()
