from evaluation_metrics.calculate_auuc import calculate_auuc
from evaluation_metrics.calculate_qini import calculate_qini
from evaluation_metrics.plot_uplift_curves import plot_uplift_curves

def display_uplift_metrics(results, title, y_test, treatment_test):
    """
    Displays the AUUC, Qini, and uplift curves for all models in the results 
    dictionary.
    
    :param results: Dictionary of model results.
    :param title: Title to use for the metric and uplift curve displays.
    :param y_test: Dependent variable test dataset.
    :param treatment_test: Treatment variable test dataset.
    """
    print(f"{title} - Uplift Metrics")

    for name, scores in results.items():
        auuc = calculate_auuc(y_test.values, treatment_test.values, scores)
        qini = calculate_qini(y_test.values, treatment_test.values, scores)
        print(f"\n{name}:")
        print(f"  AUUC: {auuc:.2f}")
        print(f"  Qini: {qini:.2f}")

    plot_uplift_curves(
        results,
        y_test.values,
        treatment_test.values,
        title = f"{title} - Uplift Curves"
    )
    