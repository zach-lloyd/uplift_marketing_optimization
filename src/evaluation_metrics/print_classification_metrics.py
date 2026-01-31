from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_classification_metrics(predictions, y_test, title):
    """
    Prints accuracy, precision, recall, and F1 score of the model.
    
    :param predictions: The model's predictions.
    :param y_test: The actual outcomes for the test data.
    :param title: A string representing the title of the model.
    """
    print(f"{title} Classification Metrics")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division = 0)
    recall = recall_score(y_test, predictions, zero_division = 0)
    f1 = f1_score(y_test, predictions, zero_division = 0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
