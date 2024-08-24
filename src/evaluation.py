import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate_models(models, X_test, y_test):
    """Evaluate multiple models on test data."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = calculate_metrics(y_test, y_pred)
    return results

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'mean_absolute_percentage_error': mean_absolute_percentage_error(y_true, y_pred)
    }

def save_results(results, filename):
    """Save evaluation results to a CSV file."""
    df = pd.DataFrame(results).T
    df.to_csv(filename)