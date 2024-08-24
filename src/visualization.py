import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, feature_names, filename):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    else:
        print(f"Model {type(model).__name__} doesn't have feature_importances_ attribute.")

def plot_prediction_error(y_true, y_pred, model_name, filename):
    """Create a scatter plot of true vs predicted values."""
    plt.figure(figsize=(10, 7))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True vs Predicted Values - {model_name}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_residuals(y_true, y_pred, model_name, filename):
    """Create a residual plot."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()