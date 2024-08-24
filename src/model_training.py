from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import VotingRegressor
from joblib import Parallel, delayed
import numpy as np

def train_models(X_train, y_train, model_params, n_jobs=-1, cv=5):
    """Train multiple regression models with given parameters using parallel processing."""
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor(),
        'SVR': SVR(),
        'MLPRegressor': MLPRegressor()
    }
    
    def train_model(name, model, params):
        if params:
            return name, grid_search(model.__class__, params, X_train, y_train, n_jobs, cv)
        else:
            model.fit(X_train, y_train)
            return name, model

    trained_models = dict(
        Parallel(n_jobs=n_jobs)(
            delayed(train_model)(name, model, model_params.get(name))
            for name, model in models.items()
        )
    )
    
    return trained_models

def grid_search(model, params, X_train, y_train, n_jobs=-1, cv=5):
    """Perform grid search for hyperparameter tuning with parallel processing."""
    grid_search = GridSearchCV(model(), params, cv=cv, n_jobs=n_jobs, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def create_ensemble(models):
    """Create an ensemble of models using VotingRegressor."""
    return VotingRegressor(estimators=[(name, model) for name, model in models.items()])

def perform_cross_validation(model, X, y, cv=5, n_jobs=-1):
    """Perform k-fold cross-validation using parallel processing."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=n_jobs)
    return -scores.mean(), scores.std()

def parallel_model_evaluation(models, X, y, cv=5, n_jobs=-1):
    """Perform parallel model evaluation with k-fold cross-validation."""
    def evaluate_model(name, model):
        mean_score, std_score = perform_cross_validation(model, X, y, cv, n_jobs=1)
        return name, (mean_score, std_score)

    results = dict(
        Parallel(n_jobs=n_jobs)(
            delayed(evaluate_model)(name, model)
            for name, model in models.items()
        )
    )
    return results