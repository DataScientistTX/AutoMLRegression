from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score

def train_models(X_train, y_train, model_params):
    """Train multiple regression models with given parameters."""
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor(),
        'SVR': SVR(),
        'MLPRegressor': MLPRegressor()
    }
    
    trained_models = {}
    for name, model in models.items():
        if name in model_params:
            model = grid_search(model.__class__, model_params[name], X_train, y_train)
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def grid_search(model, params, X_train, y_train):
    """Perform grid search for hyperparameter tuning."""
    grid_search = GridSearchCV(model(), params, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_mlp_with_early_stopping(X, y, **params):
    """Train MLP with early stopping."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    mlp = MLPRegressor(early_stopping=True, validation_fraction=0.2, **params)
    mlp.fit(X_train, y_train)
    return mlp

def create_ensemble(models):
    """Create an ensemble of models using VotingRegressor."""
    return VotingRegressor(estimators=[(name, model) for name, model in models.items()])

def perform_cross_validation(model, X, y, cv=5):
    """Perform cross-validation and return mean negative MSE."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    return -scores.mean()