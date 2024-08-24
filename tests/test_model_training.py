import unittest
import numpy as np
from sklearn.datasets import make_regression
from src.model_training import train_models, grid_search

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        self.X_train = X[:80]
        self.X_test = X[80:]
        self.y_train = y[:80]
        self.y_test = y[80:]

        # Sample configuration for testing
        self.config = {
            'model_params': {
                'LinearRegression': {},
                'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                'Lasso': {'alpha': [0.1, 1.0, 10.0]},
                'RandomForestRegressor': {'n_estimators': [10, 50, 100]},
                'XGBRegressor': {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1]},
                'SVR': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
                'MLPRegressor': {'hidden_layer_sizes': [(10,), (50,), (100,)]}
            }
        }

    def test_train_models(self):
        models = train_models(self.X_train, self.y_train, self.config['model_params'])
        
        # Check if all expected models are trained
        expected_models = ['LinearRegression', 'Ridge', 'Lasso', 'RandomForestRegressor', 
                           'XGBRegressor', 'SVR', 'MLPRegressor']
        self.assertEqual(set(models.keys()), set(expected_models))
        
        # Check if models can make predictions
        for model_name, model in models.items():
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.y_test))

    def test_grid_search(self):
        from sklearn.ensemble import RandomForestRegressor
        
        params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
        best_params = grid_search(RandomForestRegressor, params, self.X_train, self.y_train)
        
        # Check if best parameters are returned
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        
        # Check if best parameters are within the specified ranges
        self.assertIn(best_params['n_estimators'], [10, 50, 100])
        self.assertIn(best_params['max_depth'], [None, 10, 20])

if __name__ == '__main__':
    unittest.main()