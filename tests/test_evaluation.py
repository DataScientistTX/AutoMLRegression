import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.evaluation import evaluate_models, calculate_metrics, save_results

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.X_test = np.random.rand(100, 5)
        self.y_test = np.random.rand(100)
        
        # Create sample models
        self.models = {
            'LinearRegression': LinearRegression().fit(self.X_test, self.y_test),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42).fit(self.X_test, self.y_test)
        }

    def test_calculate_metrics(self):
        y_pred = self.models['LinearRegression'].predict(self.X_test)
        metrics = calculate_metrics(self.y_test, y_pred)
        
        # Check if all expected metrics are calculated
        expected_metrics = ['explained_variance', 'max_error', 'mean_absolute_error', 
                            'mean_squared_error', 'r2_score', 'mean_absolute_percentage_error']
        self.assertEqual(set(metrics.keys()), set(expected_metrics))
        
        # Check if metrics are within reasonable ranges
        self.assertGreaterEqual(metrics['r2_score'], -1)
        self.assertLessEqual(metrics['r2_score'], 1)
        self.assertGreaterEqual(metrics['mean_absolute_error'], 0)

    def test_evaluate_models(self):
        results = evaluate_models(self.models, self.X_test, self.y_test)
        
        # Check if results are returned for all models
        self.assertEqual(set(results.keys()), set(self.models.keys()))
        
        # Check if all metrics are calculated for each model
        for model_results in results.values():
            self.assertIn('explained_variance', model_results)
            self.assertIn('max_error', model_results)
            self.assertIn('mean_absolute_error', model_results)
            self.assertIn('mean_squared_error', model_results)
            self.assertIn('r2_score', model_results)
            self.assertIn('mean_absolute_percentage_error', model_results)

    def test_save_results(self):
        results = evaluate_models(self.models, self.X_test, self.y_test)
        filename = 'test_results.csv'
        save_results(results, filename)
        
        # Check if file is created
        self.assertTrue(os.path.exists(filename))
        
        # Check if file contains correct data
        df = pd.read_csv(filename, index_col=0)
        self.assertEqual(set(df.index), set(self.models.keys()))
        self.assertEqual(set(df.columns), set(results[list(results.keys())[0]].keys()))
        
        # Clean up
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()