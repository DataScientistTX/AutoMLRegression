import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import load_data, preprocess_data, split_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 1, 2, 2, 3],
            'target': [10, 20, 30, 40, 50]
        })
        self.sample_data.to_csv('test_data.csv', index=False)

    def tearDown(self):
        # Clean up the test CSV file
        import os
        os.remove('test_data.csv')

    def test_load_data(self):
        df = load_data('test_data.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 4))
        self.assertEqual(list(df.columns), ['A', 'B', 'C', 'target'])

    def test_preprocess_data(self):
        X, y = preprocess_data(self.sample_data)
        
        # Check if X and y have correct shapes
        self.assertEqual(X.shape, (5, 3))
        self.assertEqual(y.shape, (5,))
        
        # Check if X is scaled
        scaler = StandardScaler()
        X_expected = scaler.fit_transform(self.sample_data[['A', 'B', 'C']])
        np.testing.assert_array_almost_equal(X, X_expected)
        
        # Check if y is correct
        np.testing.assert_array_equal(y, self.sample_data['target'].values)

    def test_split_data(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # Check shapes
        self.assertEqual(X_train.shape, (4, 2))
        self.assertEqual(X_test.shape, (1, 2))
        self.assertEqual(y_train.shape, (4,))
        self.assertEqual(y_test.shape, (1,))
        
        # Check if the split is deterministic (given the random_state)
        expected_X_test = np.array([[7, 8]])
        expected_y_test = np.array([4])
        np.testing.assert_array_equal(X_test, expected_X_test)
        np.testing.assert_array_equal(y_test, expected_y_test)

if __name__ == '__main__':
    unittest.main()