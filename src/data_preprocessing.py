import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by separating features and target, and scaling features."""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X = normalize_features(X)
    
    return X, y

def normalize_features(X):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)