from sklearn.feature_selection import SelectKBest, f_regression

def select_features(X, y, k=10):
    """Select top k features based on f_regression score."""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features