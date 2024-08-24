import yaml
import os
from src import data_preprocessing, model_training, evaluation, visualization, feature_selection
import multiprocessing

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Determine the number of CPU cores to use
    n_jobs = config.get('n_jobs', -1)
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Get the number of folds for cross-validation
    cv = config.get('cross_validation', {}).get('n_folds', 5)

    print(f"Using {n_jobs} CPU cores for parallel processing")
    print(f"Performing {cv}-fold cross-validation")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = data_preprocessing.load_data(os.path.join('data', 'raw', config['data']['file_name']))
    X, y = data_preprocessing.preprocess_data(df)

    # Feature selection
    if config.get('feature_selection', {}).get('enabled', False):
        print("Performing feature selection...")
        k = config['feature_selection'].get('k', 10)
        X_selected, selected_features = feature_selection.select_features(X, y, k=k)
        feature_names = df.columns[:-1][selected_features]
    else:
        print("Skipping feature selection...")
        X_selected = X
        feature_names = df.columns[:-1]

    # Split data
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(
        X_selected, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Train models
    print("Training models...")
    models = model_training.train_models(X_train, y_train, config['model_params'], n_jobs=n_jobs, cv=cv)

    # Create ensemble
    print("Creating ensemble model...")
    ensemble = model_training.create_ensemble(models)
    ensemble.fit(X_train, y_train)
    models['Ensemble'] = ensemble

    # Evaluate models
    print("Evaluating models...")
    results = evaluation.evaluate_models(models, X_test, y_test)

    # Perform k-fold cross-validation in parallel
    print(f"Performing {cv}-fold cross-validation...")
    cv_results = model_training.parallel_model_evaluation(models, X_selected, y, cv=cv, n_jobs=n_jobs)
    for name, (cv_mean, cv_std) in cv_results.items():
        results[name]['cross_val_mse_mean'] = cv_mean
        results[name]['cross_val_mse_std'] = cv_std

    # Save results
    print("Saving results...")
    os.makedirs('results/metrics', exist_ok=True)
    evaluation.save_results(results, os.path.join('results', 'metrics', config['output']['results_file']))

    # Create visualizations
    print("Creating visualizations...")
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot feature importance for tree-based models
    for model_name in ['RandomForestRegressor', 'Ensemble']:
        if model_name in models:
            visualization.plot_feature_importance(
                models[model_name],
                feature_names,
                os.path.join('results', 'plots', f'feature_importance_{model_name}.png')
            )

    # Plot prediction error and residuals for each model
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        visualization.plot_prediction_error(
            y_test, y_pred, model_name,
            os.path.join('results', 'plots', f'prediction_error_{model_name}.png')
        )
        visualization.plot_residuals(
            y_test, y_pred, model_name,
            os.path.join('results', 'plots', f'residuals_{model_name}.png')
        )

    print("AutoML process completed. Results and visualizations are saved in the 'results' directory.")

if __name__ == "__main__":
    main()