import yaml
import os
from src import data_preprocessing, model_training, evaluation, visualization, feature_selection

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = data_preprocessing.load_data(os.path.join('data', 'raw', config['data']['file_name']))
    X, y = data_preprocessing.preprocess_data(df)

    # Feature selection
    print("Performing feature selection...")
    X_selected, selected_features = feature_selection.select_features(X, y, k=config['feature_selection']['k'])
    feature_names = df.columns[:-1][selected_features]  # Update feature names

    # Split data
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(
        X_selected, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Train models
    print("Training models...")
    models = model_training.train_models(X_train, y_train, config['model_params'])

    # Create ensemble
    print("Creating ensemble model...")
    ensemble = model_training.create_ensemble(models)
    ensemble.fit(X_train, y_train)
    models['Ensemble'] = ensemble

    # Evaluate models
    print("Evaluating models...")
    results = evaluation.evaluate_models(models, X_test, y_test)

    # Perform cross-validation
    print("Performing cross-validation...")
    for name, model in models.items():
        cv_score = model_training.perform_cross_validation(model, X_selected, y)
        results[name]['cross_val_mse'] = cv_score

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