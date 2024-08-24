# AutoMLRegression User Guide

## Overview

AutoMLRegression is an automated machine learning tool for regression problems. It automates the process of data preprocessing, model selection, hyperparameter tuning, and model evaluation.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AutoMLRegression.git
   ```

2. Navigate to the project directory:
   ```
   cd AutoMLRegression
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Ensure your data is in CSV format.
   - Place your data file in the `data/raw/` directory.

2. Configure the project:
   - Open `config/config.yaml`.
   - Adjust the hyperparameter grids for each model as needed.
   - Set the name of your data file and other configuration options.

3. Run the main script:
   ```
   python main.py
   ```

4. View the results:
   - Model performance metrics will be saved in `results/metrics/model_performance.csv`.
   - Feature importance plot will be saved in `results/plots/feature_importance.png`.
   - Prediction error and residual plots for each model will be saved in `results/plots/`.

## Understanding the Output

### Model Performance Metrics

The `model_performance.csv` file contains the following metrics for each model:

- Explained Variance Score
- Max Error
- Mean Absolute Error
- Mean Squared Error
- R2 Score
- Mean Absolute Percentage Error

A higher Explained Variance and R2 Score, and lower error metrics indicate better model performance.

### Feature Importance Plot

The feature importance plot shows the relative importance of each feature in predicting the target variable. This is based on the Random Forest model's feature importances.

### Prediction Error and Residual Plots

For each model, two plots are generated:

1. Prediction Error Plot: Shows true values vs. predicted values. Points closer to the diagonal line indicate better predictions.
2. Residual Plot: Shows predicted values vs. residuals (true value - predicted value). Look for random scatter around the horizontal line at y=0.

## Customization

To add new models or modify existing ones:

1. Edit `src/model_training.py` to include new models or change hyperparameter grids.
2. Update `config/config.yaml` with new model parameters if necessary.

To add new evaluation metrics:

1. Edit `src/evaluation.py` to include new metrics in the `calculate_metrics` function.

To create new visualizations:

1. Add new plotting functions in `src/visualization.py`.
2. Call these new functions from `main.py`.

## Troubleshooting

If you encounter any issues:

1. Ensure all required packages are installed correctly.
2. Check that your data file is in the correct format and location.
3. Verify that the configuration in `config.yaml` matches your data and requirements.

For further assistance, please open an issue on the GitHub repository.

## Contributing

Contributions to AutoMLRegression are welcome! Please refer to the `CONTRIBUTING.md` file in the repository for guidelines on how to contribute.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file in the repository for full details.