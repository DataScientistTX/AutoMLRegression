# AutoMLRegression

AutoMLRegression is an automated machine learning tool for regression problems. It automates the process of data preprocessing, feature selection, model training, hyperparameter tuning, and model evaluation.

## Features

- Automated data preprocessing
- Optional feature selection
- Training and evaluation of multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Support Vector Regression (SVR)
  - Multi-layer Perceptron (MLP)
- Ensemble model creation
- Hyperparameter tuning using GridSearchCV
- Cross-validation for robust model evaluation
- Visualization of results including feature importance and residual plots

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AutoMLRegression.git
   cd AutoMLRegression
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
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
   - Adjust the parameters as needed (data filename, feature selection settings, model parameters, etc.).

3. Run the main script:
   ```
   python main.py
   ```

4. View the results:
   - Model performance metrics will be saved in `results/metrics/model_performance.csv`.
   - Visualization plots will be saved in `results/plots/`.

## Project Structure

```
AutoMLRegression/
│
├── data/
│   └── raw/             # Place your input CSV file here
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
│
├── config/
│   └── config.yaml      # Configuration file
│
├── results/
│   ├── metrics/         # Performance metrics will be saved here
│   └── plots/           # Visualization plots will be saved here
│
├── main.py              # Main execution script
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Configuration

The `config/config.yaml` file allows you to customize various aspects of the AutoML process:

- Data settings (filename, test size, random state)
- Feature selection options
- Model hyperparameters for grid search
- Evaluation metrics
- Output file names

## Contributing

Contributions to AutoMLRegression are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.