# Auto Machine Learning Regression

This project provides an Automated Supervised Machine Learning Regression program that automatically tunes hyperparameters and outputs final results as tables and visualizations.

## Table of Contents
- [Auto Machine Learning Regression](#auto-machine-learning-regression)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Models](#models)
  - [Output](#output)
  - [Testing](#testing)
  - [TODO](#todo)
  - [Contributing](#contributing)
  - [License](#license)

## Features
- Automated hyperparameter tuning for multiple regression models
- Comprehensive error metric evaluation
- Feature importance analysis
- Easy-to-read output in CSV format and visualizations

## Project Structure
```
AutoMLRegression/
│
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
├── tests/
├── config/
├── results/
│   ├── models/
│   ├── plots/
│   └── metrics/
├── docs/
├── README.md
├── requirements.txt
├── setup.py
└── main.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sercangul/AutoMLRegression.git
   cd AutoMLRegression
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your raw data file (CSV format) in the `data/raw/` directory.

2. Update the configuration in `config/config.yaml` if needed.

3. Run the main script:
   ```bash
   python main.py
   ```

## Configuration

You can adjust the hyperparameter grids and other settings in the `config/config.yaml` file. This allows you to customize the model training process without changing the code.

## Models

The following regression models are implemented:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)
- Multi-layer Perceptron (MLP) Regressor

## Output

- `results/metrics/model_performance.csv`: A CSV file containing evaluation metrics for all models
- `results/plots/feature_importance.png`: A plot showing the importance of each feature

## Testing

To run the tests:
```bash
python -m unittest discover tests
```

## TODO

- Implement Seaborn pairwise plots
- Integrate with pandas profiling
- Visualize best 3 fits (predicted vs test)
- Add K-fold cross-validation results for each model
- Visualize error analysis (comparison boxplots)
- Implement data imputation for missing data points (using KNN)
- Deploy as a web application using Streamlit + Heroku

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the [Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/).