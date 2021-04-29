# Auto Machine Learning Regression

This script provides an Automated Supervised Machine Learning Regression program, which automatically tunes the hyperparameters and prints out the final results as tables,graphs and boxplots.

## Installation

In the  terminal from the home directory, use the command git clone, then paste the link from your clipboard, or copy the command and link from below:

```bash
git clone https://github.com/sercangul/AutoMachineLearning.git
```

Change directories to the new ~/Herschel-Bulkley-GUI directory:

```bash
cd AutoMachineLearning
```

To ensure that your master branch is up-to-date, use the pull command:

```bash
git pull https://github.com/sercangul/AutoMachineLearning
```

Install required python packages using requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

Change directories to the new /AutoMachineLearning directory:

```bash
cd AutoMachineLearning
```

Run the script using Python:

```bash
python AutoRegression.py
```

## Step by step explanation of the script

This notebook explains the steps to develop an Automated Supervised Machine Learning Regression program, which automatically tunes the hyperparameters and prints out the final accuracy results as a tables together with feature importance results.

Let's import all libraries.


```python
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_absolute_percentage_error

from itertools import repeat
import matplotlib.pyplot as plt
```

Lets import our dataset from the csv files as a dataframe.


```python
df = pd.read_csv('data.csv')  
```

Let's take a look at dataset. I like using df.describe() function to have some statistics about each column.


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>27.0</td>
      <td>12.438889</td>
      <td>0.181518</td>
      <td>12.050000</td>
      <td>12.350000</td>
      <td>12.400000</td>
      <td>12.525000</td>
      <td>12.800000</td>
    </tr>
    <tr>
      <th>B</th>
      <td>27.0</td>
      <td>13.240741</td>
      <td>1.476550</td>
      <td>11.500000</td>
      <td>12.000000</td>
      <td>13.000000</td>
      <td>14.250000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>C</th>
      <td>27.0</td>
      <td>722.518519</td>
      <td>124.861884</td>
      <td>496.000000</td>
      <td>632.500000</td>
      <td>720.000000</td>
      <td>832.000000</td>
      <td>885.000000</td>
    </tr>
    <tr>
      <th>D</th>
      <td>27.0</td>
      <td>19.259259</td>
      <td>1.631248</td>
      <td>16.000000</td>
      <td>18.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>E</th>
      <td>27.0</td>
      <td>12.296296</td>
      <td>2.267069</td>
      <td>9.000000</td>
      <td>11.000000</td>
      <td>12.000000</td>
      <td>14.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>F</th>
      <td>27.0</td>
      <td>68.832028</td>
      <td>13.160644</td>
      <td>40.321912</td>
      <td>59.554258</td>
      <td>66.549069</td>
      <td>81.377992</td>
      <td>92.415572</td>
    </tr>
  </tbody>
</table>
</div>



Let's define the features as X and the column we want to predict (column F) as y. 


```python
n = len(df.columns)
X = df.iloc[:,0:n-1].to_numpy() 
y = df.iloc[:,n-1].to_numpy()
```

This defines X as all the values except the last column (columns A,B,C,D,E), and y as the last column (column numbers start from zero, hence: 0 - A, 1 - B, 2 - C, 3 - D,4 - E, 5 -F).

Some algorithms provide better accuracies with the standard scaling of the input features (i.e. normalization). Let's normalize the data. 


```python
scaler = StandardScaler()
scaler.fit(X)
X= scaler.transform(X)
```

We have to split our dataset as train and test data. For this we can use train_test_split by sklearn.model_selection. Test size of 0.20 means that 20% of the data will be used as test data and 80% of the data will be used for training.


```python
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.20)
```

We might not always want to tune the parameters of models, or only tune for some models. For this I have defined a basic input. When it is set to "True", the program will perform the tuning for all the models.


```python
Perform_tuning = True
Lassotuning, Ridgetuning, randomforestparametertuning, XGboostparametertuning, SVMparametertuning, MLPparametertuning = repeat(Perform_tuning,6)
```

Let's define the grid search function to be used with our models. The values of grid might need to be changed regarding the problem (i.e., some problems might require higher values of n_estimators, while some might require lower ranges).


```python
def grid_search(model,grid):
    # Instantiate the grid search model
    print ("Performing gridsearch for {}".format(model))
    grid_search = GridSearchCV(estimator = model(), param_grid=grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print("Grid Search Best Parameters for {}".format(model))
    print (grid_search.best_params_)
    return grid_search.best_params_
```

Performing Lasso parameter tuning.


```python
if Lassotuning:
    # Create the parameter grid based on the results of random search 
    grid = {
        'alpha': [1,0.9,0.75,0.5,0.1,0.01,0.001,0.0001] , 
        "fit_intercept": [True, False]
    }
    Lasso_bestparam = grid_search(Lasso,grid) 
```

    Performing gridsearch for <class 'sklearn.linear_model._coordinate_descent.Lasso'>
    Fitting 3 folds for each of 16 candidates, totalling 48 fits
    Grid Search Best Parameters for <class 'sklearn.linear_model._coordinate_descent.Lasso'>
    {'alpha': 0.1, 'fit_intercept': True}
    

Performing Ridge parameter tuning.


```python
if Ridgetuning:
    # Create the parameter grid based on the results of random search 
    grid = {
        'alpha': [1,0.9,0.75,0.5,0.1,0.01,0.001,0.0001] , 
        "fit_intercept": [True, False]
    }
    Ridge_bestparam = grid_search(Ridge,grid) 
```

    Performing gridsearch for <class 'sklearn.linear_model._ridge.Ridge'>
    Fitting 3 folds for each of 16 candidates, totalling 48 fits
    Grid Search Best Parameters for <class 'sklearn.linear_model._ridge.Ridge'>
    {'alpha': 0.5, 'fit_intercept': True}
    

Performing Random Forest parameter tuning.


```python
if randomforestparametertuning:
    # Create the parameter grid based on the results of random search 
    grid = {
        'bootstrap': [True,False],
        'max_depth': [40, 50, 60, 70],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1,2,3,],
        'min_samples_split': [3, 4, 5,6,7],
        'n_estimators': [5,10,15]
        }
    RF_bestparam = grid_search(RandomForestRegressor,grid) 
```

    Performing gridsearch for <class 'sklearn.ensemble._forest.RandomForestRegressor'>
    Fitting 3 folds for each of 720 candidates, totalling 2160 fits
    Grid Search Best Parameters for <class 'sklearn.ensemble._forest.RandomForestRegressor'>
    {'bootstrap': True, 'max_depth': 60, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 15}
    

Performing XGBoost parameter tuning.


```python
if XGboostparametertuning:
    # Create the parameter grid based on the results of random search 
    grid = {'colsample_bytree': [0.9,0.7],
                    'gamma': [2,5],
                    'learning_rate': [0.1,0.2,0.3],
                    'max_depth': [8,10,12],
                    'n_estimators': [5,10],
                    'subsample': [0.8,1],
                    'reg_alpha': [15,20],
                    'min_child_weight':[3,5]}
    XGB_bestparam = grid_search(XGBRegressor,grid) 
```

    Performing gridsearch for <class 'xgboost.sklearn.XGBRegressor'>
    Fitting 3 folds for each of 576 candidates, totalling 1728 fits
    Grid Search Best Parameters for <class 'xgboost.sklearn.XGBRegressor'>
    {'colsample_bytree': 0.9, 'gamma': 5, 'learning_rate': 0.3, 'max_depth': 8, 'min_child_weight': 3, 'n_estimators': 10, 'reg_alpha': 15, 'subsample': 0.8}
    

Performing SVM parameter tuning.


```python
#SVM Parameter Tuning----------------------------------------------------------
if SVMparametertuning:
    grid = {'gamma': 10. ** np.arange(-5, 3),
            'C': 10. ** np.arange(-3, 3)}
    SVR_bestparam = grid_search(SVR,grid)
```

    Performing gridsearch for <class 'sklearn.svm._classes.SVR'>
    Fitting 3 folds for each of 48 candidates, totalling 144 fits
    Grid Search Best Parameters for <class 'sklearn.svm._classes.SVR'>
    {'C': 100.0, 'gamma': 0.01}
    

Performing MLP parameter tuning.


```python
if MLPparametertuning:
    grid = {
        'hidden_layer_sizes': [2,5,8,10],
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['lbfgs', 'sgd','adam'],
        'learning_rate': ['constant','invscaling','adaptive']}
    MLP_bestparam = grid_search(MLPRegressor,grid)   
```

    Performing gridsearch for <class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>
    Fitting 3 folds for each of 144 candidates, totalling 432 fits
    Grid Search Best Parameters for <class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>
    {'activation': 'relu', 'hidden_layer_sizes': 8, 'learning_rate': 'constant', 'solver': 'lbfgs'}
    

Now we obtained the best parameters for all the models using the training data. Let's define the error metrics that will be used in analyzing the accuracy of each model. 


```python
error_metrics = (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_absolute_percentage_error        
)
```

Let's define fit_model function to predict the results, and analyze the error metrics for each model.


```python
def fit_model(model,X_train, X_test, y_train, y_test,error_metrics):
    fitted_model = model.fit(X_train,y_train)
    y_predicted = fitted_model.predict(X_test)
    calculations = []
    for metric in error_metrics:
        calc = metric(y_test, y_predicted)
        calculations.append(calc)
    return calculations
```

Provide a summary of each model and their GridSearch best parameter results. If tuning is not performed, the script will use the default values as best parameters. 


```python
try:
    trainingmodels = (
        LinearRegression(), 
        Ridge(**Ridge_bestparam), 
        RandomForestRegressor(**RF_bestparam), 
        XGBRegressor(**XGB_bestparam), 
        Lasso(**Lasso_bestparam),
        SVR(**SVR_bestparam),
        MLPRegressor(**MLP_bestparam)
    )
    
except:
    trainingmodels = (
        LinearRegression(), 
        Ridge(), 
        RandomForestRegressor(), 
        XGBRegressor(), 
        Lasso(),
        SVR(),
        MLPRegressor()
    )    

calculations = []
```

Below loop performes training, testing and error metrics calculations for each model. 


```python
for trainmodel in trainingmodels:
    errors = fit_model(trainmodel,X_train, X_test, y_train, y_test,error_metrics)
    calculations.append(errors)
```

Let's organize these results, and summarize them all in a dataframe. 


```python
errors = (
    'Explained variance score',
    'Max error',
    'Mean  absolute error',
    'Mean squared error',
    'Mean squared log error',
    'Median absolute error',
    'r2 score',
    'Mean poisson deviance',
    'Mean gamma deviance',
    'Mean absolute percentage error'        
)

model_names = (
    'LinearRegression', 
    'Ridge', 
    'RandomForestRegressor', 
    'XGBRegressor', 
    'Lasso',
    'SVR',
    'MLPRegressor'
)

df_error = pd.DataFrame(calculations, columns=errors)
df_error["Model"] = model_names

cols = df_error.columns.tolist() 
cols = cols[-1:] + cols[:-1]
df_error = df_error[cols]
df_error = df_error.sort_values(by=['Mean squared error'],ascending=True)
df_error = (df_error.set_index('Model')
        .astype(float)
        .applymap('{:,.3f}'.format))
df_error.to_csv("errors.csv")
df_error
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Explained variance score</th>
      <th>Max error</th>
      <th>Mean  absolute error</th>
      <th>Mean squared error</th>
      <th>Mean squared log error</th>
      <th>Median absolute error</th>
      <th>r2 score</th>
      <th>Mean poisson deviance</th>
      <th>Mean gamma deviance</th>
      <th>Mean absolute percentage error</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression</th>
      <td>0.998</td>
      <td>0.647</td>
      <td>0.314</td>
      <td>0.143</td>
      <td>0.000</td>
      <td>0.255</td>
      <td>0.998</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.999</td>
      <td>0.744</td>
      <td>0.327</td>
      <td>0.145</td>
      <td>0.000</td>
      <td>0.291</td>
      <td>0.998</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>0.997</td>
      <td>0.731</td>
      <td>0.412</td>
      <td>0.243</td>
      <td>0.000</td>
      <td>0.520</td>
      <td>0.997</td>
      <td>0.004</td>
      <td>0.000</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>SVR</th>
      <td>0.995</td>
      <td>1.180</td>
      <td>0.497</td>
      <td>0.479</td>
      <td>0.000</td>
      <td>0.207</td>
      <td>0.994</td>
      <td>0.008</td>
      <td>0.000</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>MLPRegressor</th>
      <td>0.931</td>
      <td>5.460</td>
      <td>1.771</td>
      <td>6.871</td>
      <td>0.002</td>
      <td>0.803</td>
      <td>0.911</td>
      <td>0.112</td>
      <td>0.002</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>RandomForestRegressor</th>
      <td>0.875</td>
      <td>5.233</td>
      <td>2.555</td>
      <td>9.634</td>
      <td>0.002</td>
      <td>2.735</td>
      <td>0.875</td>
      <td>0.149</td>
      <td>0.002</td>
      <td>0.040</td>
    </tr>
    <tr>
      <th>XGBRegressor</th>
      <td>0.939</td>
      <td>7.474</td>
      <td>4.334</td>
      <td>23.491</td>
      <td>0.005</td>
      <td>3.405</td>
      <td>0.696</td>
      <td>0.363</td>
      <td>0.006</td>
      <td>0.066</td>
    </tr>
  </tbody>
</table>
</div>



Moreover, we can analyze the feature importance results using the Random Forest regressor. 


```python
#Principal Component Analysis
features = df.columns[:-1]
try:
    randreg = RandomForestRegressor(**RF_bestparam).fit(X,y)
except:
    randreg = RandomForestRegressor().fit(X,y)    
importances = randreg.feature_importances_
indices = np.argsort(importances)
plt.figure(3) #the axis number
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.savefig('Feature Importance.png', 
              bbox_inches='tight', dpi = 500)
```


    
![png](output_41_0.png)
    



```python

```

![](FeatureImportance.png)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
