#!/usr/bin/env python
# coding: utf-8

# This script explains the steps to develop an Automated Supervised Machine Learning Regression program, 
# which automatically tunes the hyperparameters and prints out the final accuracy results 
# as a table together with feature importance results.
# Let's import all libraries.

import pandas as pd
import numpy as np
from   xgboost import XGBRegressor
from   itertools import repeat
import matplotlib.pyplot as plt

from   sklearn.model_selection import GridSearchCV,train_test_split
from   sklearn.preprocessing import StandardScaler
from   sklearn.linear_model import LinearRegression,Ridge,Lasso
from   sklearn.ensemble import RandomForestRegressor
from   sklearn.svm import SVR
from   sklearn.neural_network import MLPRegressor
from   sklearn.metrics import explained_variance_score,max_error,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,r2_score,mean_poisson_deviance,mean_gamma_deviance,mean_absolute_percentage_error

# Import  dataset from the csv file as a dataframe.

df = pd.read_csv('data.csv')  

n = len(df.columns)
X = df.iloc[:,0:n-1].to_numpy() 
y = df.iloc[:,n-1].to_numpy()

# This defines X as all the values except the last column (columns A,B,C,D,E), and y as the last column (column numbers start from zero, hence: 0 - A, 1 - B, 2 - C, 3 - D,4 - E, 5 -F).
# Some algorithms provide better accuracies with the standard scaling of the input features (i.e. normalization). Let's normalize the data. 

scaler = StandardScaler()
scaler.fit(X)
X= scaler.transform(X)

# We have to split our dataset as train and test data. For this we can use train_test_split by sklearn.model_selection. Test size of 0.20 means that 20% of the data will be used as test data and 80% of the data will be used for training.

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.20)

# Set perform tuning= "True" to perform hyperparameter tuning for all the models.

Perform_tuning = True
Lassotuning, Ridgetuning, randomforestparametertuning, XGboostparametertuning, SVMparametertuning, MLPparametertuning = repeat(Perform_tuning,6)

# Define the grid search function to be used with models. 
# The values of grid might need to be changed based on the problem 

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

# Performing Lasso parameter tuning.
if Lassotuning:
    # Create the parameter grid based on the results of random search 
    grid = {
        'alpha': [1,0.9,0.5,0.1,0.01,0.001,0.0001], 
        "fit_intercept": [True, False]
    }
    Lasso_bestparam = grid_search(Lasso,grid) 

# Performing Ridge parameter tuning.

if Ridgetuning:
    # Create the parameter grid based on the results of random search 
    grid = {
        'alpha': [1,0.9,0.5,0.1,0.01,0.001,0.0001], 
        "fit_intercept": [True, False]
    }
    Ridge_bestparam = grid_search(Ridge,grid) 

# Performing Random Forest parameter tuning.

if randomforestparametertuning:
    # Create the parameter grid based on the results of random search 
    grid = {
        'bootstrap': [True,False],
        'max_depth': [40, 50, 60, 70],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1,2,3,],
        'min_samples_split': [3, 4, 5,6,7],
        'n_estimators': [5,10,15,40,80]
        }
    RF_bestparam = grid_search(RandomForestRegressor,grid) 

# Performing XGBoost parameter tuning.

if XGboostparametertuning:
    # Create the parameter grid based on the results of random search 
    grid = {'colsample_bytree': [0.9,0.7],
                    'gamma': [2,5],
                    'learning_rate': [0.1,0.2,0.3],
                    'max_depth': [8,10,12],
                    'n_estimators': [5,10,20,40,80],
                    'subsample': [0.8,1],
                    'reg_alpha': [15,20],
                    'min_child_weight':[3,5]}
    XGB_bestparam = grid_search(XGBRegressor,grid) 

# Performing SVM parameter tuning.

#SVM Parameter Tuning----------------------------------------------------------
if SVMparametertuning:
    grid = {'gamma': 10. ** np.arange(-5, 3),
            'C': 10. ** np.arange(-3, 3)}
    SVR_bestparam = grid_search(SVR,grid)

# Performing MLP parameter tuning.

if MLPparametertuning:
    grid = {
        'hidden_layer_sizes': [2,5,8,10],
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['lbfgs', 'sgd','adam'],
        'learning_rate': ['constant','invscaling','adaptive']}
    MLP_bestparam = grid_search(MLPRegressor,grid)   

# Now we obtained the best parameters for all the models using the training data. 
# Let's define the error metrics that will be used in analyzing the accuracy of each model. 

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

# Let's define fit_model function to predict the results, and analyze the error metrics for each model.

def fit_model(model,X_train, X_test, y_train, y_test,error_metrics):
    fitted_model = model.fit(X_train,y_train)
    y_predicted = fitted_model.predict(X_test)
    calculations = []
    for metric in error_metrics:
        calc = metric(y_test, y_predicted)
        calculations.append(calc)
    return calculations

# Provide a summary of each model and their GridSearch best parameter results. 
# If tuning is not performed, the script will use the default values as best parameters. 

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

# Below loop performes training, testing and error metrics calculations for each model. 

for trainmodel in trainingmodels:
    errors = fit_model(trainmodel,X_train, X_test, y_train, y_test,error_metrics)
    calculations.append(errors)

# Let's organize these results, and summarize them all in a dataframe. 

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

df_error = (df_error.sort_values(by=['Mean squared error'],
                                 ascending=True)
                    .set_index('Model')
                    .astype(float)
                    .applymap('{:,.3f}'.format))
df_error.to_csv("errors.csv")

# Analyze the feature importance results using the Random Forest regressor. 

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