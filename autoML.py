# -*- coding: utf-8 -*-

"""
Created on Tue Nov 26 13:19:33 2019
@author: Sercan Gul
Automated Machine Learning Code
"""
# Importing the libraries------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import statistics

#Import all ML algorithms
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#Define evaluate function for parameter tuning---------------------------------

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

#Importing the datasets-------------------------------------------------------
print ("Importing datasets")
dfML = pd.read_csv('data.csv')  

#Inputs------------------------------------------------------------------------
randomforestparametertuning = "no" #yes or no
XGboostparametertuning      = "no" #yes or no
SVMparametertuning          = "no" #yes or no
MLPparametertuning          = "no" #yes or no

#Preparing ML Datasets--------------------------------------------------------
#if preprocessing from original data is necessary, do it here.
print ("Preparing ML datasets")

#Define X and Y for each regression--------------------------------------------
X = dfML.iloc[:, :-1].values #X is all the value except the last column
y = dfML.iloc[:,5].values #Y is (in this case) the 4th column. The valu needs to be changed according to your dataframe.
X2 = preprocessing.scale(X) #PReprocessing provides better results especially for MLP and SVR algorihms.

#Splitting all datasets into train and test------------------------------------
print ("Train/test split")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.2,random_state = 0)
X_train2, X_test2, y_train2, y_test2= train_test_split(X2,y,test_size = 0.2,random_state = 0)

#RandomForest Parameter Tuning-------------------------------------------------
if randomforestparametertuning == "yes":
    print ("Performing gridsearch in random forest")

    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True,False],
        'max_depth': [40, 50, 60, 70],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1,2,3,],
        'min_samples_split': [3, 4, 5,6,7],
        'n_estimators': [50,100,150,200,250,300,350,400,500]
        }
    
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search_RF = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    
    # Fit the grid search to the data
    grid_search_RF.fit(X_train2, y_train2)
    print("Grid Search Best Parameters for Random Forest Regression")
    print (grid_search_RF.best_params_)
    
#XGBoost Parameter Tuning------------------------------------------------------
if XGboostparametertuning == "yes":
    print("XGBoost parameter tuning")
    # Create the parameter grid based on the results of random search 
    grid = {'colsample_bytree': [0.9,0.8,0.7],
                    'gamma': [2,3,4,5],
                    'learning_rate': [0.1,0.2,0.3],
                    'max_depth': [8,9,10,11,12],
                    'n_estimators': [150,200,250,300,350],
                    'subsample': [0.8,0.9,1],
                    'reg_alpha': [15,18,20],
                    'min_child_weight':[3,4,5]}

    # Create a based model
    XGB = XGBRegressor()
    # Instantiate the grid search model
    grid_search_XGB = GridSearchCV(estimator = XGB, param_grid = grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    
    # Fit the grid search to the data
    grid_search_XGB.fit(X_train2, y_train2)
    print("Grid Search Best Parameters for XGBoost")
    print (grid_search_XGB.best_params_)    

#SVM Parameter Tuning----------------------------------------------------------
if SVMparametertuning == "yes":
    print("SVM parameter tuning")

    C_range = 10. ** np.arange(-3, 3)
    gamma_range = 10. ** np.arange(-5, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    svr_rbf = SVR()
    # Instantiate the grid search model
    grid_search_svm = GridSearchCV(estimator = svr_rbf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search_svm.fit(X_train2, y_train2)
    print("Grid Search Best Parameters for SVM")
    print (grid_search_svm.best_params_)

#MLP Parameter Tuning----------------------------------------------------------
    
if MLPparametertuning == "yes":
    print("MLP parameter tuning")

    param_grid = {
        'hidden_layer_sizes': [50,100,150,200,250,300,350],
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['lbfgs', 'sgd','adam'],
        'learning_rate': ['constant','invscaling','adaptive']}
    MLP = MLPRegressor()
    # Instantiate the grid search model
    grid_search_MLP = GridSearchCV(estimator = MLP, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    
    # Fit the grid search to the data
    grid_search_MLP.fit(X_train2, y_train2)
    print("Grid Search Best Parameters for MLP")
    print (grid_search_MLP.best_params_)


#Fitting multi linear regression to data---------------------------------------
print ("Fit multilinear regression")
linreg = LinearRegression()
linreg.fit(X_train,y_train)

#Fitting ridge regression to data----------------------------------------------
print ("Fit ridge regression")
ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(X_train,y_train)

#Fitting random forest regression to data--------------------------------------
print ("Fit random forest regression")
try:
    randreg = RandomForestRegressor(**grid_search_RF.best_params_)
except:
    randreg = RandomForestRegressor()
randreg.fit(X_train2,y_train2)

#Fitting XGboost regression to data--------------------------------------------
print ("Fit XGBoost regression")
try:
    XGBreg = XGBRegressor(**grid_search_XGB.best_params_)
except: 
    XGBreg = XGBRegressor()
XGBreg.fit(X_train2, y_train2)

#Fitting LASSO regression to data----------------------------------------------
print ("Fit Lasso regression")
lassoreg = Lasso(alpha=0.01, max_iter=10e5)
lassoreg.fit(X_train, y_train)

#Support Vector Machines-------------------------------------------------------
print ("Fit SVR RBF regression")
try:
    svr_rbf = SVR(**grid_search_svm.best_params_)
except:
    svr_rbf = SVR()
    svr_rbf.fit(X_train2, y_train2)

#MLP Regressor-----------------------------------------------------------------
print ("Fit Multi-layer Perceptron regressor")
try:
    MLP = MLPRegressor(**grid_search_MLP.best_params_)
except:
    MLP = MLPRegressor()    
MLP.fit(X_train2, y_train2)

#Polynomial regression 2 degrees
print ("Fit polynomial regression degree=2")
poly2 = PolynomialFeatures(degree=2)
X_train_trans = poly2.fit_transform(X_train)
polyreg2 = linear_model.LinearRegression()
p=polyreg2.fit(X_train_trans,y_train)
poly2_coef = polyreg2.coef_

#Polynomial regression 3 degrees
print ("Fit polynomial regression degree=3")
poly3 = PolynomialFeatures(degree=3)
X_train_trans = poly3.fit_transform(X_train)
polyreg3 = linear_model.LinearRegression()
polyreg3.fit(X_train_trans,y_train)
poly3_coef = polyreg3.coef_

#All errors--------------------------------------------------------------------
#from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#Define the missing sklearn.metrics parameter of mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.abs((y_true - y_pred) / y_true)) * 100

def r_score(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    residuals = y_true- y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)    
    r_scores = (r_squared ** 0.5)
    return r_scores

y_predicted_RF = randreg.predict(X_test2)
y_predicted_LREG = linreg.predict(X_test)
y_predicted_RIDGE = ridgeReg.predict(X_test)
y_predicted_XGB = XGBreg.predict(X_test2)
y_predicted_LASSO = lassoreg.predict(X_test)
y_predicted_svr_rbf = svr_rbf.predict(X_test2)
y_predicted_MLP = MLP.predict(X_test2)
y_predicted_PREG2 = polyreg2.predict(poly2.fit_transform(X_test))
y_predicted_PREG3 = polyreg3.predict(poly3.fit_transform(X_test))

r_RF=  r_score(y_test2, y_predicted_RF)
MAE_RF = mean_absolute_error(y_test2, y_predicted_RF)
MSE_RF = mean_squared_error(y_test2, y_predicted_RF)
MAPE_RF = mean_absolute_percentage_error(y_test2, y_predicted_RF)

r_LREG=  r_score(y_test, y_predicted_LREG)
MAE_LREG = mean_absolute_error(y_test, y_predicted_LREG)
MSE_LREG = mean_squared_error(y_test, y_predicted_LREG)
MAPE_LREG = mean_absolute_percentage_error(y_test, y_predicted_LREG)

r_RIDGE=  r_score(y_test, y_predicted_RIDGE)
MAE_RIDGE = mean_absolute_error(y_test, y_predicted_RIDGE)
MSE_RIDGE = mean_squared_error(y_test, y_predicted_RIDGE)
MAPE_RIDGE = mean_absolute_percentage_error(y_test, y_predicted_RIDGE)

r_XGB=  r_score(y_test2, y_predicted_XGB)
MAE_XGB = mean_absolute_error(y_test2, y_predicted_XGB)
MSE_XGB = mean_squared_error(y_test2, y_predicted_XGB)
MAPE_XGB = mean_absolute_percentage_error(y_test2, y_predicted_XGB)

r_LASSO=  r_score(y_test, y_predicted_LASSO)
MAE_LASSO = mean_absolute_error(y_test, y_predicted_LASSO)
MSE_LASSO = mean_squared_error(y_test, y_predicted_LASSO)
MAPE_LASSO = mean_absolute_percentage_error(y_test, y_predicted_LASSO)

r_svr_rbf=  r_score(y_test2, y_predicted_svr_rbf)
MAE_svr_rbf = mean_absolute_error(y_test2, y_predicted_svr_rbf)
MSE_svr_rbf = mean_squared_error(y_test2, y_predicted_svr_rbf)
MAPE_svr_rbf = mean_absolute_percentage_error(y_test2, y_predicted_svr_rbf)

r_MLP=  r_score(y_test2, y_predicted_MLP)
MAE_MLP = mean_absolute_error(y_test2, y_predicted_MLP)
MSE_MLP = mean_squared_error(y_test2, y_predicted_MLP)
MAPE_MLP = mean_absolute_percentage_error(y_test2, y_predicted_MLP)

r_PREG2= r_score(y_test, y_predicted_PREG2)
MAE_PREG2= mean_absolute_error(y_test, y_predicted_PREG2)
MSE_PREG2= mean_squared_error(y_test, y_predicted_PREG2)
MAPE_PREG2= np.mean(mean_absolute_percentage_error(y_test, y_predicted_PREG2))

r_PREG3= r_score(y_test, y_predicted_PREG3)
MAE_PREG3= mean_absolute_error(y_test, y_predicted_PREG3)
MSE_PREG3= mean_squared_error(y_test, y_predicted_PREG3)
MAPE_PREG3= np.mean(mean_absolute_percentage_error(y_test, y_predicted_PREG3))

errors = [{'Model Name': 'Random Forest Regression', 'R2': r_RF, 'MAE': MAE_RF, 'MSE': MSE_RF, 'MAPE (%)': np.mean(MAPE_RF), 'Median Error (%)': statistics.median(MAPE_RF)},
          {'Model Name': 'Linear Regression', 'R2': r_LREG, 'MAE': MAE_LREG, 'MSE': MSE_LREG, 'MAPE (%)': np.mean(MAPE_LREG), 'Median Error (%)': statistics.median(MAPE_LREG)},
          {'Model Name': 'Ridge Regression', 'R2': r_RIDGE, 'MAE': MAE_RIDGE, 'MSE': MSE_RIDGE, 'MAPE (%)': np.mean(MAPE_RIDGE), 'Median Error (%)': statistics.median(MAPE_RIDGE)},
          {'Model Name': 'XGBoost Regression', 'R2': r_XGB, 'MAE': MAE_XGB, 'MSE': MSE_XGB, 'MAPE (%)': np.mean(MAPE_XGB), 'Median Error (%)': statistics.median(MAPE_XGB)},
          {'Model Name': 'Lasso Regression', 'R2': r_LASSO, 'MAE': MAE_LASSO, 'MSE': MSE_LASSO, 'MAPE (%)': np.mean(MAPE_LASSO), 'Median Error (%)': statistics.median(MAPE_LASSO)},
          {'Model Name': 'Support Vector Machine', 'R2': r_svr_rbf, 'MAE': MAE_svr_rbf, 'MSE': MSE_svr_rbf, 'MAPE (%)': np.mean(MAPE_svr_rbf), 'Median Error (%)': statistics.median(MAPE_svr_rbf)},
          {'Model Name': 'Multi-layer Perceptron', 'R2': r_MLP, 'MAE': MAE_MLP, 'MSE': MSE_MLP, 'MAPE (%)': np.mean(MAPE_MLP), 'Median Error (%)': statistics.median(MAPE_MLP)}]

df_estimationerrors = pd.DataFrame(errors)
df_estimationerrors= df_estimationerrors.sort_values(by=['Median Error (%)'])
print(df_estimationerrors)
#df_estimationerrors.to_csv("errors-test5.csv")

#Boxplot 
import matplotlib.pyplot as plt 

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)
data_to_plot = [MAPE_RF,MAPE_XGB, MAPE_svr_rbf, MAPE_MLP,MAPE_LASSO,MAPE_RIDGE,MAPE_LREG]


#Principal Component Analysis
features = dfML.columns[:-1]
importances = randreg.feature_importances_
indices = np.argsort(importances)
plt.figure(3) #the axis number
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.savefig('Feature Importance-filtered2.png', 
              bbox_inches='tight', dpi = 500,figsize=(8,6))
df_estimationerrors.to_csv("errors-filtered.csv")

predictions = [
    {'Test Data1':y_test},
    {'Test Data2':y_test2},
    {'Random Forest Regression' : y_predicted_RF},
    {'Linear Regression' : y_predicted_LREG},
    {'XGBoost Regression': y_predicted_XGB },
    {'Lasso Regression':y_predicted_LASSO},
    {'Support Vector Machine': y_predicted_svr_rbf},
    {'Multi-layer Perceptron': y_predicted_MLP}]

predictions = np.array([y_test,y_test2,y_predicted_RF,y_predicted_LREG,y_predicted_XGB,y_predicted_LASSO,y_predicted_svr_rbf,y_predicted_MLP])
df_predictions = pd.DataFrame(data = predictions, index =["y_test","y_test2","y_predicted_RF","y_predicted_LREG","y_predicted_XGB","y_predicted_LASSO","y_predicted_svr_rbf","y_predicted_MLP"])
df_predictions.to_csv("predictionresults-test-filtered2.csv")