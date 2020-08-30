In this article, I will explain how to develop an Automated Supervised Machine
Learning Regression program, which automatically tunes the hyperparameters and
prints out the final results as tables, graphs and boxplots.

I always like to keep my libraries together, hence I import all of them at once
in the beginning of the code.

```{.python .input  n=17}
import pandas as pd
import numpy as np
import statistics
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```

Lets import our dataset and define the features and the predictors.

```{.python .input  n=2}
#Importing the datasets-------------------------------------------------------
print ("Importing datasets")
df = pd.read_csv('DATA.csv')  
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Importing datasets\n"
 }
]
```

Let's take a look at how the dataset looks. I like using df.describe() function
to have some statistics about each column.

```{.python .input  n=3}
df.describe()
```

```{.json .output n=3}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>A</th>\n      <th>B</th>\n      <th>C</th>\n      <th>D</th>\n      <th>E</th>\n      <th>F</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>27.000000</td>\n      <td>27.000000</td>\n      <td>27.000000</td>\n      <td>27.000000</td>\n      <td>27.000000</td>\n      <td>27.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>12.438889</td>\n      <td>13.240741</td>\n      <td>722.518519</td>\n      <td>19.259259</td>\n      <td>12.296296</td>\n      <td>6883.202781</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>0.181518</td>\n      <td>1.476550</td>\n      <td>124.861884</td>\n      <td>1.631248</td>\n      <td>2.267069</td>\n      <td>1316.064353</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>12.050000</td>\n      <td>11.500000</td>\n      <td>496.000000</td>\n      <td>16.000000</td>\n      <td>9.000000</td>\n      <td>4032.191235</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>12.350000</td>\n      <td>12.000000</td>\n      <td>632.500000</td>\n      <td>18.000000</td>\n      <td>11.000000</td>\n      <td>5955.425767</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>12.400000</td>\n      <td>13.000000</td>\n      <td>720.000000</td>\n      <td>19.000000</td>\n      <td>12.000000</td>\n      <td>6654.906860</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>12.525000</td>\n      <td>14.250000</td>\n      <td>832.000000</td>\n      <td>20.000000</td>\n      <td>14.000000</td>\n      <td>8137.799244</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>12.800000</td>\n      <td>17.000000</td>\n      <td>885.000000</td>\n      <td>23.000000</td>\n      <td>18.000000</td>\n      <td>9241.557194</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "               A          B           C          D          E            F\ncount  27.000000  27.000000   27.000000  27.000000  27.000000    27.000000\nmean   12.438889  13.240741  722.518519  19.259259  12.296296  6883.202781\nstd     0.181518   1.476550  124.861884   1.631248   2.267069  1316.064353\nmin    12.050000  11.500000  496.000000  16.000000   9.000000  4032.191235\n25%    12.350000  12.000000  632.500000  18.000000  11.000000  5955.425767\n50%    12.400000  13.000000  720.000000  19.000000  12.000000  6654.906860\n75%    12.525000  14.250000  832.000000  20.000000  14.000000  8137.799244\nmax    12.800000  17.000000  885.000000  23.000000  18.000000  9241.557194"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let's define the features as X and the column we want to predict (column F) as
y.

```{.python .input  n=4}
X = df.iloc[:,:-1].values 
y = df.iloc[:,5].values
```

This defines X as all the values except the last column (columns A,B,C,D,E), and
y as the last column (column numbers start from zero, hence: 0 - A, 1 - B, 2 -
C, 3 - D,4 - E, 5 -F).

For some algorithms we might want to use X as preprocessed (normalized) values
(X2). This mostly provides higher accuracies for algorithms such as Multi Layer
Perceptron (MLP) or Support Vector Machines (SVM). Hence, for the rest of the
program, X2 will be used for non-linear ML algorithms such as random forest,
XGBoost, MLP, SVM. However, X will be used for polynomial regressions and linear
regression to evaluate the regression constants easier.

```{.python .input  n=5}
X2 = preprocessing.scale(X)
```

We have to split our dataset as train and test data. For this we can use
train_test_split by sklearn.model_selection. We will do this for both X and X2.
Test size of 0.20 means that 20% of the data will be used as test data and 80%
of the data will be used for training.

```{.python .input  n=6}
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.20)
X_train2, X_test2, y_train2, y_test2= train_test_split(X2,y,test_size = 0.20)

```

We might not always want to tune the parameters of models, or only tune for some
models. For this I have defined basic inputs. When they are set to "yes", the
program will perform the tuning.

```{.python .input  n=7}
#Inputs------------------------------------------------------------------------
randomforestparametertuning = "no" #yes or no
XGboostparametertuning      = "no" #yes or no
SVMparametertuning          = "no" #yes or no
MLPparametertuning          = "no" #yes or no
```

The first one is training, testing and tuning the random forest regression. The
values of param_grid might be updated regarding the problem (i.e., some problems
might require higher values of n_estimators, while some might require lower
ranges).

```{.python .input  n=8}
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
```

The second one is training, testing and tuning the XGBoost regression. The
values of grid might be updated regarding the problem (i.e., some problems might
require higher values of n_estimators, while some might require lower ranges).

```{.python .input  n=9}
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
```

The third one is training, testing and tuning the SVM regression. The values of
C_range or gamma_range might be updated regarding the problem.

```{.python .input  n=10}
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
```

The fourth one is training, testing and tuning the MLP algorithm. The values of
param_grid might be updated regarding the problem.

```{.python .input  n=11}
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

```

The below commands provide a summary of the best parameters obtained from the
GridSearch of all these four algortihms.

Next thing, we will be fitting 9 different algortihms to our data to see which
one performs the best. These are namely: multi linear regression, ridge
regression, lasso regression, polynomial regression (degree=2), polynomial
regression (degree=3), random forest regression (with the best parameters
obtained from GridSearch), XGBoost regression (with the best parameters obtained
from GridSearch), SVM regression (with the best parameters obtained from
GridSearch) and MLP (with the best parameters obtained from GridSearch).

```{.python .input  n=12}
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
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Fit multilinear regression\nFit ridge regression\nFit random forest regression\nFit XGBoost regression\nFit Lasso regression\nFit SVR RBF regression\nFit Multi-layer Perceptron regressor\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Users\\Admin\\anaconda3\\lib\\site-packages\\sklearn\\neural_network\\_multilayer_perceptron.py:571: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.\n  % self.max_iter, ConvergenceWarning)\n"
 },
 {
  "data": {
   "text/plain": "MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,\n             beta_2=0.999, early_stopping=False, epsilon=1e-08,\n             hidden_layer_sizes=(100,), learning_rate='constant',\n             learning_rate_init=0.001, max_fun=15000, max_iter=200,\n             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,\n             power_t=0.5, random_state=None, shuffle=True, solver='adam',\n             tol=0.0001, validation_fraction=0.1, verbose=False,\n             warm_start=False)"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

For the error analysis, we are using four different statistics. The first one is
r_score, which is the r2 (coefficient of determination) of the test data and the
predicted data. The second is MAE = Mean Absolute Error, the third one is MSE =
Mean Squared Error and the third one is MAPE = Mean Absolute Percentage Error.
The MAE and MSE calculations come directly from sklearn.metrics /
mean_absolute_error and mean_squared_error. For the r_score and MAPE, we are
defining below functions.

```{.python .input  n=13}
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
    return r_squared
```

Now that the error statistics are defined, lets predict the predicted values by
the algorithm and calculate the errors.

```{.python .input  n=14}
y_predicted_RF = randreg.predict(X_test2)
y_predicted_LREG = linreg.predict(X_test)
y_predicted_RIDGE = ridgeReg.predict(X_test)
y_predicted_XGB = XGBreg.predict(X_test2)
y_predicted_LASSO = lassoreg.predict(X_test)
y_predicted_svr_rbf = svr_rbf.predict(X_test2)
y_predicted_MLP = MLP.predict(X_test2)

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

errors = [{'Model Name': 'Random Forest Regression', 'R2': r_RF, 'MAE': MAE_RF, 'MSE': MSE_RF, 'MAPE (%)': np.mean(MAPE_RF), 'Median Error (%)': statistics.median(MAPE_RF)},
          {'Model Name': 'Linear Regression', 'R2': r_LREG, 'MAE': MAE_LREG, 'MSE': MSE_LREG, 'MAPE (%)': np.mean(MAPE_LREG), 'Median Error (%)': statistics.median(MAPE_LREG)},
          {'Model Name': 'Ridge Regression', 'R2': r_RIDGE, 'MAE': MAE_RIDGE, 'MSE': MSE_RIDGE, 'MAPE (%)': np.mean(MAPE_RIDGE), 'Median Error (%)': statistics.median(MAPE_RIDGE)},
          {'Model Name': 'XGBoost Regression', 'R2': r_XGB, 'MAE': MAE_XGB, 'MSE': MSE_XGB, 'MAPE (%)': np.mean(MAPE_XGB), 'Median Error (%)': statistics.median(MAPE_XGB)},
          {'Model Name': 'Lasso Regression', 'R2': r_LASSO, 'MAE': MAE_LASSO, 'MSE': MSE_LASSO, 'MAPE (%)': np.mean(MAPE_LASSO), 'Median Error (%)': statistics.median(MAPE_LASSO)},
          {'Model Name': 'Support Vector Machine', 'R2': r_svr_rbf, 'MAE': MAE_svr_rbf, 'MSE': MSE_svr_rbf, 'MAPE (%)': np.mean(MAPE_svr_rbf), 'Median Error (%)': statistics.median(MAPE_svr_rbf)},
          {'Model Name': 'Multi-layer Perceptron', 'R2': r_MLP, 'MAE': MAE_MLP, 'MSE': MSE_MLP, 'MAPE (%)': np.mean(MAPE_MLP), 'Median Error (%)': statistics.median(MAPE_MLP)}]

df_estimationerrors = pd.DataFrame(errors)
df_estimationerrors= df_estimationerrors.sort_values(by=['Median Error (%)'])
```

Let's take a look at our error table:

```{.python .input  n=15}
print(df_estimationerrors)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "                 Model Name         R2          MAE           MSE   MAPE (%)  \\\n1         Linear Regression   0.988445   109.671023  1.941709e+04   1.427712   \n4          Lasso Regression   0.988437   109.709483  1.943049e+04   1.428133   \n2          Ridge Regression   0.984014   130.192410  2.686210e+04   1.620047   \n5    Support Vector Machine  -0.055287   742.351076  8.867127e+05  13.179532   \n3        XGBoost Regression   0.465497   607.003915  4.491201e+05   9.098057   \n0  Random Forest Regression   0.387519   710.505778  5.146419e+05  11.476428   \n6    Multi-layer Perceptron -49.154381  6426.365156  4.214259e+07  99.851249   \n\n   Median Error (%)  \n1          1.464946  \n4          1.465053  \n2          1.632919  \n5          8.310402  \n3          9.071413  \n0         10.304616  \n6         99.869765  \n"
 }
]
```

Moreover, perform a principal component analysis (PCA) using the random forest
regression results:

```{.python .input  n=18}
#Principal Component Analysis
features = df.columns[:-1]
importances = randreg.feature_importances_
indices = np.argsort(importances)
plt.figure(3) #the axis number
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.savefig('Feature Importance.png', 
              bbox_inches='tight', dpi = 500)

df_estimationerrors.to_csv("errors.csv")
```

```{.json .output n=18}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```
