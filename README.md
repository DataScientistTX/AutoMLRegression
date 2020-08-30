{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this article, I will explain how to develop an Automated Supervised Machine Learning Regression program, which automatically tunes the hyperparameters and prints out the final results as tables, graphs and boxplots."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I always like to keep my libraries together, hence I import all of them at once in the beginning of the code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import statistics\n",
    "from xgboost import XGBRegressor\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn import preprocessing\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.neural_network import MLPRegressor\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn import linear_model\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets import our dataset and define the features and the predictors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Importing datasets\n"
     ]
    }
   ],
   "source": [
    "#Importing the datasets-------------------------------------------------------\n",
    "print (\"Importing datasets\")\n",
    "df = pd.read_csv('DATA.csv')  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's take a look at how the dataset looks. I like using df.describe() function to have some statistics about each column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>A</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "      <th>D</th>\n",
       "      <th>E</th>\n",
       "      <th>F</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5102.000000</td>\n",
       "      <td>5102.000000</td>\n",
       "      <td>5102.000000</td>\n",
       "      <td>5102.000000</td>\n",
       "      <td>5102.000000</td>\n",
       "      <td>5102.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>10.009189</td>\n",
       "      <td>18.306174</td>\n",
       "      <td>715.321835</td>\n",
       "      <td>14.227754</td>\n",
       "      <td>9.535280</td>\n",
       "      <td>12.287221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.508632</td>\n",
       "      <td>3.607368</td>\n",
       "      <td>161.961775</td>\n",
       "      <td>4.902647</td>\n",
       "      <td>2.577904</td>\n",
       "      <td>3.869418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>7.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>261.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>9.012500</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>605.000000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9.600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>9.700000</td>\n",
       "      <td>18.500000</td>\n",
       "      <td>698.000000</td>\n",
       "      <td>13.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>10.550000</td>\n",
       "      <td>20.500000</td>\n",
       "      <td>814.750000</td>\n",
       "      <td>17.000000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>14.675000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>14.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>1245.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>29.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 A            B            C            D            E  \\\n",
       "count  5102.000000  5102.000000  5102.000000  5102.000000  5102.000000   \n",
       "mean     10.009189    18.306174   715.321835    14.227754     9.535280   \n",
       "std       1.508632     3.607368   161.961775     4.902647     2.577904   \n",
       "min       7.000000     1.000000   261.000000     2.000000     1.000000   \n",
       "25%       9.012500    16.000000   605.000000    11.000000     8.000000   \n",
       "50%       9.700000    18.500000   698.000000    13.000000     9.000000   \n",
       "75%      10.550000    20.500000   814.750000    17.000000    11.000000   \n",
       "max      14.000000    40.000000  1245.000000    38.000000    19.000000   \n",
       "\n",
       "                 F  \n",
       "count  5102.000000  \n",
       "mean     12.287221  \n",
       "std       3.869418  \n",
       "min       1.000000  \n",
       "25%       9.600000  \n",
       "50%      12.000000  \n",
       "75%      14.675000  \n",
       "max      29.000000  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's define the features as X and the column we want to predict (column D) as y. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.iloc[:,:-1].values \n",
    "y = df.iloc[:,5].values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This defines X as all the values except the last column (columns A,B,C,D,E), and y as the last column (column numbers start from zero, hence: 0 - A, 1 - B, 2 - C, 3 - D,4 - E, 5 -F)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For some algorithms we might want to use X as preprocessed (normalized) values (X2). This mostly provides higher accuracies for algorithms such as Multi Layer Perceptron (MLP) or Support Vector Machines (SVM). Hence, for the rest of the program, X2 will be used for non-linear ML algorithms such as random forest, XGBoost, MLP, SVM. However, X will be used for polynomial regressions and linear regression to evaluate the regression constants easier."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "X2 = preprocessing.scale(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have to split our dataset as train and test data. For this we can use train_test_split by sklearn.model_selection. We will do this for both X and X2. Test size of 0.20 means that 20% of the data will be used as test data and 80% of the data will be used for training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.20)\n",
    "X_train2, X_test2, y_train2, y_test2= train_test_split(X2,y,test_size = 0.20)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We might not always want to tune the parameters of models, or only tune for some models. For this I have defined basic inputs. When they are set to \"yes\", the program will perform the tuning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Inputs------------------------------------------------------------------------\n",
    "randomforestparametertuning = \"no\" #yes or no\n",
    "XGboostparametertuning      = \"no\" #yes or no\n",
    "SVMparametertuning          = \"no\" #yes or no\n",
    "MLPparametertuning          = \"no\" #yes or no"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The first one is training, testing and tuning the random forest regression. The values of param_grid might be updated regarding the problem (i.e., some problems might require higher values of n_estimators, while some might require lower ranges)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "if randomforestparametertuning == \"yes\":\n",
    "    print (\"Performing gridsearch in random forest\")\n",
    "\n",
    "    # Create the parameter grid based on the results of random search \n",
    "    param_grid = {\n",
    "        'bootstrap': [True,False],\n",
    "        'max_depth': [40, 50, 60, 70],\n",
    "        'max_features': ['auto', 'sqrt'],\n",
    "        'min_samples_leaf': [1,2,3,],\n",
    "        'min_samples_split': [3, 4, 5,6,7],\n",
    "        'n_estimators': [50,100,150,200,250,300,350,400,500]\n",
    "        }\n",
    "    \n",
    "    # Create a based model\n",
    "    rf = RandomForestRegressor()\n",
    "    # Instantiate the grid search model\n",
    "    grid_search_RF = GridSearchCV(estimator = rf, param_grid = param_grid, \n",
    "                              cv = 3, n_jobs = -1, verbose = 2)\n",
    "    \n",
    "    # Fit the grid search to the data\n",
    "    grid_search_RF.fit(X_train2, y_train2)\n",
    "    print(\"Grid Search Best Parameters for Random Forest Regression\")\n",
    "    print (grid_search_RF.best_params_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The second one is training, testing and tuning the XGBoost regression. The values of grid might be updated regarding the problem (i.e., some problems might require higher values of n_estimators, while some might require lower ranges)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#XGBoost Parameter Tuning------------------------------------------------------\n",
    "if XGboostparametertuning == \"yes\":\n",
    "    print(\"XGBoost parameter tuning\")\n",
    "    # Create the parameter grid based on the results of random search \n",
    "    grid = {'colsample_bytree': [0.9,0.8,0.7],\n",
    "                    'gamma': [2,3,4,5],\n",
    "                    'learning_rate': [0.1,0.2,0.3],\n",
    "                    'max_depth': [8,9,10,11,12],\n",
    "                    'n_estimators': [150,200,250,300,350],\n",
    "                    'subsample': [0.8,0.9,1],\n",
    "                    'reg_alpha': [15,18,20],\n",
    "                    'min_child_weight':[3,4,5]}\n",
    "\n",
    "    # Create a based model\n",
    "    XGB = XGBRegressor()\n",
    "    # Instantiate the grid search model\n",
    "    grid_search_XGB = GridSearchCV(estimator = XGB, param_grid = grid, \n",
    "                              cv = 3, n_jobs = -1, verbose = 2)\n",
    "    \n",
    "    # Fit the grid search to the data\n",
    "    grid_search_XGB.fit(X_train2, y_train2)\n",
    "    print(\"Grid Search Best Parameters for XGBoost\")\n",
    "    print (grid_search_XGB.best_params_) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The third one is training, testing and tuning the SVM regression. The values of C_range or gamma_range might be updated regarding the problem."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#SVM Parameter Tuning----------------------------------------------------------\n",
    "if SVMparametertuning == \"yes\":\n",
    "    print(\"SVM parameter tuning\")\n",
    "\n",
    "    C_range = 10. ** np.arange(-3, 3)\n",
    "    gamma_range = 10. ** np.arange(-5, 3)\n",
    "    param_grid = dict(gamma=gamma_range, C=C_range)\n",
    "    svr_rbf = SVR()\n",
    "    # Instantiate the grid search model\n",
    "    grid_search_svm = GridSearchCV(estimator = svr_rbf, param_grid = param_grid, \n",
    "                              cv = 3, n_jobs = -1, verbose = 2)\n",
    "    # Fit the grid search to the data\n",
    "    grid_search_svm.fit(X_train2, y_train2)\n",
    "    print(\"Grid Search Best Parameters for SVM\")\n",
    "    print (grid_search_svm.best_params_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The fourth one is training, testing and tuning the MLP algorithm. The values of param_grid might be updated regarding the problem."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "if MLPparametertuning == \"yes\":\n",
    "    print(\"MLP parameter tuning\")\n",
    "\n",
    "    param_grid = {\n",
    "        'hidden_layer_sizes': [50,100,150,200,250,300,350],\n",
    "        'activation': ['identity','logistic','tanh','relu'],\n",
    "        'solver': ['lbfgs', 'sgd','adam'],\n",
    "        'learning_rate': ['constant','invscaling','adaptive']}\n",
    "    MLP = MLPRegressor()\n",
    "    # Instantiate the grid search model\n",
    "    grid_search_MLP = GridSearchCV(estimator = MLP, param_grid = param_grid, \n",
    "                              cv = 3, n_jobs = -1, verbose = 2)\n",
    "    \n",
    "    # Fit the grid search to the data\n",
    "    grid_search_MLP.fit(X_train2, y_train2)\n",
    "    print(\"Grid Search Best Parameters for MLP\")\n",
    "    print (grid_search_MLP.best_params_)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The below commands provide a summary of the best parameters obtained from the GridSearch of all these four algortihms."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next thing, we will be fitting 9 different algortihms to our data to see which one performs the best. These are namely: multi linear regression, ridge regression, lasso regression, polynomial regression (degree=2), polynomial regression (degree=3), random forest regression (with the best parameters obtained from GridSearch), XGBoost regression (with the best parameters obtained from GridSearch), SVM regression (with the best parameters obtained from GridSearch) and MLP (with the best parameters obtained from GridSearch)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fit multilinear regression\n",
      "Fit ridge regression\n",
      "Fit random forest regression\n",
      "Fit XGBoost regression\n",
      "[18:07:04] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.\n",
      "Fit Lasso regression\n",
      "Fit SVR RBF regression\n",
      "Fit Multi-layer Perceptron regressor\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\anaconda3\\lib\\site-packages\\sklearn\\neural_network\\_multilayer_perceptron.py:571: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.\n",
      "  % self.max_iter, ConvergenceWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,\n",
       "             beta_2=0.999, early_stopping=False, epsilon=1e-08,\n",
       "             hidden_layer_sizes=(100,), learning_rate='constant',\n",
       "             learning_rate_init=0.001, max_fun=15000, max_iter=200,\n",
       "             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,\n",
       "             power_t=0.5, random_state=None, shuffle=True, solver='adam',\n",
       "             tol=0.0001, validation_fraction=0.1, verbose=False,\n",
       "             warm_start=False)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fitting multi linear regression to data---------------------------------------\n",
    "print (\"Fit multilinear regression\")\n",
    "linreg = LinearRegression()\n",
    "linreg.fit(X_train,y_train)\n",
    "\n",
    "#Fitting ridge regression to data----------------------------------------------\n",
    "print (\"Fit ridge regression\")\n",
    "ridgeReg = Ridge(alpha=0.05, normalize=True)\n",
    "ridgeReg.fit(X_train,y_train)\n",
    "\n",
    "#Fitting random forest regression to data--------------------------------------\n",
    "print (\"Fit random forest regression\")\n",
    "try:\n",
    "    randreg = RandomForestRegressor(**grid_search_RF.best_params_)\n",
    "except:\n",
    "    randreg = RandomForestRegressor()\n",
    "randreg.fit(X_train2,y_train2)\n",
    "\n",
    "#Fitting XGboost regression to data--------------------------------------------\n",
    "print (\"Fit XGBoost regression\")\n",
    "try:\n",
    "    XGBreg = XGBRegressor(**grid_search_XGB.best_params_)\n",
    "except: \n",
    "    XGBreg = XGBRegressor()\n",
    "XGBreg.fit(X_train2, y_train2)\n",
    "\n",
    "#Fitting LASSO regression to data----------------------------------------------\n",
    "print (\"Fit Lasso regression\")\n",
    "lassoreg = Lasso(alpha=0.01, max_iter=10e5)\n",
    "lassoreg.fit(X_train, y_train)\n",
    "\n",
    "#Support Vector Machines-------------------------------------------------------\n",
    "print (\"Fit SVR RBF regression\")\n",
    "try:\n",
    "    svr_rbf = SVR(**grid_search_svm.best_params_)\n",
    "except:\n",
    "    svr_rbf = SVR()\n",
    "    svr_rbf.fit(X_train2, y_train2)\n",
    "\n",
    "#MLP Regressor-----------------------------------------------------------------\n",
    "print (\"Fit Multi-layer Perceptron regressor\")\n",
    "try:\n",
    "    MLP = MLPRegressor(**grid_search_MLP.best_params_)\n",
    "except:\n",
    "    MLP = MLPRegressor()    \n",
    "MLP.fit(X_train2, y_train2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the error analysis, we are using four different statistics. The first one is r_score, which is the r2 (coefficient of determination) of the test data and the predicted data. The second is MAE = Mean Absolute Error, the third one is MSE = Mean Squared Error and the third one is MAPE = Mean Absolute Percentage Error. The MAE and MSE calculations come directly from sklearn.metrics / mean_absolute_error and mean_squared_error. For the r_score and MAPE, we are defining below functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define the missing sklearn.metrics parameter of mean absolute percentage error\n",
    "def mean_absolute_percentage_error(y_true, y_pred): \n",
    "    y_true, y_pred = np.array(y_true), np.array(y_pred)\n",
    "    return (np.abs((y_true - y_pred) / y_true)) * 100\n",
    "\n",
    "def r_score(y_true, y_pred): \n",
    "    y_true, y_pred = np.array(y_true), np.array(y_pred)\n",
    "    residuals = y_true- y_pred\n",
    "    ss_res = np.sum(residuals**2)\n",
    "    ss_tot = np.sum((y_true-np.mean(y_true))**2)\n",
    "    r_squared = 1 - (ss_res / ss_tot)    \n",
    "    return r_squared"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that the error statistics are defined, lets predict the predicted values by the algorithm and calculate the errors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_predicted_RF = randreg.predict(X_test2)\n",
    "y_predicted_LREG = linreg.predict(X_test)\n",
    "y_predicted_RIDGE = ridgeReg.predict(X_test)\n",
    "y_predicted_XGB = XGBreg.predict(X_test2)\n",
    "y_predicted_LASSO = lassoreg.predict(X_test)\n",
    "y_predicted_svr_rbf = svr_rbf.predict(X_test2)\n",
    "y_predicted_MLP = MLP.predict(X_test2)\n",
    "\n",
    "r_RF=  r_score(y_test2, y_predicted_RF)\n",
    "MAE_RF = mean_absolute_error(y_test2, y_predicted_RF)\n",
    "MSE_RF = mean_squared_error(y_test2, y_predicted_RF)\n",
    "MAPE_RF = mean_absolute_percentage_error(y_test2, y_predicted_RF)\n",
    "\n",
    "r_LREG=  r_score(y_test, y_predicted_LREG)\n",
    "MAE_LREG = mean_absolute_error(y_test, y_predicted_LREG)\n",
    "MSE_LREG = mean_squared_error(y_test, y_predicted_LREG)\n",
    "MAPE_LREG = mean_absolute_percentage_error(y_test, y_predicted_LREG)\n",
    "\n",
    "r_RIDGE=  r_score(y_test, y_predicted_RIDGE)\n",
    "MAE_RIDGE = mean_absolute_error(y_test, y_predicted_RIDGE)\n",
    "MSE_RIDGE = mean_squared_error(y_test, y_predicted_RIDGE)\n",
    "MAPE_RIDGE = mean_absolute_percentage_error(y_test, y_predicted_RIDGE)\n",
    "\n",
    "r_XGB=  r_score(y_test2, y_predicted_XGB)\n",
    "MAE_XGB = mean_absolute_error(y_test2, y_predicted_XGB)\n",
    "MSE_XGB = mean_squared_error(y_test2, y_predicted_XGB)\n",
    "MAPE_XGB = mean_absolute_percentage_error(y_test2, y_predicted_XGB)\n",
    "\n",
    "r_LASSO=  r_score(y_test, y_predicted_LASSO)\n",
    "MAE_LASSO = mean_absolute_error(y_test, y_predicted_LASSO)\n",
    "MSE_LASSO = mean_squared_error(y_test, y_predicted_LASSO)\n",
    "MAPE_LASSO = mean_absolute_percentage_error(y_test, y_predicted_LASSO)\n",
    "\n",
    "r_svr_rbf=  r_score(y_test2, y_predicted_svr_rbf)\n",
    "MAE_svr_rbf = mean_absolute_error(y_test2, y_predicted_svr_rbf)\n",
    "MSE_svr_rbf = mean_squared_error(y_test2, y_predicted_svr_rbf)\n",
    "MAPE_svr_rbf = mean_absolute_percentage_error(y_test2, y_predicted_svr_rbf)\n",
    "\n",
    "r_MLP=  r_score(y_test2, y_predicted_MLP)\n",
    "MAE_MLP = mean_absolute_error(y_test2, y_predicted_MLP)\n",
    "MSE_MLP = mean_squared_error(y_test2, y_predicted_MLP)\n",
    "MAPE_MLP = mean_absolute_percentage_error(y_test2, y_predicted_MLP)\n",
    "\n",
    "errors = [{'Model Name': 'Random Forest Regression', 'R2': r_RF, 'MAE': MAE_RF, 'MSE': MSE_RF, 'MAPE (%)': np.mean(MAPE_RF), 'Median Error (%)': statistics.median(MAPE_RF)},\n",
    "          {'Model Name': 'Linear Regression', 'R2': r_LREG, 'MAE': MAE_LREG, 'MSE': MSE_LREG, 'MAPE (%)': np.mean(MAPE_LREG), 'Median Error (%)': statistics.median(MAPE_LREG)},\n",
    "          {'Model Name': 'Ridge Regression', 'R2': r_RIDGE, 'MAE': MAE_RIDGE, 'MSE': MSE_RIDGE, 'MAPE (%)': np.mean(MAPE_RIDGE), 'Median Error (%)': statistics.median(MAPE_RIDGE)},\n",
    "          {'Model Name': 'XGBoost Regression', 'R2': r_XGB, 'MAE': MAE_XGB, 'MSE': MSE_XGB, 'MAPE (%)': np.mean(MAPE_XGB), 'Median Error (%)': statistics.median(MAPE_XGB)},\n",
    "          {'Model Name': 'Lasso Regression', 'R2': r_LASSO, 'MAE': MAE_LASSO, 'MSE': MSE_LASSO, 'MAPE (%)': np.mean(MAPE_LASSO), 'Median Error (%)': statistics.median(MAPE_LASSO)},\n",
    "          {'Model Name': 'Support Vector Machine', 'R2': r_svr_rbf, 'MAE': MAE_svr_rbf, 'MSE': MSE_svr_rbf, 'MAPE (%)': np.mean(MAPE_svr_rbf), 'Median Error (%)': statistics.median(MAPE_svr_rbf)},\n",
    "          {'Model Name': 'Multi-layer Perceptron', 'R2': r_MLP, 'MAE': MAE_MLP, 'MSE': MSE_MLP, 'MAPE (%)': np.mean(MAPE_MLP), 'Median Error (%)': statistics.median(MAPE_MLP)}]\n",
    "\n",
    "df_estimationerrors = pd.DataFrame(errors)\n",
    "df_estimationerrors= df_estimationerrors.sort_values(by=['Median Error (%)'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's look at how our error table looks like:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                 Model Name        R2       MAE        MSE   MAPE (%)  \\\n",
      "0  Random Forest Regression  0.394357  2.213911   9.303641  22.560231   \n",
      "5    Support Vector Machine  0.200438  2.663320  12.282532  26.982137   \n",
      "3        XGBoost Regression  0.249081  2.641985  11.535310  27.109934   \n",
      "6    Multi-layer Perceptron  0.202111  2.736434  12.256838  27.674362   \n",
      "1         Linear Regression  0.140352  2.782276  13.082425  27.047742   \n",
      "4          Lasso Regression  0.140528  2.782735  13.079752  27.052476   \n",
      "2          Ridge Regression  0.138746  2.786991  13.106871  27.126285   \n",
      "\n",
      "   Median Error (%)  \n",
      "0         13.560000  \n",
      "5         16.776997  \n",
      "3         17.611227  \n",
      "6         18.078627  \n",
      "1         18.147954  \n",
      "4         18.255871  \n",
      "2         18.389583  \n"
     ]
    }
   ],
   "source": [
    "print(df_estimationerrors)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also visualize our erros by BoxPlots:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Boxplot \n",
    "import matplotlib.pyplot as plt \n",
    "\n",
    "# Create a figure instance\n",
    "fig = plt.figure(1, figsize=(9, 6))\n",
    "\n",
    "# Create an axes instance\n",
    "ax = fig.add_subplot(111)\n",
    "data_to_plot = [MAPE_RF,MAPE_MLP,MAPE_svr_rbf,MAPE_XGB,MAPE_RIDGE,MAPE_LASSO,MAPE_LREG]\n",
    "\n",
    "# Create the boxplot\n",
    "bp = ax.boxplot(data_to_plot)\n",
    "## add patch_artist=True option to ax.boxplot() \n",
    "## to get fill color\n",
    "bp = ax.boxplot(data_to_plot, patch_artist=True)\n",
    "\n",
    "## change outline color, fill color and linewidth of the boxes\n",
    "for box in bp['boxes']:\n",
    "    # change outline color\n",
    "    box.set( color='#7570b3', linewidth=2)\n",
    "    # change fill color\n",
    "    box.set( facecolor = '#1b9e77' )\n",
    "\n",
    "## change color and linewidth of the whiskers\n",
    "for whisker in bp['whiskers']:\n",
    "    whisker.set(color='#7570b3', linewidth=2)\n",
    "\n",
    "## change color and linewidth of the caps\n",
    "for cap in bp['caps']:\n",
    "    cap.set(color='#7570b3', linewidth=2)\n",
    "\n",
    "## change color and linewidth of the medians\n",
    "for median in bp['medians']:\n",
    "    median.set(color='#b2df8a', linewidth=2)\n",
    "\n",
    "## change the style of fliers and their fill\n",
    "for flier in bp['fliers']:\n",
    "    flier.set(marker='o', color='#e7298a', alpha=0.5)\n",
    "              \n",
    "## Custom x-axis labels\n",
    "ax.set_xticklabels(['Random Forest',  'MLP','SVM', 'XGBoost','Ridge','Lasso','Multi Linear'])\n",
    "ax.set_ylim(0,100)\n",
    "ax.set_ylabel(\"Percentage Error (%)\")\n",
    "## Remove top axes and right axes ticks\n",
    "ax.get_xaxis().tick_bottom()\n",
    "ax.get_yaxis().tick_left()\n",
    "\n",
    "fig.savefig('boxplots.png', dpi=1000)\n",
    "fig.savefig('boxplots.pdf')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Moreover, perform a principal component analysis (PCA) using the random forest regression results:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Principal Component Analysis\n",
    "features = df.columns[:-1]\n",
    "importances = randreg.feature_importances_\n",
    "indices = np.argsort(importances)\n",
    "plt.figure(3) #the axis number\n",
    "plt.title('Feature Importance')\n",
    "plt.barh(range(len(indices)), importances[indices], color='b', align='center')\n",
    "plt.yticks(range(len(indices)), features[indices])\n",
    "plt.xlabel('Relative Importance')\n",
    "plt.savefig('Feature Importance.png', \n",
    "              bbox_inches='tight', dpi = 500,figsize=(8,6))\n",
    "\n",
    "df_estimationerrors.to_csv(\"errors.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
