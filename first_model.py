# Import

import pandas as pd
import numpy as np
from statistics import mean

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Import data
train = pd.read_csv('train.csv', header=0, sep=",", encoding="ISO-8859-1")
test = pd.read_csv('test.csv', header=0, sep=",", encoding="ISO-8859-1")

target = train['relevance']
piv_train = train.shape[0]
df = pd.read_csv('dataframe.csv', header=0, sep=",", encoding="ISO-8859-1")

df_model = df
for col in df_model.columns:
    df_model[col].fillna(value=df_model[col].mean(), inplace=True)

X = df_model.iloc[0:piv_train].values
y = target.values

# Best parameters GridSearch
xgb = XGBRegressor(objective='reg:squarederror', seed=42)

param_grid = {'n_estimators' : [100, 250, 500],
              'colsample_bytree' : [0.6, 0.8, 1.0],
              'subsample' : [0.6, 0.8, 1],
              'min_child_weight' : [1, 2],
              'max_depth' : [3, 5, 7],
              'gamma' : [0, 0.1, 1]}

n_splits = 10
CV_xgb = RandomizedSearchCV(estimator=xgb,
                            param_distributions=param_grid,
                            cv=n_splits,
                            n_iter=50, 
                            scoring='neg_mean_squared_error')

CV_xgb.fit(X, y)
report(CV_xgb.cv_results_)

# Model with rank: 1
# Mean validation score: -0.219 (std: 0.004)
# Parameters: {'subsample': 1, 'n_estimators': 100, 'min_child_weight': 2, 'max_depth': 7, 'gamma': 0.1, 'colsample_bytree': 1.0}

# Model with rank: 2
# Mean validation score: -0.219 (std: 0.004)
# Parameters: {'subsample': 1, 'n_estimators': 100, 'min_child_weight': 2, 'max_depth': 7, 'gamma': 0, 'colsample_bytree': 1.0}

# Model with rank: 3
# Mean validation score: -0.220 (std: 0.004)
# Parameters: {'subsample': 0.8, 'n_estimators': 100, 'min_child_weight': 2, 'max_depth': 7, 'gamma': 1, 'colsample_bytree': 1.0}


X = df_model.iloc[0:piv_train].values
y = target.values
n_splits = 10

kf = KFold(n_splits=n_splits, random_state=2019)
rmse = []

for train_index, cv_index in kf.split(X, y):
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]

    xgb = XGBRegressor(n_estimators=CV_xgb.best_estimator_.n_estimators,
                       colsample_bytree=CV_xgb.best_estimator_.colsample_bytree, 
                       subsample=CV_xgb.best_estimator_.subsample,
                       min_child_weight=CV_xgb.best_estimator_.min_child_weight,
                       max_depth=CV_xgb.best_estimator_.max_depth,
                       gamma=CV_xgb.best_estimator_.gamma,
                       objective='reg:squarederror',
                       seed=42)
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_cv)
    mse = mean_squared_error(y_cv, y_pred)
    rmse.append(sqrt(mse))
    print("mse: ", mse, " RMSE: ", sqrt(mse))    

print('RMSE: ' + str(sum(rmse) / n_splits), " std : ", str(stdev(rmse)))

# mse:  0.21419101562050336  RMSE:  0.46280775233405863
# mse:  0.223653294898621  RMSE:  0.47291996669481084
# mse:  0.22359022486031924  RMSE:  0.47285328047959996
# mse:  0.2148441287738957  RMSE:  0.46351281403419226
# mse:  0.22040656315386137  RMSE:  0.4694747737140531
# mse:  0.21854502539893303  RMSE:  0.46748799492493176
# mse:  0.22183177753706154  RMSE:  0.470990209597887
# mse:  0.21801368355967649  RMSE:  0.4669193544496485
# mse:  0.21199266733479735  RMSE:  0.46042661449442446
# mse:  0.22412299306743497  RMSE:  0.47341629995959683
# RMSE: 0.4680809060683204  std :  0.004643130501752707

# Model
xgb = XGBRegressor(n_estimators=CV_xgb.best_estimator_.n_estimators,
                   colsample_bytree=CV_xgb.best_estimator_.colsample_bytree, 
                   subsample=CV_xgb.best_estimator_.subsample,
                   min_child_weight=CV_xgb.best_estimator_.min_child_weight,
                   max_depth=CV_xgb.best_estimator_.max_depth,
                   gamma=CV_xgb.best_estimator_.gamma,
                   objective='reg:squarederror',
                   seed=42)
xgb.fit(X, y)

test['relevance'] = xgb.predict(df_model.iloc[piv_train:df_model.shape[0]].values)
test[['id', 'relevance']].to_csv('xgb_submission.csv', index=False)

# Exploration résultats
print(min(test['relevance']))
print(max(test['relevance']))
print(mean(test['relevance']))

print(min(train['relevance']))
print(max(train['relevance']))
print(mean(train['relevance']))